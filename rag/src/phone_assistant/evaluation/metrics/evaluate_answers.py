"""Small, explicit-budget generation/judge run with cached raw answers and manual-label sheet."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from phone_assistant.evaluation.metrics.answer_evaluation import judge_answer
from phone_assistant.conversation.co_guiding import CoGuidingEngine
from phone_assistant.retrieval.search.dense_search import load_encoder
from phone_assistant.llm.gemini_client import generate_answer
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256
from phone_assistant.recommendation.recsys_service import RecommendationService


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/results_answers')
    parser.add_argument('--rejudge',type=Path,help='Rejudge saved answers, without regenerating them; use a new output')
    args=parser.parse_args()
    if not 1<=args.limit<=10:
        parser.error('limit must be 1..10; this run uses two modes and up to four API calls per query')
    if (args.output/'records.jsonl').exists():
        raise FileExistsError('Use a new --output to retain answers and judgments.')
    documents=read_jsonl(ROOT/'artifacts/runtime/catalog/documents.jsonl')
    index=json.loads((ROOT/'artifacts/runtime/catalog/bm25_index.json').read_text(encoding='utf-8'))
    engine=CoGuidingEngine(documents,index,RecommendationService())
    queries_path=ROOT/'artifacts/evaluation/queries.json'
    queries=json.loads(queries_path.read_text(encoding='utf-8'))[:args.limit]
    config=json.loads((ROOT/'artifacts/runtime/catalog/dense_index/manifest.json').read_text())['channels']['text']
    encoder=load_encoder(config['model'],revision=config['revision'])
    records=[]
    args.output.mkdir(parents=True,exist_ok=True)
    cases = read_jsonl(args.rejudge)[:args.limit*2] if args.rejudge else [
        {'query_id':item['id'],'query':item['query'],'mode':mode} for item in queries for mode in ('decoupled','coupled')]
    for case in cases:
            mode=case['mode']
            if args.rejudge:
                result={'results':case['contexts']}
                generation={'answer':case['answer'],'model':case.get('generator_model')}
            else:
                result=engine.run(case['query'],user_index=0,mode=mode,top_k=3)
                generation=generate_answer(case['query'],result['results'],include_images=True,timeout=25,retries=0) if result['results'] else {'answer':None,'error':'no_verified_evidence'}
            judgment=judge_answer(case['query'],generation['answer'],result['results'],encoder) if generation.get('answer') else {'status':'generation_failed','error':generation.get('error')}
            row={'query_id':case['query_id'],'query':case['query'],'mode':mode,'user_index':0,
                 'answer':generation.get('answer'),'generator_model':generation.get('model'),
                 'contexts':result['results'],'judgment':judgment}
            records.append(row)
            with (args.output/'records.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(case['query_id'],mode,judgment['status'],flush=True)
    summary={}
    for mode in ('decoupled','coupled'):
        selected=[row for row in records if row['mode']==mode]
        judged=[row['judgment'] for row in selected if row['judgment']['status']=='judged']
        summary[mode]={'generated':sum(bool(row.get('answer')) for row in selected),'requested':len(selected),'judged':len(judged)}
        for metric in ('faithfulness','context_unsupported_rate','answer_relevance_reverse_question_cosine','judge_coverage'):
            values=[row['metrics'][metric] for row in judged if row['metrics'][metric] is not None]
            summary[mode][metric]=sum(values)/len(values) if values else None
            summary[mode][metric+'_denominator']=len(values)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'queries_sha256':sha256(queries_path),
              'summary':summary,'limit':args.limit,'max_api_calls':(2 if args.rejudge else 4)*args.limit,
              'judge_configuration':{'thinking_budget':512,'response_mime_type':'application/json'},
              'rejudged_source_sha256':sha256(args.rejudge) if args.rejudge else None,
              'scope':'custom same-provider LLM judge; small diagnostic, not official Ragas or human labels',
              'hallucination_definition':'context_unsupported_rate among judged claims; unknown labels excluded with coverage reported',
              'records_sha256':sha256(args.output/'records.jsonl')}
    (args.output/'summary.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (args.output/'manual_claim_labels.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=('query_id','mode','claim','doc_id','quote','auto_supported','manual_supported','notes')
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in records:
            for claim in row['judgment'].get('claims',[]):
                writer.writerow({'query_id':row['query_id'],'mode':row['mode'],'claim':claim['text'],
                    'doc_id':claim.get('doc_id'),'quote':claim.get('quote'),'auto_supported':claim.get('supported'),
                    'manual_supported':'','notes':''})
    (args.output/'answer_evaluation_report.md').write_text(
        '# Answer evaluation diagnostic\n\n'+manifest['scope']+'\n\n'+
        'Hallucination proxy: '+manifest['hallucination_definition']+'\n\n'+
        '```json\n'+json.dumps(summary,indent=2)+'\n```\n\n'+
        'Generation/judge errors remain failures, never converted to zero hallucination. Manual claim labels are blank.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
