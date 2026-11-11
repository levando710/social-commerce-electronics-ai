"""Inference ablation on fixed query criteria; no invented user-query relevance labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from phone_assistant.conversation.co_guiding import CoGuidingEngine, MODES
from phone_assistant.evaluation.metrics.evaluate_retrieval import is_relevant, metrics_for_ranking
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256
from phone_assistant.recommendation.recsys_service import RecommendationService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/evaluation/results_co_guiding')
    parser.add_argument('--user-index', type=int, default=0)
    args = parser.parse_args()
    if (args.output/'ablation.json').exists():
        raise FileExistsError('Use a new --output; retain completed experiments.')
    documents = read_jsonl(ROOT/'artifacts/runtime/catalog/documents.jsonl')
    index = json.loads((ROOT/'artifacts/runtime/catalog/bm25_index.json').read_text(encoding='utf-8'))
    queries_path = ROOT/'artifacts/evaluation/queries.json'
    queries = json.loads(queries_path.read_text(encoding='utf-8'))
    service = RecommendationService()
    engine = CoGuidingEngine(documents,index,service)
    rows = []
    for query in queries:
        relevant = [doc['doc_id'] for doc in documents if is_relevant(doc,query.get('relevance',{}))]
        for mode in MODES:
            started = time.perf_counter()
            response = engine.run(query['query'],user_index=args.user_index,mode=mode,top_k=5)
            metrics = metrics_for_ranking([r['doc_id'] for r in response['results']],relevant,(1,5))
            rows.append({'query_id':query['id'],'mode':mode,'user_index':args.user_index,
                'metrics':metrics,'elapsed_ms':(time.perf_counter()-started)*1000,
                'feedback':response['feedback'],'result_ids':[r['doc_id'] for r in response['results']],
                'feed_ids':[r['doc_id'] for r in response['recommendations']]})
    summary = {}
    for mode in MODES:
        selected = [row for row in rows if row['mode']==mode]
        summary[mode] = {key:sum(row['metrics'][key] for row in selected)/len(selected)
                         for key in selected[0]['metrics']}
        summary[mode]['mean_ms'] = sum(row['elapsed_ms'] for row in selected)/len(selected)
    output = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'queries_sha256':sha256(queries_path),
        'scope':'mechanistic inference ablation with one fixed real offline profile; query assignment is not observed user behavior',
        'user_index':args.user_index,'summary':summary,'per_query':rows,
        'personalized_recsys_hr':None,
        'missing_supervision':'No observed user-query-item triples; do not fabricate HR/NDCG for conversational recommendation.',
        'model':'graph_multimodal_gcl seed 42; legacy candidate retrieval; fixed RRF preference weight 0.2 and evidence reward 0.03'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'ablation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    lines=['# Co-Guiding inference ablation','',output['scope'],'',
           '| Mode | P@5 | NDCG@5 | MRR@5 | Mean ms |','|---|---:|---:|---:|---:|']
    for mode,values in summary.items():
        lines.append(f"| {mode} | {values['precision@5']:.4f} | {values['ndcg@5']:.4f} | {values['mrr@5']:.4f} | {values['mean_ms']:.2f} |")
    lines.extend(['',output['missing_supervision'],
        'All modes use the same conversational query plan and conservative catalog constraints. Unknown required facts abstain.',
        'Differences demonstrate ranking mechanics on this query set, not a causal personalization improvement or learned joint objective.',''])
    (args.output/'ablation_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
