"""Researcher-authored functional scenarios, not a synthetic user study or training set."""
from datetime import datetime, timezone
import json

from phone_assistant.conversation.co_guiding import CoGuidingEngine
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl
from phone_assistant.recommendation.recsys_service import RecommendationService


def main():
    documents=read_jsonl(ROOT/'artifacts/runtime/catalog/documents.jsonl')
    index=json.loads((ROOT/'artifacts/runtime/catalog/bm25_index.json').read_text(encoding='utf-8'))
    engine=CoGuidingEngine(documents,index,RecommendationService())
    cases=[
        ('Samsung 128GB unlocked',{'brands':['samsung'],'storage_gb':128,'network':'unlocked'}),
        ('con 256GB thi sao',{'brands':['samsung'],'storage_gb':256,'network':'unlocked'}),
        ('first one battery',{'brands':['samsung'],'storage_gb':256}),
        ('new search iphone 13 128GB',{'brands':['apple'],'storage_gb':128,'model_terms':'iphone 13'}),
        ('Samsung',{'brands':['samsung'],'storage_gb':128}),
        ('new search nokia',{'brands':['nokia']}),
        ('new search Samsung duoi 5 trieu',{'brands':['samsung'],'max_price':5000000}),
        ('new search Samsung 8GB RAM 128GB',{'brands':['samsung'],'ram_gb':8,'storage_gb':128}),
    ]
    previous=None
    rows=[]
    for query,expected in cases:
        result=engine.run(query,user_index=0,previous_plan=previous,top_k=3)
        constraints=result['query_plan']['constraints']
        assert all(constraints.get(key)==value for key,value in expected.items()),(query,constraints)
        assert all(row['constraint_check']['eligible'] for row in result['results'])
        if 'max_price' in expected:
            assert result['abstention'],'Unknown VND pricing must not be invented'
        if query=='Samsung':
            assert 'model_terms' not in constraints
        if query=='new search nokia':
            assert 'storage_gb' not in constraints
        rows.append({'query':query,'expected_state':expected,'actual_state':constraints,
                     'verified_result_ids':[row['doc_id'] for row in result['results']],
                     'feed_ids':[row['doc_id'] for row in result['recommendations']],
                     'abstention':result['abstention'],'passed':True})
        previous=result['query_plan']
    out=ROOT/'artifacts/evaluation/results_conversations';out.mkdir(parents=True,exist_ok=True)
    record={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'8 researcher-authored functional cases; no LLM calls, not observed user dialogs, not a training set',
            'passed':len(rows),'cases':rows}
    (out/'functional_cases.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(record['scope']);print('Passed:',len(rows))


if __name__=='__main__':
    main()
