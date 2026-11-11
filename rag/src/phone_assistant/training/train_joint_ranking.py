"""Paired shared-ranking ablation; validation only checkpoint selection."""
from phone_assistant.common.project_layout import source_path
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from phone_assistant.recommendation.joint_ranking import JointGraphRanker
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_recsys import evaluate_vectors, exported

VARIANTS={'joint_control':(0.,0.),'joint_retrieval':(.1,0.),'joint_bidirectional':(.1,.02)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/results_joint_ranking')
    parser.add_argument('--models',type=Path,default=ROOT/'artifacts/models/joint')
    args=parser.parse_args()
    if args.epochs<=0 or len(set(args.seeds))!=len(args.seeds):
        parser.error('Positive epochs and distinct seeds required')
    if (args.output/'experiment.json').exists() or any((args.models/name/f'seed_{seed}').exists() for name in VARIANTS for seed in args.seeds):
        raise FileExistsError('Use new --output and --models directories')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=InteractionData(ROOT/'data/fixtures/recommendation')
    features,feature_manifest=load_features(ROOT/'artifacts/features/legacy',data)
    users=torch.tensor(data.pairs['train'][:,0].copy(),dtype=torch.long)
    positives=torch.tensor(data.pairs['train'][:,1].copy(),dtype=torch.long)
    args.output.mkdir(parents=True,exist_ok=True)
    runs=[]
    for name,(retrieval_weight,alignment_weight) in VARIANTS.items():
        for seed in args.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=JointGraphRanker(data,features,retrieval_weight=retrieval_weight,alignment_weight=alignment_weight)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            best_key,best_state,best_epoch=None,None,None
            history=[]
            for epoch in range(1,args.epochs+1):
                model.train();optimizer.zero_grad(set_to_none=True)
                negatives=torch.tensor(data.negatives(data.pairs['train'][:,0],rng),dtype=torch.long)
                losses=model.loss(users,positives,negatives)
                if not torch.isfinite(losses['total']):
                    raise ValueError('Nonfinite joint loss')
                losses['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                if epoch%10==0 or epoch==args.epochs:
                    validation,_=evaluate_vectors(*exported(model),data,'validation')
                    key=(validation['all']['ndcg@10'],validation['all']['hr@10'])
                    if best_key is None or key>best_key:
                        best_key,best_epoch=key,epoch
                        best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                    history.append({'epoch':epoch,'loss':{key:float(value.detach()) for key,value in losses.items()},'validation':validation})
            model.load_state_dict(best_state)
            vectors=exported(model)
            validation,_=evaluate_vectors(*vectors,data,'validation')
            test,ranks=evaluate_vectors(*vectors,data,'test')
            folder=args.models/name/f'seed_{seed}';folder.mkdir(parents=True)
            torch.save(model.state_dict(),folder/'checkpoint.pt')
            np.savez_compressed(folder/'embeddings.npz',users=vectors[0],items=vectors[1],bias=vectors[2],
                query_weight=model.query_projection.weight.detach().numpy(),query_bias=model.query_projection.bias.detach().numpy())
            runs.append({'method':name,'seed':seed,'best_epoch':best_epoch,'validation':validation,'test':test,
                'test_ranks':ranks,'loss_history':history,'train_seconds':None,
                'retrieval_weight':retrieval_weight,'alignment_weight':alignment_weight,
                'artifacts_sha256':{file:sha256(folder/file) for file in ('checkpoint.pt','embeddings.npz')}})
            (args.output/'completed_runs.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
            print(name,seed,'selected_epoch',best_epoch,'test_HR@10',test['all']['hr@10'],flush=True)
    summary={}
    for name in VARIANTS:
        selected=[run for run in runs if run['method']==name]
        summary[name]={key:{'mean':float(np.mean([run['test']['all'][key] for run in selected])),
                           'std':float(np.std([run['test']['all'][key] for run in selected],ddof=1)) if len(selected)>1 else 0.}
                       for key in ('hr@10','ndcg@10')}
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'summary':summary,'runs':runs,
        'split_manifest_sha256':sha256(data.directory/'manifest.json'),
        'feature_manifest_sha256':sha256(ROOT/'artifacts/features/legacy/manifest.json'),
        'feature_manifest':feature_manifest,
        'supervision':'positive TRAIN item metadata as proxy query; no held-out target features supplied as queries',
        'objective':'BPR + GCL + lambda_retrieval * metadata CE + lambda_alignment * JS(query ranking, user ranking)',
        'scope':'shared graph/retrieval training prototype; not conversational joint supervision; generator frozen',
        'source_sha256':{name:sha256(source_path(name)) for name in ('joint_ranking.py','train_joint_ranking.py')}}
    (args.output/'experiment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Shared ranking joint-loss prototype','',report['scope'],'',report['supervision'],'',
           '| Variant | HR@10 mean | Std | NDCG@10 mean | Std |','|---|---:|---:|---:|---:|']
    for name,metrics in summary.items():
        lines.append(f"| {name} | {metrics['hr@10']['mean']:.4f} | {metrics['hr@10']['std']:.4f} | {metrics['ndcg@10']['mean']:.4f} | {metrics['ndcg@10']['std']:.4f} |")
    lines.extend(['','Only 5 validation users / 58 test users. No claims of conversational recommendation improvement.',
                  'No test items/reviews are used as query supervision. Metadata inputs are catalog-alignment proxies, not observed conversations.',''])
    (args.output/'joint_ranking_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
