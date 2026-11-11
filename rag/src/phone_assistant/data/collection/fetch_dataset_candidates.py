"""Fetch bounded public benchmark candidates at pinned revisions, without running remote code."""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

ROOT=MODULE_ROOT
SIM_REPO='swap-uniba/ecommerce-user-sim'


def request_json(url):
    response=requests.get(url,timeout=40)
    response.raise_for_status()
    return response.json()


def fetch_file(job):
    root,relative,url,expected_size=job
    target=root/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_suffix(target.suffix+'.part')
        received=0
        with requests.get(url,stream=True,timeout=(15,90)) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    if chunk:
                        received+=len(chunk)
                        if received>max(expected_size+1024,1024*1024):
                            raise ValueError('Download larger than declared size: '+relative)
                        handle.write(chunk)
        if expected_size and received!=expected_size:
            raise ValueError('Size mismatch: '+relative)
        temporary.replace(target)
    content=target.read_bytes()
    if expected_size and len(content)!=expected_size:
        raise ValueError('Existing file has a size mismatch: '+relative)
    return {'path':relative,'source_url':url,'bytes':len(content),
            'sha256':hashlib.sha256(content).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/external')
    parser.add_argument('--max-mb',type=float,default=80.)
    parser.add_argument('--all-vogue-images',action='store_true',help='Also fetch the full 76-image VOGUE catalog')
    parser.add_argument('--sim-revision',help='Pinned Hugging Face commit; defaults to existing manifest or current revision')
    parser.add_argument('--vogue-revision',help='Pinned GitHub tree; defaults to existing manifest or main')
    args=parser.parse_args()
    if args.max_mb<=0:
        parser.error('max-mb must be positive')
    info=request_json(f'https://huggingface.co/api/datasets/{SIM_REPO}')
    old_manifest_path=args.output/'download_manifest.json'
    old=json.loads(old_manifest_path.read_text(encoding='utf-8')) if old_manifest_path.exists() else {}
    revision=args.sim_revision or old.get('ecommerce_user_sim',{}).get('revision') or info['sha']
    if old and revision!=old['ecommerce_user_sim']['revision']:
        raise ValueError('Use a new --output when changing the dataset revision')
    tree=request_json(f'https://huggingface.co/api/datasets/{SIM_REPO}/tree/{revision}?recursive=true&limit=1000')
    jobs=[]
    for record in tree:
        if record['type']!='file':
            continue
        name=record['path']
        corpus=(name.startswith(('corpus/dialogues/','corpus/users/','corpus/items/'))
                and name.endswith(('Electronics.parquet','Cell_Phones_and_Accessories.parquet')))
        evaluation=(name.startswith(('eval/users/','eval/items/','eval/catalog/')) and name.endswith('/Electronics.parquet'))
        personalwab=name.startswith('eval/personalwab/Electronics/')
        if name=='README.md' or corpus or evaluation or personalwab:
            jobs.append((args.output/'ecommerce_user_sim',name,
                f'https://huggingface.co/datasets/{SIM_REPO}/resolve/{revision}/{name}',record['size']))
    vogue_ref=args.vogue_revision or old.get('vogue',{}).get('revision') or 'main'
    vogue_tree=request_json(f'https://api.github.com/repos/D3Mlab/vogue/git/trees/{vogue_ref}?recursive=1')
    vogue_revision=vogue_tree['sha']
    if old and vogue_revision!=old['vogue']['revision']:
        raise ValueError('Use a new --output when changing the VOGUE revision')
    for record in vogue_tree['tree']:
        if record['type']!='blob':
            continue
        name=record['path']
        selected=(name in ('README.md','LICENSE.txt','data/conversation_trials/scenarios.json')
            or (name.startswith('data/conversation_trials/transcripts/') and name.endswith('.json'))
            or (name.startswith('data/metadata/') and name.endswith('.json'))
            or (name.endswith('.csv') and not name.startswith('data/supplements/'))
            or name in ('data/metadata/item_1.png','data/metadata/item_37.png')
            or (args.all_vogue_images and name.startswith('data/metadata/') and name.endswith('.png')))
        if selected:
            jobs.append((args.output/'vogue',name,
                f'https://raw.githubusercontent.com/D3Mlab/vogue/{vogue_revision}/{name}',record['size']))
    declared=sum(job[3] for job in jobs)
    if declared>args.max_mb*1000000:
        raise ValueError(f'Selected {declared/1000000:.1f} MB exceeds --max-mb')
    print(f'Fetching {len(jobs)} files, {declared/1000000:.2f} MB (Amazon image URLs fetched separately)',flush=True)
    artifacts=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch_file,jobs):
            artifacts.append(result)
    manifest={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'ecommerce_user_sim':{'repo':SIM_REPO,'revision':revision,'gated':info.get('gated'),
            'license':info.get('cardData',{}).get('license'),
            'files':[row for row,job in zip(artifacts,jobs) if job[0].name=='ecommerce_user_sim']},
        'vogue':{'repo':'D3Mlab/vogue','revision':vogue_revision,
            'license':'See downloaded LICENSE.txt: dialogue CC BY 4.0; product images/metadata separate research-only terms',
            'files':[row for row,job in zip(artifacts,jobs) if job[0].name=='vogue']},
        'total_bytes':sum(row['bytes'] for row in artifacts),
        'notes':['Public files only; no access applications or emails submitted.',
                 'E-commerce image URLs are not downloaded. VOGUE images: '+('all 76 catalog images' if args.all_vogue_images else 'two samples'),
                 'No existing train/validation/test splits or models modified.']}
    (args.output/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'total_bytes':manifest['total_bytes'],'files':len(artifacts),'sim_revision':revision,'vogue_revision':vogue_revision},indent=2))


if __name__=='__main__':
    main()
