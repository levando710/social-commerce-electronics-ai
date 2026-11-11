"""Inspect downloaded schemas, joins and leakage risks before choosing a benchmark."""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

import csv
from datetime import datetime, timezone
import hashlib
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
import io
import requests

import pyarrow.parquet as pq
from PIL import Image

ROOT=MODULE_ROOT


def parquet(path):
    return pq.read_table(path).to_pylist()


def audit_sim(root):
    domains={}
    for domain in ('Cell_Phones_and_Accessories','Electronics'):
        users=parquet(root/f'corpus/users/{domain}.parquet')
        dialogues=parquet(root/f'corpus/dialogues/{domain}.parquet')
        items=parquet(root/f'corpus/items/{domain}.parquet')
        item_ids={row['parent_asin'] for row in items}
        user_ids={row['user_id'] for row in users}
        history=[event for row in users for event in row['interaction_history']]
        lengths=[len(row['interaction_history']) for row in users]
        def join_user(value):
            if value in user_ids:
                return value
            base=re.sub(r'_\d+$','',value)
            return base if base in user_ids else None
        canonical=[join_user(row['user_id']) for row in dialogues]
        usable=[]
        for row in users:
            target=row['target']
            clean={event['item'] for event in row['interaction_history']
                   if event['item']!=target['item'] and event['ts']<target['ts'] and event['item'] in item_ids}
            usable.append(len(clean))
        domains[domain]={
            'users':len(users),'dialogues':len(dialogues),'catalog_items':len(items),
            'history_events':len(history),'history_length':{'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths),'mean':statistics.mean(lengths)},
            'users_with_no_history':sum(not row['interaction_history'] for row in users),
            'users_with_target_in_history':sum(row['target']['item'] in {event['item'] for event in row['interaction_history']} for row in users),
            'users_with_nonpast_history':sum(any(event['ts']>=row['target']['ts'] for event in row['interaction_history']) for row in users),
            'history_events_missing_metadata':sum(event['item'] not in item_ids for event in history),
            'targets_with_metadata':sum(row['target']['item'] in item_ids for row in users),
            'dialogues_resolving_user':sum(value is not None for value in canonical),
            'users_with_at_least_3_unique_strictly_past_nontarget_items':sum(value>=3 for value in usable),
            'items_with_image_urls':sum(any(image.get('hi_res') or image.get('large') or image.get('thumb') for image in row['images']) for row in items),
            'nature':'LLM-simulated dialogues grounded in Amazon histories; not human conversation logs',
            'actor_input_policy':'Never expose hidden target, target review, or simulator prompt to the recommender agent.',
        }
    evaluation=parquet(root/'eval/users/Electronics.parquet')
    items=parquet(root/'eval/items/Electronics.parquet')
    catalog=parquet(root/'eval/catalog/Electronics.parquet')
    corpus=parquet(root/'corpus/users/Electronics.parquet')
    by_user={row['user_id']:row for row in corpus}
    item_ids={row['parent_asin'] for row in items}
    catalog_ids={row['parent_asin'] for row in catalog}
    eval_summary={'users':len(evaluation),'item_details':len(items),'candidate_catalog_items':len(catalog),
        'history_events':sum(len(row['history']) for row in evaluation),
        'targets_with_item_details':sum(row['target_asin'] in item_ids for row in evaluation),
        'targets_in_candidate_catalog':sum(row['target_asin'] in catalog_ids for row in evaluation),
        'users_overlapping_corpus':sum(row['user_id'] in by_user for row in evaluation),
        'targets_same_as_corpus_target':sum(by_user.get(row['user_id'],{}).get('target',{}).get('item')==row['target_asin'] for row in evaluation),
        'users_with_target_in_history':sum(row['target_asin'] in {event['parent_asin'] for event in row['history']} for row in evaluation),
        'history_timestamps_released':all(all('timestamp' in event or 'ts' in event for event in row['history']) for row in evaluation),
        'warning':'corpus/eval folders are simulator artifacts, not a leak-free CRS training split. Prevent target-label and dialogue leakage.',
        'nature':'user-simulator evaluation protocol; online simulation logs are not released here'}
    return {'corpus':domains,'eval_electronics':eval_summary}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def audit_vogue(root):
    paths=sorted((root/'data/conversation_trials/transcripts').glob('*.json'))
    dialogs=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
    products=[json.loads(path.read_text(encoding='utf-8')) for path in (root/'data/metadata').glob('*.json')]
    metadata_ids={int(row['id']) for row in products}
    profiles=read_csv(root/'data/fashion_profiles/profiles.csv')
    ratings=read_csv(root/'data/conversation_trials/item_ratings/seeker_ratings.csv')
    conversations={int(row['conversation_id']) for row in dialogs}
    all_targets=[int(item) for row in dialogs for item in row['gt_items']]
    valid_images=0
    for path in (root/'data/metadata').glob('*.png'):
        with Image.open(path) as image:
            image.verify()
        valid_images+=1
    return {'conversations':len(dialogs),'utterances':sum(len(row['conversation_content']) for row in dialogs),
        'profile_participants':len(profiles),'trial_item_metadata':len(products),'target_mentions':len(all_targets),
        'targets_with_metadata':sum(item in metadata_ids for item in all_targets),
        'seeker_rating_rows':len(ratings),
        'rating_rows_joining_conversation':sum(int(row['conversation_id']) in conversations for row in ratings),
        'downloaded_images_decodable':valid_images,
        'nature':'60 human-human fashion dialogues with grounded item selections, participant profiles and ratings',
        'limitations':['Different domain from electronics.',
            'Profile survey item IDs 37..76 differ from dialogue catalog item IDs 1..36; do not assume the same ratings matrix.',
            'Small human benchmark, not a large-scale collaborative-filtering training corpus.',
            'Transcripts/annotations and product images/metadata have different license terms.']}


def main():
    external=ROOT/'data/external'
    manifest=json.loads((external/'download_manifest.json').read_text())
    for dataset in ('ecommerce_user_sim','vogue'):
        for record in manifest[dataset]['files']:
            path=external/dataset/record['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Downloaded artifact changed: '+str(path))
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'download_manifest_sha256':hashlib.sha256((external/'download_manifest.json').read_bytes()).hexdigest(),
        'ecommerce_user_sim':audit_sim(external/'ecommerce_user_sim'),
        'vogue':audit_vogue(external/'vogue'),
        'not_done':['No model retraining or changes to existing experiments.',
                    'No academic data access applications submitted.',
                    'Amazon image URL availability not exhaustively tested.']}
    sample_items=parquet(external/'ecommerce_user_sim/corpus/items/Cell_Phones_and_Accessories.parquet')[:5]
    sample_items+=parquet(external/'ecommerce_user_sim/corpus/items/Electronics.parquet')[:5]
    def check_image(row):
        url=next((image.get('hi_res') or image.get('large') or image.get('thumb')
                  for image in row['images'] if image.get('hi_res') or image.get('large') or image.get('thumb')),None)
        try:
            with requests.get(url,stream=True,timeout=(8,15)) as response:
                response.raise_for_status()
                chunks=[];size=0
                for chunk in response.iter_content(65536):
                    size+=len(chunk)
                    if size>5000000:
                        raise ValueError('Image exceeds audit limit')
                    chunks.append(chunk)
                with Image.open(io.BytesIO(b''.join(chunks))) as image:
                    image.verify()
            return {'item_id':row['parent_asin'],'url':url,'decodable':True}
        except (requests.RequestException,OSError,ValueError) as exc:
            return {'item_id':row['parent_asin'],'url':url,'decodable':False,'error':str(exc)[:160]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        image_samples=list(pool.map(check_image,sample_items))
    result['ecommerce_image_url_sample']={'sample_size':len(image_samples),
        'decodable':sum(row['decodable'] for row in image_samples),'samples':image_samples,
        'scope':'first five catalog records per domain, not a representative availability estimate'}
    out=ROOT/'data/discovery';out.mkdir(exist_ok=True)
    (out/'candidate_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
