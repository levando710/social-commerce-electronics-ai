# Pipeline RAG demo

Pipeline hiá»‡n táº¡i dÃ¹ng dá»¯ liá»‡u Ä‘Ã£ lá»c trong `data_phone_clean_v2_filtered`.
Má»¥c tiÃªu lÃ  chá»©ng minh bÃ i toÃ¡n Multimodal RAG kháº£ thi trÃªn dataset cÃ´ng khai:
metadata sáº£n pháº©m + review + áº£nh sáº£n pháº©m.

## 1. Chuáº©n bá»‹ corpus

```powershell
cd D:\DATN\rag
..\.venvs\rag\Scripts\python.exe src\run.py prepare_corpus --data data_phone_clean_v2_filtered --output rag_ready
```

Äáº§u ra:

- `artifacts/runtime/catalog/documents.jsonl`: má»—i dÃ²ng lÃ  má»™t sáº£n pháº©m Ä‘Ã£ Ä‘Æ°á»£c gom metadata, thÃ´ng sá»‘, review vÃ  Ä‘Æ°á»ng dáº«n áº£nh.
- `artifacts/runtime/catalog/summary.json`: thá»‘ng kÃª sá»‘ document, sá»‘ document cÃ³ review, sá»‘ document cÃ³ áº£nh.

Káº¿t quáº£ hiá»‡n táº¡i:

```text
documents: 1136
documents_with_reviews: 482
documents_with_product_images: 1043
documents_with_review_images: 0
```

## 2. Build chá»‰ má»¥c BM25 ná»n

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py build_index --corpus artifacts/runtime/catalog\documents.jsonl --output artifacts/runtime/catalog\bm25_index.json
```

Äáº§u ra hiá»‡n táº¡i:

```text
documents: 1136
avg_doc_length: 375.82
vocabulary: 13686
```

BM25 Ä‘Æ°á»£c dÃ¹ng lÃ m ná»n vÃ¬ dá»… giáº£i thÃ­ch, khÃ´ng cáº§n API, khÃ´ng cáº§n GPU vÃ  phÃ¹ há»£p Ä‘á»ƒ chá»©ng minh
luá»“ng RAG trÆ°á»›c khi thay báº±ng embedding model.

Trong báº£n demo hiá»‡n táº¡i, káº¿t quáº£ cuá»‘i khÃ´ng cÃ²n lÃ  BM25 thuáº§n mÃ  lÃ  **hybrid retrieval**:

- BM25 láº¥y nhÃ³m á»©ng viÃªn ban Ä‘áº§u.
- Má»Ÿ rá»™ng truy váº¥n Viá»‡t-Anh cho cÃ¡c Ã½ Ä‘á»‹nh phá»• biáº¿n nhÆ° `pin`, `quá»‘c táº¿`, `má»Ÿ khÃ³a`, `giÃ¡ ráº»`, `Ä‘Ã¡nh giÃ¡`.
- Boost title/brand/model khi trÃ¹ng vá»›i truy váº¥n.
- Æ¯u tiÃªn sáº£n pháº©m cÃ³ review vÃ  áº£nh khi cÃ¢u há»i cáº§n evidence hoáº·c áº£nh.
- Boost mÃ¡y `unlocked/quá»‘c táº¿`, Ä‘á»“ng thá»i pháº¡t cÃ¡c sáº£n pháº©m cÃ³ dáº¥u hiá»‡u khÃ³a máº¡ng khi ngÆ°á»i dÃ¹ng há»i mÃ¡y quá»‘c táº¿.
- Æ¯u tiÃªn rating tá»‘t khi cÃ¢u há»i cÃ³ Ã½ Ä‘á»‹nh `good/tá»‘t`.
- Chá»n evidence theo cÃ¢u cÃ³ nhiá»u token trÃ¹ng nháº¥t thay vÃ¬ cáº¯t thÃ´ vÄƒn báº£n.

## 3. Query demo

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py query_rag "samsung galaxy unlocked good battery" --top-k 3
..\.venvs\rag\Scripts\python.exe src\run.py query_rag "iphone unlocked renewed good camera" --top-k 3
..\.venvs\rag\Scripts\python.exe src\run.py query_rag "Ä‘iá»‡n thoáº¡i quá»‘c táº¿ pin tá»‘t" --top-k 3
```

Káº¿t quáº£ tráº£ vá» gá»“m:

- `score`: Ä‘iá»ƒm hybrid sau rerank.
- `bm25_score`: Ä‘iá»ƒm BM25 gá»‘c Ä‘á»ƒ giáº£i thÃ­ch baseline.
- `title`, `brand`, `average_rating`, `review_count`.
- `product_images`: Ä‘Æ°á»ng dáº«n áº£nh sáº£n pháº©m local.
- `review_images`: Ä‘Æ°á»ng dáº«n áº£nh review náº¿u cÃ³.
- `evidence`: Ä‘oáº¡n vÄƒn báº£n Ä‘Æ°á»£c truy xuáº¥t Ä‘á»ƒ LLM dÃ¹ng lÃ m cÄƒn cá»©.

## Ã nghÄ©a chá»©ng minh kháº£ thi

Pipeline nÃ y Ä‘Ã£ chá»©ng minh Ä‘Æ°á»£c cÃ¡c Ä‘iá»ƒm chÃ­nh:

- CÃ³ thá»ƒ láº¥y dá»¯ liá»‡u cÃ´ng khai miá»n Ä‘iá»‡n thoáº¡i tá»« Amazon Reviews 2023.
- CÃ³ thá»ƒ lá»c thÃ nh táº­p sáº£n pháº©m Ä‘iá»‡n thoáº¡i tÆ°Æ¡ng Ä‘á»‘i sáº¡ch.
- CÃ³ thá»ƒ gom metadata, review vÃ  áº£nh vÃ o document cáº¥p sáº£n pháº©m.
- CÃ³ thá»ƒ truy xuáº¥t sáº£n pháº©m liÃªn quan theo cÃ¢u há»i tá»± nhiÃªn.
- CÃ³ thá»ƒ tráº£ vá» evidence vÃ  áº£nh Ä‘i kÃ¨m, lÃ m Ä‘áº§u vÃ o cho LLM/Gemini á»Ÿ bÆ°á»›c sinh cÃ¢u tráº£ lá»i.

## BÆ°á»›c tiáº¿p theo

Sau baseline BM25, cÃ³ thá»ƒ nÃ¢ng cáº¥p theo hai hÆ°á»›ng:

- Text embedding: dÃ¹ng SentenceTransformers hoáº·c API embedding Ä‘á»ƒ thay BM25.
- Multimodal reranking: dÃ¹ng Gemini hoáº·c CLIP Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ thÃªm áº£nh sáº£n pháº©m cÃ¹ng vá»›i text evidence.

## Giao diá»‡n chat demo

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py chat_app --port 7860
```

Má»Ÿ:

```text
http://127.0.0.1:7860
```

Giao diá»‡n chat dÃ¹ng cÃ¹ng hybrid retriever, hiá»ƒn thá»‹ card sáº£n pháº©m, áº£nh local, rating, sá»‘ review,
Ä‘iá»ƒm hybrid, Ä‘iá»ƒm BM25 vÃ  Ä‘oáº¡n evidence truy xuáº¥t Ä‘Æ°á»£c.

## TÃ­ch há»£p Gemini

Giao diá»‡n hiá»‡n Ä‘Ã£ cÃ³ bÆ°á»›c LLM generation:

```text
User query
-> Hybrid retrieval
-> Top evidence + áº£nh sáº£n pháº©m
-> Gemini
-> CÃ¢u tráº£ lá»i tiáº¿ng Viá»‡t + danh sÃ¡ch evidence
```

á»¨ng dá»¥ng Ä‘á»c key Gemini theo thá»© tá»±:

- biáº¿n mÃ´i trÆ°á»ng `GEMINI_API_KEY` hoáº·c `GOOGLE_API_KEY`;
- `rag/.env`;
- `rag/.streamlit/secrets.toml`;
- file secrets cÅ© cá»§a project model: `D:\DATN\model\.streamlit\secrets.toml`.

Gemini chá»‰ Ä‘Æ°á»£c dÃ¹ng sau khi retrieval Ä‘Ã£ chá»n evidence. Náº¿u API lá»—i hoáº·c thiáº¿u key, giao diá»‡n váº«n hiá»ƒn thá»‹
káº¿t quáº£ truy xuáº¥t vÃ  card evidence Ä‘á»ƒ demo khÃ´ng bá»‹ dá»«ng hoÃ n toÃ n.

## ÄÃ¡nh giÃ¡ retrieval theo metric IR

Äá»ƒ Ä‘Ã¡p á»©ng yÃªu cáº§u Ä‘Ã¡nh giÃ¡ thá»±c nghiá»‡m, pipeline cÃ³ thÃªm bá»™ Ä‘Ã¡nh giÃ¡ offline cho
retrieval. BÆ°á»›c nÃ y khÃ´ng gá»i Gemini, khÃ´ng cÃ o dá»¯ liá»‡u vÃ  khÃ´ng train mÃ´ hÃ¬nh.
Má»¥c tiÃªu lÃ  so sÃ¡nh cháº¥t lÆ°á»£ng truy xuáº¥t giá»¯a BM25 thuáº§n vÃ  hybrid retrieval trÃªn
cÃ¹ng táº­p document.

Cháº¡y toÃ n bá»™ pipeline Ä‘Ã¡nh giÃ¡ offline:

```powershell
.\scripts\run_offline_evaluation.ps1
```

Hoáº·c cháº¡y riÃªng bÆ°á»›c Ä‘Ã¡nh giÃ¡ retrieval:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py evaluate_retrieval
```

Äáº§u vÃ o:

- `artifacts/evaluation/queries.json`: bá»™ truy váº¥n Ä‘Ã¡nh giÃ¡ vÃ  tiÃªu chÃ­ liÃªn quan.
- `artifacts/runtime/catalog/documents.jsonl`: document cáº¥p sáº£n pháº©m gá»“m text, metadata, review vÃ  áº£nh.
- `artifacts/runtime/catalog/bm25_index.json`: chá»‰ má»¥c BM25.

Äáº§u ra:

- `artifacts/evaluation/results/retrieval_metrics.json`: káº¿t quáº£ Ä‘áº§y Ä‘á»§ dáº¡ng JSON.
- `artifacts/evaluation/results/retrieval_metrics_by_query.csv`: metric theo tá»«ng truy váº¥n.
- `artifacts/evaluation/results/retrieval_top_results.csv`: top káº¿t quáº£ cá»§a tá»«ng phÆ°Æ¡ng Ã¡n.
- `artifacts/evaluation/results/retrieval_evaluation_report.md`: báº£ng tÃ³m táº¯t Ä‘Æ°a vÃ o bÃ¡o cÃ¡o.

CÃ¡c metric hiá»‡n cÃ³:

- Precision@K: tá»· lá»‡ káº¿t quáº£ liÃªn quan trong K káº¿t quáº£ Ä‘áº§u.
- Recall@K: tá»· lá»‡ tÃ i liá»‡u liÃªn quan Ä‘Æ°á»£c tÃ¬m tháº¥y trong K káº¿t quáº£ Ä‘áº§u.
- NDCG@K: Ä‘Ã¡nh giÃ¡ cháº¥t lÆ°á»£ng thá»© háº¡ng, Æ°u tiÃªn káº¿t quáº£ liÃªn quan á»Ÿ vá»‹ trÃ­ cao.
- MRR@K: Ä‘o vá»‹ trÃ­ cá»§a káº¿t quáº£ liÃªn quan Ä‘áº§u tiÃªn.
- Coverage@K: tá»· lá»‡ truy váº¥n cÃ³ Ã­t nháº¥t má»™t káº¿t quáº£ liÃªn quan trong K káº¿t quáº£ Ä‘áº§u.

Káº¿t quáº£ thá»­ nghiá»‡m hiá»‡n táº¡i trÃªn 1.136 document vÃ  50 truy váº¥n Ä‘Ã¡nh giÃ¡:

| PhÆ°Æ¡ng Ã¡n | P@5 | R@5 | NDCG@5 | MRR@5 | Coverage@5 |
|---|---:|---:|---:|---:|---:|
| BM25 text-only | 0,724 | 0,057 | 0,739 | 0,839 | 0,900 |
| BM25 cÃ³ visual evidence | 0,768 | 0,058 | 0,778 | 0,845 | 0,880 |
| Hybrid retrieval | 0,864 | 0,063 | 0,869 | 0,920 | 0,960 |

So vá»›i BM25 cÃ³ visual evidence, hybrid retrieval cáº£i thiá»‡n Precision@5 khoáº£ng
12,50%, Recall@5 khoáº£ng 9,03%, NDCG@5 khoáº£ng 11,79%, MRR@5 khoáº£ng 8,88% vÃ 
Coverage@5 khoáº£ng 9,09%.
Recall@5 váº«n cÃ²n tháº¥p vÃ¬ táº­p liÃªn quan theo tiÃªu chÃ­ cÃ³ nhiá»u document, trong khi
top-k nhá».

## PhÃ¢n tÃ­ch áº£nh cho Multimodal RAG

Äá»ƒ áº£nh trá»Ÿ thÃ nh nguá»“n tri thá»©c truy xuáº¥t Ä‘Æ°á»£c, pipeline há»— trá»£ hai cÃ¡ch phÃ¢n tÃ­ch:

- Local visual features: dÃ¹ng Pillow Ä‘á»ƒ trÃ­ch xuáº¥t kÃ­ch thÆ°á»›c, tá»‰ lá»‡ áº£nh, Ä‘á»™ sÃ¡ng,
  Ä‘á»™ tÆ°Æ¡ng pháº£n, mÃ u chá»§ Ä‘áº¡o vÃ  tÃ­n hiá»‡u rá»§i ro cÆ¡ báº£n. CÃ¡ch nÃ y khÃ´ng cáº§n API vÃ 
  cháº¡y á»•n Ä‘á»‹nh.
- Gemini Vision/VLM: sinh caption/tag/risk giÃ u ngá»¯ nghÄ©a hÆ¡n khi API/quota kháº£ dá»¥ng.

Cháº¡y nhÃ¡nh local cho batch áº£nh:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py analyze_product_images --local-only --limit 200 --retry-errors
```

Cháº¡y nhÃ¡nh Gemini Vision:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py analyze_product_images --limit 30
```

Cáº­p nháº­t corpus vÃ  index sau khi cÃ³ káº¿t quáº£ áº£nh:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py prepare_corpus
..\.venvs\rag\Scripts\python.exe src\run.py build_index
```

Káº¿t quáº£ Ä‘Æ°á»£c cache vÃ o:

```text
artifacts/runtime/multimodal/image_analysis.jsonl
```

Sau Ä‘Ã³ `prepare_corpus.py` tá»± Ä‘á»™ng Ä‘Æ°a cÃ¡c trÆ°á»ng nhÆ° `caption_vi`, `caption_en`,
`visual_tags`, `condition_tags`, `risk_tags` vÃ  `evidence` vÃ o `visual_evidence`
cá»§a document, Ä‘á»“ng thá»i thÃªm pháº§n `Visual evidence` vÃ o text Ä‘á»ƒ BM25/hybrid cÃ³
thá»ƒ truy xuáº¥t.

Kiá»ƒm tra Ä‘á»™ phá»§ dá»¯ liá»‡u Ä‘a phÆ°Æ¡ng thá»©c:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py audit_multimodal_coverage
```

Káº¿t quáº£ hiá»‡n táº¡i:

```text
documents: 1136
documents_with_reviews: 482
documents_with_product_images: 1043
documents_with_review_images: 0
documents_with_visual_evidence: 856
image_analysis_unique_images: 856
image_analysis_success: 856
image_analysis_failed: 0
```

Tá»· lá»‡ áº£nh sáº£n pháº©m sáºµn cÃ³ cao, khoáº£ng 91,81%. Hiá»‡n 856 document Ä‘Ã£ cÃ³
`visual_evidence`, tÆ°Æ¡ng Ä‘Æ°Æ¡ng 75,35% toÃ n corpus. Gemini Vision Ä‘Ã£ káº¿t ná»‘i Ä‘Æ°á»£c
nhÆ°ng gáº·p quota 429/503 á»Ÿ má»™t sá»‘ lÆ°á»£t gá»i, nÃªn pipeline dÃ¹ng local visual features
lÃ m lá»›p ná»n á»•n Ä‘á»‹nh vÃ  dÃ¹ng VLM Ä‘á»ƒ lÃ m giÃ u ngá»¯ nghÄ©a khi quota cho phÃ©p.

## Pipeline Multimodal RAG tá»•ng quÃ¡t

Báº£n thá»­ nghiá»‡m hiá»‡n táº¡i Ä‘Ã£ xá»­ lÃ½ text, metadata, review vÃ  áº£nh sáº£n pháº©m. Äá»ƒ Ä‘i
Ä‘Ãºng hÆ°á»›ng Multimodal RAG, kiáº¿n trÃºc tá»•ng quÃ¡t Ä‘Æ°á»£c má»Ÿ rá»™ng nhÆ° sau:

```text
User query
  -> Query understanding
  -> Text retriever: title, description, metadata, review
  -> Image retriever: product image, issue image, visual caption, CLIP/VLM embedding
  -> Video retriever: sampled frames, transcript, OCR, detected product state
  -> Audio retriever: ASR transcript from review/test video
  -> Fusion and reranking
  -> LLM/VLM answer generation grounded by retrieved evidence
  -> Answer + product cards + evidence + modality source
```

Vai trÃ² tá»«ng modality:

- Text: cung cáº¥p thÃ´ng tin trá»±c tiáº¿p tá»« tiÃªu Ä‘á», mÃ´ táº£, review vÃ  thÃ´ng sá»‘.
- Image: há»— trá»£ nháº­n diá»‡n ngoáº¡i hÃ¬nh, mÃ u sáº¯c, phá»¥ kiá»‡n, tÃ¬nh tráº¡ng váº¿t xÆ°á»›c hoáº·c lá»—i hiá»ƒn thá»‹.
- Video: bá»• sung báº±ng chá»©ng Ä‘á»™ng nhÆ° thao tÃ¡c test mÃ¡y, test camera, loa, cáº£m á»©ng, Face ID.
- Audio: Ä‘Æ°á»£c chuyá»ƒn thÃ nh transcript Ä‘á»ƒ khai thÃ¡c nháº­n xÃ©t trong video review hoáº·c video test mÃ¡y.

Trong Ä‘á»“ Ã¡n tháº­t, cÃ¡c modality nÃ y khÃ´ng nháº¥t thiáº¿t pháº£i Ä‘Æ°a tháº³ng vÃ o mÃ´ hÃ¬nh
Ä‘á»‹nh giÃ¡ dáº¡ng báº£ng. ChÃºng cÃ³ thá»ƒ Ä‘Æ°á»£c chuyá»ƒn thÃ nh nhÃ£n cÃ³ cáº¥u trÃºc nhÆ°
`camera tá»‘t`, `pin tá»‘t`, `mÃ n hÃ¬nh lá»—i`, `ngoáº¡i hÃ¬nh xÆ°á»›c`, `loa hoáº¡t Ä‘á»™ng`,
`Ä‘Ã£ test cáº£m á»©ng`, sau Ä‘Ã³ dÃ¹ng cho chatbot tÆ° váº¥n, lá»c sáº£n pháº©m vÃ  bá»• sung Ä‘áº·c
trÆ°ng cho mÃ´ hÃ¬nh Ä‘á»‹nh giÃ¡.
## Cap nhat: Query understanding truoc retrieval

Pipeline hien co da bo sung mot lop **query understanding** truoc retrieval. Lop
nay doc cau hoi tu nhien, loc bot tu nhieu va trich xuat cac truong co cau truc:

- `query_rewrite`: cau truy van ngan gon hon de dua vao retrieval.
- `keywords`: tu khoa quan trong.
- `tags`: cac y dinh nhu `battery_good`, `camera_good`, `unlocked`, `review_rich`, `has_image`.
- `filters`: dieu kien mem nhu brand, yeu cau co review, yeu cau co anh san pham.
- `must_have`, `nice_to_have`, `avoid`: tin hieu dung cho reranking.

Trong giao dien demo, neu co Gemini API key thi he thong co the dung Gemini de
sinh query plan. Khi danh gia offline, he thong dung rule-based query
understanding de ket qua lap lai duoc va khong phu thuoc quota API.

Nam phuong an retrieval dang duoc so sanh:

| Phuong an | Mo ta |
|---|---|
| `bm25_text_only` | Chi dung text, bo visual evidence |
| `bm25` | Dung text co them visual evidence |
| `semantic_tfidf` | Baseline semantic nhe dua tren TF-IDF concept, khong can API/model ngoai |
| `hybrid` | BM25 + reranking theo brand/title/review/image/unlocked/rating |
| `intent_hybrid` | Query understanding + hybrid retrieval |

Ket qua moi tren 1.136 document va 50 query cho thay `intent_hybrid` dat
Precision@1 = 0,920 va MRR@5 = 0,940, cao hon `hybrid` o hai chi so lien quan
den vi tri ket qua dau tien. Precision@5 giam nhe tu 0,864 xuong 0,852, vi vay
buoc query understanding can tiep tuc tinh chinh trong giai doan sau.

## Cap nhat: Manual relevance labeling

De tang do tin cay cua thuc nghiem, pipeline co them buoc sinh file gan nhan thu
cong tu top results:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py make_manual_label_sheet --max-queries 30 --top-k 10
```

File dau ra:

```text
artifacts/evaluation/manual_relevance_labels.csv
```

Dien cot `manual_relevant` bang:

- `1`: ket qua lien quan voi query.
- `0`: ket qua khong lien quan.
- de trong: chua gan nhan.

Sau khi gan nhan, chay:

```powershell
..\.venvs\rag\Scripts\python.exe src\run.py evaluate_manual_labels
```

Ket qua duoc luu vao:

```text
artifacts/evaluation/manual_results/manual_retrieval_metrics.json
artifacts/evaluation/manual_results/manual_retrieval_report.md
```

