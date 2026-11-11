# Nhánh RecSys: baseline, đồ thị dị thể và GCL

**Giới hạn.** BM3 từng dùng chung vùng nhớ với đặc trưng NumPy, làm ảnh hưởng seed và mô hình chạy sau trong thí nghiệm cũ; đầu vào nay được sao chép và có kiểm tra bất biến. Các so sánh BM3 nhiều seed hoặc graph trong `results_recsys` và `results_connected_multimodal_v1` không dùng làm kết luận (xem `PROVENANCE_WARNING.md` và `INVALIDATED.md`); thí nghiệm thay thế là `results_connected_multimodal_v2`.

Đây là triển khai nghiên cứu cục bộ trên split điện thoại đã cố định. Chưa phải phiên bản hoàn thành toàn bộ đề cương GNN–GCL–RAG.

## 1. Dữ liệu và đặc trưng đầu vào

- 1.500 người dùng, 1.136 sản phẩm; train/validation/test có 1.500/5/58 tương tác.
- `artifacts/features/legacy/features.npz` chứa 1.136 vector văn bản metadata và vector CLIP ảnh; 93 sản phẩm không có ảnh dùng vector 0 và mask riêng.
- Văn bản được tạo lại từ title, store, features, description, categories và một allowlist các trường thông số. Không dùng `reviews`, `average_rating`, `rating_number`, `helpful_vote` hoặc text RAG đã tổng hợp review.
- Metadata được chia thành 5.475 đoạn, encode bằng MiniLM đã cache, lấy trung bình các vector đoạn theo sản phẩm rồi chuẩn hóa L2. Ảnh tái sử dụng CLIP đã tạo ở đợt trước.
- Feature store, split và checkpoint đều có checksum. Đây là catalog tĩnh transductive: chưa xác nhận rằng mỗi metadata/ảnh đã có tại thời điểm từng review.

## 2. Mô hình và hàm mục tiêu

Ký hiệu: $u$ là người dùng, $i$ là sản phẩm, $\mathcal E_{tr}$ là tập cạnh train, $d$ là số chiều latent. $x_i^t,x_i^v$ là đặc trưng text/image, $\operatorname{sg}$ là stop-gradient.

### 2.1. LightGCN

Đồ thị hai phía chỉ chứa $\mathcal E_{tr}$. Lan truyền tuyến tính bằng adjacency đối xứng đã chuẩn hóa, lấy trung bình embedding đầu vào và các lớp:

$$H^{(\ell+1)}=D^{-1/2}AD^{-1/2}H^{(\ell)},\qquad Z=\frac{1}{L+1}\sum_{\ell=0}^{L}H^{(\ell)}.\tag{1}$$

Điểm là tích vô hướng embedding user/item. Huấn luyện bằng BPR và regularization trên ID embeddings. Tham khảo [bài LightGCN](https://arxiv.org/abs/2002.02126).

### 2.2. VBPR

Tách sở thích ID và sở thích thị giác; sử dụng CLIP cố định thay cho CNN features của thí nghiệm gốc:

$$s(u,i)=b_i+p_u^Tq_i+\theta_u^T E_vx_i^v+\beta_v^Tx_i^v.\tag{2}$$

Học bằng BPR, có item bias, visual bias và regularization. Nhánh VBPR này chỉ dùng ảnh; không đưa text vào rồi gọi đó là VBPR gốc. Tham khảo [bài VBPR](https://arxiv.org/abs/1510.01784).

### 2.3. BM3

Sử dụng LightGCN backbone, residual item ID, predictor và dropout latent targets đã detach. Mục tiêu bootstrap ghép user–item cùng các thành phần alignment/masking text và ảnh. Không thêm BPR hoặc InfoNCE vào BM3.

$$D(a,b)=1-\cos(a,\operatorname{sg}(b)),\qquad \widetilde z=\operatorname{Dropout}(\operatorname{sg}(z)).\tag{3}$$

Loss gồm hai chiều user–item, alignment modality→item và modality→target của chính modality. Phần ảnh chỉ tính trên item có ảnh. Tham khảo [mã của tác giả BM3](https://github.com/enoche/BM3). Đây là local reimplementation: encoder features, xử lý ảnh thiếu, cấu hình và giao thức đánh giá khác benchmark gốc.

### 2.4. Đồ thị dị thể đa phương thức

Schema của cấu hình đầy đủ đã chạy:

| Loại node | Số node |
|---|---:|
| User | 1.500 |
| Item | 1.136 |
| Metadata text | 1.136 |
| Image | 1.043 |
| Giá trị thuộc tính kỹ thuật | 2.294 |

Các cạnh không hướng: user–item (1.500 cạnh train), item–text (1.136), item–image (1.043), item–spec (9.846). Các giá trị spec trùng nhau dùng chung node. Không tạo cạnh xã hội, click hoặc follow giả.

ID user/item và node spec có embedding học được; text/image đi qua projection riêng về $d$ chiều. Lan truyền trên đồ thị đã chuẩn hóa như (1), sau đó lấy embedding user/item để tính BPR. Đây là heterogeneous schema với type-specific input projections, chưa phải attention theo quan hệ hoặc một cây phân cấp thông số hoàn chỉnh.

### 2.5. Graph Contrastive Learning

Sinh hai graph views bằng edge dropout; hai chiều của một cạnh luôn bị giữ/bỏ cùng nhau. Tính InfoNCE riêng theo loại user và item, mỗi loại tối đa 256 node train được lấy mẫu:

$$\mathcal L_{NCE}(Z_1,Z_2)=-\frac{1}{|S|}\sum_{a\in S}\log\frac{\exp(\cos(z_a^1,z_a^2)/\tau)}{\sum_{b\in S}\exp(\cos(z_a^1,z_b^2)/\tau)}.\tag{4}$$

Lấy trung bình hai chiều $Z_1\leftrightarrow Z_2$, cộng loss user và item. Cấu hình hiện tại dùng edge dropout 0,2, temperature 0,2 và trọng số GCL 0,05:

$$\mathcal L=\mathcal L_{BPR}+\lambda_{reg}\mathcal L_{reg}+\lambda_{GCL}(\mathcal L_{NCE}^{user}+\mathcal L_{NCE}^{item}).\tag{5}$$

Hướng augmentation tham khảo [SGL](https://arxiv.org/abs/2010.10783). Loss này kết hợp các mục tiêu trong RecSys; **chưa phải joint loss giữa RecSys và RAG, chưa có Co-Guiding**.

## 3. Giao thức và kết quả

Cấu hình cố định: 100 epoch, d=32, 2 lớp, Adam lr=0,001, seeds 42/43/44. Cùng một split và catalog cho mọi phương án. Negative sampling chỉ loại các positives trong train; không đọc validation/test để chọn negative. Chọn checkpoint bằng NDCG@10 validation, rồi HR@10; nếu bằng nhau giữ checkpoint sớm nhất. Chỉ đánh giá test sau khi chọn.

Kết quả HR@10 trung bình ± sample standard deviation qua 3 seed:

| Phương án | HR@10 | NDCG@10 |
|---|---:|---:|
| Pop | 0,0517 | 0,0201 |
| LightGCN | 0,0345 ± 0,0299 | 0,0211 ± 0,0229 |
| VBPR | 0,0575 ± 0,0359 | 0,0308 ± 0,0231 |
| BM3 | 0,0172 ± 0,0172 | 0,0112 ± 0,0113 |
| Graph text-only | 0,0230 ± 0,0100 | 0,0130 ± 0,0112 |
| Graph multimodal | 0,0230 ± 0,0100 | 0,0130 ± 0,0112 |
| Graph multimodal + GCL | 0,0460 ± 0,0263 | 0,0281 ± 0,0205 |

Trong lượt này GCL tăng HR@10 của graph từ 2,30% lên 4,60%, nhưng chưa vượt Pop ở HR@10. Thêm ảnh chưa làm thay đổi HR@10/NDCG@10 của graph không GCL. Chưa có cơ sở khẳng định mô hình đề xuất vượt trội.

15/18 lượt có validation NDCG@10 bằng 0 ở mọi checkpoint đã xét, nên giữ checkpoint epoch 10 theo quy tắc tie-break. Chỉ 5 người validation không đủ tin cậy để lựa chọn mô hình; mỗi người chỉ có một tương tác train. Cần mở rộng lịch sử tương tác trước khi tối ưu mô hình nghiêm túc. Không đổi cấu hình để chạy đuổi theo kết quả trên 58 người test.

Chi tiết từng seed, rank từng mẫu test, loss history và checksum nằm trong `artifacts/evaluation/results_recsys/experiment.json`. Bảng đầy đủ: `artifacts/evaluation/results_recsys/recsys_evaluation_report.md`.

## 4. Chạy và sử dụng

Từ `D:\DATN`, dùng môi trường `.venvs/rag` đã cài `requirements-dense.txt`:

```powershell
.\.venvs\rag\Scripts\python.exe -B rag\src\run.py build_recsys_features
.\.venvs\rag\Scripts\python.exe -B rag\src\run.py train_recsys
.\.venvs\rag\Scripts\python.exe -B rag\src\run.py recommend_products --user-index 0 --top-k 5
```

Feature store/checkpoint/experiment đã tồn tại sẽ không bị ghi đè. Khi chạy một thí nghiệm mới, dùng `--output` mới cho feature store và dùng đồng thời `--models`/`--output` mới cho trainer. Lệnh trên chỉ dùng lại được sau khi chỉ rõ các thư mục mới hoặc trên checkout chưa có artifact.

CLI gợi ý dùng cố định `graph_multimodal_gcl`, seed 42 để minh họa; không tự chọn seed tốt nhất theo test. Không truyền `--user-index` thì fallback Pop. Chỉ loại history train khi serving; đây là hồ sơ offline, chưa có đăng nhập và cập nhật hành vi thực.

```powershell
.\.venvs\rag\Scripts\python.exe -B rag\src\run.py chat_app --enable-recommendations
```

Endpoint độc lập `POST /api/recommend` nhận `{"user_index":0,"top_k":5}`. Bỏ `user_index` để lấy Pop. Giao diện chat hiện vẫn gọi `/api/query`; hai nhánh **chưa tự phản hồi cho nhau**. Không diễn giải việc cùng tồn tại API thành Co-Guiding.

## 5. Kiểm thử

- Unit tests kiểm tra propagation bằng kết quả tính tay; graph chỉ chứa cạnh train; edge dropout đối xứng; InfoNCE đúng cặp thấp hơn hoán vị; gradient BM3 target bị chặn; mọi model backpropagate và export vectors đúng scoring.
- Giao thức đánh giá Pop của trainer khớp baseline đã lưu. Đã smoke-test CLI và API với model thật, cold start, loại sản phẩm đã thấy và đầu vào lỗi.
