# Huấn luyện chung từ câu hỏi hội thoại và đánh giá Co-Guiding

Tài liệu này nối tiếp công thức (1)–(13) trong [CONNECTED_MULTIMODAL_METHOD.md](CONNECTED_MULTIMODAL_METHOD.md). Mã thực thi: `conversational_joint.py`, `train_conversational_joint.py`; kế hoạch được lưu trước khi chạy tại `data_discovery/CONVERSATIONAL_EXPERIMENT_PLAN_2026_10_29.md`.

## Đầu vào và đầu ra

Mỗi mẫu giám sát là $(q,u,i^*)$: câu hỏi đầu tiên của người dùng mô phỏng, ID người dùng có lịch sử trong graph và sản phẩm đích được phát hành cùng dữ liệu. Chỉ cohort train có 564 câu hỏi/282 người được lấy mẫu cho loss. Validation có 70 câu/35 người; test có 72 câu/36 người. Hai persona cùng người phải ở cùng cohort. Lịch sử trước đích của cả ba cohort là ngữ cảnh train; đây là thiết lập transductive, không phải kiểm tra người dùng hoàn toàn mới.

Input của bộ mã hóa chỉ gồm câu hỏi. Nhãn đích dùng làm supervision, không được ghép vào câu hỏi, dùng để viết lại truy vấn, chọn negative, hoặc tạo đặc trưng. Không dùng prompt bí mật của simulator, review của mục tiêu hay câu trả lời tương lai. Encoder MiniLM/CLIP và bộ sinh câu trả lời giữ nguyên trọng số. Phần được train là graph, các projection đa phương thức và query head.

Đầu ra gồm vector user/item, ma trận và bias của query head, checkpoint đầy đủ để tiếp tục train, thứ hạng và metric theo user. Đánh giá query đo khả năng lấy lại mục tiêu mô phỏng trong toàn catalog; không đo độ trung thực của văn bản được sinh.

## Query tower và loss

Ký hiệu $n(x)=x/\max(\|x\|_2,\epsilon)$, $f_T$ là MiniLM giữ nguyên trọng số. Chia câu hỏi dài thành đoạn 80 từ, overlap 16; lấy trung bình vector chuẩn hóa rồi chuẩn hóa lại thành $x_q$. Query head có chiều ra 16:

$$
z_q=n(W_Qx_q+b_Q),\qquad z_i=n(\bar h_i),\qquad a_{q,i}=z_q^\top z_i/0.2.
\tag{14}
$$

Với query minibatch $B_Q$ gồm 32 mẫu, cross-entropy dùng toàn bộ 4.435 item:

$$
\mathcal L_Q=-\frac1{|B_Q|}\sum_{(q,u,i^*)\in B_Q}
\log\frac{\exp a_{q,i^*}}{\sum_{j\in\mathcal I}\exp a_{q,j}}.
\tag{15}
$$

Tạo phân phối query–item $p_{q}=\mathrm{softmax}(a_q)$ và user–item $p_u=\mathrm{softmax}(n(\bar h_u)^\top z_i/0.2)$. Hạng alignment là Jensen–Shannon đối xứng, với $m=(p_q+p_u)/2$:

$$
\mathcal L_{JS}=\frac1{|B_Q|}\sum_{(q,u,i^*)\in B_Q}
\tfrac12\left[D_{KL}(p_q\|m)+D_{KL}(p_u\|m)\right].
\tag{16}
$$

Mục tiêu chung dùng graph objective từ (12):

$$
\mathcal L_{joint}=\mathcal L_{graph}+0.1\mathcal L_Q+\lambda_{JS}\mathcal L_{JS},
\qquad\lambda_{JS}\in\{0,0.02\}.
\tag{17}
$$

Ba biến thể giữ cùng graph đa phương thức, GCL, lịch lấy mẫu và cấu hình:

| Biến thể | Query head được train | Query CE cập nhật graph | JS |
|---|---|---|---|
| `gradient_decoupled` | Có | Không; dùng stop-gradient trên item trong (14) | 0 |
| `joint_retrieval` | Có | Có | 0 |
| `joint_bidirectional` | Có | Có | 0,02 |

Đối chứng vẫn có query head được học; không dùng head ngẫu nhiên để tạo lợi thế giả cho joint. Graph và head được clip gradient riêng, tránh head làm thay đổi tỷ lệ clipping của graph trong đối chứng. CE vẫn hiển thị trong total loss của đối chứng nhưng không truyền gradient về graph. JS liên kết hai phân phối xếp hạng; đây là prototype local, không phải tuyên bố đã tái lập một hệ thống RAG huấn luyện cả retriever và LLM generator.

## Chọn checkpoint và tái lập

Adam lr=0,001, RecSys batch=1.024, 100 epoch, CPU 2 threads; seeds 42/43/44. Kiểm tra validation mỗi 10 epoch. Tiêu chí checkpoint:

$$
S_{val}=0.5\,\mathrm{NDCG}^{Rec}_{val}@10+0.5\,\mathrm{NDCG}^{Q}_{val}@10.
\tag{18}
$$

Khi hòa, ưu tiên query NDCG rồi RecSys NDCG, cuối cùng epoch sớm. Không so trực tiếp với checkpoint của bảng RecSys v2 như thể cùng giao thức: bảng cũ chọn theo RecSys validation riêng. Query metric trung bình hai persona trong từng user trước khi trung bình tất cả user. RecSys kiểm tra trên 1.229 người; query kiểm tra trên 36 người độc lập.

Checkpoint mỗi epoch lưu tham số, optimizer, RNG của graph/dropout và hai sampler. Static features/adjacency được tái dựng từ file có hash, không lưu buffer mutable chung với NumPy. Cuối run kiểm tra fingerprint features, dựng lại checkpoint đã chọn và so toàn bộ vector/head. Script `audit_conversational_results.py` làm lại bước này từ các artifact đã lưu và tính khoảng bootstrap theo user. Khoảng này điều kiện trên các seed đã train, không bao phủ bất định do lấy lại mẫu dữ liệu train.

## Phép thử Co-Guiding lúc suy luận

Đây là phép thử riêng với joint gradient training. Dùng cố định graph+GCL v2 seed 42, bốn chế độ `decoupled`, `rag_to_rec`, `rec_to_rag`, `coupled`. Hai nhóm đều bật hoặc đều tắt kiểm tra ràng buộc trong cả bốn chế độ, để việc lọc không bị nhầm là lợi ích feedback.

Truy hồi kết hợp BM25, MiniLM và CLIP với trọng số (13); lấy 80 ứng viên. Chiều Rec→RAG cộng prior theo thứ hạng feed, và thêm tối đa 40 item từ feed vào tập ứng viên. Chiều RAG→Rec xét feed gốc 100 item, thêm item có evidence, bỏ item đã xem và item không qua ràng buộc, rồi cộng reward:

$$
s_{RAG}(q,i)=\frac{\mathbf1[i\in C_q]}{60+r_q(i)}+
\frac{0.2\,\mathbf1[\mathrm{Rec}\to\mathrm{RAG}]\,\mathbf1[i\in F_u]}{60+r_u(i)},
\qquad
s_{feed}(i)=\frac1{60+r_{base}(i)}+\frac{0.03\,\mathbf1[i\in E_q]}{r_E(i)}.
\tag{19}
$$

Các điểm chỉ áp dụng cho ứng viên hợp lệ. Phép thử giữ nguyên câu hỏi gốc và vector cache ở mọi nhánh, **tắt query rewriting**; đo tác động guard và feedback, chưa đo lợi ích rewriting. Chỉ số feed là khả năng đưa lại mục tiêu theo câu hỏi hiện tại; không gọi đây là chất lượng gợi ý dài hạn. Rank ngoài top-20 chỉ được mã hóa là “không tìm thấy trong top-20”, không dùng để báo cáo MRR không giới hạn.

Guard phân biệt nhà sản xuất và thiết bị tương thích: “Spigen case for iPhone” không phải điện thoại Apple. RAM/dung lượng ghi trong tên máy trên ốp không được coi là phần cứng của ốp. Thiếu dữ kiện dẫn tới không xác minh được, không tự suy ra là đúng. Bộ phân tích là heuristic và vẫn có thể sai; cần xem các mục tiêu bị loại do unknown/violation. Bật guard không tự chứng minh câu trả lời tốt hơn.

Ứng dụng dùng `--connected` có rewriting theo trạng thái hội thoại và fusion giới hạn ứng viên/kênh; vì vậy bảng offline full-catalog và không rewriting không phải cam kết metric cho mọi câu hỏi trong giao diện. Khi bật `--conversational-joint`, query head mới được thêm vào xếp hạng; giao diện dùng mean pooling như quá trình train.

## Giới hạn cần ghi trong luận văn

Hội thoại được LLM mô phỏng dựa trên lịch sử review, không phải hội thoại người thật. Qrel có một mục tiêu và chưa đánh giá đầy đủ các sản phẩm liên quan khác. Catalog tĩnh, chưa có mốc thời gian toàn cục; nhiều mục là phụ kiện và thiếu cấu hình phần cứng. Kết quả âm vẫn được giữ. Faithfulness/relevance/hallucination của câu trả lời, đánh giá nhiều lượt và người dùng thật cần được thực hiện riêng.
