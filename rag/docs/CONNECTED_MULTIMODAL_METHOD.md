# Biểu diễn đa phương thức và đồ thị phân cấp thông số

Mô tả phương pháp cho các mục 2.2 và 2.4 của báo cáo. Công thức được soạn theo mã triển khai; phần này không thay cho khảo sát tài liệu hay chứng minh đóng góp mới. Các baseline là bản triển khai nội bộ đã mô tả trong `RECSYS_MODELS.md`.

## Đầu vào, xử lý và đầu ra

Đầu vào gồm catalog sản phẩm, metadata có cấu trúc, ảnh catalog, lịch sử positive review thuộc train và câu hỏi hiện tại. Nhãn mục tiêu validation/test chỉ dành cho evaluator. Không có cạnh bạn bè/follow hoặc nhãn tình trạng thiết bị cũ được quan sát trong các nguồn đang dùng.

Với item $i$, ký hiệu $M_i$ là metadata cho phép, $I_i$ là ảnh đã tải/kiểm tra, $S_i$ là các thuộc tính được ghi rõ. Với user $u$, $H_u^{train}$ là các tương tác train. Bài toán đầu vào:

$$
\mathcal X=(\mathcal U,\mathcal I,\{M_i,I_i,S_i\}_{i\in\mathcal I},\mathcal E_{UI}^{train},q_u).
\tag{1}
$$

Đầu ra nhánh RecSys là thứ tự item theo điểm user–item; đầu ra IR là danh sách sản phẩm/tài liệu có điểm truy hồi. Việc có thứ tự đúng chưa đồng nghĩa câu trả lời sinh ra trung thực; RAG generation cần đánh giá riêng.

## Biểu diễn văn bản và ảnh

Metadata text chỉ gồm title, store, features, description, categories và các trường thông số có trong allowlist cố định. Không đưa target review, rating, review count, Best Sellers Rank, visual_description hoặc generated_description vào text features. Chia văn bản thành đoạn 80 từ, overlap 16 từ; encoder có giới hạn token nên đây là biện pháp giảm truncation, không bảo đảm mọi đoạn kỹ thuật đều dưới giới hạn tokenizer.

Cho $C_i$ là các đoạn metadata, $f_T$ là MiniLM có sẵn, và $n(x)=x/\max(\|x\|_2,\epsilon)$:

$$
\mathbf t_i=n\!\left(\frac1{|C_i|}\sum_{c\in C_i}n(f_T(c))\right)\in\mathbb R^{384}.
\tag{2}
$$

Với CLIP ViT-B/32 có sẵn:

$$
\mathbf v_i=\begin{cases}n(f_V(I_i))\in\mathbb R^{512},&\text{ảnh hợp lệ},\\\mathbf0,&\text{thiếu/lỗi ảnh},\end{cases}
\qquad m_i=\mathbb1[\text{ảnh hợp lệ}].
\tag{3}
$$

Không huấn luyện lại hai encoder trong bước tạo chỉ mục. Chỉ giữ một encoder trong mỗi tiến trình; lưu vector theo item và kiểm tra hash đầu vào khi tiếp tục. Ảnh catalog không phải bằng chứng xác minh tình trạng của một máy cũ cụ thể. Mask thiếu ảnh quyết định có tạo image node hay không.

## Phân cấp thông số và đồ thị

Chỉ từ trường catalog có sẵn, tạo bộ ba `(nhóm, thuộc tính, giá trị)`. Ví dụ: compute → RAM → `8 GB`; storage → Memory Storage Capacity → `128 GB`. Thiếu RAM thì không tạo giá trị RAM. Các nhóm do người triển khai định nghĩa, không phải nhóm sở thích đã học từ người dùng.

$$
\mathcal V=\mathcal U\cup\mathcal I\cup\mathcal T\cup\mathcal V_{image}
\cup\mathcal S_{value}\cup\mathcal S_{attribute}\cup\mathcal S_{group}.
\tag{4}
$$

$$
\mathcal E=\mathcal E_{UI}^{train}\cup\mathcal E_{IT}\cup\mathcal E_{IV}
\cup\mathcal E_{IS}\cup\mathcal E_{IA}\cup\mathcal E_{SA}\cup\mathcal E_{AG}.
\tag{5}
$$

Trong đó $IS$: item–giá trị, $IA$: item–thuộc tính hiện diện, $SA$: giá trị–thuộc tính, $AG$: thuộc tính–nhóm. Cạnh $IA$ cho phép nhóm thông số ảnh hưởng item trong hai bước lan truyền; không ngụ ý biết giá trị thuộc tính còn thiếu. Tất cả cạnh đối xứng trong graph propagation. Đây là đồ thị dị thể về loại nút và biểu diễn ban đầu; chưa có trọng số ma trận riêng cho mỗi loại cạnh như các mô hình relational GNN chuyên biệt.

## Lan truyền thông tin và học gợi ý

User/item/giá trị/thuộc tính/nhóm có embedding học được. Text và image nodes được chiếu tuyến tính, không bias, vào cùng không gian $d=16$:

$$
\mathbf h_x^{(0)}=\begin{cases}
\mathbf e_x,&x\in\mathcal U\cup\mathcal I\cup\mathcal S_{value}\cup\mathcal S_{attribute}\cup\mathcal S_{group},\\
W_T\mathbf t_i,&x=t_i,\\W_V\mathbf v_i,&x=v_i,\;m_i=1.
\end{cases}
\tag{6}
$$

Không cộng trực tiếp vector gốc 384 chiều với 512 chiều. Sau phép chiếu, lan truyền tuyến tính trên adjacency chuẩn hóa đối xứng, không thêm self-loop; lấy trung bình cả trạng thái ban đầu:

$$
\widehat A=D^{-1/2}AD^{-1/2},\qquad
H^{(\ell+1)}=\widehat A H^{(\ell)},\qquad
\bar H=\frac1{L+1}\sum_{\ell=0}^{L}H^{(\ell)}.
\tag{7}
$$

$$
s(u,i)=\bar{\mathbf h}_u^\top\bar{\mathbf h}_i.
\tag{8}
$$

Với minibatch positive train $B$ và item negative $j$ lấy ngoài train positives của user:

$$
\mathcal L_{BPR}=-\frac1{|B|}\sum_{(u,i,j)\in B}\log\sigma(s(u,i)-s(u,j)).
\tag{9}
$$

Negative sampler không được biết nhãn validation/test, vì vậy vẫn có thể lấy trúng một item người dùng sẽ thích trong tương lai; không dùng evaluator để tránh điều đó. Regularization hiện áp dụng cho các ID embedding user, positive item và negative item được lấy trong batch:

$$
R_B=\frac1{2|B|}\sum_{(u,i,j)\in B}(\|\mathbf e_u\|^2+\|\mathbf e_i\|^2+\|\mathbf e_j\|^2).
\tag{10}
$$

## Học tương phản đồ thị

Tạo hai graph views bằng cách bỏ cạnh đối xứng với xác suất 0,2. Tính hai biểu diễn chuẩn hóa của cùng node; lấy tối đa 256 user và 256 positive item duy nhất trong batch. Với tập node $B_X$, temperature $\tau=0,2$:

$$
\ell_X(1,2)=-\frac1{|B_X|}\sum_{x\in B_X}
\log\frac{\exp(n(\mathbf h_x^1)^\top n(\mathbf h_x^2)/\tau)}
{\sum_{y\in B_X}\exp(n(\mathbf h_x^1)^\top n(\mathbf h_y^2)/\tau)},
\quad\mathcal L_X=\tfrac12[\ell_X(1,2)+\ell_X(2,1)].
\tag{11}
$$

$$
\mathcal L=\mathcal L_{BPR}+10^{-4}R_B+0.05(\mathcal L_U+\mathcal L_I).
\tag{12}
$$

Đây là objective riêng của nhánh graph+GCL. **Không gọi công thức này là joint RAG–RecSys objective**: phần kết hợp có giám sát hội thoại cần mô tả và đánh giá riêng. Áp dụng GCL không bảo đảm cải thiện; phải so với cùng graph không GCL.

## Truy hồi đa phương thức và đánh giá

Text query và image-text query được encode riêng bằng encoder tương ứng, không dùng target/simulator prompt. Query dài được chia đoạn 80 từ cho MiniLM và 32 từ cho CLIP rồi mean/normalize. Dùng cosine trong từng không gian riêng và BM25 trên metadata. Kết hợp theo rank:

$$
s_{RRF}(q,i)=\sum_{m\in\{BM25,T,V\}}\frac{w_m}{60+r_m(q,i)},
\qquad(w_{BM25},w_T,w_V)=(1,1,0.5).
\tag{13}
$$

Chỉ cộng kênh có item trong ranking. Raw Pillow color/brightness tags là baseline riêng, không được dùng thay CLIP trong graph đề xuất hoặc suy luận tình trạng máy.

Checkpoint được chọn bằng validation. RecSys test báo cáo full-catalog HR/NDCG, cold-item và hai nhóm user riêng. IR dùng câu hỏi đầu tiên mô phỏng với một nhãn đích đã phát hành: P/R/NDCG/MRR theo qrel không phải relevance được đánh giá đầy đủ; Coverage là tỷ lệ item duy nhất xuất hiện trong top-K trên catalog. Persona cùng user được trung bình trong user trước khi tổng hợp.

Giới hạn: thời gian chia theo từng user; catalog chưa time-versioned; hội thoại là mô phỏng; chưa có khảo sát người dùng thật; các baseline nội bộ không phải mã benchmark chính thức. Cần giữ các kết quả âm và phân tích lỗi, không điều chỉnh siêu tham số dựa trên test.
