# CHƯƠNG 1. XÁC ĐỊNH BÀI TOÁN VÀ YÊU CẦU CỦA NỀN TẢNG MẠNG XÃ HỘI THƯƠNG MẠI ĐỒ ĐIỆN TỬ THÔNG MINH

## 1.1. Bối cảnh và nhu cầu thực tế của mạng xã hội thương mại đồ điện tử

- Xu hướng hội tụ giữa mạng xã hội (Social Media) và thương mại điện tử (E-commerce) đối với ngành hàng thiết bị điện tử, công nghệ.
- Thực trạng dữ liệu đồ điện tử trên mạng xã hội: sự đan xen giữa thông số kỹ thuật chuẩn xác (*Hard Specs*, dạng bảng/thuộc tính) và trải nghiệm thực tế phi cấu trúc của người dùng (*Soft Experiences*: bài đăng review, ảnh chụp ngoại quan, unbox, video test máy).
- Nhu cầu cốt lõi của người dùng: vừa tìm kiếm tư vấn kỹ thuật chính xác theo ngôn ngữ tự nhiên (triệt tiêu ảo giác thông số), vừa nhận được các đề xuất mua sắm/bài đăng cá nhân hóa.

## 1.2. Bài toán tư vấn hội thoại đa phương thức — Conversational Multimodal Recommendation (C-MMRec)

Làm rõ đầu vào (Input), quá trình xử lý (Process) và đầu ra (Output) của bài toán. Mô hình hóa từng thành phần này dưới dạng các ký hiệu toán học.

## 1.3. Khảo sát các hướng tiếp cận liên quan

### 1.3.1. Multimodal Information Retrieval (Multimodal IR)

Truy hồi thông tin đa phương thức: kết hợp văn bản, hình ảnh, âm thanh…

### 1.3.2. Multimodal Retrieval-Augmented Generation (Multimodal RAG)

Sinh văn bản tăng cường bằng truy xuất đa phương thức.

### 1.3.3. Multimodal Recommender Systems

Hệ thống gợi ý đa phương thức.

> **Yêu cầu trình bày:** Với mỗi hướng tiếp cận trên, nêu kiến trúc hệ thống, các nghiên cứu SOTA gần nhất, các kỹ thuật xử lý, các hạn chế còn tồn tại và thách thức cần xử lý.

## 1.4. Hạn chế của các phương pháp hiện hành và thách thức

- Đa số CRS chỉ xử lý dữ liệu văn bản thuần túy, bỏ qua thông tin thị giác sản phẩm.
- Các hệ thống Multimodal RAG hiện nay (như ColPali, LangChain RAG) chủ yếu truy xuất phẳng trong không gian vector, dễ gặp hiện tượng ảo giác khi suy diễn thông số phần cứng phức tạp.
- Thiếu cơ chế kết hợp hai chiều (closed-loop) giữa truy vấn hội thoại (RAG) và hành vi mạng xã hội (RecSys).

## 1.5. Mục tiêu, phạm vi và giải pháp đề xuất của đề tài

- **Mục tiêu nghiên cứu:** Xây dựng khung kiến trúc lai GNN + GCL + Multimodal RAG cho nền tảng mạng xã hội thương mại đồ điện tử.
- **Phạm vi đề tài:** Tập trung vào các dòng thiết bị điện tử tiêu dùng (điện thoại, máy tính bảng, laptop và phụ kiện); dữ liệu bao gồm bài đăng bán, thông số kỹ thuật, đánh giá của người dùng và hình ảnh thực tế.

**Đóng góp:**

1. Xây dựng đồ thị dị thể đa phương thức (Heterogeneous Multimodal Graph) liên kết giữa thông số kỹ thuật, bài đăng mạng xã hội và hình ảnh sản phẩm.
2. Áp dụng học tương phản đồ thị (Graph Contrastive Learning — GCL) để khử nhiễu tương tác xã hội và giải quyết vấn đề thưa thớt dữ liệu (data sparsity).
3. Cơ chế Co-Guiding phản hồi tương hỗ giữa RAG và RecSys giúp tối ưu đồng thời chất lượng hội thoại và độ chính xác gợi ý.

# CHƯƠNG 2. ĐỀ XUẤT MÔ HÌNH LAI MULTIMODAL GNN – GCL – MULTIMODAL RAG CHO NỀN TẢNG MẠNG XÃ HỘI THƯƠNG MẠI

## 2.1. Mô hình kiến trúc tổng quát

- Giới thiệu hướng tiếp cận lai: kết hợp hệ gợi ý đa phương thức (Multimodal Graph Recommender System) và sinh phản hồi ngữ cảnh đa phương thức (Multimodal RAG).
- Mục tiêu: nâng cao độ chính xác gợi ý và khả năng hội thoại đa phương thức tự nhiên.

**Yêu cầu về hình kiến trúc tổng quát:**

Vẽ một hình kiến trúc tổng quát gồm các khối mô-đun với các vai trò khác nhau và có kết nối với nhau. Vẽ bằng các công cụ vector (Visio, LaTeX TikZ, draw.io), không sao chép hình.

Tham khảo cách vẽ kiến trúc từ hai bài báo cô đã gửi, đặc biệt bài [A Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/html/2504.08748v1).

Các khối mô-đun cần thể hiện:

- Biểu diễn tri thức và chỉ mục đa phương thức (Multimodal Knowledge Representation).
- Khung truy hồi Hybrid Multimodal RAG kết hợp Query Understanding.
- Mô hình hệ gợi ý đa phương thức (Multimodal Graph Recommender System).
- Cơ chế phản hồi tương hỗ hai chiều RAG ↔ RecSys (The Novel Co-Guiding Mechanism).

> **Yêu cầu diễn giải:** Từ mục 2.2 trở đi, trình bày chi tiết từng mô-đun và cơ chế xử lý. Các công thức phải tự soạn, không sao chép dưới dạng ảnh; sử dụng đúng ký hiệu và đánh số thứ tự.

## 2.2. Biểu diễn tri thức và chỉ mục đa phương thức (Multimodal Knowledge Representation)

### 2.2.1. Xử lý văn bản và thông số

Chuẩn hóa văn bản bài đăng, tách thực thể kỹ thuật (RAM, CPU, dung lượng pin, tình trạng máy like-new/vỡ màn). Sinh embedding bằng Sentence-BERT hoặc Domain-specific LLM.

> **Định hướng dữ liệu:** Làm với một tập dữ liệu Amazon Reviews 2023 trước. Sau khi làm rõ luồng thực nghiệm, bổ sung thực nghiệm với các tập dữ liệu khác tại [Datasets — Multimodal-RAG-Survey](https://deepwiki.com/llm-lab-org/Multimodal-RAG-Survey/5-datasets).

### 2.2.2. Trích xuất đặc trưng thị giác ngữ nghĩa cấp độ sâu

Thay thế xử lý ảnh thủ công đang sử dụng (Pillow brightness/color tags) bằng mô hình ViT/CLIP Embeddings.

### 2.2.3. Xây dựng không gian biểu diễn liên kết

Kết hợp cấu trúc phân cấp thông số (Specs Hierarchy) và đặc trưng cảm quan (Visual Embeddings).

## 2.3. Khung truy hồi Hybrid Multimodal RAG kết hợp Query Understanding

### 2.3.1. Mô-đun hiểu truy vấn

Intent Analysis, Constraint Extraction & Query Rewriting.

### 2.3.2. Cơ chế truy hồi lai (Hybrid Retrieval)

Kết hợp Sparse Lexical (BM25 trên Specs/Review) và Dense Semantic (CLIP/Vector DB).

### 2.3.3. Cơ chế xếp hạng lại có kiểm soát ràng buộc thực tế (Fact-Constrained Reranking)

Nhằm triệt tiêu ảo giác thông số.

### 2.3.4. Mô-đun sinh câu trả lời có viện dẫn bằng chứng thị giác (Visual Evidence Grounding)

## 2.4. Mô hình hệ gợi ý đa phương thức (Multimodal Graph Recommender System)

### 2.4.1. Thiết kế đồ thị tương tác không đồng nhất (Heterogeneous User-Item-Content Graph)

### 2.4.2. Lan truyền thông điệp đồ thị (Graph Convolution) tích hợp đặc trưng đa phương thức (Visual-Textual Features)

## 2.5. Cơ chế phản hồi tương hỗ hai chiều RAG ↔ RecSys (The Novel Co-Guiding Mechanism)

### 2.5.1. Luồng RAG → RecSys

Chuyển đổi ý định hội thoại tức thời thành trọng số phạt/thưởng (Penalty/Reward) để tái xếp hạng bảng tin.

### 2.5.2. Luồng RecSys → RAG

Sử dụng phân phối sở thích người dùng để định hướng không gian truy hồi evidence của chatbot.

### 2.5.3. Hàm mất mát kết hợp (Joint Objective / Multi-task Loss Function)

## 2.6. Kiểm thử phương pháp đề xuất

### 2.6.1. Dữ liệu thực nghiệm

Sử dụng Amazon Electronics và một số bộ dữ liệu công khai về bài toán này, tham khảo tại [Datasets — Multimodal-RAG-Survey](https://deepwiki.com/llm-lab-org/Multimodal-RAG-Survey/5-datasets).

Miêu tả rõ thông tin và tính đặc thù của mỗi bộ dữ liệu.

### 2.6.2. Môi trường và phương pháp thực nghiệm

#### 2.6.2.1. Môi trường thực nghiệm

#### 2.6.2.2. Chia dữ liệu thực nghiệm, thiết lập siêu tham số

#### 2.6.2.3. Độ đo thực nghiệm

- **Chỉ số IR:** Precision@K, Recall@K, NDCG@K, MRR@K, Coverage@K.
- **Chỉ số RecSys:** Hit Ratio (HR@K), NDCG@K.
- **Chỉ số chất lượng RAG:** Faithfulness, Answer Relevance, Hallucination Rate.

#### 2.6.2.4. Các phương pháp cơ sở được sử dụng để so sánh với phương pháp đề xuất

- **Nhánh RAG:** BM25 text-only, BM25 + Visual Tags thô, Dense Retrieval (CLIP), Standard Hybrid RAG.
- **Nhánh RecSys:** Pop, LightGCN, VBPR, BM3.

### 2.6.3. Kết quả thực nghiệm và phân tích so sánh

#### 2.6.3.1. Đánh giá hiệu năng truy hồi của Hybrid Multimodal RAG so với các baselines

#### 2.6.3.2. Đánh giá độ chính xác của hệ gợi ý đa phương thức

#### 2.6.3.3. Đánh giá hiệu quả của cơ chế tương hỗ Co-Guiding RAG ↔ RecSys

### 2.6.4. Nghiên cứu triệt tiêu (Ablation Study)

#### 2.6.4.1. Đánh giá đóng góp của thành phần ảnh (Vision Embedding vs. Text-only)

#### 2.6.4.2. Đánh giá ảnh hưởng của mô-đun Query Understanding và Reranking

#### 2.6.4.3. Đánh giá tính độc lập khi ngắt kết nối vòng lặp Co-Guiding (Decoupled vs. Joint)

### 2.6.5. Phân tích định tính và phân tích lỗi (Qualitative Analysis & Error Analysis)

#### 2.6.5.1. Phân tích tình huống mẫu (Case Study)

Khả năng tư vấn so sánh cấu hình và xử lý ảnh thực tế từ bài đăng người dùng.

#### 2.6.5.2. Nhận diện các ca thất bại (Failure Cases) và hạn chế còn tồn tại

### 2.6.6. Đánh giá độ trễ và chi phí tính toán (Latency & Computational Complexity)

## Yêu cầu bổ sung khi hoàn thiện và nộp bản thảo

- Bổ sung phụ lục cài đặt và thực thi thực nghiệm.
- Đưa đường dẫn mã nguồn thực nghiệm vào bản thảo.
- Gửi lại cô bản có thể chỉnh sửa (Word, LaTeX hoặc định dạng đang sử dụng để soạn quyển).
- Với các hình ảnh, gửi kèm tệp gốc hoặc đường dẫn cho phép chỉnh sửa.

