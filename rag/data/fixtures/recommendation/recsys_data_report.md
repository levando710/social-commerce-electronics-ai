# Kiểm tra dữ liệu và baseline RecSys

- Review nguồn: 2255; sản phẩm trong catalog: 1136.
- Tương tác tích cực (rating >= 4, sau gộp trùng): 1563.
- Người dùng: 1500; người có ít nhất 2 sản phẩm: 58.
- Train / validation / test: 1500 / 5 / 58 tương tác.
- 2-core (user và item cùng bậc >= 2): {'interactions': 8, 'users': 4, 'items': 4}.

## Pop: full-catalog ranking, chỉ học từ train

| Split / nhóm | Số người | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| validation / all | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| validation / warm_item | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| validation / cold_item | 0 | N/A | N/A | N/A | N/A |
| test / all | 58 | 0.0172 | 0.0086 | 0.0517 | 0.0201 |
| test / warm_item | 51 | 0.0196 | 0.0098 | 0.0588 | 0.0229 |
| test / cold_item | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Giới hạn diễn giải

- Reviews are feedback proxies, not observed clicks, follows or confirmed transactions.
- The collected product/review subset is not representative of Amazon or a social network.
- Per-user chronology does not enforce a shared global time cutoff across different users.
- Low-degree users remain train-only. Do not fabricate interactions to increase evaluation size.
- Cold-item results are included and also reported separately; no sampled negatives.
- Historical product text/reviews are not a temporally frozen feature store for future GNN evaluation.

User ID được băm SHA-256; cạnh đồ thị chỉ lấy từ train.
Pop được fit bằng train; khi đánh giá test, loại các item đã thấy ở train và validation.
Chưa huấn luyện GNN/GCL. Cần mở rộng tương tác trước khi kết luận hiệu quả cá nhân hóa.
