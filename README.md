# SPIN × Qwen3-0.6B × GSM8K-VI

## 1. Mục tiêu

Fine-tune **Qwen3-0.6B** trên bộ toán tiếng Việt **GSM8K-VI** bằng **SPIN**
(Self-Play Preference Optimization), dựa trên official
[`uclaml/SPIN`](https://github.com/uclaml/SPIN).

- Dữ liệu: 4000 train / 500 validation / 500 test
- Cấu hình: `k=1`, `enable_thinking=False`
- Phần cứng: 1× A100 40GB (Modal)
- Quy trình: SFT → SPIN-1 → SPIN-2 → SPIN-3 → SPIN-4 → Đánh giá

## 2. Kết quả

Đánh giá trên 500 mẫu test, greedy decoding, exact-match sau khi chuẩn hóa số.

| Model             |        Accuracy |
| ----------------- | --------------: |
| base (Qwen3-0.6B) |            6.0% |
| SFT               |           33.6% |
| SPIN-1            |           32.8% |
| **SPIN-2**  | **34.8%** |
| SPIN-3            |           34.2% |
| SPIN-4            |           32.6% |

**Nhận xét:** SPIN tăng khoảng 1.2 điểm so với SFT rồi đi ngang. Với model
0.6B, chất lượng bão hòa sau vòng 2; chênh lệch giữa các vòng chủ yếu là nhiễu.

## 3. Quy trình

1. **Chuẩn bị dữ liệu**

   - Nguồn: `namfam/gsm8k-vietnamese`
   - Chia split với seed 42: 4000 train / 500 validation / 500 test
   - Chuyển sang JSONL và format chat của Qwen3
2. **SFT**

   - Full fine-tune 3 epochs
   - Learning rate `2e-5`, max length 1024, BF16
3. **SPIN** (4 vòng)

   - Sinh câu trả lời từ checkpoint trước với `k=1`
   - Chuyển dữ liệu sang format của official SPIN
   - Train 2 epochs (~238 steps/vòng), `beta=0.1`, lr `5e-7`
   - Đánh giá sau mỗi vòng
4. **Đánh giá cuối**

   - So sánh base, SFT và 4 vòng SPIN trên cùng test set

## 4. Cấu trúc thư mục

```
SPIN/
├── src/                 # Mã nguồn chính (chạy trên Modal)
├── docs/                # Tài liệu và nhật ký
├── scripts/             # Script hỗ trợ (upload Hugging Face)
├── tests/               # Unit tests
├── artifacts/           # Checkpoint và output (không commit)
├── requirements.txt
└── README.md
```

## 5. Cách chạy

```bash
# Kiểm tra môi trường (CPU, không tốn GPU)
modal run modal_app.py::preflight

# Chạy từng giai đoạn
modal run modal_app.py::session1
modal run modal_app.py::session2
modal run modal_app.py::session3
modal run modal_app.py::session4
```

Mỗi session chạy xong sẽ tự thoát và giải phóng GPU.

## License

MIT
