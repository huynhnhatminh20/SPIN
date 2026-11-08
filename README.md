# SPIN × Qwen3-0.6B × GSM8K-VI

## 1. Mục tiêu

Fine-tune **Qwen3-0.6B** trên bộ toán tiếng Việt **GSM8K-VI** bằng **SPIN**
(Self-Play Preference Optimization), dựa trên official
[`uclaml/SPIN`](https://github.com/uclaml/SPIN).

- Dữ liệu: 4000 train / 500 validation / 500 test
- Cấu hình: `k=1`, `enable_thinking=False`
- Phần cứng khuyến nghị: 1× GPU A100 (40GB)
- Quy trình: SFT → SPIN-1 → SPIN-2 → SPIN-3 → SPIN-4 → Đánh giá

## 2. Kết quả

Đánh giá trên 500 mẫu test, greedy decoding, exact-match sau khi chuẩn hóa số.

| Model             |        Accuracy |
| ----------------- | --------------: |
| base (Qwen3-0.6B) |            6.0% |
| SFT               |           33.6% |
| SPIN-1            |           32.8% |
| **SPIN-2**        |       **34.8%** |
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
├── src/                 # Mã nguồn chính và pipeline điều phối huấn luyện
├── docs/                # Tài liệu mô hình và dataset card
├── tests/               # Unit tests kiểm tra hệ thống
├── artifacts/           # Checkpoints, dữ liệu trung gian và kết quả đánh giá
├── requirements.txt     # Các thư viện phụ thuộc
└── README.md
```

## 5. Hướng dẫn cài đặt & Cách chạy

### Cài đặt môi trường

1. Cài đặt các thư viện yêu cầu:

```bash
pip install -r requirements.txt
```

2. Clone repository chính thức của SPIN và thiết lập biến môi trường:

```bash
git clone https://github.com/uclaml/SPIN.git official_spin
export OFFICIAL_SPIN_DIR=$(pwd)/official_spin
# Trên Windows PowerShell:
# $env:OFFICIAL_SPIN_DIR = "$PWD/official_spin"
```

### Chạy qua Orchestrator (`app.py`)

Hệ thống cung cấp file điều phối `src/app.py` để tự động hóa toàn bộ pipeline:

```bash
# 1. Kiểm tra môi trường và tính tương thích (smoke test, không tốn tài nguyên GPU)
python src/app.py --stage preflight

# 2. Huấn luyện từng giai đoạn (hỗ trợ resume tự động nếu đã có checkpoint)
python src/app.py --stage session1   # Chuẩn bị dữ liệu -> SFT -> Sinh dữ liệu gen 1 -> SPIN-1
python src/app.py --stage session2   # Sinh dữ liệu gen 2 -> SPIN-2 -> Đánh giá
python src/app.py --stage session3   # Sinh dữ liệu gen 3 -> SPIN-3 -> Đánh giá
python src/app.py --stage session4   # Sinh dữ liệu gen 4 -> SPIN-4 -> Đánh giá

# 3. Đánh giá riêng biệt
python src/app.py --stage eval
```

### Chạy từng bước độc lập (Manual)

Bạn cũng có thể chạy độc lập từng script trong thư mục `src/`:

- **Chuẩn bị dữ liệu**: `python src/split_data.py` và `python src/reformat_gsm8k_vi.py`
- **SFT**: `python src/sft_train.py`
- **Sinh dữ liệu đối nghịch**: `python src/generate_qwen3.py`
- **Chuyển đổi dữ liệu cho SPIN**: `python src/convert_qwen3.py`
- **Đánh giá benchmark**: `python src/eval_gsm8k_vi.py`

## License

MIT
