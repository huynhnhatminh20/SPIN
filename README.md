# SPIN × Qwen3-0.6B × GSM8K-VI

Wrapper/orchestrator cho dự án fine-tune Qwen3-0.6B trên GSM8K-VI bằng thuật toán SPIN (Self-Play Preference Optimization), chạy trên Modal (A100 40GB).

## Cấu trúc

```
SPIN/
├── src/                 # Mã nguồn chính (chạy trên Modal dưới /src/app)
├── data/                # Dữ liệu mẫu (chỉ .gitkeep, dữ liệu lớn lưu ngoài repo)
├── artifacts/           # Output khi chạy local (được .gitignore)
├── docs/                # Lịch sử phiên làm việc (SESSION.md)
├── tests/               # Unit tests
├── requirements.txt     # Dependencies
└── .gitignore
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

```bash
# Local preflight (không tốn GPU)
python src/app.py --stage preflight

# Chạy trên Modal
modal deploy modal_app.py
modal run modal_app.py --stage session1
```

## Kết quả chính (n=500 test GSM8K-VI)

| Model | Accuracy |
|-------|----------|
| base (Qwen/Qwen3-0.6B) | 6.0% |
| SFT | 33.6% |
| SPIN-1 | 32.8% |
| SPIN-2 | 34.8% |
| SPIN-3 | 34.2% |
| SPIN-4 | 32.6% |

## Lưu ý

- Các file model weights (`.safetensors`, `.bin`, `.pt`) và dữ liệu lớn **không được commit lên GitHub**. Sử dụng Hugging Face Hub, Git LFS, hoặc Modal Volume để lưu trữ.
- Code SPIN loss không được implement lại trong wrapper. Điểm duy nhất gọi official SPIN là `accelerate launch /src/official_spin/spin/run_spin.py <config>`.
- Official SPIN: https://github.com/uclaml/SPIN

## License

MIT