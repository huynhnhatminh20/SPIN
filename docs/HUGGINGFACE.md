# Hướng dẫn upload SPIN lên Hugging Face

Mục tiêu: đưa checkpoint và artifact của đồ án lên Hugging Face để không mất
khi xóa local, đồng thời có link chia sẻ.

Kết quả mong đợi:

- Model repo: https://huggingface.co/NhatMinh20/SPIN-Qwen3-0.6B-GSM8K-VI
- Dataset repo: https://huggingface.co/datasets/NhatMinh20/GSM8K-VI-SPIN-artifacts

Model repo chứa 5 thư mục con (`sft/`, `spin1/` … `spin4/`), mỗi thư mục là một
model hoàn chỉnh. Dataset repo chứa `data/`, `evaluation/`, metadata và config.

## 1. Tạo token

1. Mở https://huggingface.co/settings/tokens
2. Bấm **Create new token**
3. Chọn role **Write** (cần quyền tạo repo + upload file)
4. Đặt tên, ví dụ `spin-upload`, rồi copy token (`hf_...`)

Không gửi token vào chat. Chỉ dán vào lệnh `hf auth login` bên dưới.

## 2. Đăng nhập

```powershell
hf auth login
# dán token khi được hỏi
```

Kiểm tra:

```powershell
hf auth whoami
```

Kết quả phải hiện `NhatMinh20`.

> Nếu `hf auth login` báo lỗi, dùng cách thay thế:
> ```powershell
> $env:HF_TOKEN = "hf_..."
> ```

## 3. Kiểm tra trước khi upload

```powershell
cd C:\Users\ASUS\Downloads\SPIN
python scripts/upload_to_hf.py --dry-run
```

Bước này chỉ render model card vào `artifacts/README.md`, không upload gì.

## 4. Upload

```powershell
python scripts/upload_to_hf.py
```

Script sẽ:

1. Kiểm tra đủ 5 checkpoint và kích thước `model.safetensors`.
2. Tạo model repo (public) và upload `README.md` + `sft/` … `spin4/`.
3. Tạo dataset repo (public) và upload `data/`, `evaluation/`, metadata.
4. In link và số file đã upload.

Muốn repo private:

```powershell
python scripts/upload_to_hf.py --private
```

Chỉ upload model hoặc chỉ dataset:

```powershell
python scripts/upload_to_hf.py --only model
python scripts/upload_to_hf.py --only dataset
```

Upload lần đầu khoảng **5.6 GB**. Thời gian tùy mạng; nếu terminal timeout thì
chạy lại cùng lệnh — file đã có sẽ được bỏ qua.

## 5. Cách load model sau khi upload

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "NhatMinh20/SPIN-Qwen3-0.6B-GSM8K-VI"
subfolder = "spin2"  # best: 34.8%

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder=subfolder,
    torch_dtype="bfloat16",
    device_map="auto",
)
```

Nhớ dùng `enable_thinking=False` khi `apply_chat_template` (xem model card).

## Cập nhật / upload lại

Sửa file trong `artifacts/` rồi chạy lại:

```powershell
python scripts/upload_to_hf.py
```

Hoặc upload thủ công một file:

```powershell
hf upload NhatMinh20/SPIN-Qwen3-0.6B-GSM8K-VI `
  artifacts/spin2/model.safetensors spin2/model.safetensors
```

## Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|-----------|
| `401 Unauthorized` | Chưa login hoặc token sai → `hf auth login` lại |
| `403 Forbidden` | Token thiếu quyền **Write**, hoặc repo thuộc tài khoản khác |
| `Repository not found` | Sai tên user/repo; kiểm tra `hf auth whoami` |
| Upload chậm / treo | Chạy lại lệnh; `hf_xet` đã cài nên file lớn được chia nhỏ |
| Hết dung lượng | Chỉ upload `spin2` hoặc dùng repo private có quota |

## File liên quan

- `docs/hf_model_card.md` — template model card
- `docs/hf_dataset_card.md` — template dataset card
- `scripts/upload_to_hf.py` — script upload
- `artifacts/` — dữ liệu nguồn (đã gitignore)
