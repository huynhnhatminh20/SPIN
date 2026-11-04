---
language:
  - vi
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-0.6B
pipeline_tag: text-generation
datasets:
  - {{DATASET_REPO}}
tags:
  - spin
  - self-play
  - gsm8k
  - math-word-problems
  - qwen3
  - vietnamese
---

# SPIN-Qwen3-0.6B-GSM8K-VI

Fine-tune **Qwen3-0.6B** trên bộ toán tiếng Việt **GSM8K-VI** bằng **SPIN**
(Self-Play Preference Optimization), theo official implementation
[`uclaml/SPIN`](https://github.com/uclaml/SPIN) (commit `a12ba80`).

Repo chứa toàn bộ pipeline: checkpoint SFT và 4 vòng SPIN. Mỗi thư mục con là
một model hoàn chỉnh (có `config.json`, tokenizer, `model.safetensors`), có thể
load trực tiếp bằng `subfolder=`.

## Kết quả (test 500 mẫu GSM8K-VI)

| Model | Accuracy | Ghi chú |
|-------|---------:|---------|
| `base` (Qwen/Qwen3-0.6B) | 6.0% | zero-shot, chưa fine-tune |
| `sft` | 33.6% | SFT 3 epochs |
| `spin1` | 32.8% | SPIN vòng 1 |
| **`spin2`** | **34.8%** | **tốt nhất** |
| `spin3` | 34.2% | SPIN vòng 3 |
| `spin4` | 32.6% | SPIN vòng 4 (đã bão hòa) |

SPIN chỉ cải thiện ~1.2 pp so với SFT rồi đi ngang; chênh lệch giữa các vòng
nằm trong nhiễu (McNemar p > 0.05). Với model 0.6B, SPIN đạt trần khá sớm.

## Cấu trúc repo

```
.
├── README.md
├── sft/       # checkpoint sau SFT
├── spin1/     # SPIN vòng 1
├── spin2/     # SPIN vòng 2  (best)
├── spin3/     # SPIN vòng 3
└── spin4/     # SPIN vòng 4
```

Mỗi thư mục chứa `model.safetensors` (BF16, 310 tensors),
`config.json`, `tokenizer.json`, `chat_template.jinja`, `generation_config.json`
và metadata train (`train_results.json`, `trainer_state.json`).

## Cách dùng

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{{MODEL_REPO}}"
subfolder = "spin2"  # đổi thành sft / spin1 / spin3 / spin4 nếu muốn

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder=subfolder,
    torch_dtype="bfloat16",   # transformers >= 5 có thể dùng dtype="bfloat16"
    device_map="auto",
)

SYSTEM_VI = (
    "Bạn là trợ lý toán học tiếng Việt. Giải từng bước rồi chốt đáp án "
    "trên dòng cuối theo đúng định dạng 'Answer: <số>'."
)
problem = "Một cửa hàng có 45 quả táo, bán 12 quả. Hỏi còn lại bao nhiêu quả?"

messages = [
    {"role": "system", "content": SYSTEM_VI},
    {"role": "user", "content": problem},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Qwen3: tắt thinking cho đúng format train
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Model sinh ra dạng:

```
Solution:
Reason:
...các bước giải...
Answer:
<số>
```

## Cấu hình train

**Dữ liệu**
- Nguồn: [`namfam/gsm8k-vietnamese`](https://huggingface.co/datasets/namfam/gsm8k-vietnamese)
- Split seed 42: train 4000 / validation 500 / test 500 (tổng 8792 mẫu)
- SFT target: `solution` + `\nAnswer: <answer>`, không chèn thinking trace.

**SFT**
- Qwen3-0.6B full fine-tune, 3 epochs, lr `2e-5`, max length 1024, BF16.

**SPIN**
- Official `uclaml/SPIN`, commit `a12ba808a51c23bb98c7cc892d261e9c2f71ff8e`
- `beta=0.1`, `k=1`, `learning_rate=5e-7`, 2 epochs (~238 steps/vòng)
- `per_device_train_batch_size=4`, gradient accumulation 8, max length 1024
- `enable_thinking=False`

**Hạ tầng**
- 1× A100 40GB (Modal), torch 2.14, transformers 5.17, trl 1.13, accelerate 1.15
- Mỗi vòng SPIN: generate 4000 mẫu → chuyển sang format official → train → eval.

**Đánh giá**
- Greedy decoding, `max_new_tokens=256`, batch 16.
- Exact-match sau khi chuẩn hóa số (bỏ dấu phẩy/khoảng trắng/`$`).

## Giới hạn

- Model 0.6B, tiếng Việt, chủ yếu toán tiểu học/trung học cơ sở (GSM8K-VI).
- Có thể sai số học nhiều bước, hoặc sinh đáp án sai format.
- Không dùng cho quyết định quan trọng; luôn kiểm tra lại kết quả.
- Kế thừa thiên lệch của dữ liệu nguồn GSM8K-VI.

## License

- Code của dự án: MIT.
- Weights: kế thừa Qwen3-0.6B (Apache-2.0).
- Dữ liệu: xem license của `namfam/gsm8k-vietnamese`.
