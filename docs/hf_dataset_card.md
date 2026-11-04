---
language:
  - vi
license: other
task_categories:
  - text-generation
  - question-answering
tags:
  - gsm8k
  - math-word-problems
  - vietnamese
  - spin
  - self-play
source_datasets:
  - namfam/gsm8k-vietnamese
---

# GSM8K-VI — SPIN artifacts

Dữ liệu và kết quả đánh giá đi kèm model
[`{{MODEL_REPO}}`](https://huggingface.co/{{MODEL_REPO}}).

Dataset gốc: [`namfam/gsm8k-vietnamese`](https://huggingface.co/datasets/namfam/gsm8k-vietnamese).
Repo này chia lại split với seed 42 và lưu thêm các file sinh ra trong quá
trình chạy SPIN.

## Nội dung

```
data/
├── train.jsonl             # 4000 mẫu
├── validation.jsonl        # 500 mẫu
├── test.jsonl              # 500 mẫu
├── sft_train.jsonl         # train.jsonl ở format chat Qwen3 (messages)
├── sft_validation.jsonl    # validation.jsonl ở format chat Qwen3
├── train.parquet
├── validation.parquet
├── test.parquet
└── split_metadata.json     # seed, số mẫu, sha256

evaluation/
├── summary.json            # accuracy 6 model (n=500)
└── preds_{base,sft,spin1..spin4}.jsonl

run_metadata_session{1..4}.json
spin{1..4}_config.yaml
run_spin_with_shim.py
```

## Định dạng

`data/{train,validation,test}.jsonl` — mỗi dòng:

```json
{
  "problem": "câu hỏi toán tiếng Việt",
  "solution": "lời giải từng bước",
  "answer": "đáp án số"
}
```

`data/sft_*.jsonl` — mỗi dòng:

```json
{
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý toán học tiếng Việt..."},
    {"role": "user", "content": "câu hỏi"},
    {"role": "assistant", "content": "lời giải\nAnswer: <số>"}
  ]
}
```

`evaluation/preds_*.jsonl` — mỗi dòng gồm `problem`, `gold`, `pred`,
`correct`, `raw` (output đầy đủ của model).

## Split

| Split | Số mẫu |
|-------|-------:|
| train | 4000 |
| validation | 500 |
| test | 500 |
| **tổng dùng** | **5000** |

Tổng dataset gốc: 8792 mẫu; phần còn lại không dùng. Seed: `42`.

## Kết quả tham chiếu (test 500)

| Model | Accuracy |
|-------|---------:|
| base | 6.0% |
| sft | 33.6% |
| spin1 | 32.8% |
| spin2 | 34.8% |
| spin3 | 34.2% |
| spin4 | 32.6% |

## License

Dữ liệu phái sinh từ `namfam/gsm8k-vietnamese`; vui lòng kiểm tra license của
dataset gốc trước khi dùng lại. Code xử lý của dự án: MIT.
