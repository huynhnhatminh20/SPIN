"""Reformat GSM8K-VI splits into SFT training rows with Qwen3 chat format.

Project adapter (NOT official SPIN).
Input : artifacts/data/{train,validation}.jsonl  (fields: problem, solution, answer)
Output: artifacts/data/sft_train.jsonl + sft_validation.jsonl
Each row: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Ground-truth target = `solution` + "\\nAnswer: <answer>" so eval parsing is stable.
No thinking traces are injected (enable_thinking=False is applied at
tokenizer.apply_chat_template time in sft_train.py, not stored in data).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_VI = (
    "Bạn là trợ lý toán học tiếng Việt. Giải từng bước rồi chốt đáp án "
    "trên dòng cuối theo đúng định dạng 'Answer: <số>'."
)


def to_messages(problem: str, solution: str, answer) -> dict:
    target = solution.strip()
    ans = str(answer).strip()
    if "Answer:" not in target:
        target = f"{target}\nAnswer: {ans}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_VI},
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": target},
        ]
    }


def convert(in_path: Path, out_path: Path) -> int:
    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            row = to_messages(r.get("problem", ""), r.get("solution", ""), r.get("answer", ""))
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data")
    args = ap.parse_args()
    d = Path(args.data_dir)
    for src, dst in [("train.jsonl", "sft_train.jsonl"), ("validation.jsonl", "sft_validation.jsonl")]:
        n = convert(d / src, d / dst)
        print(f"[reformat] {src} -> {dst}: {n} rows")


if __name__ == "__main__":
    main()
