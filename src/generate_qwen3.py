"""Qwen3 generation (k=1, fixed). Project adapter (NOT official SPIN).

Generates exactly ONE response per prompt (k=1 -- DO NOT change to k=4).
Qwen3 main experiment: enable_thinking=False, Temperature=0.7, TopP=0.8, TopK=20.
Output rows: {"real": [...], "generated": [...]} matching official SPIN input
(UCLA-AGI/SPIN_* format), prompt taken from `problem`, ground truth from solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SYSTEM_VI = (
    "Bạn là trợ lý toán học tiếng Việt. Giải từng bước rồi chốt đáp án "
    "trên dòng cuối theo đúng định dạng 'Answer: <số>'."
)


def load_prompts(train_file: Path):
    rows = [json.loads(l) for l in open(train_file, encoding="utf-8") if l.strip()]
    return rows


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="SFT ckpt (iter0) or SPIN-1 ckpt (iter1)")
    ap.add_argument("--train_file", default="artifacts/data/train.jsonl")
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)  # GSM8K-VI đáp án ngắn; 256 đủ, tiết kiệm giờ GPU
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)  # 0.6B nhẹ, batch lớn cho nhanh
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    rows = load_prompts(Path(args.train_file))
    print(f"[gen] model={args.model_path} prompts={len(rows)} k=1 thinking=False")

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), args.batch_size)):
            chunk = rows[i:i + args.batch_size]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_VI},
                     {"role": "user", "content": r["problem"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for r in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
            decoded = tok.batch_decode(
                gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, g in zip(chunk, decoded):
                g = g.strip()
                import re as _re
                _m = _re.search(r"<think>(.*?)</think>", g, _re.S)
                assert _m is None or not _m.group(1).strip(), \
                    "thinking content leaked despite enable_thinking=False"
                gt = r["solution"].strip()
                if "Answer:" not in gt:
                    gt = f"{gt}\nAnswer: {r['answer']}"
                fout.write(json.dumps({
                    "real": [{"role": "user", "content": r["problem"]},
                             {"role": "assistant", "content": gt}],
                    "generated": [{"role": "user", "content": r["problem"]},
                                  {"role": "assistant", "content": g}],
                }, ensure_ascii=False) + "\n")
    print(f"[gen] saved k=1 generations -> {out}")


if __name__ == "__main__":
    main()
