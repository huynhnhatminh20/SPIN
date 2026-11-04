"""Evaluate checkpoints on fixed test 500 (GSM8K-VI). Project adapter.

Greedy decode (do_sample=False) with enable_thinking=False, extract the last
number after 'Answer:' (fallback: last number in text), normalize
(commas/spaces/$ removed) and exact-match. Reports per-model accuracy +
JSON results file. Never touches official SPIN code.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SYSTEM_VI = (
    "Bạn là trợ lý toán học tiếng Việt. Giải từng bước rồi chốt đáp án "
    "trên dòng cuối theo đúng định dạng 'Answer: <số>'."
)


def extract(text: str) -> str:
    m = re.findall(r"[Aa]nswer\s*:\s*([\-]?[\d\.,\s]+)", text)
    cand = m[-1] if m else None
    if cand is None:
        nums = re.findall(r"\-?[\d][\d\.,]*", text)
        cand = nums[-1] if nums else ""
    return cand.replace(",", "").replace(" ", "").replace("$", "").strip().rstrip(".")


def norm(a: str) -> str:
    return a.replace(",", "").replace(" ", "").replace("$", "").strip().rstrip(".")


@torch.inference_mode()
def evaluate(model_path: str, test_file: Path, max_new_tokens: int, batch_size: int):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    rows = [json.loads(l) for l in open(test_file, encoding="utf-8") if l.strip()]
    correct, preds = 0, []
    for i in tqdm(range(0, len(rows), batch_size), desc=model_path):
        chunk = rows[i:i + batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_VI},
             {"role": "user", "content": r["problem"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, o in zip(chunk, outs):
            p, g = extract(o), norm(str(r["answer"]))
            ok = (p == g) and p != ""
            correct += ok
            preds.append({"problem": r["problem"], "gold": g, "pred": p,
                          "correct": ok, "raw": o})
    return correct / max(1, len(rows)), preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="'name=path' pairs, e.g. base=Qwen/Qwen3-0.6B sft=artifacts/sft")
    ap.add_argument("--test_file", default="artifacts/data/test.jsonl")
    ap.add_argument("--out_dir", default="artifacts/evaluation")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="re-eval even models already present in summary.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Merge mode: keep previous results, only eval models missing from summary
    # (greedy decode is deterministic, re-eval of old models only burns GPU).
    summary: dict = {}
    sp = out / "summary.json"
    try:
        prev = json.loads(sp.read_text(encoding="utf-8"))
        if isinstance(prev, dict):
            summary.update(prev)
    except (OSError, ValueError):
        pass
    for item in args.models:
        name, path = item.split("=", 1)
        if (not args.force and isinstance(summary.get(name), dict)
                and isinstance(summary[name].get("accuracy"), (int, float))):
            print(f"[eval] {name}: skip (already {summary[name]['accuracy']*100:.2f}%)")
            continue
        acc, preds = evaluate(path, Path(args.test_file),
                              args.max_new_tokens, args.batch_size)
        summary[name] = {"accuracy": acc, "n": len(preds), "path": path}
        with open(out / f"preds_{name}.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[eval] {name}: {acc*100:.2f}% ({len(preds)} samples)")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
