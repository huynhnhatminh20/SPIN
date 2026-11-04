"""SFT Qwen/Qwen3-0.6B on GSM8K-VI. Project adapter (NOT official SPIN).

Uses HF TRL SFTTrainer with Qwen3 chat template (enable_thinking=False).
Single A100 40GB: full finetune of 0.6B in bf16 fits easily (~2-4 GB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

MODEL_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="artifacts/data/sft_train.jsonl")
    ap.add_argument("--val_file", default="artifacts/data/sft_validation.jsonl")
    ap.add_argument("--out_dir", default="artifacts/sft")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--epochs", type=float, default=3.0)  # budget $5/session: train rẻ, ưu tiên chất lượng
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per_device_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    try:
        from trl import SFTTrainer, SFTConfig
        has_sft_config = True
    except ImportError:
        from trl import SFTTrainer
        has_sft_config = False

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # Smoke-test Qwen3 chat template with thinking disabled (main experiment).
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "1+1=?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # Qwen3 non-thinking still renders EMPTY <think></think> tags -- correct
    # behavior. Fail only if reasoning CONTENT leaks inside the block.
    import re as _re
    _m = _re.search(r"<think>(.*?)</think>", probe, _re.S)
    assert _m is None or not _m.group(1).strip(), \
        "thinking content leaked despite enable_thinking=False"
    print(f"[sft] chat-template probe OK: {probe[:120]!r}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    def fmt(ex):
        # Qwen3 template applied inside SFTTrainer via tokenizer; store text w/o thinking.
        text = tok.apply_chat_template(
            ex["messages"], tokenize=False, enable_thinking=False)
        return {"text": text}

    train_ds = train_ds.map(fmt, remove_columns=["messages"])
    val_ds = val_ds.map(fmt, remove_columns=["messages"])

    targs = dict(
        output_dir=args.out_dir, num_train_epochs=args.epochs,
        learning_rate=args.lr, per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16, fp16=not args.bf16,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, seed=args.seed, report_to="none",
        gradient_checkpointing=True, optim="adamw_torch",
    )
    if has_sft_config:
        # trl>=~0.15 (SFTConfig): length cap lives in the config as `max_length`.
        targs = SFTConfig(**targs, max_length=args.max_len)
        length_kwargs: dict = {}
    else:
        # old trl: TrainingArguments has no length cap; old SFTTrainer takes it.
        targs = TrainingArguments(**targs)
        length_kwargs = {"max_seq_length": args.max_len}

    trainer = SFTTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, **length_kwargs,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    Path(args.out_dir, "sft_metadata.json").write_text(
        json.dumps({"model_id": args.model_id, "epochs": args.epochs,
                    "lr": args.lr, "max_len": args.max_len, "seed": args.seed,
                    "thinking": False}, indent=2),
        encoding="utf-8")
    print(f"[sft] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
