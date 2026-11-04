"""Dataset split: namfam/gsm8k-vietnamese -> fixed train 4000 / val 500 / test 500.

Project adapter (NOT official SPIN). Only prepares data; never touches SPIN loss.
Source: HF `namfam/gsm8k-vietnamese` (~8.79k rows, splits train/test/val,
fields: problem, solution, answer). We concatenate all available rows,
shuffle with a fixed seed, then take the first 5000 -> deterministic split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
N_TRAIN, N_VAL, N_TEST = 4000, 500, 500


def load_all_rows(dataset_id: str = "namfam/gsm8k-vietnamese"):
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(dataset_id)
    print(f"[split] available splits: {list(ds.keys())}")
    parts = [ds[s] for s in ds.keys()]
    full = concatenate_datasets(parts)
    return full, ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="namfam/gsm8k-vietnamese")
    ap.add_argument("--out_dir", default="artifacts/data")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full, _ = load_all_rows(args.dataset_id)
    total = len(full)
    need = N_TRAIN + N_VAL + N_TEST
    assert total >= need, f"Dataset has {total} rows < needed {need}"

    shuffled = full.shuffle(seed=args.seed)
    train = shuffled.select(range(0, N_TRAIN))
    val = shuffled.select(range(N_TRAIN, N_TRAIN + N_VAL))
    test = shuffled.select(range(N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST))

    for name, split in [("train", train), ("validation", val), ("test", test)]:
        rows = [dict(r) for r in split]
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        split.to_parquet(out / f"{name}.parquet")
        print(f"[split] {name}: {len(rows)} -> {out / (name + '.jsonl')}")

    # Reproducibility metadata (+ content hash of selected indices)
    meta = {
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "n_train": N_TRAIN,
        "n_validation": N_VAL,
        "n_test": N_TEST,
        "total_available": total,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_train": hashlib.sha256(
            json.dumps([r["problem"] for r in train], ensure_ascii=False).encode()
        ).hexdigest()[:16],
    }
    with open(out / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[split] metadata: {meta}")


if __name__ == "__main__":
    main()
