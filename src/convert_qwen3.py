"""Convert Qwen3 generations to official SPIN parquet format. Project adapter.

Official SPIN (spin/convert_data.py + run_spin.py) expects a dataset (local dir
or Hub) with splits train/test where each row is:
  {"real": [{"role":"user",...},{"role":"assistant",...}],
   "generated": [{"role":"user",...},{"role":"assistant",...}]}
This script validates that schema, drops empty generations, and writes:
  out_dir/train/train_prefs-00000-of-00001.parquet
  out_dir/test/test_prefs-00000-of-00001.parquet
(test split = small slice of train for run_spin.py's mandatory eval dataset;
real eval is done separately by eval_gsm8k_vi.py on the fixed test 500.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq  # noqa: F401  (ensures engine present)
from datasets import Dataset


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def check_row(r: dict, i: int) -> bool:
    if not all(k in r for k in ("real", "generated")):
        print(f"[convert] row {i}: missing real/generated -> drop");
        return False
    for k in ("real", "generated"):
        msgs = r[k]
        if not (isinstance(msgs, list) and len(msgs) == 2
                and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant"):
            print(f"[convert] row {i}: bad schema in '{k}' -> drop")
            return False
        if not msgs[1].get("content", "").strip():
            print(f"[convert] row {i}: empty assistant in '{k}' -> drop")
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generation .jsonl from generate_qwen3.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    clean = [r for i, r in enumerate(rows) if check_row(r, i)]
    print(f"[convert] {len(rows)} -> {len(clean)} valid (dropped {len(rows)-len(clean)})")
    assert clean, "no valid rows left after validation"

    ds = Dataset.from_list(clean).shuffle(seed=args.seed)
    n_eval = max(1, int(len(ds) * args.eval_frac))
    test, train = ds.select(range(n_eval)), ds.select(range(n_eval, len(ds)))

    out = Path(args.out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    # save_to_disk (Arrow dir), NOT bare parquet: official run_spin.py loads
    # local data via load_from_disk(os.path.join(ds, split)), which requires
    # Arrow dataset dirs. Bare *.parquet would fail there.
    # (Official convert_data.py writes flat parquet for Hub upload; the Hub
    # path goes through load_dataset(ds, split=...). Local path needs this.)
    train.save_to_disk(str(out / "train"))
    test.save_to_disk(str(out / "test"))
    print(f"[convert] train={len(train)} test={len(test)} -> {out}")
    print(f"[convert] dataset_mixer entry for config: \"{out.as_posix()}: 1.0\"")


if __name__ == "__main__":
    main()
