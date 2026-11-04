"""Modal entrypoint. Project file (NOT official SPIN).

Builds an image that clones official uclaml/SPIN -> /src/official_spin and
mounts THIS directory (the wrapper) -> /src/app. Artifacts persist across
Session 1 / Session 2 via a Modal Volume (this is what makes the two
sessions independent processes).

Deploy / run (from this directory, with `modal` installed + logged in):
    modal run modal_app.py::preflight
    modal run modal_app.py::session1     # then container exits, GPU released
    modal run modal_app.py::session2     # loads Volume artifacts, no Session-1 process needed

Cost safety: one A100 40GB per function, timeout 3h each, no retries.
"""
from __future__ import annotations

import modal

OFFICIAL_SPIN_URL = "https://github.com/uclaml/SPIN"
OFFICIAL_SPIN_DIR = "/src/official_spin"
APP_DIR = "/src/app"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(f"git clone {OFFICIAL_SPIN_URL} {OFFICIAL_SPIN_DIR}")
    # NOTE: official setup.py pins transformers==4.37 (no Qwen3 support).
    # Deliberately NOT `pip install /src/official_spin/.`; install a Qwen3-capable
    # stack instead. SPIN algorithm code (trainer.py) is version-agnostic here.
    .pip_install(
        "torch>=2.6",
        "transformers>=4.51",
        "datasets>=2.14",
        "accelerate>=0.23",
        "peft>=0.6",
        "trl>=0.9",
        "tqdm",
        "pyarrow",
        "huggingface_hub",
        "modal",  # mid-stage Volume.commit() from inside app.py (session4+)
    )
    .add_local_dir(".", remote_path=APP_DIR)
)

app = modal.App("qwen3-spin-gsm8k-vi")
artifacts = modal.Volume.from_name("spin-artifacts", create_if_missing=True)
ART = "/artifacts"
GPU = "A100-40GB"


def _run(stage: str) -> None:
    import os
    import subprocess

    env = dict(
        os.environ,
        ARTIFACTS_DIR=ART,
        OFFICIAL_SPIN_DIR=OFFICIAL_SPIN_DIR,
        HF_HOME="/root/.cache/huggingface",
    )
    r = subprocess.run(
        ["python", f"{APP_DIR}/app.py", "--stage", stage],
        env=env,
        cwd=APP_DIR,
    )
    artifacts.commit()
    if r.returncode != 0:
        raise SystemExit(f"stage {stage} failed ({r.returncode})")


@app.function(image=image, volumes={ART: artifacts}, timeout=1800, retries=0)
def preflight() -> None:
    # CPU là đủ (chỉ load tokenizer + validate schema, không train/gen) -> $0 GPU.
    _run("preflight")


@app.function(image=image, gpu=GPU, volumes={ART: artifacts}, timeout=2 * 3600, retries=0)
def session1() -> None:
    _run("session1")


@app.function(image=image, gpu=GPU, volumes={ART: artifacts}, timeout=2 * 3600, retries=0)
def session2() -> None:
    _run("session2")


@app.function(image=image, gpu=GPU, volumes={ART: artifacts}, timeout=3 * 3600, retries=0)
def session3() -> None:
    _run("session3")


@app.function(image=image, gpu=GPU, volumes={ART: artifacts}, timeout=3 * 3600, retries=0)
def session4() -> None:
    _run("session4")


@app.function(image=image, gpu="T4", volumes={ART: artifacts}, timeout=1800, retries=0)
def spottest(models: str = "spin2", n: int = 50) -> str:
    """Quick GPU spot-eval (cheap T4): accuracy over first n test rows.

    Usage: python -m modal run modal_app.py::spottest --models spin1,spin2 --n 50
    Uses the EXACT eval logic (greedy, same prompt/extract). Read-only.
    """
    import sys

    sys.path.insert(0, APP_DIR)
    from eval_gsm8k_vi import SYSTEM_VI, evaluate  # noqa
    from eval_gsm8k_vi import extract, norm  # noqa

    import json
    from pathlib import Path

    test_file = Path(f"{ART}/data/test.jsonl")
    rows_all = [json.loads(l) for l in open(test_file, encoding="utf-8")
                if l.strip()]
    # reuse evaluate() on a truncated test file (keeps batching/extract identical)
    tmp = Path("/tmp/spottest.jsonl")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows_all[:n]), encoding="utf-8")
    out_lines = []
    for name in [m.strip() for m in models.split(",") if m.strip()]:
        acc, preds = evaluate(f"{ART}/{name}", tmp, 256, 16)
        ok = sum(1 for p in preds if p["correct"])
        out_lines.append(f"{name}: {acc * 100:.1f}% ({ok}/{len(preds)})")
    print("[spottest]\n" + "\n".join(out_lines), flush=True)
    return "\n".join(out_lines)


@app.function(image=image, volumes={ART: artifacts}, timeout=600, retries=0)
def status() -> str:
    """CPU: liệt kê artifacts + metadata để kiểm tra tiến trình từ xa."""
    import json
    import subprocess
    from pathlib import Path

    r = subprocess.run(["find", ART, "-maxdepth", "3"], capture_output=True, text=True)
    lines = [f"ARTIFACTS:\n{r.stdout}"]
    for meta in sorted(Path(ART).glob("run_metadata_*.json")):
        try:
            d = json.loads(meta.read_text())
            lines.append(f"{meta.name}: stage={d.get('stage')} utc={d.get('utc')}")
        except Exception as e:
            lines.append(f"{meta.name}: unreadable ({e})")
    return "\n".join(lines)
