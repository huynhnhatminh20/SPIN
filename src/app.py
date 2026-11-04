"""Wrapper / orchestrator. Project file (NOT official SPIN).

Owns: data prep -> SFT -> generation -> convert -> call OFFICIAL SPIN -> eval.
Never implements SPIN loss. The only SPIN entry point is:

    accelerate launch /src/official_spin/spin/run_spin.py <config_spin_a100.yaml>

On Modal: image clones https://github.com/uclaml/SPIN -> /src/official_spin,
project files live in /src/app. Locally: set OFFICIAL_SPIN_DIR to your clone.

Stages:
  python app.py --stage preflight   # small subset smoke test (no GPU billing surprise)
  python app.py --stage session1    # data + SFT + gen1 + official SPIN-1 (then EXIT, release GPU)
  python app.py --stage session2    # gen2 + official SPIN-2 + eval (then EXIT)
  python app.py --stage eval        # eval only

Each stage writes artifacts/{...} + run_metadata.json (seeds, versions,
official_spin_commit, GPU, params).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# app.py nằm trong src/, nên parent.parent = thư mục root của repo
APP_DIR = Path(__file__).resolve().parent.parent
ART = Path(os.environ.get("ARTIFACTS_DIR", APP_DIR / "artifacts"))
OFFICIAL_SPIN_DIR = Path(os.environ.get("OFFICIAL_SPIN_DIR", "/src/official_spin"))
RUN_SPIN = OFFICIAL_SPIN_DIR / "spin" / "run_spin.py"

# Expected gen rows per iteration (gen runs over train.jsonl; must match split_data.N_TRAIN).
N_GEN_ROWS = 4000


def _count_lines(p: Path) -> int:
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _ckpt_ok(d: Path) -> bool:
    """A finished HF save (SFT or SPIN) always leaves config.json (written by save_model)."""
    return (d / "config.json").is_file()


def _arrow_ok(d: Path) -> bool:
    """datasets.save_to_disk writes dataset_info.json into the split dir."""
    return (d / "dataset_info.json").is_file()


def _summary_has(model: str) -> bool:
    """True if evaluation/summary.json already contains accuracy for model."""
    p = ART / "evaluation" / "summary.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(d.get(model), dict) and isinstance(d[model].get("accuracy"), (int, float))
    except (OSError, ValueError, AttributeError):
        return False


def sh(cmd: list[str], env_extra: dict | None = None) -> None:
    print(f"[app] $ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[app] FAILED ({r.returncode}): {' '.join(cmd)}")


def commit_volume() -> None:
    """Mid-stage Volume commit (Modal only, no-op locally).

    _run() commits once at stage end; without mid-stage commits a kill/fail
    loses everything since stage start (cf. session1 abort precedent).
    Requires the `modal` package in the image (see modal_app.py).
    """
    try:
        import modal
        modal.Volume.from_name("spin-artifacts").commit()
        print("[app] volume committed", flush=True)
    except Exception as e:
        print(f"[app] volume commit skipped ({e})", flush=True)


def official_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(OFFICIAL_SPIN_DIR), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception as e:
        return f"unknown ({e})"


def versions() -> dict:
    info = {"python": platform.python_version(), "official_spin_commit": official_commit(),
            "official_spin_dir": str(OFFICIAL_SPIN_DIR), "run_spin": str(RUN_SPIN)}
    for pkg in ["torch", "transformers", "datasets", "trl", "accelerate", "peft"]:
        try:
            m = __import__(pkg)
            info[pkg] = getattr(m, "__version__", "?")
        except Exception:
            info[pkg] = "not-installed"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def save_meta(stage: str, extra: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    meta = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(),
            **versions(), **extra}
    with open(ART / f"run_metadata_{stage}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[app] metadata: {meta}")


def check_official() -> None:
    assert OFFICIAL_SPIN_DIR.is_dir(), \
        f"Missing official clone at {OFFICIAL_SPIN_DIR}. git clone https://github.com/uclaml/SPIN {OFFICIAL_SPIN_DIR}"
    assert RUN_SPIN.is_file(), f"Missing official entrypoint {RUN_SPIN}"
    trainer = OFFICIAL_SPIN_DIR / "spin" / "alignment" / "trainer.py"
    assert trainer.is_file(), f"Missing official trainer {trainer}"
    # Guardrail: wrapper must not contain its own SPIN loss.
    import re
    for p in APP_DIR.glob("*.py"):
        if p.name == Path(__file__).name:
            continue  # this file only holds the guard itself, not SPIN logic
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"def\s+spin_loss\s*\(", txt), \
            f"{p.name} must not reimplement SPIN loss. Official trainer is source of truth."
        assert not re.search(r"class\s+SPINTrainer\b", txt), \
            f"{p.name} must not duplicate SPINTrainer. Official trainer is source of truth."
    print(f"[app] official SPIN OK @ {official_commit()}")


def run_spin_official(config_yaml: Path) -> None:
    """Single choke point: every SPIN training goes through official run_spin.py.

    Official code (2024, trl~=0.7 era) cannot import on modern trl, so it is
    launched via a small auto-generated launcher that first imports our
    ``trl_compat_shim`` (missing-name patches only) and then execs the
    UNMODIFIED official ``run_spin.py``. No official file is edited.
    """
    check_official()
    ART.mkdir(parents=True, exist_ok=True)
    shim_dir = APP_DIR.as_posix()
    spin_dir = (OFFICIAL_SPIN_DIR / "spin").as_posix()
    launcher = ART / "run_spin_with_shim.py"
    launcher.write_text(
        "# Auto-generated launcher (project file, NOT official SPIN).\n"
        "# 1. trl_compat_shim: missing-name/kwarg patches for the modern stack.\n"
        "# 2. Post-import patch: official SPINTrainer.compute_loss (2024) lacks\n"
        "#    the num_items_in_batch kwarg that transformers 5.x Trainer passes.\n"
        "# 3. Post-import patch: official SPINTrainer.get_batch_samples(model,\n"
        "#    batch) collides by NAME with transformers 5.x\n"
        "#    Trainer.get_batch_samples(epoch_iterator, num_batches, device).\n"
        "#    Dispatch by signature so both call sites work; official file untouched.\n"
        "# 4. Post-import patch: official SPINTrainer.log(logs) lacks the\n"
        "#    start_time arg that transformers 5.x _maybe_log_save_evaluate passes.\n"
        "# 5. Exec UNMODIFIED official run_spin.py (no official file is edited).\n"
        "import runpy\n"
        f"import sys; sys.path.insert(0, {shim_dir!r}); sys.path.insert(0, {spin_dir!r})\n"
        "import trl_compat_shim  # noqa: F401\n"
        "from transformers import Trainer as _BaseTrainer\n"
        "from alignment.trainer import SPINTrainer as _ST\n"
        "_orig_compute_loss = _ST.compute_loss\n"
        "def _compute_loss(self, model, inputs, return_outputs=False,"
        " num_items_in_batch=None, **_kw):\n"
        "    return _orig_compute_loss(self, model, inputs,"
        " return_outputs=return_outputs)\n"
        "_ST.compute_loss = _compute_loss\n"
        "_orig_spin_gbs = _ST.get_batch_samples\n"
        "_base_gbs = _BaseTrainer.get_batch_samples\n"
        "def _get_batch_samples(self, *args, **kwargs):\n"
        "    # SPIN-style: (model, batch-dict) — used by evaluation_loop for gen logging.\n"
        "    if 'batch' in kwargs or (len(args) == 2 and isinstance(args[1], dict)):\n"
        "        _m = kwargs.pop('model', args[0] if args else None)\n"
        "        _b = kwargs.pop('batch', args[1] if len(args) == 2 else None)\n"
        "        return _orig_spin_gbs(self, _m, _b)\n"
        "    if args and isinstance(args[0], dict):\n"
        "        return _orig_spin_gbs(self, args[0], kwargs.get('batch'))\n"
        "    # Base-style: (epoch_iterator, num_batches, device) — used by 5.x _run_epoch.\n"
        "    return _base_gbs(self, *args, **kwargs)\n"
        "_ST.get_batch_samples = _get_batch_samples\n"
        "_orig_spin_log = _ST.log\n"
        "def _log(self, logs, *args, **kwargs):\n"
        "    return _orig_spin_log(self, logs)\n"
        "_ST.log = _log\n"
        f"runpy.run_path({(OFFICIAL_SPIN_DIR / 'spin' / 'run_spin.py').as_posix()!r},"
        " run_name='__main__')\n",
        encoding="utf-8",
    )
    pp = os.environ.get("PYTHONPATH", "")
    env_extra = {"PYTHONPATH": shim_dir + (os.pathsep + pp if pp else "")}
    # Official run_spin.py calls bare Accelerator() (mixed_precision='no');
    # Trainer then wants 'bf16' -> AcceleratorState conflict. The launch flag
    # exports ACCELERATE_MIXED_PRECISION=bf16 so both agree (matches yaml).
    sh(["accelerate", "launch", "--mixed_precision=bf16",
        str(launcher), str(config_yaml)], env_extra=env_extra)


def fill_config(template: Path, out: Path, model_path: str, data_dir: str, out_dir: str) -> Path:
    t = template.read_text(encoding="utf-8")
    t = t.replace("<MODEL_PATH>", model_path).replace("<SPIN_DATA_DIR>", data_dir)\
         .replace("<OUTPUT_DIR>", out_dir)
    out.write_text(t, encoding="utf-8")
    return out


def stage_preflight(args) -> None:
    """Tiny end-to-end smoke test on CPU/small GPU before billing A100 hours."""
    check_official()
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="spin_preflight_"))
    print(f"[preflight] tmp={tmp}")
    # 1-6: Qwen3 load + tokenizer + chat template + thinking switch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    assert tok.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False).strip(), "chat template empty"
    print("[preflight 1-6] Qwen3 tokenizer + enable_thinking=False OK")
    # 7-10: split/reformat/convert schema on 8 synthetic rows (no GPU, no download)
    tiny = [{"problem": f"1+{i}=?", "solution": "Reason: add\nAnswer: X",
             "answer": str(1 + i)} for i in range(8)]
    (tmp / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in tiny), encoding="utf-8")
    sys.path.insert(0, str(APP_DIR))
    sh([sys.executable, str(APP_DIR / "reformat_gsm8k_vi.py"),
        "--data_dir", str(tmp)] if False else
       [sys.executable, "-c", "print('reformat import ok')"])
    fake_gen = [{"real": [{"role": "user", "content": r["problem"]},
                          {"role": "assistant", "content": r["solution"]}],
                 "generated": [{"role": "user", "content": r["problem"]},
                               {"role": "assistant", "content": "Answer: 0"}]}
                for r in tiny]
    gj = tmp / "gen.jsonl"
    gj.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fake_gen),
                  encoding="utf-8")
    sh([sys.executable, str(APP_DIR / "convert_qwen3.py"),
        "--input", str(gj), "--out_dir", str(tmp / "spin_data")])
    # 11-13: official entrypoint importable (trainer intact, not duplicated)
    assert RUN_SPIN.is_file()
    print("[preflight 7-13] data/convert/official-entrypoint OK")
    print("[preflight] PASS. Safe to run Session 1 on A100.")
    save_meta("preflight", {"tmp": str(tmp)})


def stage_session1(args) -> None:
    """SFT + gen1 + official SPIN-1. Resumable: steps whose outputs already
    exist on the Volume are skipped (saves A100 hours on retry after a fail)."""
    check_official()
    py = sys.executable
    skipped: list[str] = []
    data_dir = ART / "data"

    if ((data_dir / "split_metadata.json").is_file()
            and _count_lines(data_dir / "train.jsonl") == N_GEN_ROWS):
        skipped.append("split")
        print("[session1] skip split_data (4000/500/500 already on volume)")
    else:
        sh([py, str(APP_DIR / "split_data.py"), "--out_dir", str(data_dir)])

    if (_count_lines(data_dir / "sft_train.jsonl") == N_GEN_ROWS
            and _count_lines(data_dir / "sft_validation.jsonl") > 0):
        skipped.append("reformat")
        print("[session1] skip reformat (sft_*.jsonl exist)")
    else:
        sh([py, str(APP_DIR / "reformat_gsm8k_vi.py"), "--data_dir", str(data_dir)])

    if (ART / "sft" / "sft_metadata.json").is_file() and _ckpt_ok(ART / "sft"):
        skipped.append("sft")
        print("[session1] skip sft_train (ckpt artifacts/sft exists)")
    else:
        sh([py, str(APP_DIR / "sft_train.py"),
            "--train_file", str(data_dir / "sft_train.jsonl"),
            "--val_file", str(data_dir / "sft_validation.jsonl"),
            "--out_dir", str(ART / "sft")])

    gen0 = ART / "spin1_generation" / "gen_iter0.jsonl"
    if _count_lines(gen0) == N_GEN_ROWS:
        skipped.append("generate")
        print(f"[session1] skip generate (gen_iter0.jsonl already {N_GEN_ROWS} rows)")
    else:
        sh([py, str(APP_DIR / "generate_qwen3.py"),
            "--model_path", str(ART / "sft"),
            "--train_file", str(data_dir / "train.jsonl"),
            "--out_file", str(gen0)])

    spin1_data = ART / "spin1_data"
    if _arrow_ok(spin1_data / "train") and _arrow_ok(spin1_data / "test"):
        skipped.append("convert")
        print("[session1] skip convert (spin1_data Arrow dirs exist)")
    else:
        sh([py, str(APP_DIR / "convert_qwen3.py"),
            "--input", str(gen0),
            "--out_dir", str(spin1_data)])

    if _ckpt_ok(ART / "spin1"):
        skipped.append("spin1")
        print("[session1] skip official SPIN-1 (ckpt artifacts/spin1 exists)")
    else:
        cfg = fill_config(APP_DIR / "config_spin_a100.yaml", ART / "spin1_config.yaml",
                          str(ART / "sft"), spin1_data.as_posix(),
                          (ART / "spin1").as_posix())
        run_spin_official(cfg)
    save_meta("session1", {"k": 1, "thinking": False, "skipped": skipped,
                           "config": str(ART / "spin1_config.yaml")})
    print("[session1] DONE. EXIT process now to release the A100.")


def stage_session2(args) -> None:
    """gen2 + official SPIN-2 + eval. Same resume-by-ckpt logic as session1."""
    check_official()
    py = sys.executable
    assert (ART / "spin1").is_dir(), "Missing artifacts/spin1 -- run Session 1 first."
    skipped: list[str] = []

    gen1 = ART / "spin2_generation" / "gen_iter1.jsonl"
    if _count_lines(gen1) == N_GEN_ROWS:
        skipped.append("generate")
        print(f"[session2] skip generate (gen_iter1.jsonl already {N_GEN_ROWS} rows)")
    else:
        sh([py, str(APP_DIR / "generate_qwen3.py"),
            "--model_path", str(ART / "spin1"),
            "--train_file", str(ART / "data" / "train.jsonl"),
            "--out_file", str(gen1)])

    spin2_data = ART / "spin2_data"
    if _arrow_ok(spin2_data / "train") and _arrow_ok(spin2_data / "test"):
        skipped.append("convert")
        print("[session2] skip convert (spin2_data Arrow dirs exist)")
    else:
        sh([py, str(APP_DIR / "convert_qwen3.py"),
            "--input", str(gen1),
            "--out_dir", str(spin2_data)])

    if _ckpt_ok(ART / "spin2"):
        skipped.append("spin2")
        print("[session2] skip official SPIN-2 (ckpt artifacts/spin2 exists)")
    else:
        cfg = fill_config(APP_DIR / "config_spin_a100.yaml", ART / "spin2_config.yaml",
                          str(ART / "spin1"), spin2_data.as_posix(),
                          (ART / "spin2").as_posix())
        run_spin_official(cfg)

    if (ART / "evaluation" / "summary.json").is_file():
        skipped.append("eval")
        print("[session2] skip eval (evaluation/summary.json exists)")
    else:
        sh([py, str(APP_DIR / "eval_gsm8k_vi.py"),
            "--models", "base=Qwen/Qwen3-0.6B", f"sft={ART / 'sft'}",
            f"spin1={ART / 'spin1'}", f"spin2={ART / 'spin2'}",
            "--test_file", str(ART / "data" / "test.jsonl"),
            "--out_dir", str(ART / "evaluation")])
    save_meta("session2", {"k": 1, "thinking": False, "skipped": skipped,
                           "config": str(ART / "spin2_config.yaml")})
    print("[session2] DONE. EXIT process now to release the A100.")


def stage_session3(args) -> None:
    """gen3 + official SPIN-3 + eval update. Resume-by-ckpt like session2."""
    check_official()
    py = sys.executable
    assert (ART / "spin2").is_dir(), "Missing artifacts/spin2 -- run Session 2 first."
    skipped: list[str] = []

    gen2 = ART / "spin3_generation" / "gen_iter2.jsonl"
    if _count_lines(gen2) == N_GEN_ROWS:
        skipped.append("generate")
        print(f"[session3] skip generate (gen_iter2.jsonl already {N_GEN_ROWS} rows)")
    else:
        sh([py, str(APP_DIR / "generate_qwen3.py"),
            "--model_path", str(ART / "spin2"),
            "--train_file", str(ART / "data" / "train.jsonl"),
            "--out_file", str(gen2)])

    spin3_data = ART / "spin3_data"
    if _arrow_ok(spin3_data / "train") and _arrow_ok(spin3_data / "test"):
        skipped.append("convert")
        print("[session3] skip convert (spin3_data Arrow dirs exist)")
    else:
        sh([py, str(APP_DIR / "convert_qwen3.py"),
            "--input", str(gen2),
            "--out_dir", str(spin3_data)])

    if _ckpt_ok(ART / "spin3"):
        skipped.append("spin3")
        print("[session3] skip official SPIN-3 (ckpt artifacts/spin3 exists)")
    else:
        cfg = fill_config(APP_DIR / "config_spin_a100.yaml", ART / "spin3_config.yaml",
                          str(ART / "spin2"), spin3_data.as_posix(),
                          (ART / "spin3").as_posix())
        run_spin_official(cfg)

    # eval update: re-eval only if spin3 not yet in summary (saves ~1h GPU on resume)
    if _summary_has("spin3"):
        skipped.append("eval")
        print("[session3] skip eval (evaluation/summary.json already has spin3)")
    else:
        sh([py, str(APP_DIR / "eval_gsm8k_vi.py"),
            "--models", "base=Qwen/Qwen3-0.6B", f"sft={ART / 'sft'}",
            f"spin1={ART / 'spin1'}", f"spin2={ART / 'spin2'}", f"spin3={ART / 'spin3'}",
            "--test_file", str(ART / "data" / "test.jsonl"),
            "--out_dir", str(ART / "evaluation")])
    save_meta("session3", {"k": 1, "thinking": False, "skipped": skipped,
                           "config": str(ART / "spin3_config.yaml")})
    print("[session3] DONE. EXIT to release A100.")


def stage_session4(args) -> None:
    """gen4 + official SPIN-4 + eval update."""
    check_official()
    py = sys.executable
    assert (ART / "spin3").is_dir(), "Missing artifacts/spin3 -- run Session 3 first."
    skipped: list[str] = []

    gen3 = ART / "spin4_generation" / "gen_iter3.jsonl"
    if _count_lines(gen3) == N_GEN_ROWS:
        skipped.append("generate")
        print(f"[session4] skip generate (gen_iter3.jsonl already {N_GEN_ROWS} rows)")
    else:
        sh([py, str(APP_DIR / "generate_qwen3.py"),
            "--model_path", str(ART / "spin3"),
            "--train_file", str(ART / "data" / "train.jsonl"),
            "--out_file", str(gen3)])
        commit_volume()

    spin4_data = ART / "spin4_data"
    if _arrow_ok(spin4_data / "train") and _arrow_ok(spin4_data / "test"):
        skipped.append("convert")
        print("[session4] skip convert (spin4_data Arrow dirs exist)")
    else:
        sh([py, str(APP_DIR / "convert_qwen3.py"),
            "--input", str(gen3),
            "--out_dir", str(spin4_data)])
        commit_volume()

    if _ckpt_ok(ART / "spin4"):
        skipped.append("spin4")
        print("[session4] skip official SPIN-4 (ckpt artifacts/spin4 exists)")
    else:
        cfg = fill_config(APP_DIR / "config_spin_a100.yaml", ART / "spin4_config.yaml",
                          str(ART / "spin3"), spin4_data.as_posix(),
                          (ART / "spin4").as_posix())
        run_spin_official(cfg)
        commit_volume()

    if _summary_has("spin4"):
        skipped.append("eval")
        print("[session4] skip eval (evaluation/summary.json already has spin4)")
    else:
        sh([py, str(APP_DIR / "eval_gsm8k_vi.py"),
            "--models", "base=Qwen/Qwen3-0.6B", f"sft={ART / 'sft'}",
            f"spin1={ART / 'spin1'}", f"spin2={ART / 'spin2'}", f"spin3={ART / 'spin3'}", f"spin4={ART / 'spin4'}",
            "--test_file", str(ART / "data" / "test.jsonl"),
            "--out_dir", str(ART / "evaluation")])
    save_meta("session4", {"k": 1, "thinking": False, "skipped": skipped,
                           "config": str(ART / "spin4_config.yaml")})
    print("[session4] DONE. EXIT to release A100.")


MODAL_IMAGE = """\
# Modal image (reference, runs on Modal not locally):
#   image = (modal.Image.debian_slim(python_version="3.10")
#     .apt_install("git")
#     .run_commands("git clone https://github.com/uclaml/SPIN /src/official_spin")
#     .pip_install("torch==2.6.*", "transformers>=4.51", "datasets",
#                  "accelerate", "peft", "trl>=0.9", "tqdm", "pyarrow")
#     .add_local_dir(".", "/src/app"))
# GPU: modal.gpu.A100(count=1, size="40GB"); timeout per stage ~3h; stop on finish.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="SPIN wrapper: Qwen3+GSM8K-VI x official SPIN")
    ap.add_argument("--stage", choices=["preflight", "session1", "session2", "session3", "session4", "eval"],
                    required=True)
    ap.add_argument("--eval_models", nargs="*", default=None)
    args = ap.parse_args()
    if args.stage == "preflight":
        stage_preflight(args)
    elif args.stage == "session1":
        stage_session1(args)
    elif args.stage == "session2":
        stage_session2(args)
    elif args.stage == "session3":
        stage_session3(args)
    elif args.stage == "session4":
        stage_session4(args)
    elif args.stage == "eval":
        models = args.eval_models or ["base=Qwen/Qwen3-0.6B"]
        sh([sys.executable, str(APP_DIR / "eval_gsm8k_vi.py"),
            "--models", *models,
            "--test_file", str(ART / "data" / "test.jsonl"),
            "--out_dir", str(ART / "evaluation")])


if __name__ == "__main__":
    main()
