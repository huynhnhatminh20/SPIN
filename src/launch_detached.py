"""Phóng job Modal DETACHED: thoát local/chat mà job vẫn chạy độc lập trên cloud.

Cách dùng (từ thư mục này):
    python launch_detached.py session1   # SFT + gen + SPIN-1 (~2-2.5h A100)
    python launch_detached.py session2   # gen + SPIN-2 + eval
    python launch_detached.py session3   # gen3 + SPIN-3 + eval (cần spin2)
    python launch_detached.py session4   # gen4 + SPIN-4 + eval (cần spin3)
    python launch_detached.py status     # xem artifacts (chạy ngay, CPU)

Yêu cầu: đã `python -m modal deploy modal_app.py` ít nhất 1 lần.
In ra call ID — giữ lại để tra log: python -m modal app logs <app-id>.
"""
from __future__ import annotations

import sys

import modal

APP_NAME = "qwen3-spin-gsm8k-vi"


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("session1", "session2", "session3", "session4", "status", "preflight"):
        print(__doc__)
        raise SystemExit(1)
    stage = sys.argv[1]
    fn = modal.Function.from_name(APP_NAME, stage)
    if stage == "status":
        print(fn.remote())
        return
    call = fn.spawn()  # detached: client thoát, function vẫn chạy tới timeout
    print(f"[launch] {stage} spawned detached. call_id={call.object_id}")
    print("[launch] Tra log: python -m modal app logs -f <app-id> (xem URL khi deploy/run)")


if __name__ == "__main__":
    main()
