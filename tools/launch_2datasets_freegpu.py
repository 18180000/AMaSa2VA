#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Task:
    dataset: str
    gpu: int
    proc: subprocess.Popen
    log_path: str
    log_file: object
    start_ts: float
    startup_checked: bool = False


def parse_gpu_state() -> List[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, util, mem = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "util": float(util), "mem": float(mem)})
    return rows


def get_free_gpus(exclude: Optional[set] = None) -> List[int]:
    exclude = exclude or set()
    free = []
    for g in parse_gpu_state():
        if g["index"] == 0:
            continue
        if g["index"] in exclude:
            continue
        if g["mem"] < 2000 and g["util"] < 10:
            free.append(g["index"])
    return sorted(free)


def tail_text(path: str, n: int = 80) -> str:
    if not os.path.exists(path):
        return f"<missing log: {path}>"
    cmd = ["tail", "-n", str(n), path]
    return subprocess.check_output(cmd, text=True, errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--modes", default="core,feedback")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--poll_sec", type=float, default=5.0)
    args = ap.parse_args()

    req_modes = [x.strip().lower() for x in args.modes.split(",") if x.strip()]
    if req_modes != ["core", "feedback"]:
        raise SystemExit("[FAIL_FAST] ONLY_CORE_FEEDBACK_SUPPORTED expected=core,feedback")

    all_ds = [x.strip() for x in args.datasets.split(",") if x.strip()]
    pending = []
    for d in all_ds:
        if d == "rvos_valid":
            print("[PROGRESS] SKIP dataset=rvos_valid reason=disabled_by_user", flush=True)
            continue
        if d == "mevis_valid_u":
            print("[PROGRESS] SKIP_ALREADY_DONE dataset=mevis_valid_u reason=final_feedback_54.64", flush=True)
            continue
        if d not in ("davis17_valid", "revos_valid"):
            raise SystemExit(f"[FAIL_FAST] UNKNOWN_DATASET dataset={d}")
        pending.append(d)

    os.makedirs(os.path.join(args.out_root, "logs"), exist_ok=True)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runner = os.path.join(root_dir, "scripts", "run_one_dataset_2modes.sh")
    assert_tool = os.path.join(root_dir, "tools", "assert_no_unknown_args.py")
    flags_cache = os.path.join(root_dir, "artifacts", "help_flags.txt")

    subprocess.check_call(
        [
            args.python,
            "-u",
            assert_tool,
            "--python",
            args.python,
            "--entry",
            "/home/usergjf/sallm_final/scripts/sa2va_eval_ref_vos_patched.py",
            "--allowed-flags-file",
            flags_cache,
            "--",
        ]
    )

    running: Dict[str, Task] = {}

    while pending or running:
        done = []
        for ds, task in running.items():
            ret = task.proc.poll()
            now = time.time()

            if not task.startup_checked:
                dt = now - task.start_ts
                if ret is not None and dt < 30:
                    print(f"[FAIL_FAST] EARLY_EXIT dataset={ds} gpu={task.gpu} code={ret}", flush=True)
                    print(f"[FAIL_FAST] EARLY_EXIT_LOGTAIL dataset={ds}\n{tail_text(task.log_path, 80)}", flush=True)
                    return 1
                if dt >= 30 and ret is None:
                    task.startup_checked = True
                    print(f"[PROGRESS] STARTUP_OK_30S dataset={ds} gpu={task.gpu}", flush=True)

            if ret is not None:
                print(f"[PROGRESS] DATASET_EXIT dataset={ds} gpu={task.gpu} code={ret}", flush=True)
                done.append(ds)

        for ds in done:
            task = running.pop(ds)
            try:
                task.log_file.close()
            except Exception:
                pass

        if pending:
            used = {t.gpu for t in running.values()}
            free = get_free_gpus(used)
            # Prefer the most idle card for revos first.
            if "revos_valid" in pending and free:
                pending.remove("revos_valid")
                pending.insert(0, "revos_valid")
            while pending and free:
                ds = pending.pop(0)
                gpu = free.pop(0)
                log_path = os.path.join(args.out_root, "logs", f"{ds}_full.log")
                logf = open(log_path, "a", buffering=1)
                cmd = ["bash", runner, ds, str(gpu), args.python, args.out_root]
                print(f"[PROGRESS] LAUNCH dataset={ds} gpu={gpu} log={log_path}", flush=True)
                proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=os.environ.copy())
                running[ds] = Task(
                    dataset=ds,
                    gpu=gpu,
                    proc=proc,
                    log_path=log_path,
                    log_file=logf,
                    start_ts=time.time(),
                )
            if pending and not free:
                print(f"[PROGRESS] WAIT_FREE_GPU pending={pending}", flush=True)

        time.sleep(args.poll_sec)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
