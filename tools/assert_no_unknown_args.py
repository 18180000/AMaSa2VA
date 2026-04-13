#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from typing import List, Set

BANNED = {
    "--max_frames_per_video",
    "--frame_stride",
    "--pl_adapter_alpha",
}


def extract_help_flags(help_text: str) -> Set[str]:
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", help_text))


def extract_argv_flags(argv: List[str]) -> Set[str]:
    flags: Set[str] = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
            flags.add(flag)
        i += 1
    return flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--allowed-flags-file", default="")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    argv = args.argv
    if argv and argv[0] == "--":
        argv = argv[1:]

    help_flags: Set[str]
    cache_path = args.allowed_flags_file.strip()
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            help_flags = {x.strip() for x in f if x.strip()}
    else:
        help_cmd = [args.python, "-u", args.entry, "--help"]
        env = dict(os.environ)
        fallback_py = "/home/usergjf/Sa2VA/Sa2VA-main:/home/usergjf/sallm_final"
        env["PYTHONPATH"] = f"{fallback_py}:{env.get('PYTHONPATH', '')}".rstrip(":")
        proc = subprocess.run(
            help_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            print(f"[FAIL_FAST] HELP_QUERY_FAILED code={proc.returncode} cmd={' '.join(help_cmd)}")
            print(proc.stdout)
            return 2

        help_flags = extract_help_flags(proc.stdout)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                for flag in sorted(help_flags):
                    f.write(flag + "\n")
    argv_flags = extract_argv_flags(argv)

    unknown = sorted(f for f in argv_flags if f not in help_flags)
    banned_present = sorted(f for f in argv_flags if f in BANNED)

    if unknown or banned_present:
        print(
            "[FAIL_FAST] UNKNOWN_ARGS "
            f"unknown={unknown} banned={banned_present} entry={args.entry}"
        )
        return 1

    print(
        "[PROGRESS] ASSERT_NO_UNKNOWN_ARGS_PASS "
        f"n_argv_flags={len(argv_flags)} n_help_flags={len(help_flags)} cache={cache_path or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
