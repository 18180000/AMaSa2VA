#!/usr/bin/env python3
import argparse
import os
import re

import pandas as pd


def _pick_letter(pred):
    if pred is None:
        return None
    if not isinstance(pred, str):
        pred = str(pred)
    m = re.search(r"\b([ABCD])\b", pred)
    if m:
        return m.group(1)
    m = re.match(r"\s*([ABCD])[\.\)]?", pred.strip())
    if m:
        return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0, help="limit denylist size (0 = no limit)")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    if "index" not in df.columns or "answer" not in df.columns or "prediction" not in df.columns:
        raise SystemExit("missing required columns: index/answer/prediction")

    deny = []
    for _, row in df.iterrows():
        ans = str(row["answer"]).strip()
        pred = row["prediction"]
        idx = int(row["index"])
        pred_letter = _pick_letter(pred)
        if pred_letter is None or pred_letter != ans:
            deny.append(idx)

    if args.max and len(deny) > args.max:
        deny = deny[: args.max]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for idx in deny:
            f.write(f"{idx}\n")

    print(f"[PROGRESS] DENYLIST_WRITTEN n={len(deny)} out={args.out}")


if __name__ == "__main__":
    main()
