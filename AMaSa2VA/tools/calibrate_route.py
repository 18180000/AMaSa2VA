#!/usr/bin/env python3
"""Calibrate the paper route threshold from disjoint Base/Core unit metrics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_units(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["unit_id"]: item for item in data["units"]}


def load_mean_gates(path):
    gates = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("event") != "prompt_route":
                continue
            gates[item["video_id"]].append(float(item["cosine_gate"]))
    return {sample_id: float(np.mean(values)) for sample_id, values in gates.items() if values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_metrics", required=True)
    parser.add_argument("--core_metrics", required=True)
    parser.add_argument("--core_audit", required=True)
    parser.add_argument("--calibration_split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = load_units(args.base_metrics)
    core = load_units(args.core_metrics)
    gates = load_mean_gates(args.core_audit)
    unit_ids = sorted(set(base) & set(core) & set(gates))
    if not unit_ids:
        raise RuntimeError("no aligned Base/Core/g_cos calibration units")
    candidates = sorted({0.0, 1.0, *[gates[unit_id] for unit_id in unit_ids]})
    trials = []
    for threshold in candidates:
        scores = []
        core_count = 0
        for unit_id in unit_ids:
            use_core = gates[unit_id] >= threshold
            scores.append(core[unit_id]["JF"] if use_core else base[unit_id]["JF"])
            core_count += int(use_core)
        trials.append({
            "threshold": threshold,
            "JF": float(np.mean(scores)),
            "core_count": core_count,
            "base_count": len(unit_ids) - core_count,
        })
    # Resolve equal-JF thresholds conservatively: prefer fewer Core units and
    # then the higher threshold.
    best = max(trials, key=lambda item: (item["JF"], -item["core_count"], item["threshold"]))
    output = {
        "calibration_split": args.calibration_split,
        "calibration_disjoint_from_reported_splits": True,
        "count": len(unit_ids),
        "compatibility_feature": "mean frame-level g_cos per expression",
        "direction": "core_if_ge",
        "selected": best,
        "base_JF": float(np.mean([base[unit_id]["JF"] for unit_id in unit_ids])),
        "core_JF": float(np.mean([core[unit_id]["JF"] for unit_id in unit_ids])),
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
