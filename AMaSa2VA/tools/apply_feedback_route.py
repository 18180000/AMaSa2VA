#!/usr/bin/env python3
"""Create Feedback results by routing each expression between Base and Core."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_mean_gates(path):
    gates = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("event") == "prompt_route":
                gates[item["video_id"]].append(float(item["cosine_gate"]))
    return {key: float(np.mean(values)) for key, values in gates.items() if values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results", required=True)
    parser.add_argument("--core_results", required=True)
    parser.add_argument("--core_audit", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = load(args.base_results)
    core = load(args.core_results)
    threshold_config = load(args.threshold)
    threshold = float(threshold_config["selected"]["threshold"])
    gates = load_mean_gates(args.core_audit)
    routed = copy.deepcopy(base)
    audit = {"threshold": threshold, "core_count": 0, "base_count": 0, "missing_gate": 0, "units": []}
    for video_id, expressions in routed.items():
        for exp_id in expressions:
            unit_id = f"{video_id}::{exp_id}"
            gate = gates.get(unit_id)
            use_core = gate is not None and gate >= threshold
            if use_core:
                routed[video_id][exp_id] = core[video_id][exp_id]
                audit["core_count"] += 1
            else:
                audit["base_count"] += 1
                audit["missing_gate"] += int(gate is None)
            audit["units"].append({"unit_id": unit_id, "mean_g_cos": gate, "branch": "core" if use_core else "base"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(routed), encoding="utf-8")
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "units"}, indent=2))


if __name__ == "__main__":
    main()
