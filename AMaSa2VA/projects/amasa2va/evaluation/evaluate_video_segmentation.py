#!/usr/bin/env python3
"""Evaluate video-segmentation predictions with J, F, and J&F metrics.

The script follows the upstream Sa2VA prediction JSON contract:
{video_id: {expression_id: {"prediction_masks": [COCO RLE, ...]}}}.
Dataset paths are explicit CLI arguments so this file is portable.
"""

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as cocomask

sa2va_repo_root = os.environ.get("SA2VA_REPO_ROOT", "").strip()
if sa2va_repo_root:
    root_path = Path(sa2va_repo_root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"SA2VA_REPO_ROOT does not exist: {root_path}")
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

try:
    from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Sa2VA video-segmentation metrics are unavailable; set SA2VA_REPO_ROOT "
        "to the upstream Sa2VA checkout"
    ) from exc

try:
    from third_parts.revos.utils.metircs import (
        get_r2vos_accuracy,
        get_r2vos_robustness,
    )
except ImportError:
    get_r2vos_accuracy = None
    get_r2vos_robustness = None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode_mask(rle, shape):
    if not rle:
        return np.zeros(shape, dtype=np.uint8)
    decoded = cocomask.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.sum(axis=2)
    if decoded.shape != shape:
        raise ValueError(f"mask shape mismatch: expected={shape} actual={decoded.shape}")
    return (decoded > 0).astype(np.uint8)


def assign_davis_anno_ids(exp_dict):
    anno_count = 0
    for video_id in sorted(exp_dict):
        video = exp_dict[video_id]
        video["frames"] = sorted(video["frames"])
        for exp_id in sorted(video["expressions"]):
            video["expressions"][exp_id]["anno_id"] = [anno_count]
            anno_count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="Sa2VA prediction JSON")
    parser.add_argument("--expressions", type=Path, required=True, help="metadata JSON")
    parser.add_argument("--masks", type=Path, required=True, help="GT mask JSON or DAVIS pickle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foreground", type=Path, help="optional foreground masks for A/R metrics")
    parser.add_argument("--dataset", required=True, help="dataset label stored in the output")
    parser.add_argument("--davis-pickle", action="store_true", help="load GT masks with pickle")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16")
    cv2.setNumThreads(1)

    exp_dict = load_json(args.expressions)["videos"]
    if args.davis_pickle:
        assign_davis_anno_ids(exp_dict)
        with args.masks.open("rb") as handle:
            mask_dict = pickle.load(handle)
    else:
        mask_dict = load_json(args.masks)
    foreground_dict = load_json(args.foreground) if args.foreground else None
    predictions = load_json(args.predictions)

    work_items = []
    for video_id, video_predictions in predictions.items():
        if video_id not in exp_dict:
            continue
        video = exp_dict[video_id]
        for exp_id, prediction in video_predictions.items():
            if exp_id in video["expressions"]:
                work_items.append((video_id, exp_id, prediction))

    def evaluate_unit(item):
        video_id, exp_id, prediction = item
        video = exp_dict[video_id]
        frames = video["frames"]
        pred_rles = prediction.get("prediction_masks", [])
        if len(pred_rles) != len(frames):
            raise ValueError(
                f"{video_id}::{exp_id}: expected {len(frames)} masks, got {len(pred_rles)}"
            )
        if not pred_rles:
            return None
        shape = tuple(pred_rles[0]["size"])
        gt_masks = np.zeros((len(frames), *shape), dtype=np.uint8)
        pred_masks = np.zeros_like(gt_masks)
        foreground_masks = np.zeros_like(gt_masks) if foreground_dict is not None else None
        anno_ids = video["expressions"][exp_id].get("anno_id", [int(exp_id)])

        for frame_idx in range(len(frames)):
            for anno_id in anno_ids:
                gt_masks[frame_idx] |= decode_mask(mask_dict[str(anno_id)][frame_idx], shape)
            pred_masks[frame_idx] = decode_mask(pred_rles[frame_idx], shape)
            if foreground_masks is not None:
                foreground_masks[frame_idx] = decode_mask(
                    foreground_dict[video_id]["masks_rle"][frame_idx], shape
                )

        j_score = float(db_eval_iou(gt_masks, pred_masks).mean())
        f_score = float(db_eval_boundary(gt_masks, pred_masks).mean())
        result = {
            "unit_id": f"{video_id}::{exp_id}",
            "video_id": video_id,
            "exp_id": exp_id,
            "J": 100.0 * j_score,
            "F": 100.0 * f_score,
            "JF": 50.0 * (j_score + f_score),
        }
        if foreground_masks is not None:
            if get_r2vos_accuracy is None or get_r2vos_robustness is None:
                raise RuntimeError("ReVOS A/R metrics are unavailable in this evaluator environment")
            result["A"] = 100.0 * float(get_r2vos_accuracy(gt_masks, pred_masks).mean())
            result["R"] = 100.0 * float(
                get_r2vos_robustness(gt_masks, pred_masks, foreground_masks).mean()
            )
            result["type_id"] = int(video["expressions"][exp_id]["type_id"])
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        units = [unit for unit in executor.map(evaluate_unit, work_items) if unit is not None]

    result = {
        "dataset": args.dataset,
        "count": len(units),
        "J": float(np.mean([item["J"] for item in units])) if units else None,
        "F": float(np.mean([item["F"] for item in units])) if units else None,
        "JF": float(np.mean([item["JF"] for item in units])) if units else None,
        "units": units,
    }
    if foreground_dict is not None and units:
        for label, type_ids in (("referring", {0}), ("reason", {1}), ("overall", {0, 1})):
            selected = [item for item in units if item["type_id"] in type_ids]
            if selected:
                result[label] = {
                    key: float(np.mean([item[key] for item in selected]))
                    for key in ("J", "F", "JF", "A", "R")
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "units"}, indent=2))


if __name__ == "__main__":
    main()
