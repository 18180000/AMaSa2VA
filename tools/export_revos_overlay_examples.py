#!/opt/anaconda3/envs/torch/bin/python
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def decode_mask(mask_rle):
    mask = cocomask.decode(mask_rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(image).astype(np.float32)
    out = arr.copy()

    color = np.array([255, 64, 64], dtype=np.float32)
    alpha = 0.45
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    # Simple boundary highlight.
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    out[boundary] = np.array([255, 255, 0], dtype=np.float32)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def export_one_sample(results_by_mode, image_root: Path, out_root: Path, video_id: str, exp_id: str):
    sample_dir = out_root / f"{sanitize(video_id)}__exp_{sanitize(exp_id)}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = None
    for mode, results in results_by_mode.items():
        if video_id in results and exp_id in results[video_id]:
            meta = results[video_id][exp_id]
            break
    if meta is None:
        raise KeyError(f"missing sample video_id={video_id} exp_id={exp_id}")

    info_path = sample_dir / "info.txt"
    with info_path.open("w", encoding="utf-8") as f:
        f.write(f"video_id={video_id}\n")
        f.write(f"exp_id={exp_id}\n")
        f.write(f"expression={meta.get('exp', '')}\n")

    for mode, results in results_by_mode.items():
        if video_id not in results or exp_id not in results[video_id]:
            continue

        item = results[video_id][exp_id]
        mode_dir = sample_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        frames = item["frames"]
        pred_masks = item["prediction_masks"]
        for frame_name, mask_rle in zip(frames, pred_masks):
            img_path = image_root / video_id / f"{frame_name}.jpg"
            if not img_path.exists():
                img_path = image_root / video_id / f"{frame_name}.png"
            if not img_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            mask = decode_mask(mask_rle)
            overlay = overlay_mask(image, mask)
            overlay.save(mode_dir / f"{frame_name}_overlay.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--core-results", required=True)
    parser.add_argument("--feedback-results", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--sample-keys",
        default="",
        help="Comma-separated explicit sample keys like video_id|exp_id",
    )
    args = parser.parse_args()

    if args.sample_keys.strip():
        picked = [(key.strip(), 0.0) for key in args.sample_keys.split(",") if key.strip()]
    else:
        unit = json.load(open(args.unit, "r", encoding="utf-8"))
        ranked = [
            (k, v["mean_JF"])
            for k, v in unit.items()
            if isinstance(v, dict) and "mean_JF" in v
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        picked = ranked[: args.topk]

    results_by_mode = {
        "base": json.load(open(args.base_results, "r", encoding="utf-8")),
        "core": json.load(open(args.core_results, "r", encoding="utf-8")),
        "feedback": json.load(open(args.feedback_results, "r", encoding="utf-8")),
    }

    out_root = Path(args.out_root)
    image_root = Path(args.image_root)
    out_root.mkdir(parents=True, exist_ok=True)

    picked_info = []
    for sample_key, score in picked:
        video_id, exp_id = sample_key.split("|", 1)
        export_one_sample(results_by_mode, image_root, out_root, video_id, exp_id)
        picked_info.append({"sample_key": sample_key, "mean_JF": score})

    with (out_root / "picked_samples.json").open("w", encoding="utf-8") as f:
        json.dump(picked_info, f, indent=2, ensure_ascii=False)

    print(f"[PROGRESS] EXPORTED_SAMPLES n={len(picked_info)} out={out_root}")
    for item in picked_info:
        print(f"[PROGRESS] PICKED {item['sample_key']} mean_JF={item['mean_JF']:.4f}")


if __name__ == "__main__":
    main()
