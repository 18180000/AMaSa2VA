#!/opt/anaconda3/envs/torch/bin/python
import argparse
import json
import os
from shutil import copy2
from typing import Dict, List


def bounce_indices(n_frames: int, factor: int) -> List[int]:
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    target = n_frames * factor
    if n_frames == 1:
        return [0] * target
    period = 2 * n_frames - 2
    out: List[int] = []
    for t in range(target):
        p = t % period
        idx = p if p < n_frames else (period - p)
        out.append(idx)
    return out


def ensure_anno_ids(videos: Dict) -> Dict[str, str]:
    anno_to_vid = {}
    need_fill = False
    for vid_data in videos.values():
        for exp in vid_data.get("expressions", {}).values():
            if "anno_id" not in exp:
                need_fill = True
                break
        if need_fill:
            break

    if not need_fill:
        for vid, vid_data in videos.items():
            for exp in vid_data.get("expressions", {}).values():
                for anno_id in exp.get("anno_id", []):
                    anno_to_vid[str(anno_id)] = vid
        return anno_to_vid

    anno_count = 0
    for vid in sorted(videos.keys()):
        exps = videos[vid].get("expressions", {})
        for exp_id in sorted(exps.keys(), key=lambda x: str(x)):
            exps[exp_id]["anno_id"] = [anno_count]
            anno_to_vid[str(anno_count)] = vid
            anno_count += 1
    print(f"[PROGRESS] ANNO_ID_FILLED count={anno_count}")
    return anno_to_vid


def main() -> int:
    p = argparse.ArgumentParser(description="Build XMem-style copied ref-VOS dataset.")
    p.add_argument("--src-meta", required=True)
    p.add_argument("--src-mask", required=True)
    p.add_argument("--src-images", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--image-link-mode", type=str, default="hardlink", choices=["hardlink", "copy"])
    args = p.parse_args()

    meta = json.load(open(args.src_meta, "r", encoding="utf-8"))
    mask = json.load(open(args.src_mask, "r", encoding="utf-8"))
    src_meta_obj = json.load(open(args.src_meta, "r", encoding="utf-8"))
    videos = meta.get("videos", {})
    if not videos:
      raise SystemExit("[FAIL_FAST] EMPTY_META_VIDEOS")

    os.makedirs(args.out_dir, exist_ok=True)
    anno_to_vid = ensure_anno_ids(videos)

    src_len_by_vid = {}
    vid_to_idx = {}
    for vid, v in videos.items():
        frames = v["frames"]
        src_len_by_vid[vid] = len(frames)
        idx = bounce_indices(len(frames), args.factor)
        vid_to_idx[vid] = idx
        v["frames"] = [f"{t:05d}" for t in range(len(idx))]

    new_mask = {}
    for anno_id, seq in mask.items():
        vid = anno_to_vid.get(str(anno_id))
        if vid is None:
            new_mask[str(anno_id)] = seq
            continue
        idx = vid_to_idx[vid]
        new_mask[str(anno_id)] = [seq[i] for i in idx]

    out_meta = os.path.join(args.out_dir, "meta_expressions.json")
    out_mask = os.path.join(args.out_dir, "mask_dict.json")
    json.dump(meta, open(out_meta, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(new_mask, open(out_mask, "w", encoding="utf-8"))

    out_img = os.path.join(args.out_dir, "JPEGImages")
    os.makedirs(out_img, exist_ok=True)
    for vid, v in videos.items():
        idx = vid_to_idx[vid]
        src_frames = src_meta_obj["videos"][vid]["frames"]
        out_vid_dir = os.path.join(out_img, vid)
        os.makedirs(out_vid_dir, exist_ok=True)
        for t, src_i in enumerate(idx):
            src_name = src_frames[src_i]
            src_file = os.path.join(args.src_images, vid, f"{src_name}.jpg")
            dst_file = os.path.join(out_vid_dir, f"{t:05d}.jpg")
            if os.path.exists(dst_file):
                continue
            if args.image_link_mode == "hardlink":
                try:
                    os.link(src_file, dst_file)
                except OSError:
                    copy2(src_file, dst_file)
            else:
                copy2(src_file, dst_file)

    sample_vid = next(iter(videos))
    print(
        f"[PROGRESS] COPYSET_OK factor={args.factor} out={args.out_dir} "
        f"videos={len(videos)} sample_vid={sample_vid} sample_len={len(videos[sample_vid]['frames'])} "
        f"sample_src_len={src_len_by_vid[sample_vid]}"
    )
    print(f"[PROGRESS] COPYSET_FILES meta={out_meta} mask={out_mask} images={out_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
