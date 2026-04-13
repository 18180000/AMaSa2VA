#!/opt/anaconda3/envs/torch/bin/python
import argparse
import json
import os
from shutil import copy2
from typing import List


def bounce_indices(n_frames: int, factor: int) -> List[int]:
    """
    XMem paper protocol: "playing the video back and forth" to n× length.
    We build a mirrored timeline over period (2*n_frames-2):
      0,1,...,n-1,n-2,...,1,0,1,...
    and then truncate to exactly n_frames * factor.
    """
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


def main() -> int:
    p = argparse.ArgumentParser(description="Build XMem-style back-and-forth copied MEVIS_U set.")
    p.add_argument("--src_dir", required=True, help="Source MEVIS_U dir with meta_expressions.json, mask_dict.json, JPEGImages/")
    p.add_argument("--out_dir", required=True, help="Output copied dataset dir")
    p.add_argument("--factor", type=int, default=3, help="Length factor n×")
    p.add_argument("--materialize_images", type=int, default=1, choices=[0, 1],
                   help="If 1, create expanded JPEGImages/<vid>/<new_frame>.jpg with unique frame names.")
    p.add_argument("--image_link_mode", type=str, default="hardlink", choices=["hardlink", "copy"],
                   help="How to materialize frames when materialize_images=1.")
    args = p.parse_args()

    src_meta = os.path.join(args.src_dir, "meta_expressions.json")
    src_mask = os.path.join(args.src_dir, "mask_dict.json")
    src_img = os.path.join(args.src_dir, "JPEGImages")
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(src_meta, "r", encoding="utf-8"))
    src_meta_obj = json.load(open(src_meta, "r", encoding="utf-8"))
    mask = json.load(open(src_mask, "r", encoding="utf-8"))
    videos = meta.get("videos", {})
    if not videos:
        raise SystemExit("[FAIL_FAST] EMPTY_META_VIDEOS")

    if args.factor < 1:
        raise SystemExit("[FAIL_FAST] factor must be >= 1")

    # Keep source frame order for diagnostics.
    src_len_by_vid = {}
    # Build per-video mirrored index sequence.
    vid_to_idx = {}
    for vid, v in videos.items():
        frames = v["frames"]
        src_len_by_vid[vid] = len(frames)
        idx = bounce_indices(len(frames), args.factor)
        vid_to_idx[vid] = idx
        # IMPORTANT: dataset loader sorts frames lexicographically.
        # Use monotonic unique frame names so sorted(frames) keeps expanded timeline.
        v["frames"] = [f"{t:05d}" for t in range(len(idx))]

    # Remap mask timeline for each anno_id based on its video.
    # anno_id in expressions can be int or str.
    anno_to_vid = {}
    for vid, v in videos.items():
        for exp in v["expressions"].values():
            for a in exp.get("anno_id", []):
                anno_to_vid[str(a)] = vid

    new_mask = {}
    for anno_id, seq in mask.items():
        vid = anno_to_vid.get(str(anno_id))
        if vid is None:
            # Keep untouched if anno_id is not referenced.
            new_mask[str(anno_id)] = seq
            continue
        idx = vid_to_idx[vid]
        new_mask[str(anno_id)] = [seq[i] for i in idx]

    out_meta = os.path.join(args.out_dir, "meta_expressions.json")
    out_mask = os.path.join(args.out_dir, "mask_dict.json")
    json.dump(meta, open(out_meta, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(new_mask, open(out_mask, "w", encoding="utf-8"))

    out_img = os.path.join(args.out_dir, "JPEGImages")
    if args.materialize_images == 1:
        os.makedirs(out_img, exist_ok=True)
        for vid, v in videos.items():
            idx = vid_to_idx[vid]
            src_frames = src_meta_obj["videos"][vid]["frames"]
            out_vid_dir = os.path.join(out_img, vid)
            os.makedirs(out_vid_dir, exist_ok=True)
            for t, src_i in enumerate(idx):
                src_name = src_frames[src_i]
                src_file = os.path.join(src_img, vid, f"{src_name}.jpg")
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
    else:
        if os.path.lexists(out_img):
            if os.path.islink(out_img) or os.path.isfile(out_img):
                os.unlink(out_img)
        if not os.path.exists(out_img):
            os.symlink(src_img, out_img)

    sample_vid = next(iter(videos))
    sample_src_len = src_len_by_vid[sample_vid]
    sample_idx_preview = vid_to_idx[sample_vid][:20]
    print(
        f"[PROGRESS] COPYSET_OK factor={args.factor} out={args.out_dir} "
        f"videos={len(videos)} sample_vid={sample_vid} sample_len={len(videos[sample_vid]['frames'])} "
        f"sample_src_len~={sample_src_len}"
    )
    print(
        f"[PROGRESS] COPYSET_PROTOCOL xmem_back_and_forth "
        f"sample_idx_preview={sample_idx_preview}"
    )
    print(f"[PROGRESS] COPYSET_FILES meta={out_meta} mask={out_mask} images={out_img} materialize_images={args.materialize_images} image_link_mode={args.image_link_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
