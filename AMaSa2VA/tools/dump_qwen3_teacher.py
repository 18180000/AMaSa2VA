#!/usr/bin/env python3
"""Dump Qwen3 paper-aligned prompt-adapter records from MeViS train."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer

sa2va_repo_root = os.environ.get("SA2VA_REPO_ROOT", "").strip()
if sa2va_repo_root:
    root_path = Path(sa2va_repo_root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"SA2VA_REPO_ROOT does not exist: {root_path}")
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

from projects.sa2va.evaluation.dataset import RefVOSDataset

from projects.amasa2va.models.memory.sam2_patch import (
    PaperMemoryConfig,
    apply_paper_memory_patch,
    locate_sam2_module,
)


DATASET = {
    "image_folder": "video_datas/mevis/train/JPEGImages",
    "expression_file": "video_datas/mevis/train/meta_expressions.json",
    "mask_file": "video_datas/mevis/train/mask_dict.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--start_index", type=int, default=200)
    parser.add_argument("--end_index", type=int, default=700)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--max_empty_teacher_fraction", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_shard(records, output_dir: Path, index: int):
    path = output_dir / f"shard_{index:05d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(records, temporary)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"[PAPER_DUMP_SHARD] path={path} count={len(records)} sha256={digest}", flush=True)


def save_progress(path: Path, state: dict):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if not 0.0 <= args.max_empty_teacher_fraction < 1.0:
        raise ValueError("max_empty_teacher_fraction must be in [0, 1)")
    torch.set_grad_enabled(False)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "resume_state.json"
    if not args.resume:
        progress_path.unlink(missing_ok=True)
        for stale_shard in output_dir.glob("shard_*.pt"):
            stale_shard.unlink()
    manifest = {
        **vars(args),
        "memory_capacity": 8,
        "retrieval_k": 4,
        "retrieval_temperature": 0.4,
        "compression": "similarity_merge",
        "query_source": "projected_seg_prompt",
        "teacher_source": "offline_full_sequence_object_memory_residual_prompt",
        "teacher_memory_budget": 64,
        "teacher_top_k": 8,
        "teacher_temperature": 0.4,
        "teacher_residual_alpha": 0.5,
        "qwen_vlm_frame_policy": "official Sa2VA-Qwen path uses the first five frames",
        "sam2_frame_policy": "all video frames",
        "memory_mask_source": "baseline_predicted_mask",
        "sharding": "one atomic shard per successful expression for resumable generation",
    }
    (output_dir / "dump_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print("[QWEN_PROCESSOR] enabled=1 official_predict_forward=1 vlm_frames=first5 sam2_frames=all", flush=True)
    sam2 = locate_sam2_module(model)
    bank = apply_paper_memory_patch(
        sam2,
        PaperMemoryConfig(
            capacity=8,
            retrieval_k=4,
            temperature=0.4,
            compression="similarity_merge",
            adapter_checkpoint=None,
            capture_training_records=True,
            inject_enabled=False,
        ),
    )

    captured_prompts = []
    original_prompt_forward = model.text_hidden_fcs.forward

    def capture_prompt(hidden):
        prompt = original_prompt_forward(hidden)
        captured_prompts.append(prompt.detach().float().cpu())
        return prompt

    model.text_hidden_fcs.forward = capture_prompt
    root = Path(args.data_root)
    dataset = RefVOSDataset(
        image_folder=str(root / DATASET["image_folder"]),
        expression_file=str(root / DATASET["expression_file"]),
        mask_file=str(root / DATASET["mask_file"]),
    )
    end_index = args.end_index if args.end_index > 0 else len(dataset)
    end_index = min(end_index, len(dataset))
    if args.max_samples > 0:
        end_index = min(end_index, args.start_index + args.max_samples)
    if not 0 <= args.start_index < end_index:
        raise ValueError(f"invalid range [{args.start_index}, {end_index})")

    total = 0
    successful_expressions = 0
    skipped_empty_teacher = []
    teacher_delta_norms = []
    teacher_original_cosines = []
    future_evidence_records = 0
    selected_evidence_records = 0
    next_index = args.start_index
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["start_index"] != args.start_index or progress["end_index"] != end_index:
            raise RuntimeError("resume state range does not match the requested teacher range")
        next_index = int(progress["next_index"])
        total = int(progress["training_records"])
        successful_expressions = int(progress["successful_expressions"])
        skipped_empty_teacher = list(progress["skipped_empty_teacher"])
        teacher_delta_norms = list(progress["teacher_delta_norms"])
        teacher_original_cosines = list(progress["teacher_original_cosines"])
        future_evidence_records = int(progress["future_evidence_records"])
        selected_evidence_records = int(progress["selected_evidence_records"])
        print(
            f"[PAPER_DUMP_RESUME] next_index={next_index} records={total} "
            f"successful={successful_expressions} skipped={len(skipped_empty_teacher)}",
            flush=True,
        )

    def checkpoint_progress(next_dataset_index):
        save_progress(progress_path, {
            "start_index": args.start_index,
            "end_index": end_index,
            "next_index": next_dataset_index,
            "training_records": total,
            "successful_expressions": successful_expressions,
            "skipped_empty_teacher": skipped_empty_teacher,
            "teacher_delta_norms": teacher_delta_norms,
            "teacher_original_cosines": teacher_original_cosines,
            "future_evidence_records": future_evidence_records,
            "selected_evidence_records": selected_evidence_records,
        })

    for index in range(next_index, end_index):
        item = dataset[index]
        sample_id = f"{item['video_id']}::{item['exp_id']}"
        bank.clear()
        captured_prompts.clear()
        setattr(sam2, "_amasa2va_sample_id", sample_id)
        model.predict_forward(
            video=item["images"],
            text=item["text_prompt"],
            tokenizer=tokenizer,
            processor=processor,
        )
        if not captured_prompts or captured_prompts[-1].numel() == 0:
            raise RuntimeError(f"no projected SEG prompt for index={index} sample={sample_id}")
        original_prompt = captured_prompts[-1].reshape(-1, 256)[0]
        teacher_retrieval = bank.retrieve_offline(
            sample_id,
            original_prompt,
            retrieval_k=8,
            temperature=0.4,
            candidate_capacity=64,
        )
        if teacher_retrieval is None:
            skipped_empty_teacher.append({"dataset_index": index, "sample_id": sample_id})
            print(
                f"[PAPER_DUMP_SKIP] reason=empty_offline_object_memory "
                f"index={index} sample={sample_id}",
                flush=True,
            )
            checkpoint_progress(index + 1)
            torch.cuda.empty_cache()
            continue
        teacher = original_prompt + 0.5 * teacher_retrieval.evidence
        successful_expressions += 1
        teacher_delta_norms.append(float(torch.linalg.vector_norm(teacher - original_prompt).item()))
        teacher_original_cosines.append(
            float(F.cosine_similarity(teacher[None], original_prompt[None], dim=1)[0].item())
        )
        selected = [
            {
                "object_id": entry.object_id,
                "frame_start": entry.frame_start,
                "frame_end": entry.frame_end,
                "score": float(score),
                "weight": float(weight),
            }
            for entry, score, weight in zip(
                teacher_retrieval.entries,
                teacher_retrieval.scores,
                teacher_retrieval.weights,
            )
        ]
        expression_records = []
        for record in bank.training_records:
            frame_idx = int(record["frame_idx"])
            selected_evidence_records += 1
            if any(entry["frame_end"] > frame_idx for entry in selected):
                future_evidence_records += 1
            expression_records.append({
                "sample_id": f"{sample_id}::{frame_idx}",
                "frame_idx": frame_idx,
                "evidence": record["evidence"],
                "current_visual": record["current_visual"],
                "original_prompt": original_prompt,
                "teacher_evidence": teacher_retrieval.evidence,
                "teacher_prompt": teacher,
                "teacher_selected": selected,
                "query_source": "projected_seg_prompt",
                "teacher_source": "offline_full_sequence_topk8_residual",
            })
            total += 1
        if expression_records:
            save_shard(expression_records, output_dir, index)
        checkpoint_progress(index + 1)
        print(f"[PAPER_DUMP_PROGRESS] done={index - args.start_index + 1}/{end_index - args.start_index} records={total}", flush=True)
        torch.cuda.empty_cache()

    if total <= 0:
        raise RuntimeError("teacher dump produced zero training records")
    attempted_expressions = end_index - args.start_index
    empty_teacher_fraction = len(skipped_empty_teacher) / attempted_expressions
    if empty_teacher_fraction > args.max_empty_teacher_fraction:
        raise RuntimeError(
            f"empty offline object-memory fraction {empty_teacher_fraction:.6f} exceeds "
            f"limit {args.max_empty_teacher_fraction:.6f}"
        )
    if not teacher_delta_norms or min(teacher_delta_norms) <= 1e-6:
        raise RuntimeError("offline teacher did not produce a non-zero prompt target")
    if future_evidence_records <= 0:
        raise RuntimeError("offline teacher never selected evidence from a future frame")
    manifest.update({
        "attempted_expressions": attempted_expressions,
        "processed_expressions": successful_expressions,
        "skipped_empty_teacher_count": len(skipped_empty_teacher),
        "skipped_empty_teacher_fraction": empty_teacher_fraction,
        "skipped_empty_teacher": skipped_empty_teacher,
        "training_records": total,
        "teacher_delta_norm_min": min(teacher_delta_norms),
        "teacher_delta_norm_mean": sum(teacher_delta_norms) / len(teacher_delta_norms),
        "teacher_delta_norm_max": max(teacher_delta_norms),
        "teacher_original_cosine_min": min(teacher_original_cosines),
        "teacher_original_cosine_mean": sum(teacher_original_cosines) / len(teacher_original_cosines),
        "teacher_original_cosine_max": max(teacher_original_cosines),
        "selected_evidence_records": selected_evidence_records,
        "future_evidence_records": future_evidence_records,
    })
    (output_dir / "dump_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[PAPER_DUMP_COMPLETE] attempted={attempted_expressions} "
        f"processed={successful_expressions} skipped_empty={len(skipped_empty_teacher)} "
        f"records={total} future_evidence_records={future_evidence_records}",
        flush=True,
    )


if __name__ == "__main__":
    main()
