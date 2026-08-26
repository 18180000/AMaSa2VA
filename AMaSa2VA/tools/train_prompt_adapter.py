#!/usr/bin/env python3
"""Train the paper-aligned 512-to-256 prompt adapter."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from projects.amasa2va.models.memory.adapter import PaperPromptAdapter


class RecordDataset(Dataset):
    def __init__(self, shard_dir: Path):
        self.records = []
        for path in sorted(shard_dir.glob("shard_*.pt")):
            self.records.extend(torch.load(path, map_location="cpu", weights_only=False))
        if not self.records:
            raise RuntimeError(f"no records found in {shard_dir}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        return (
            item["evidence"].float().reshape(256),
            item["current_visual"].float().reshape(256),
            item["teacher_prompt"].float().reshape(256),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RecordDataset(Path(args.shard_dir))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    model = PaperPromptAdapter(dim=256, hidden_dim=1024).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    iterator = iter(loader)
    started = time.time()
    last_loss = None
    for step in range(1, args.steps + 1):
        try:
            evidence, current_visual, teacher = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            evidence, current_visual, teacher = next(iterator)
        evidence = evidence.cuda(non_blocking=True)
        current_visual = current_visual.cuda(non_blocking=True)
        teacher = teacher.cuda(non_blocking=True)
        prediction = model(evidence, current_visual)
        cosine_loss = (1.0 - F.cosine_similarity(prediction, teacher, dim=1)).mean()
        mse_loss = F.mse_loss(prediction, teacher)
        loss = cosine_loss + 0.1 * mse_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {float(loss)}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        if step == 1 or step % 100 == 0:
            print(
                f"[PAPER_TRAIN] step={step}/{args.steps} loss={last_loss:.6f} "
                f"cos={float(cosine_loss):.6f} mse={float(mse_loss):.6f}",
                flush=True,
            )

    checkpoint = output_dir / "paper_prompt_adapter.pt"
    torch.save(model.state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        **vars(args),
        "records": len(dataset),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "final_loss": last_loss,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "loss": "1-cosine+0.1*mse",
        "input": "concat(topk_evidence_256,current_fpn_mean_256)",
        "output": "projected_seg_prompt_256",
    }
    (output_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PAPER_TRAIN_COMPLETE] checkpoint={checkpoint} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
