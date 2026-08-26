from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryEntry:
    object_id: int
    frame_start: int
    frame_end: int
    token: torch.Tensor


@dataclass(frozen=True)
class RetrievalResult:
    evidence: torch.Tensor
    entries: Tuple[MemoryEntry, ...]
    scores: torch.Tensor
    weights: torch.Tensor


class ObjectCentricMemoryBank:
    """Equation (1)-(5) object memory with explicit causal bookkeeping."""

    def __init__(
        self,
        capacity: int = 8,
        retrieval_k: Optional[int] = 4,
        temperature: float = 0.4,
        compression: str = "similarity_merge",
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if retrieval_k is not None and retrieval_k < 1:
            raise ValueError("retrieval_k must be positive or None")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if compression not in {"none", "fifo", "similarity_merge"}:
            raise ValueError(f"unsupported compression: {compression}")
        self.capacity = capacity
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.compression = compression
        self._bank: DefaultDict[str, DefaultDict[int, List[MemoryEntry]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._staging: DefaultDict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        self._raw_entries: DefaultDict[str, List[MemoryEntry]] = defaultdict(list)
        self.audit_events: List[dict] = []
        self.training_records: List[dict] = []

    def clear(self) -> None:
        self._bank.clear()
        self._staging.clear()
        self._raw_entries.clear()
        self.audit_events.clear()
        self.training_records.clear()

    def stage(self, video_id: str, frame_idx: int, feature_map: torch.Tensor) -> None:
        feature_map = feature_map.detach()
        if feature_map.ndim == 4:
            if feature_map.shape[0] != 1:
                raise ValueError("stage expects a single frame feature map")
            feature_map = feature_map[0]
        if feature_map.ndim != 3:
            raise ValueError(f"expected CxHxW feature map, got {tuple(feature_map.shape)}")
        self._staging[video_id][int(frame_idx)] = feature_map.to("cpu")

    def commit(
        self,
        video_id: str,
        frame_idx: int,
        object_id: int,
        predicted_mask: torch.Tensor,
    ) -> bool:
        # One frame can be committed once per object. Keep the staged feature
        # until the sample is cleared so later object IDs do not lose it.
        staged = self._staging.get(video_id, {}).get(int(frame_idx))
        if staged is None:
            self.audit_events.append(
                {"event": "memory_skip", "reason": "missing_staged_feature", "video_id": video_id,
                 "frame_idx": int(frame_idx), "object_id": int(object_id)}
            )
            return False

        mask = predicted_mask.detach().to("cpu")
        while mask.ndim > 2:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"expected 2D mask, got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            mask = mask > 0
        if tuple(mask.shape) != tuple(staged.shape[-2:]):
            mask = F.interpolate(
                mask[None, None].float(), size=staged.shape[-2:], mode="nearest"
            )[0, 0].bool()
        if not bool(mask.any()):
            self.audit_events.append(
                {"event": "memory_skip", "reason": "empty_mask", "video_id": video_id,
                 "frame_idx": int(frame_idx), "object_id": int(object_id)}
            )
            return False

        token = staged[:, mask].mean(dim=1).contiguous()
        entry = MemoryEntry(int(object_id), int(frame_idx), int(frame_idx), token)
        self._raw_entries[video_id].append(entry)
        trajectory = self._bank[video_id][int(object_id)]
        trajectory.append(entry)
        while len(trajectory) > self.capacity:
            if self.compression == "none":
                break
            if self.compression == "fifo":
                trajectory.pop(0)
            else:
                self._merge_most_similar_adjacent(trajectory)
        self.audit_events.append(
            {"event": "memory_commit", "video_id": video_id, "frame_idx": int(frame_idx),
             "object_id": int(object_id), "trajectory_length": len(trajectory),
             "mask_pixels_feature_grid": int(mask.sum().item())}
        )
        return True

    @staticmethod
    def _merge_most_similar_adjacent(trajectory: List[MemoryEntry]) -> None:
        if len(trajectory) < 2:
            return
        similarities = torch.stack(
            [
                F.cosine_similarity(trajectory[i].token[None], trajectory[i + 1].token[None])[0]
                for i in range(len(trajectory) - 1)
            ]
        )
        index = int(torch.argmax(similarities).item())
        left, right = trajectory[index], trajectory[index + 1]
        trajectory[index:index + 2] = [
            MemoryEntry(
                object_id=left.object_id,
                frame_start=left.frame_start,
                frame_end=right.frame_end,
                token=(left.token + right.token) / 2,
            )
        ]

    def _historical_entries(self, video_id: str, frame_idx: int) -> Iterable[MemoryEntry]:
        for trajectory in self._bank.get(video_id, {}).values():
            for entry in trajectory:
                if entry.frame_end < frame_idx:
                    yield entry

    def retrieve(
        self, video_id: str, frame_idx: int, query: torch.Tensor
    ) -> Optional[RetrievalResult]:
        entries = list(self._historical_entries(video_id, int(frame_idx)))
        if not entries:
            self.audit_events.append(
                {"event": "memory_retrieve", "video_id": video_id, "frame_idx": int(frame_idx),
                 "candidate_count": 0, "selected_count": 0}
            )
            return None

        query = query.detach().float().reshape(-1).to("cpu")
        tokens = torch.stack([entry.token.float() for entry in entries])
        if query.numel() != tokens.shape[1]:
            raise ValueError(
                f"task query dimension {query.numel()} does not match memory dimension {tokens.shape[1]}"
            )
        scores = F.cosine_similarity(query[None], tokens, dim=1)
        k = len(entries) if self.retrieval_k is None else min(self.retrieval_k, len(entries))
        selected_scores, selected_indices = torch.topk(scores, k=k, largest=True, sorted=True)
        selected_entries = tuple(entries[int(i)] for i in selected_indices.tolist())
        selected_tokens = tokens[selected_indices]
        weights = torch.softmax(selected_scores / self.temperature, dim=0)
        evidence = torch.sum(selected_tokens * weights[:, None], dim=0)
        self.audit_events.append(
            {"event": "memory_retrieve", "video_id": video_id, "frame_idx": int(frame_idx),
             "candidate_count": len(entries), "selected_count": k,
             "selected": [
                 {"object_id": e.object_id, "frame_start": e.frame_start, "frame_end": e.frame_end,
                  "score": float(s), "weight": float(w)}
                 for e, s, w in zip(selected_entries, selected_scores, weights)
             ]}
        )
        return RetrievalResult(evidence, selected_entries, selected_scores, weights)

    def retrieve_offline(
        self,
        video_id: str,
        query: torch.Tensor,
        retrieval_k: int = 8,
        temperature: Optional[float] = None,
        candidate_capacity: int = 64,
    ) -> Optional[RetrievalResult]:
        """Retrieve from the complete sequence for offline teacher construction."""
        if candidate_capacity < 1:
            raise ValueError("offline teacher candidate capacity must be positive")
        trajectories: DefaultDict[int, List[MemoryEntry]] = defaultdict(list)
        for entry in self._raw_entries.get(video_id, []):
            trajectories[entry.object_id].append(entry)
        for trajectory in trajectories.values():
            while len(trajectory) > candidate_capacity:
                self._merge_most_similar_adjacent(trajectory)
        entries = [entry for trajectory in trajectories.values() for entry in trajectory]
        if not entries:
            return None
        query = query.detach().float().reshape(-1).to("cpu")
        tokens = torch.stack([entry.token.float() for entry in entries])
        if query.numel() != tokens.shape[1]:
            raise ValueError(
                f"teacher query dimension {query.numel()} does not match memory dimension {tokens.shape[1]}"
            )
        scores = F.cosine_similarity(query[None], tokens, dim=1)
        k = min(int(retrieval_k), len(entries))
        selected_scores, selected_indices = torch.topk(scores, k=k, largest=True, sorted=True)
        selected_entries = tuple(entries[int(index)] for index in selected_indices.tolist())
        selected_tokens = tokens[selected_indices]
        tau = self.temperature if temperature is None else float(temperature)
        if tau <= 0:
            raise ValueError("offline teacher temperature must be positive")
        weights = torch.softmax(selected_scores / tau, dim=0)
        evidence = torch.sum(selected_tokens * weights[:, None], dim=0)
        return RetrievalResult(evidence, selected_entries, selected_scores, weights)

    def trajectory_lengths(self, video_id: str) -> Dict[int, int]:
        return {object_id: len(entries) for object_id, entries in self._bank.get(video_id, {}).items()}
