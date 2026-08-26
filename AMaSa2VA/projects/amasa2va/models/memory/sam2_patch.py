from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .adapter import load_paper_adapter
from .fusion import FusionConfig, fuse_and_route
from .object_memory import ObjectCentricMemoryBank


@dataclass(frozen=True)
class PaperMemoryConfig:
    capacity: int = 8
    retrieval_k: Optional[int] = 4
    temperature: float = 0.4
    compression: str = "similarity_merge"
    alpha: float = 0.5
    gate_enabled: bool = True
    cosine_c0: float = 0.80
    cosine_c1: float = 0.95
    route_threshold: Optional[float] = None
    route_threshold_source: str = "disabled"
    adapter_checkpoint: Optional[str] = None
    capture_training_records: bool = False
    inject_enabled: bool = True
    audit_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.route_threshold is not None and not self.gate_enabled:
            raise ValueError("routing requires the cosine gate")
        if self.route_threshold is not None and self.route_threshold_source in {"", "disabled"}:
            raise ValueError("a routing threshold requires an explicit disjoint-calibration source")


def locate_sam2_module(model):
    candidates = [
        model,
        getattr(model, "sam2_model", None),
        getattr(model, "grounding_encoder", None),
        getattr(getattr(model, "grounding_encoder", None), "sam2_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "_get_image_feature") and hasattr(
            candidate, "_run_single_frame_inference"
        ):
            return candidate
    raise RuntimeError("could not locate the SAM2 per-frame inference module")


def _video_id(inference_state, sam2_model=None) -> str:
    if sam2_model is not None:
        sample_id = getattr(sam2_model, "_amasa2va_sample_id", None)
        if sample_id is not None:
            return str(sample_id)
    if isinstance(inference_state, dict):
        for key in ("video_id", "video_name", "video_path"):
            if inference_state.get(key) is not None:
                return str(inference_state[key])
    return f"inference_state:{id(inference_state)}"


def _object_id(inference_state, output_dict) -> int:
    if isinstance(inference_state, dict):
        for object_id, candidate in inference_state.get("output_dict_per_obj", {}).items():
            if candidate is output_dict:
                return int(object_id)
    return 0


def _current_visual(bank: ObjectCentricMemoryBank, video_id: str, frame_idx: int) -> Optional[torch.Tensor]:
    staged = bank._staging.get(video_id, {}).get(int(frame_idx))
    return staged.mean(dim=(1, 2)) if staged is not None else None


def apply_paper_memory_patch(sam2_model, config: PaperMemoryConfig) -> ObjectCentricMemoryBank:
    if getattr(sam2_model, "_amasa2va_paper_memory_patched", False):
        raise RuntimeError("paper memory patch is already applied")

    bank = ObjectCentricMemoryBank(
        capacity=config.capacity,
        retrieval_k=config.retrieval_k,
        temperature=config.temperature,
        compression=config.compression,
    )
    adapter: Optional[nn.Module] = None
    if config.adapter_checkpoint:
        device = next(sam2_model.parameters()).device if isinstance(sam2_model, nn.Module) else torch.device("cpu")
        adapter = load_paper_adapter(config.adapter_checkpoint, device)

    original_get_image_feature = sam2_model._get_image_feature
    original_run_single_frame = sam2_model._run_single_frame_inference
    prompt_cache = {}
    query_source_logged = False

    def persist_audit(record: dict) -> None:
        if config.audit_path:
            with open(config.audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_audit(record: dict) -> None:
        bank.audit_events.append(record)
        persist_audit(record)

    def patched_get_image_feature(self, inference_state, frame_idx, batch_size):
        features = original_get_image_feature(inference_state, frame_idx, batch_size)
        if features is None:
            return features
        _, backbone_out, current_vision_feats, _, feat_sizes = features
        feature_map = None
        if isinstance(backbone_out, dict) and backbone_out.get("backbone_fpn"):
            feature_map = backbone_out["backbone_fpn"][-1]
        elif current_vision_feats:
            top = current_vision_feats[-1]
            height, width = feat_sizes[-1]
            feature_map = top.permute(1, 2, 0).contiguous().view(batch_size, top.shape[-1], height, width)
        if feature_map is None:
            raise RuntimeError("SAM2 did not expose a final FPN feature map for object memory")
        bank.stage(_video_id(inference_state, self), int(frame_idx), feature_map[:1])
        return features

    def patched_run_single_frame(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        language_embd=None,
    ):
        nonlocal query_source_logged
        video_id = _video_id(inference_state, self)
        object_id = _object_id(inference_state, output_dict)
        prompt_key = (video_id, object_id)
        prompt_source = "provided"
        if language_embd is not None:
            prompt_cache[prompt_key] = language_embd.detach().clone()
        else:
            language_embd = prompt_cache.get(prompt_key)
            prompt_source = "cached_task_prompt" if language_embd is not None else "unavailable"
        routed_prompt = language_embd
        if language_embd is not None:
            if int(frame_idx) not in bank._staging.get(video_id, {}):
                self._get_image_feature(inference_state, frame_idx, batch_size)
            query = language_embd.detach().reshape(-1, language_embd.shape[-1]).mean(dim=0)
            if not query_source_logged:
                print(
                    f"[PAPER_QUERY] source=projected_seg_prompt dim={query.numel()} "
                    f"prompt_shape={tuple(language_embd.shape)}",
                    flush=True,
                )
                query_source_logged = True
            retrieval = bank.retrieve(video_id, int(frame_idx), query)
            if bank.audit_events and bank.audit_events[-1].get("event") == "memory_retrieve":
                persist_audit(bank.audit_events[-1])
            current_visual = _current_visual(bank, video_id, int(frame_idx))
            if retrieval is not None and current_visual is not None:
                evidence = retrieval.evidence.to(language_embd.device, language_embd.dtype)
                current_visual = current_visual.to(language_embd.device, language_embd.dtype)
                if adapter is None:
                    adapted = evidence
                    adapter_kind = "identity_control"
                else:
                    adapted = adapter(evidence.float()[None], current_visual.float()[None])[0]
                    adapted = adapted.to(language_embd.dtype)
                    adapter_kind = "trained_paper_adapter"
                adapted_prompt = adapted.reshape(*([1] * (language_embd.ndim - 1)), -1).expand_as(language_embd)
                raw_compatibility = float(
                    F.cosine_similarity(
                        adapted_prompt.reshape(-1, adapted_prompt.shape[-1]).mean(dim=0)[None].float(),
                        language_embd.reshape(-1, language_embd.shape[-1]).mean(dim=0)[None].float(),
                        dim=1,
                    )[0]
                )
                if not config.inject_enabled:
                    routed_prompt = language_embd
                    gate, route_memory = 0.0, False
                elif config.gate_enabled:
                    routed_prompt, gate, route_memory = fuse_and_route(
                        adapted_prompt,
                        language_embd,
                        FusionConfig(
                            alpha=config.alpha,
                            cosine_c0=config.cosine_c0,
                            cosine_c1=config.cosine_c1,
                            route_threshold=config.route_threshold,
                        ),
                    )
                else:
                    routed_prompt = config.alpha * adapted_prompt + (1 - config.alpha) * language_embd
                    gate, route_memory = 1.0, True
                write_audit(
                    {"event": "prompt_route", "video_id": video_id, "frame_idx": int(frame_idx),
                     "object_id": object_id, "adapter": adapter_kind, "cosine_gate": gate,
                     "raw_prompt_cosine": raw_compatibility,
                     "prompt_source": prompt_source,
                     "query_source": "projected_seg_prompt",
                     "route_memory": route_memory, "route_threshold": config.route_threshold,
                     "route_threshold_source": config.route_threshold_source}
                )
                if config.capture_training_records:
                    bank.training_records.append(
                        {
                            "frame_idx": int(frame_idx),
                            "object_id": object_id,
                            "sample_id": video_id,
                            "evidence": retrieval.evidence.detach().float().cpu(),
                            "current_visual": current_visual.detach().float().cpu(),
                            "original_prompt": language_embd.detach().reshape(
                                -1, language_embd.shape[-1]
                            ).mean(dim=0).float().cpu(),
                        }
                    )

        output = original_run_single_frame(
            inference_state=inference_state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            batch_size=batch_size,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            language_embd=routed_prompt,
        )
        compact_current_out, predicted_masks = output
        if predicted_masks is not None:
            mask = torch.sigmoid(predicted_masks[:, 0]) > 0.5
            bank.commit(video_id, int(frame_idx), object_id, mask)
            if bank.audit_events and bank.audit_events[-1].get("event") in {"memory_commit", "memory_skip"}:
                persist_audit(bank.audit_events[-1])
        return compact_current_out, predicted_masks

    patched_get_image_feature._amasa2va_paper_memory = True
    patched_run_single_frame._amasa2va_paper_memory = True
    sam2_model._get_image_feature = patched_get_image_feature.__get__(sam2_model, type(sam2_model))
    sam2_model._run_single_frame_inference = patched_run_single_frame.__get__(sam2_model, type(sam2_model))
    sam2_model._amasa2va_paper_memory_patched = True
    sam2_model._amasa2va_memory_bank = bank
    sam2_model._amasa2va_prompt_cache = prompt_cache
    sam2_model._amasa2va_memory_config = asdict(config)
    sam2_model._amasa2va_adapter_parameter_count = (
        sum(parameter.numel() for parameter in adapter.parameters()) if adapter is not None else 0
    )
    print("[AMASA2VA_MEMORY_CONFIG] " + json.dumps(asdict(config), sort_keys=True), flush=True)
    return bank
