"""Sa2VAChatQwenMem: Question-conditioned frame retrieval for Sa2VA-Qwen2.5-VL models.

This adapter wraps Sa2VAChatModelQwen (used by Sa2VA-Qwen2_5-VL-3B) to provide
the same late-interaction frame retrieval mechanism as Sa2VAChatMem does for
the InternVL-based Sa2VA models.

Key differences from the InternVL adapter:
  - Visual features come from Qwen2.5-VL's `get_image_features` instead of
    InternVL's `extract_feature`.
  - Input construction uses Qwen's processor (chat template + vision_info)
    instead of manual token assembly with IMG_CONTEXT tokens.
  - The Qwen model's predict_forward only uses the first 5 frames; we override
    generate_v2 to use ALL retrieved frames.
"""

import os
import re
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from PIL import Image

from vlmeval.vlm.sa2va_chat import Sa2VAChat
from vlmeval.smp import *
from ..feedback import load_qa_denylist


class Sa2VAChatQwenMem(Sa2VAChat):
    """Video QA with question-conditioned frame retrieval for Qwen2.5-VL Sa2VA models.

    Environment variables (same as Sa2VAChatMem):
        QA_MEM_ENABLE: Enable question-conditioned retrieval
        QA_QUERY_MODE: 'question_only' or 'question_with_options' (default)
        QA_FUSION_MODE: 'topk_select', 'weighted_enhance' or 'budget_realloc'
        QA_TOP_K: Number of frames to retrieve (default: 4)
        QA_RET_TAU: Retrieval temperature (default: 0.4)
        QA_STRIP_BOILERPLATE: Drop fixed prompt scaffolding before encoding
        QA_KEY_CENTER: Centre frame keys within a video before matching
        QA_LATE_INTERACTION: Use ColBERT-style per-patch max-sim scoring

    budget_realloc only:
        QA_BUDGET_MODE: 'topk' (default) or 'softmax'
        QA_BUDGET_RATIO: token ratio of relevant to other frames (default: 3.0)
        QA_BUDGET_GAIN: sharpness of the softmax allocation (default: 1.5)
        QA_BUDGET_FLOOR_TOKENS: minimum tokens per frame (default: 64)
        QA_BUDGET_REF_FRAMES: spend the budget of this many full-resolution
            frames (0 = the frames given). Set below --nframe to make the
            budget bind, e.g. --nframe 16 with QA_BUDGET_REF_FRAMES=8.
    """

    # Qwen2.5/3-VL costs one visual token per (patch_size * spatial_merge_size)
    # px block. We fill this in __init__ from the model config; the class value
    # is a fallback for Qwen2.5-VL (patch 14 * merge 2 = 28).
    QWEN_TOKEN_PX = 28

    BOILERPLATE = (
        "These are the frames of a video.",
        "Select the best answer to the following multiple-choice question based on the video.",
        "Respond with only the letter (A, B, C, or D) of the correct option.",
        "This video's subtitles are listed below:",
    )

    def __init__(self, *args, **kwargs):
        # Patch the config before parent init to avoid InternVL-specific attribute access
        # Sa2VAChat.__init__ accesses self.model.config.vision_config.image_size which
        # doesn't exist on Qwen2.5-VL configs. We monkey-patch it after model load.
        # We can't easily intercept, so we override the whole init flow.
        from transformers import AutoTokenizer, AutoModel
        import transformers

        model_path = kwargs.get('model_path', args[0] if args else None)
        assert model_path is not None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        # PL adapter settings (same as Sa2VAChat)
        self.pl_adapter_ckpt = os.environ.get('PL_ADAPTER_CKPT', '')
        self.pl_adapter_alpha = float(os.environ.get('PL_ADAPTER_ALPHA', '0.5'))
        self.pl_gate_mode = os.environ.get('PL_GATE_MODE', 'off').strip().lower()
        self.pl_gate_c0 = float(os.environ.get('PL_GATE_C0', '0.80'))
        self.pl_gate_c1 = float(os.environ.get('PL_GATE_C1', '0.95'))
        self.pl_gate_t = float(os.environ.get('PL_GATE_T', '10.0'))
        self.pl_gate_min = float(os.environ.get('PL_GATE_MIN', '0.0'))
        self.pl_gate_max = float(os.environ.get('PL_GATE_MAX', '1.0'))
        self.pl_denylist_path = os.environ.get('PL_DENYLIST_PATH', '').strip()
        self._denylist = load_qa_denylist(self.pl_denylist_path)
        if self._denylist is not None:
            print(f"[PROGRESS] VQA_DENYLIST_LOADED n={len(self._denylist)} "
                  f"path={self.pl_denylist_path}")
        self.pl_adapter = None
        self.pl_enabled = False
        print('[VQA] PL_ADAPTER disabled (base mode)')

        self.pattern = r'Image(\d+)'
        self.replacement = r'Image-\1'
        self.reverse_pattern = r'Image-(\d+)'
        self.reverse_replacement = r'Image\1'

        device = torch.cuda.current_device()
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.model = self.model.to(device)
        self.model.to(torch.bfloat16)

        # Determine the effective visual-token pixel block from the vision config.
        # Qwen2.5-VL: patch 14 * spatial_merge 2 = 28; Qwen3-VL: patch 16 * 2 = 32.
        try:
            vc = self.model.config.vision_config
            patch = getattr(vc, "patch_size", 14)
            merge = getattr(vc, "spatial_merge_size", 2)
            self.QWEN_TOKEN_PX = patch * merge
            self.qwen3_vl = getattr(vc, "model_type", "").lower().startswith("qwen3")
        except Exception:
            self.QWEN_TOKEN_PX = self.QWEN_TOKEN_PX
            self.qwen3_vl = False

        # Qwen doesn't have image_size in vision_config; set a dummy
        self.image_size = 448
        self.version = 'V2.0'

        # QA memory settings
        self.qa_memory_enabled = os.environ.get("QA_MEM_ENABLE", "0").strip() == "1"
        self.qa_query_mode = os.environ.get("QA_QUERY_MODE", "question_with_options").strip()
        self.qa_fusion_mode = os.environ.get("QA_FUSION_MODE", "topk_select").strip()
        self.qa_top_k = int(os.environ.get("QA_TOP_K", "4"))
        self.qa_retrieval_temperature = float(os.environ.get("QA_RET_TAU", "0.4"))
        self.qa_restore_temporal_order = os.environ.get("QA_RESTORE_TEMPORAL_ORDER", "1").strip() == "1"
        self.qa_strip_boilerplate = os.environ.get("QA_STRIP_BOILERPLATE", "1").strip() == "1"
        self.qa_key_center = os.environ.get("QA_KEY_CENTER", "1").strip() == "1"
        self.qa_late_interaction = os.environ.get("QA_LATE_INTERACTION", "0").strip() == "1"
        self.qa_mem_alpha = float(os.environ.get("VQA_MEM_ALPHA", "0.5"))
        # Frame budget reallocation: keep total visual tokens fixed, spend more
        # of them on question-relevant frames.
        self.qa_budget_mode = os.environ.get("QA_BUDGET_MODE", "topk").strip()
        self.qa_budget_ratio = float(os.environ.get("QA_BUDGET_RATIO", "3.0"))
        self.qa_budget_gain = float(os.environ.get("QA_BUDGET_GAIN", "1.5"))
        self.qa_budget_floor_tokens = int(os.environ.get("QA_BUDGET_FLOOR_TOKENS", "64"))
        self.qa_budget_ref_frames = int(os.environ.get("QA_BUDGET_REF_FRAMES", "0"))
        self.qa_ret_frame_tokens = int(os.environ.get("QA_RET_FRAME_TOKENS", "256"))

        if self.qa_memory_enabled:
            print(f"[QA_MEMORY] Enabled (Qwen): mode={self.qa_fusion_mode}, query={self.qa_query_mode}, "
                  f"topk={self.qa_top_k}, tau={self.qa_retrieval_temperature}, "
                  f"strip_boilerplate={self.qa_strip_boilerplate}, key_center={self.qa_key_center}, "
                  f"late_interaction={self.qa_late_interaction}")

        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        return self._processor

    def _extract_question_text(self, message, dataset):
        prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])

        if self.qa_strip_boilerplate:
            for boiler in self.BOILERPLATE:
                prompt = prompt.replace(boiler, " ")
            prompt = re.sub(r"[ \t]{2,}", " ", prompt)
            prompt = re.sub(r"\n{2,}", "\n", prompt).strip()

        suffixes = [
            "Answer with the option's letter from the given choices directly.",
            "\nAnswer:",
        ]
        for suffix in suffixes:
            if suffix in prompt:
                prompt = prompt.split(suffix)[0].strip()

        if self.qa_query_mode == "question_only":
            lines = prompt.split("\n")
            question_lines = []
            for line in lines:
                if not re.match(r"^[A-D]\.\s", line.strip()):
                    question_lines.append(line)
            prompt = "\n".join(question_lines).strip()

        return prompt

    def _encode_question_tokens(self, question_text):
        q_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        if not q_ids:
            q_ids = [self.tokenizer.eos_token_id]

        device = self.device if hasattr(self, "device") else torch.cuda.current_device()
        q_ids = torch.tensor(q_ids).to(device)

        with torch.no_grad():
            q_emb = self.model.get_input_embeddings()(q_ids)
            q_emb = F.normalize(q_emb, dim=-1)

        return q_emb

    def _encode_question(self, question_text):
        q_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        if not q_ids:
            q_ids = [self.tokenizer.eos_token_id]

        device = self.device if hasattr(self, "device") else torch.cuda.current_device()
        q_ids = torch.tensor(q_ids).to(device)

        with torch.no_grad():
            q_emb = self.model.get_input_embeddings()(q_ids)
            q_emb = q_emb.mean(dim=0)
            q_emb = F.normalize(q_emb, dim=-1)

        return q_emb

    def _extract_frame_features(self, frames):
        """Extract per-frame patch-level features using Qwen2.5-VL's visual encoder.

        Returns:
            frame_features: [N_frames, P_patches, d] tensor
        """
        processor = self._get_processor()
        device = self.device if hasattr(self, "device") else torch.cuda.current_device()

        # Encode every frame at the same modest resolution. Retrieval only needs
        # coarse spatial evidence, and a fixed size keeps the per-frame token
        # count identical, which the scoring below relies on. Letting the
        # processor pick sizes breaks this: with many frames it applies a
        # cumulative pixel budget and later frames come back smaller.
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        area = self.qa_ret_frame_tokens * self.QWEN_TOKEN_PX ** 2
        w0, h0 = frames[0].size
        h_bar, w_bar = smart_resize(
            h0, w0, factor=self.QWEN_TOKEN_PX,
            min_pixels=self.QWEN_TOKEN_PX ** 2, max_pixels=area,
        )
        small = [
            img if img.size == (w_bar, h_bar) else img.resize((w_bar, h_bar), Image.BICUBIC)
            for img in frames
        ]

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in small] +
                       [{"type": "text", "text": "describe"}],
        }]

        from qwen_vl_utils import process_vision_info
        processsed_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_inputs = processor(
            text=[processsed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.QWEN_TOKEN_PX ** 2,
            max_pixels=h_bar * w_bar,
        )
        mm_inputs = mm_inputs.to(device)

        pixel_values = mm_inputs["pixel_values"]
        image_grid_thw = mm_inputs["image_grid_thw"]

        with torch.no_grad():
            out = self.model.model.get_image_features(
                pixel_values, image_grid_thw
            )
            # Qwen3-VL returns (image_embeds, deepstack_image_embeds)
            image_embeds_list = out[0] if self.qwen3_vl else out

        sizes = {e.shape[0] for e in image_embeds_list}
        if len(sizes) != 1:
            raise RuntimeError(
                f"retrieval features have mismatched token counts {sizes}; "
                "frames must encode to a uniform size"
            )

        frame_features = torch.stack(image_embeds_list, dim=0)
        return frame_features

    def _retrieve_frames_late_interaction(self, question_text, frame_features, topk, tau):
        q_tokens = self._encode_question_tokens(question_text).float()
        patches = frame_features.float()

        if self.qa_key_center and patches.shape[0] > 1:
            video_mean = patches.mean(dim=(0, 1), keepdim=True)
            patches = patches - video_mean
            patches = F.normalize(patches, dim=-1)

        frame_scores = []
        for f in range(patches.shape[0]):
            sim_matrix = torch.matmul(q_tokens, patches[f].T)
            max_per_qtoken = sim_matrix.max(dim=1).values
            frame_scores.append(max_per_qtoken.sum())

        sim = torch.stack(frame_scores)
        sim = sim / tau

        K = min(topk, sim.shape[0])
        vals, idx = torch.topk(sim, k=K)

        if self.qa_restore_temporal_order:
            idx_sorted, sort_order = torch.sort(idx)
            vals_sorted = vals[sort_order]
        else:
            idx_sorted = idx
            vals_sorted = vals

        return idx_sorted, vals_sorted, sim

    def _retrieve_frames(self, question_text, frame_features, topk, tau):
        if self.qa_late_interaction:
            return self._retrieve_frames_late_interaction(
                question_text, frame_features, topk, tau
            )

        q_repr = self._encode_question(question_text)
        key = frame_features.mean(dim=1).float()
        if self.qa_key_center and key.shape[0] > 1:
            key = key - key.mean(dim=0, keepdim=True)
        key = F.normalize(key, dim=-1)

        sim = torch.matmul(q_repr.unsqueeze(0), key.T).squeeze(0)
        sim = sim / tau

        K = min(topk, sim.shape[0])
        vals, idx = torch.topk(sim, k=K)

        if self.qa_restore_temporal_order:
            idx_sorted, sort_order = torch.sort(idx)
            vals_sorted = vals[sort_order]
        else:
            idx_sorted = idx
            vals_sorted = vals

        return idx_sorted, vals_sorted, sim

    def _baseline_frame_tokens(self, image):
        """Visual token count this frame would consume in the default pipeline."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        w, h = image.size
        h_bar, w_bar = smart_resize(
            h, w, factor=self.QWEN_TOKEN_PX,
            min_pixels=self.model.min_pixels, max_pixels=self.model.max_pixels,
        )
        return (h_bar // self.QWEN_TOKEN_PX) * (w_bar // self.QWEN_TOKEN_PX)

    def _resize_to_token_budget(self, image, target_tokens):
        """Resize so the frame costs about `target_tokens` visual tokens.

        Returns the resized image and its actual token cost. Dimensions are
        multiples of QWEN_TOKEN_PX so the processor's own smart_resize becomes a
        no-op and does not undo our allocation.
        """
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        w, h = image.size
        target_area = max(1, int(target_tokens)) * self.QWEN_TOKEN_PX ** 2
        h_bar, w_bar = smart_resize(
            h, w, factor=self.QWEN_TOKEN_PX,
            min_pixels=self.QWEN_TOKEN_PX ** 2, max_pixels=target_area,
        )
        tokens = (h_bar // self.QWEN_TOKEN_PX) * (w_bar // self.QWEN_TOKEN_PX)
        if (w_bar, h_bar) != (w, h):
            image = image.resize((w_bar, h_bar), Image.BICUBIC)
        return image, tokens

    def _allocate_frame_budget(self, images, sims, selected_indices):
        """Reallocate a fixed visual-token budget across frames by relevance.

        A frame is never upscaled past its native resolution, since that buys
        tokens without adding detail. Reallocation is therefore only meaningful
        when the budget actually binds: pass more frames than the budget can
        hold at full resolution (QA_BUDGET_REF_FRAMES), so extra temporal
        coverage is paid for by coarsening the irrelevant frames.
        """
        caps = [self._baseline_frame_tokens(img) for img in images]
        n = len(images)

        if self.qa_budget_ref_frames > 0:
            # Spend what `ref_frames` frames would have cost at full resolution.
            total = int(round(sum(caps) / n * self.qa_budget_ref_frames))
        else:
            total = sum(caps)
        total = min(total, sum(caps))

        floor = min(self.qa_budget_floor_tokens, total // max(n, 1))

        if self.qa_budget_mode == "softmax":
            # Raw retrieval scores are nearly flat, so standardise before the
            # softmax; otherwise the allocation collapses to uniform.
            s = sims.float()
            s = (s - s.mean()) / (s.std() + 1e-6)
            share = F.softmax(s * self.qa_budget_gain, dim=0).cpu().tolist()
        else:
            # Top-K frames get `ratio` times the share of the rest.
            r = self.qa_budget_ratio
            sel = set(selected_indices)
            raw = [r if i in sel else 1.0 for i in range(n)]
            share = [x / sum(raw) for x in raw]

        # Water-filling: clamp to [floor, native cap] and hand the leftover to
        # frames that can still take more, so the budget is actually spent.
        targets = [total * x for x in share]
        for _ in range(16):
            targets = [min(caps[i], max(floor, targets[i])) for i in range(n)]
            leftover = total - sum(targets)
            if leftover <= 1:
                break
            room = [caps[i] - targets[i] for i in range(n)]
            if sum(room) <= 0:
                break
            targets = [targets[i] + leftover * room[i] / sum(room) for i in range(n)]

        resized, actual = [], []
        for img, t in zip(images, targets):
            out, tok = self._resize_to_token_budget(img, t)
            resized.append(out)
            actual.append(tok)

        return resized, caps, actual, total

    @contextmanager
    def _weighted_image_features(self, frame_weights):
        """Temporarily scale per-frame image embeddings inside the Qwen forward pass.

        Qwen2_5_VLModel.forward calls self.get_image_features(pixel_values,
        image_grid_thw) and then masked_scatters the result into inputs_embeds.
        Patching that single method lets us reweight frames while leaving mrope
        position ids and the placeholder scattering to the stock HF code, which
        is why we do not hand-build inputs_embeds here.

        Only the prefill step receives pixel_values, so the scaling is applied
        exactly once per generate() call.
        """
        inner = self.model.model.model  # Qwen2_5_VLForConditionalGeneration -> Qwen2_5_VLModel
        original = inner.get_image_features

        def patched(pixel_values, image_grid_thw=None, **kwargs):
            out = original(pixel_values, image_grid_thw, **kwargs)
            if self.qwen3_vl:
                embeds, deepstack = out
            else:
                embeds = out
            if len(embeds) != len(frame_weights):
                print(f"[QA_WARN] frame count mismatch: embeds={len(embeds)} "
                      f"weights={len(frame_weights)}; skipping weighting")
                return out
            scaled = tuple(
                e * w if w != 1.0 else e
                for e, w in zip(embeds, frame_weights)
            )
            return (scaled, deepstack) if self.qwen3_vl else scaled

        had_own_attr = "get_image_features" in inner.__dict__
        inner.get_image_features = patched
        try:
            yield
        finally:
            if had_own_attr:
                inner.get_image_features = original
            else:
                del inner.get_image_features

    def generate_v2(self, message, dataset=None):
        """Override generate_v2 to inject question-conditioned frame retrieval.

        Even when QA_MEM_ENABLE is off, we still use our own inference path
        (not super().generate_v2) because the Qwen model's predict_forward
        hardcodes a 5-frame limit.
        """
        image_num = len([x for x in message if x["type"] == "image"])
        use_qa_memory = self.qa_memory_enabled and image_num > 1
        if self._denylist is not None and self._denylist.matches(
            message, getattr(self, "_current_index", None)
        ):
            use_qa_memory = False
            print(f"[PROGRESS] VQA_DENYLIST_SKIP "
                  f"idx={getattr(self, '_current_index', None)}")

        prompt = "<image>\n" + "\n".join([x["value"] for x in message if x["type"] == "text"])
        if dataset:
            prompt = self.build_video_prompt(prompt, dataset)

        image_path = [x["value"] for x in message if x["type"] == "image"]
        ori_images = []
        for p in image_path:
            if isinstance(p, str):
                ori_images.append(Image.open(p).convert("RGB"))
            else:
                ori_images.append(p.convert("RGB") if p.mode != "RGB" else p)

        frame_weights = None

        if use_qa_memory:
            question_text = self._extract_question_text(message, dataset)
            frame_features = self._extract_frame_features(ori_images)
            idx_sorted, scores, all_sims = self._retrieve_frames(
                question_text, frame_features, self.qa_top_k, self.qa_retrieval_temperature
            )
            print(f"[QA_RETRIEVAL] Question: {question_text[:80]}...")
            print(f"[QA_RETRIEVAL] Top-{len(idx_sorted)} frames: {idx_sorted.cpu().tolist()}, "
                  f"scores: {[f'{s:.3f}' for s in scores.cpu().tolist()]}")
            selected_indices = idx_sorted.cpu().tolist()

            if self.qa_fusion_mode == "weighted_enhance":
                # Keep every frame; only scale the embeddings of the retrieved ones.
                selected_frames = ori_images
                softmax_w = F.softmax(scores / self.qa_retrieval_temperature, dim=0)
                alpha = self.qa_mem_alpha
                frame_weights = [1.0] * len(ori_images)
                for pos, frame_idx in enumerate(selected_indices):
                    frame_weights[frame_idx] = 1.0 + softmax_w[pos].item() * alpha
                print(f"[PROGRESS] QA_MEMORY_APPLIED mode=weighted_enhance topk={self.qa_top_k} "
                      f"tau={self.qa_retrieval_temperature} alpha={alpha} "
                      f"num_frames={len(selected_frames)} "
                      f"weights={[f'{w:.3f}' for w in frame_weights]}")
            elif self.qa_fusion_mode == "budget_realloc":
                # Keep every frame and the total token budget; shift resolution
                # toward the relevant frames.
                selected_frames, caps, actual_tokens, budget = self._allocate_frame_budget(
                    ori_images, all_sims, selected_indices
                )
                print(f"[PROGRESS] QA_MEMORY_APPLIED mode=budget_realloc "
                      f"budget_mode={self.qa_budget_mode} ratio={self.qa_budget_ratio} "
                      f"topk={self.qa_top_k} num_frames={len(selected_frames)} "
                      f"budget={budget} spent={sum(actual_tokens)} "
                      f"native={sum(caps)} per_frame={actual_tokens}")
            else:
                selected_frames = [ori_images[i] for i in selected_indices]
                print(f"[PROGRESS] QA_MEMORY_APPLIED mode={self.qa_fusion_mode} topk={self.qa_top_k} "
                      f"tau={self.qa_retrieval_temperature} num_frames={len(selected_frames)}")
        else:
            selected_frames = ori_images
            print(f"[PROGRESS] BASELINE (no retrieval) num_frames={len(selected_frames)}")

        processor = self._get_processor()
        device = self.device if hasattr(self, "device") else torch.cuda.current_device()

        content = [{"type": "image", "image": img} for img in selected_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        processsed_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        if use_qa_memory and self.qa_fusion_mode == "budget_realloc":
            # The frames already carry their allocated resolution. The default
            # bounds would rescale them back, so widen them to keep our sizes.
            min_pixels = self.QWEN_TOKEN_PX ** 2
            max_pixels = max(
                self.model.max_pixels,
                max(img.size[0] * img.size[1] for img in selected_frames),
            )
        else:
            min_pixels = self.model.min_pixels
            max_pixels = self.model.max_pixels

        mm_inputs = processor(
            text=[processsed_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        mm_inputs = mm_inputs.to(device)

        with torch.no_grad():
            if frame_weights is not None:
                with self._weighted_image_features(frame_weights):
                    generate_output = self.model.model.generate(
                        **mm_inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                        output_hidden_states=False,
                        return_dict_in_generate=True,
                    )
            else:
                generate_output = self.model.model.generate(
                    **mm_inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                )

        generate_output_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
        ]
        response = processor.batch_decode(
            generate_output_trimmed, skip_special_tokens=False
        )[0].strip()
        response = (
            response.replace("<|end|>", "")
            .replace("", "")
            .replace("<|im_end|>", "")
            .strip()
        )

        print("Question:", prompt[:100], "...\nResponse:")
        return response
