import os
import re

import torch
import torch.nn.functional as F
from PIL import Image

from vlmeval.vlm.sa2va_chat import Sa2VAChat
from ..feedback import load_qa_denylist


class Sa2VAChatMem(Sa2VAChat):
    """Video QA with question-conditioned frame-level memory retrieval.

    Environment variables:
        QA_MEM_ENABLE: Enable question-conditioned retrieval (recommended)
        QA_QUERY_MODE: 'question_only' or 'question_with_options' (default)
        QA_FUSION_MODE: 'topk_select', 'weighted_enhance', or 'legacy_weighted_sum'
        QA_TOP_K: Number of frames to retrieve (default: 4)
        QA_RET_TAU: Retrieval temperature (default: 0.4)
        QA_STRIP_BOILERPLATE: Drop the fixed prompt scaffolding before encoding
        QA_KEY_CENTER: Centre frame keys within a video before matching
        QA_LATE_INTERACTION: Use ColBERT-style per-patch max-sim scoring instead
                       of mean-pooled frame keys (preserves spatial discrimination)
    """

    # vlmeval prepends these verbatim to every Video-MME question, so they carry
    # zero discriminative signal while dominating a mean-pooled embedding.
    BOILERPLATE = (
        "These are the frames of a video.",
        "Select the best answer to the following multiple-choice question based on the video.",
        "Respond with only the letter (A, B, C, or D) of the correct option.",
        "This video's subtitles are listed below:",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Legacy VQA memory settings
        self.vqa_mem_enable = os.environ.get("VQA_MEM_ENABLE", "0").strip() == "1"
        self.vqa_mem_topk = int(os.environ.get("VQA_MEM_TOPK", "4"))
        self.vqa_mem_tau = float(os.environ.get("VQA_MEM_TAU", "0.4"))
        self.vqa_mem_alpha = float(os.environ.get("VQA_MEM_ALPHA", "0.5"))
        self.pl_denylist_path = os.environ.get("PL_DENYLIST_PATH", "").strip()
        
        # New QA memory settings
        self.qa_memory_enabled = os.environ.get("QA_MEM_ENABLE", "0").strip() == "1"
        self.qa_query_mode = os.environ.get("QA_QUERY_MODE", "question_with_options").strip()
        self.qa_fusion_mode = os.environ.get("QA_FUSION_MODE", "topk_select").strip()
        self.qa_top_k = int(os.environ.get("QA_TOP_K", "4"))
        self.qa_retrieval_temperature = float(os.environ.get("QA_RET_TAU", "0.4"))
        self.qa_restore_temporal_order = os.environ.get("QA_RESTORE_TEMPORAL_ORDER", "1").strip() == "1"
        self.qa_strip_boilerplate = os.environ.get("QA_STRIP_BOILERPLATE", "1").strip() == "1"
        self.qa_key_center = os.environ.get("QA_KEY_CENTER", "1").strip() == "1"
        self.qa_late_interaction = os.environ.get("QA_LATE_INTERACTION", "0").strip() == "1"
        
        self._denylist = self._load_denylist(self.pl_denylist_path)
        
        if self.qa_memory_enabled:
            print(f"[QA_MEMORY] Enabled: mode={self.qa_fusion_mode}, query={self.qa_query_mode}, "
                  f"topk={self.qa_top_k}, tau={self.qa_retrieval_temperature}, "
                  f"strip_boilerplate={self.qa_strip_boilerplate}, key_center={self.qa_key_center}, "
                  f"late_interaction={self.qa_late_interaction}")

    def _build_video_input(self, video):
        dtype = getattr(self.model.vision_model, "dtype", torch.bfloat16)
        pixel_values = []
        for frame_image in video:
            pixel_values.append(self.model.transformer(frame_image))
        pixel_values = torch.stack(pixel_values, dim=0).to(dtype)
        return pixel_values

    def _encode_question(self, question_text):
        """Encode question text into LLM embedding space."""
        q_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        if not q_ids:
            q_ids = [self.tokenizer.eos_token_id]
        
        q_ids = torch.tensor(q_ids).to(self.device if hasattr(self, "device") else torch.cuda.current_device())
        
        with torch.no_grad():
            q_emb = self.model.language_model.get_input_embeddings()(q_ids)
            q_repr = q_emb.mean(dim=0)
            q_repr = F.normalize(q_repr, dim=-1)
        
        return q_repr

    def _extract_question_text(self, message, dataset):
        """Extract pure question text from message."""
        prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        prompt = re.sub(r'Frame-\d+\s*', '', prompt)
        
        if self.qa_strip_boilerplate:
            for boiler in self.BOILERPLATE:
                prompt = prompt.replace(boiler, " ")
            prompt = re.sub(r'^\s*Question:\s*', '', prompt.strip())
            prompt = re.sub(r'\n\s*Question:\s*', '\n', prompt)
        
        suffixes = [
            "Answer with the option's letter from the given choices directly.",
            "Answer with the option's letter.",
            "请直接回答选项字母。",
            "\nAnswer:",
        ]
        for suffix in suffixes:
            if suffix in prompt:
                prompt = prompt.split(suffix)[0].strip()
        
        prompt = re.sub(r'[ \t]{2,}', ' ', prompt)
        prompt = re.sub(r'\n{2,}', '\n', prompt).strip()
        
        if self.qa_query_mode == "question_only":
            lines = prompt.split('\n')
            question_lines = []
            for line in lines:
                if not re.match(r'^[A-D]\.\s', line.strip()):
                    question_lines.append(line)
            prompt = '\n'.join(question_lines).strip()
        
        return prompt

    def _encode_question_tokens(self, question_text):
        """Encode question text into per-token LLM embeddings (for late interaction)."""
        q_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        if not q_ids:
            q_ids = [self.tokenizer.eos_token_id]

        q_ids = torch.tensor(q_ids).to(self.device if hasattr(self, "device") else torch.cuda.current_device())

        with torch.no_grad():
            q_emb = self.model.language_model.get_input_embeddings()(q_ids)
            q_emb = F.normalize(q_emb, dim=-1)

        return q_emb

    def _retrieve_frames_late_interaction(self, question_text, frame_features, topk, tau):
        """ColBERT-style late interaction: per-patch max-sim scoring.

        Instead of collapsing each frame into a single mean-pooled vector, we
        keep all patch tokens and score each frame by summing, over query tokens,
        the maximum cosine similarity to any patch in that frame.  This preserves
        spatial discrimination: a query token about "person" can match a patch
        showing a person in one frame but not in another, even when the frames
        are globally similar.
        """
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
        """Question-conditioned frame retrieval."""
        if self.qa_late_interaction:
            return self._retrieve_frames_late_interaction(
                question_text, frame_features, topk, tau
            )

        q_repr = self._encode_question(question_text)
        frame_key = self._build_frame_keys(frame_features)
        sim = torch.matmul(q_repr.unsqueeze(0), frame_key.T).squeeze(0)
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

    def _build_frame_keys(self, frame_features):
        """Pool patch tokens into one key per frame.

        Plain mean pooling leaves the keys of one video near-collinear (measured
        mean off-diagonal cosine 0.935 on Video-MME), which compresses every
        query-frame similarity into a ~0.007 wide band and makes retrieval
        essentially question-independent. Subtracting the per-video mean removes
        the component all frames share and restores the dynamic range.
        """
        key = frame_features.mean(dim=1).float()
        if self.qa_key_center and key.shape[0] > 1:
            key = key - key.mean(dim=0, keepdim=True)
        return F.normalize(key, dim=-1).to(frame_features.dtype)

    def _mix_video_features_qa(self, pixel_values, question_text):
        """Question-conditioned feature mixing."""
        device = self.device if hasattr(self, "device") else torch.cuda.current_device()
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            frame_features = self.model.extract_feature(
                pixel_values.to(self.model.vision_model.dtype)
            )
        
        if frame_features.shape[0] <= 1:
            return frame_features, None
        
        idx_sorted, scores, all_sims = self._retrieve_frames(
            question_text, frame_features, self.qa_top_k, self.qa_retrieval_temperature
        )
        
        print(f"[QA_RETRIEVAL] Question: {question_text[:80]}...")
        print(f"[QA_RETRIEVAL] Top-{len(idx_sorted)} frames: {idx_sorted.cpu().tolist()}, "
              f"scores: {[f'{s:.3f}' for s in scores.cpu().tolist()]}")
        
        if self.qa_fusion_mode == "topk_select":
            return frame_features[idx_sorted], idx_sorted
        elif self.qa_fusion_mode == "weighted_enhance":
            weights = F.softmax(scores / self.qa_retrieval_temperature, dim=0)
            enhanced = []
            for i in range(frame_features.shape[0]):
                if i in idx_sorted:
                    pos = (idx_sorted == i).nonzero(as_tuple=True)[0][0]
                    weight = weights[pos].item()
                    alpha = getattr(self, 'vqa_mem_alpha', 0.5)
                    enhanced.append(frame_features[i] * (1.0 + weight * alpha))
                else:
                    enhanced.append(frame_features[i])
            return torch.stack(enhanced, dim=0), None
        else:
            return self._mix_video_features_legacy(pixel_values), None

    def _mix_video_features_legacy(self, pixel_values):
        """Legacy visual similarity retrieval."""
        device = self.device if hasattr(self, "device") else torch.cuda.current_device()
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            frame_features = self.model.extract_feature(pixel_values.to(self.model.vision_model.dtype))
        if frame_features.shape[0] <= 1:
            return frame_features

        frame_repr = F.normalize(frame_features.mean(dim=1), dim=-1)
        sim = torch.matmul(frame_repr, frame_repr.transpose(0, 1))
        sim = sim / max(self.vqa_mem_tau, 1e-6)
        k = max(1, min(self.vqa_mem_topk, sim.shape[1]))

        mixed = []
        for i in range(sim.shape[0]):
            vals, idx = torch.topk(sim[i], k=k, dim=-1)
            weights = torch.softmax(vals, dim=-1).to(frame_features.dtype)
            mem_feat = torch.sum(frame_features[idx] * weights[:, None, None], dim=0)
            mixed_feat = self.vqa_mem_alpha * mem_feat + (1.0 - self.vqa_mem_alpha) * frame_features[i]
            mixed.append(mixed_feat)
        mixed = torch.stack(mixed, dim=0)
        return self._apply_pl_adapter(mixed, frame_repr)

    def _mix_video_features(self, pixel_values):
        return self._mix_video_features_legacy(pixel_values)

    def _build_adapter_memory(self, frame_repr, sim, frame_idx):
        d = min(256, frame_repr.shape[-1])
        max_mem = 16
        scores = sim[frame_idx].clone()
        scores[frame_idx] = -1e9
        k = max(1, min(max_mem, frame_repr.shape[0] - 1))
        vals, idx = torch.topk(scores, k=k, dim=-1)
        mem = frame_repr[idx, :d]
        if mem.shape[0] < max_mem:
            pad = torch.zeros(max_mem - mem.shape[0], d, dtype=frame_repr.dtype, device=frame_repr.device)
            mem = torch.cat([mem, pad], dim=0)
        return mem, vals[0] if vals.numel() else torch.tensor(0.0, device=frame_repr.device, dtype=frame_repr.dtype)

    def _compute_gate(self, conf):
        mode = getattr(self, "pl_gate_mode", "off")
        if mode == "off":
            return conf.new_tensor(1.0)
        if mode != "cos":
            return conf.new_tensor(1.0)
        gate = torch.sigmoid((conf - self.pl_gate_c0) * self.pl_gate_t)
        gate = torch.clamp(gate, min=self.pl_gate_min, max=self.pl_gate_max)
        return gate

    def _apply_pl_adapter(self, mixed_features, frame_repr):
        if not getattr(self, "pl_enabled", False) or getattr(self, "pl_adapter", None) is None:
            return mixed_features
        if mixed_features.shape[0] <= 1:
            return mixed_features

        sim = torch.matmul(frame_repr, frame_repr.transpose(0, 1))
        adapted = []
        gate_values = []
        with torch.no_grad():
            for i in range(mixed_features.shape[0]):
                mem_in16, conf = self._build_adapter_memory(frame_repr, sim, i)
                xcur_mean = mixed_features[i].mean(dim=0)
                xcur_adapter = xcur_mean[: mem_in16.shape[-1]]
                delta = self.pl_adapter(mem_in16.unsqueeze(0), xcur_adapter.unsqueeze(0))[0]
                gate = self._compute_gate(conf)
                cur = mixed_features[i].clone()
                cur[:, : delta.shape[0]] = cur[:, : delta.shape[0]] + (
                    self.pl_adapter_alpha * gate * delta
                ).unsqueeze(0)
                adapted.append(cur)
                gate_values.append(float(gate))

        print(
            "[PROGRESS] VQA_PL_ADAPTER_APPLIED "
            f"enable=1 alpha={self.pl_adapter_alpha} "
            f"gate_mode={getattr(self, 'pl_gate_mode', 'off')} "
            f"gate_mean={sum(gate_values) / max(len(gate_values), 1):.4f}"
        )
        return torch.stack(adapted, dim=0)

    def _load_denylist(self, path: str):
        if not path:
            return None
        try:
            deny = load_qa_denylist(path)
            if deny is not None:
                print(f"[PROGRESS] VQA_DENYLIST_LOADED n={len(deny)} path={path}")
            return deny
        except Exception as e:
            print(f"[WARN] VQA_DENYLIST_LOAD_FAIL path={path} err={e}")
            return None

    def generate_v2(self, message, dataset=None):
        if self._denylist is not None:
            idx = getattr(self, "_current_index", None)
            if self._denylist.matches(message, idx):
                print(f"[PROGRESS] VQA_DENYLIST_SKIP idx={idx}")
                return super().generate_v2(message, dataset)
        
        image_num = len([x for x in message if x["type"] == "image"])
        use_qa_memory = self.qa_memory_enabled and image_num > 1
        use_vqa_memory = self.vqa_mem_enable and image_num > 1
        
        if not use_qa_memory and not use_vqa_memory:
            return super().generate_v2(message, dataset)
        
        if not self.model.init_prediction_config:
            self.model.preparing_for_generation(tokenizer=self.tokenizer)

        prompt = "<image>\n" + "\n".join([x["value"] for x in message if x["type"] == "text"])
        if dataset:
            prompt = self.build_video_prompt(prompt, dataset)

        image_path = [x["value"] for x in message if x["type"] == "image"]
        ori_image = [Image.open(_image_path).convert("RGB") for _image_path in image_path]
        pixel_values = self._build_video_input(ori_image)
        
        if use_qa_memory:
            question_text = self._extract_question_text(message, dataset)
            visual_features, keep_idx = self._mix_video_features_qa(pixel_values, question_text)
            if keep_idx is not None:
                pixel_values = pixel_values[keep_idx.to(pixel_values.device)]
        else:
            visual_features = self._mix_video_features_legacy(pixel_values)

        num_frames = visual_features.shape[0]
        num_image_tokens = self.model.patch_token
        image_token_str = f"{self.model.IMG_START_TOKEN}{self.model.IMG_CONTEXT_TOKEN * num_image_tokens}{self.model.IMG_END_TOKEN}\n"
        image_token_str = (image_token_str * num_frames).strip()
        text = prompt.replace("<image>", image_token_str)

        input_text = self.model.template["INSTRUCTION"].format(
            input=text, round=1, bot_name=self.model.bot_name
        )
        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        with torch.no_grad():
            generate_output = self.model.generate(
                pixel_values=pixel_values,
                visual_features=visual_features,
                input_ids=ids,
                attention_mask=attention_mask,
                generation_config=self.model.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.model.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        response = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False
        ).strip()
        response = (
            response.replace("<|end|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|im_end|>", "")
            .strip()
        )
        
        if use_qa_memory:
            print(f"[PROGRESS] QA_MEMORY_APPLIED mode={self.qa_fusion_mode} topk={self.qa_top_k} "
                  f"tau={self.qa_retrieval_temperature} num_frames={num_frames}")
        else:
            print(f"[PROGRESS] VQA_MEMORY_APPLIED enable=1 topk={self.vqa_mem_topk} "
                  f"tau={self.vqa_mem_tau} alpha={self.vqa_mem_alpha}")
        
        print("Question:", prompt[:100], "...\nResponse:")
        return response
