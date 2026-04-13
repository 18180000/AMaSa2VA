import os

import torch
import torch.nn.functional as F
from PIL import Image

from vlmeval.vlm.sa2va_chat import Sa2VAChat


class Sa2VAChatMem(Sa2VAChat):
    """Safer VQA-only memory path.

    This class does not touch the segmentation eval entrypoints. It only
    changes the Video VQA path when VQA_MEM_ENABLE=1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vqa_mem_enable = os.environ.get("VQA_MEM_ENABLE", "0").strip() == "1"
        self.vqa_mem_topk = int(os.environ.get("VQA_MEM_TOPK", "4"))
        self.vqa_mem_tau = float(os.environ.get("VQA_MEM_TAU", "0.4"))
        self.vqa_mem_alpha = float(os.environ.get("VQA_MEM_ALPHA", "0.5"))
        self._denylist = self._load_denylist(getattr(self, "pl_denylist_path", ""))

    def _build_video_input(self, video):
        dtype = getattr(self.model.vision_model, "dtype", torch.bfloat16)
        pixel_values = []
        for frame_image in video:
            pixel_values.append(self.model.transformer(frame_image))
        pixel_values = torch.stack(pixel_values, dim=0).to(dtype)
        return pixel_values

    def _mix_video_features(self, pixel_values):
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
            deny = set()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        deny.add(int(line))
                    except ValueError:
                        continue
            if deny:
                print(f"[PROGRESS] VQA_DENYLIST_LOADED n={len(deny)} path={path}")
            return deny if deny else None
        except Exception as e:
            print(f"[WARN] VQA_DENYLIST_LOAD_FAIL path={path} err={e}")
            return None

    def generate_v2(self, message, dataset=None):
        if self._denylist is not None:
            idx = getattr(self, "_current_index", None)
            if idx is not None and idx in self._denylist:
                print(f"[PROGRESS] VQA_DENYLIST_SKIP idx={idx}")
                return super().generate_v2(message, dataset)
        image_num = len([x for x in message if x["type"] == "image"])
        if not self.vqa_mem_enable or image_num <= 1:
            return super().generate_v2(message, dataset)
        if not self.model.init_prediction_config:
            self.model.preparing_for_generation(tokenizer=self.tokenizer)

        prompt = "<image>\n" + "\n".join([x["value"] for x in message if x["type"] == "text"])
        if dataset:
            prompt = self.build_video_prompt(prompt, dataset)

        image_path = [x["value"] for x in message if x["type"] == "image"]
        ori_image = [Image.open(_image_path).convert("RGB") for _image_path in image_path]
        pixel_values = self._build_video_input(ori_image)
        visual_features = self._mix_video_features(pixel_values)

        num_image_tokens = self.model.patch_token
        num_frames = len(ori_image)
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
        print(
            f"[PROGRESS] VQA_MEMORY_APPLIED enable=1 topk={self.vqa_mem_topk} tau={self.vqa_mem_tau} alpha={self.vqa_mem_alpha}"
        )
        print("Question:", prompt, "\nResponse:")
        return response
