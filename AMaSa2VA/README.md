# AMaSa2VA: Adaptive Memory-Augmented Video Segmentation and Understanding

Code for **Adaptive Memory-Augmented Large Multimodal Model for Unified Video
Segmentation and Understanding**.

## Motivation

Unified video models often rely on the current observation or implicit temporal
propagation, making them vulnerable to occlusion, appearance changes, fast
motion, and distractors. AMaSa2VA explicitly retrieves task-relevant historical
evidence while preserving the original Sa2VA prediction as a reliable fallback.

## Innovations

- **Object-centric memory:** stores compact mask-conditioned target features
  instead of dense frame tokens.
- **Sparse retrieval and adaptive fusion:** retrieves Top-K historical evidence,
  maps it into the prompt space, and suppresses incompatible memory.
- **Unified temporal evidence utilization:** applies object memory to RefVOS and
  question-conditioned visual resource allocation to Video-MME.
- **Plug-and-play design:** keeps the pretrained MLLM and SAM2 frozen and does
  not modify the original Sa2VA source code.

## Results

### Video Segmentation (J&F)

| Method | MeViS | ReVOS | Ref-DAVIS17 |
|---|---:|---:|---:|
| Sa2VA-1B | 53.2 | 47.6 | 69.2 |
| **AMaSa2VA-1B** | **54.6** | **54.6** | **70.1** |
| Sa2VA-InternVL3-2B | 51.2 | 56.0 | 74.5 |
| **AMaSa2VA-InternVL3-2B** | **53.2** | **57.6** | **75.8** |
| Sa2VA-Qwen2.5-VL-3B | 51.1 | 50.7 | 72.7 |
| **AMaSa2VA-Qwen2.5-VL-3B** | **51.6** | **52.5** | **73.9** |

### Video-MME Accuracy

| Method | Overall | Short | Medium | Long |
|---|---:|---:|---:|---:|
| Sa2VA-1B | 44.0 | 51.6 | 42.3 | 38.2 |
| **AMaSa2VA-1B** | **45.6** | **55.0** | **43.8** | **38.4** |
| Sa2VA-InternVL3-2B | 49.5 | 59.0 | 47.2 | 42.2 |
| **AMaSa2VA-InternVL3-2B** | **52.4** | **61.8** | **51.0** | **44.4** |
| Sa2VA-Qwen2.5-VL-3B | 55.0 | 65.7 | 51.9 | 47.3 |
| **AMaSa2VA-Qwen2.5-VL-3B** | **57.3** | **68.3** | **55.6** | **48.0** |

## Installation

Install the upstream [Sa2VA](https://github.com/magic-research/Sa2VA)
environment first, then install this repository:

```bash
git clone <repository-url>
cd AMaSa2VA
python -m pip install -e .
python -m pip install -r requirements.txt  
```

Keep Sa2VA external and configure its source and evaluation directories:

```bash
export SA2VA_REPO_ROOT=/path/to/Sa2VA
export SA2VA_EVAL_ROOT=$SA2VA_REPO_ROOT/sa2va_eval
```

## Prompt-Adapter Training

```bash
python tools/dump_qwen3_teacher.py \
  --model_path <model> --data_root <data> --out_dir <teacher-output>

python tools/train_prompt_adapter.py \
  --shard_dir <teacher-output> --out_dir <adapter-output>
```

Only the lightweight prompt adapter is optimized; Sa2VA and SAM2 remain frozen.

## Inference

### Video Segmentation

Attach memory to an already loaded Sa2VA model before running the upstream
video-segmentation inference:

```python
from projects.amasa2va.integration import apply_memory_for_backbone, begin_expression
from projects.amasa2va.models.memory import PaperMemoryConfig

context, memory = apply_memory_for_backbone(
    model,
    PaperMemoryConfig(adapter_checkpoint="checkpoints/prompt_adapter.pt"),
    model_path=model_path,
    backbone="auto",
    processor=processor,
)

begin_expression(context, memory, video_id, expression_id)
prediction = model.predict_forward(...)
```

Qwen models require their processor; InternVL/Phi models do not.

Evaluate the generated prediction JSON with the shared metric entry point:

```bash
python projects/amasa2va/evaluation/evaluate_video_segmentation.py \
  <predictions.json> --dataset <dataset> \
  --expressions <expressions.json> --masks <masks> --output <metrics.json>
```

### Video Question Answering

```bash
scripts/run_videomme_vqa.sh \
  <gpu> <output-dir> <model-path> <auto|internvl|qwen> <base|core>
```

The unified evaluator can also be called directly:

```bash
QA_MEM_ENABLE=1 python projects/amasa2va/evaluation/run_videomme.py \
  --data Video-MME \
  --work-dir <output-dir> \
  --model-path <model-path> \
  --backbone auto
```

## Data and Models

Datasets and pretrained weights are not included. Please obtain Sa2VA,
MeViS, ReVOS, Ref-DAVIS17, and Video-MME from their official sources.

## Acknowledgements

This project builds upon
[Sa2VA](https://github.com/magic-research/Sa2VA) and
[SAM2](https://github.com/facebookresearch/sam2). We thank the authors for
their excellent work.

## Citation

```bibtex
@article{amasa2va2026,
  title={Adaptive Memory-Augmented Large Multimodal Model for Unified Video Segmentation and Understanding},
  year={2026}
}
```

## License

See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) before redistribution.
