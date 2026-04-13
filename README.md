# Sa2VA Video QA Enhancement

This repository contains the implementation of video question answering enhancement methods for the Sa2VA multimodal model.

## Overview

This project enhances Sa2VA's video question answering capabilities through:

- **Frame-level Visual Feature Memory**: Computes similarity between frames and mixes features using top-k similar frames
- **Learnable Adapter**: Generates feature increments based on inter-frame similarity memory
- **Confidence-based Gating**: Controls adapter increment application via cosine gating
- **Denylist Mechanism**: Skips enhancement for specific samples and falls back to original inference

## Three Modes

| Mode | Description |
|------|-------------|
| `base` | Original model without enhancement |
| `core` | Adapter + Gate enabled |
| `feedback` | Adapter + Gate + Denylist enabled |

## Prerequisites

### 1. Upstream Repository

Download the official Sa2VA repository:
```bash
git clone <Sa2VA-repo-url>
```

### 2. Model Weights

Prepare model weights (e.g., Sa2VA-1B, Sa2VA-InternVL3-2B, Sa2VA-4B).

### 3. Python Environment

```bash
conda create -n sa2va-vqa python=3.10
conda activate sa2va-vqa
pip install -r requirements.txt
```

## Installation

### 1. Directory Structure

```
your-project/
├── Sa2VA-main/           # Upstream repository
│   └── sa2va_eval/
└── sa2va-vqa-enhancement/  # This repository
    ├── configs/
    ├── scripts/
    ├── tools/
    └── vendor_stubs/
```

### 2. Configure Paths

```bash
cd configs
cp config.template.sh config.local.sh
# Edit config.local.sh with your actual paths
```

### 3. Environment Variables

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=/path/to/sa2va-vqa-enhancement/vendor_stubs
export TMPDIR=/tmp
```

## Usage

### Video QA with Memory Enhancement

```bash
# Core mode
bash scripts/run_videomme_1b_vqa_mem_core.sh <gpu_id> <output_dir> <model_path>

# Feedback mode
bash scripts/run_videomme_1b_vqa_mem_feedback.sh <gpu_id> <output_dir> <model_path> <denylist_path>
```

### Build Denylist

```bash
python tools/build_videomme_denylist.py \
    --xlsx <prediction_file.xlsx> \
    --out <denylist.txt>
```

## Key Files

### Core Implementation
- `tools/sa2va_chat_mem.py` - VQA Memory implementation
- `tools/run_sa2va_eval_local_mem.py` - VQA evaluation entry point

### Configuration
- `configs/modes.sh` - Mode environment variable definitions
- `configs/config.template.sh` - Path configuration template

### Scripts
- `scripts/run_videomme_1b_vqa_mem_core.sh` - Video-MME core mode
- `scripts/run_videomme_1b_vqa_mem_feedback.sh` - Video-MME feedback mode

## Verification

After running, check logs for:
- `VQA_PL_ADAPTER_APPLIED` - Adapter enabled
- `VQA_MEMORY_APPLIED` - Memory enabled
- `VQA_DENYLIST_LOADED` - Denylist loaded

## Known Issues

1. **TMPDIR**: Ensure `TMPDIR` points to an existing directory
2. **CUPTI**: If `libcupti.so.12` is missing, add CUPTI path to `LD_LIBRARY_PATH`
3. **GPU Mapping**: Use `CUDA_VISIBLE_DEVICES` carefully on systems with complex CUDA mappings

## Citation

If you find this work useful, please cite:

```bibtex
@misc{sa2va-vqa-enhancement,
  title={Sa2VA Video QA Enhancement},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Sa2VA](https://github.com/...) - The upstream multimodal model
