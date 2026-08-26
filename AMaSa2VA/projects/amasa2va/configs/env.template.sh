#!/bin/bash
# Configuration template for local paths
# Copy this file to config.local.sh and modify the paths according to your environment

# =============================================================================
# Sa2VA Repository Paths
# =============================================================================
export SA2VA_REPO_ROOT="/path/to/Sa2VA-main"
export SA2VA_EVAL_ROOT="${SA2VA_REPO_ROOT}/sa2va_eval"
# Keep the original Sa2VA checkout external. Segmentation teacher/evaluation
# imports projects.* from SA2VA_REPO_ROOT; video QA imports run.py and vlmeval
# from SA2VA_EVAL_ROOT.

# =============================================================================
# Model Paths
# =============================================================================
export MODEL_1B_PATH="/path/to/pretrained/Sa2VA-1B"
export MODEL_2B_PATH="/path/to/pretrained/Sa2VA-InternVL3-2B"
export MODEL_4B_PATH="/path/to/pretrained/Sa2VA-4B"
export MODEL_QWEN_PATH="/path/to/pretrained/Sa2VA-Qwen2_5-VL-3B"

# =============================================================================
# Adapter Checkpoint
# =============================================================================
export ADAPTER_CKPT_PATH="/path/to/pl_adapter_ckpt.pt"

# =============================================================================
# Data Paths
# =============================================================================
export DATA_ROOT="/path/to/data"
export VIDEO_MME_DATA="${DATA_ROOT}/Video-MME"
export MMBENCH_VIDEO_DATA="${DATA_ROOT}/MMBench-Video"
export SA2VA_DATA_ROOT="/path/to/sa2va-data"

# Dataset specific paths
export DAVIS17_META="/path/to/davis17_valid/meta_expressions.json"
export DAVIS17_MASK="/path/to/davis17/valid/mask_dict.json"
export REVOS_META="/path/to/revos_valid/meta_expressions.json"
export REVOS_MASK="/path/to/revos/mask_dict.json"
export MEVIS_U_META="/path/to/mevis/valid_u/meta_expressions.json"
export MEVIS_U_MASK="/path/to/mevis/valid_u/mask_dict.json"

# =============================================================================
# Cache Paths
# =============================================================================
export HF_HOME="/path/to/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="/path/to/torch_home"
export XDG_CACHE_HOME="/path/to/xdg_cache"

# =============================================================================
# Output Paths
# =============================================================================
export OUTPUT_ROOT="/path/to/outputs"
export TMPDIR="/tmp"

# =============================================================================
# Python Environment
# =============================================================================
export PYTHON_BIN="/path/to/conda/envs/sa2va-vqa-clean/bin/python"

# =============================================================================
# CUDA Settings
# =============================================================================
export CUDA_VISIBLE_DEVICES="0"
export CUPTI_LIB_PATH="/usr/local/cuda/extras/CUPTI/lib64"
