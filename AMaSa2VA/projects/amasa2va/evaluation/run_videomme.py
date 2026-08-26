#!/usr/bin/env python3
"""Unified Video-MME entry with explicit InternVL/Phi and Qwen routing.

Sa2VA-1B, Sa2VA-InternVL3-2B and Sa2VA-4B use ``Sa2VAChatMem``;
Sa2VA-Qwen2.5-VL uses the separately maintained ``Sa2VAChatQwenMem``.

This is the single canonical entry point for the video-QA method and
supersedes the older per-backbone run_sa2va_eval_local*.py scripts.

Two vlmeval quirks this script works around:
  1. get_cache_path() hardcodes the relative path
     'dataset/video_vlm/video_benchmarks/Video-MME/', so we chdir into a
     workspace root and place a symlink there.
  2. Video frames are cached under $LMUData/images/<dataset>/, and LMUDataRoot()
     only honours $LMUData if the directory already exists.

All paths below can be overridden either via CLI flags or the environment
variables used as their defaults (see ``projects/amasa2va/configs/env.template.sh``).
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from projects.amasa2va.models.backbones import QWEN, resolve_backbone

SA2VA_EVAL_ROOT = Path(os.environ.get('SA2VA_EVAL_ROOT', '/path/to/Sa2VA-main/sa2va_eval'))
DEFAULT_MODEL_PATH = os.environ.get('MODEL_4B_PATH', '/path/to/pretrained/Sa2VA-4B')
DEFAULT_VIDEOMME_ROOT = Path(os.environ.get('VIDEO_MME_DATA', '/path/to/data/Video-MME'))
DEFAULT_LMUDATA = Path(os.environ.get('LMUData', '/path/to/data/LMUData'))


def ensure_link(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        return
    link_path.symlink_to(target_path.resolve())
    print(f'[SETUP] linked {link_path} -> {target_path}')


def validate_sa2va_eval_root(path: Path) -> None:
    missing = [name for name in ('run.py', 'vlmeval') if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f'SA2VA_EVAL_ROOT={path} is not an upstream Sa2VA sa2va_eval '
            f'directory; missing: {", ".join(missing)}'
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', default=['Video-MME'])
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--mode', default='all', choices=['all', 'infer'])
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--use-subtitle', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--model-name', default='Sa2VA-4B-MEM')
    parser.add_argument('--videomme-root', default=str(DEFAULT_VIDEOMME_ROOT))
    parser.add_argument('--lmudata', default=str(DEFAULT_LMUDATA))
    parser.add_argument('--baseline', action='store_true',
                        help='disable memory; use stock Sa2VAChat on InternVL/Phi')
    parser.add_argument('--backbone', default='auto',
                        choices=['auto', 'internvl', 'qwen'],
                        help='wrapper family; auto reads local checkpoint metadata')
    parser.add_argument('--qwen', action='store_true',
                        help='deprecated alias for --backbone qwen')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.qwen and args.backbone not in ('auto', 'qwen'):
        parser.error('--qwen conflicts with --backbone internvl')
    requested_backbone = 'qwen' if args.qwen else args.backbone
    try:
        backbone = resolve_backbone(args.model_path, requested_backbone)
    except ValueError as exc:
        parser.error(str(exc))
    if args.baseline:
        # Qwen still needs its processor-aware wrapper. Turning both switches
        # off makes that wrapper follow its unmodified base generation path.
        os.environ['VQA_MEM_ENABLE'] = '0'
        os.environ['QA_MEM_ENABLE'] = '0'
    validate_sa2va_eval_root(SA2VA_EVAL_ROOT)

    lmudata = Path(args.lmudata)
    lmudata.mkdir(parents=True, exist_ok=True)
    os.environ['LMUData'] = str(lmudata)
    os.environ['HF_HOME'] = os.environ.get('HF_HOME', str(lmudata / 'hf_home'))
    os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', str(lmudata / 'torch_home'))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYTHONUNBUFFERED'] = '1'

    # vlmeval's VideoMME.get_cache_path() hardcodes a path relative to the cwd;
    # we chdir into a scratch workspace and place a symlink there instead of
    # requiring the real dataset to live inside the repo.
    scratch_root = Path(os.environ.get('SA2VA_VQA_SCRATCH', str(WORKSPACE_ROOT / '.scratch')))
    ensure_link(
        scratch_root / 'dataset' / 'video_vlm' / 'video_benchmarks' / 'Video-MME',
        Path(args.videomme_root),
    )

    sys.path.insert(0, str(SA2VA_EVAL_ROOT))
    sys.path.insert(0, str(WORKSPACE_ROOT))
    os.chdir(scratch_root)

    from vlmeval.config import supported_VLM
    import run as eval_run

    if args.baseline and backbone != QWEN:
        from vlmeval.vlm.sa2va_chat import Sa2VAChat as model_cls
    elif backbone == QWEN:
        from projects.amasa2va.models.qwen2_5_vl.video_qa import Sa2VAChatQwenMem as model_cls
    else:
        from projects.amasa2va.models.internvl.video_qa import Sa2VAChatMem as model_cls

    supported_VLM[args.model_name] = partial(model_cls, model_path=args.model_path)
    print(f'[SETUP] backbone={backbone} registered {args.model_name} -> '
          f'{model_cls.__name__} ({args.model_path})')

    cli = [
        'run.py',
        '--data', *args.data,
        '--model', args.model_name,
        '--work-dir', args.work_dir,
        '--mode', args.mode,
        '--nframe', str(args.nframe),
    ]
    if args.pack:
        cli.append('--pack')
    if args.use_subtitle:
        cli.append('--use-subtitle')
    if args.verbose:
        cli.append('--verbose')

    sys.argv = cli
    eval_run.main()


if __name__ == '__main__':
    main()
