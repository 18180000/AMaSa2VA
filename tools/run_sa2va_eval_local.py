#!/opt/anaconda3/envs/torch/bin/python
import argparse
import os
import sys
from functools import partial
from pathlib import Path


def ensure_link(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        return
    os.symlink(target_path, link_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "infer"])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--nframe", type=int, default=8)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--use-subtitle", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    repo_root = Path("/home/usergjf/Sa2VA/Sa2VA-main/sa2va_eval")
    workspace_root = Path("/home/usergjf/stage12_restart_v1")

    os.environ.setdefault("HF_HOME", "/data0/hf_home")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data0/hf_home/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data0/hf_home/datasets")
    os.environ.setdefault("TORCH_HOME", "/data0/torch_home")
    os.environ.setdefault("XDG_CACHE_HOME", "/data0/xdg_cache")
    os.environ.setdefault("TMPDIR", "/data0/tmp")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONUNBUFFERED"] = "1"

    ensure_link(
        workspace_root / "dataset" / "video_vlm" / "video_benchmarks" / "Video-MME",
        Path("/data0/data/Video-MME"),
    )
    ensure_link(
        workspace_root / "dataset" / "video_vlm" / "video_benchmarks" / "MMBench-Video",
        Path("/data0/data/MMBench-Video"),
    )

    sys.path.insert(0, str(repo_root))
    os.chdir(workspace_root)

    from vlmeval.config import supported_VLM
    from vlmeval.vlm.sa2va_chat import Sa2VAChat
    import run as eval_run

    supported_VLM[args.model_name] = partial(Sa2VAChat, model_path=args.model_path)

    cli = [
        "run.py",
        "--data",
        *args.data,
        "--model",
        args.model_name,
        "--work-dir",
        args.work_dir,
        "--mode",
        args.mode,
        "--nframe",
        str(args.nframe),
    ]
    if args.pack:
        cli.append("--pack")
    if args.use_subtitle:
        cli.append("--use-subtitle")

    sys.argv = cli
    eval_run.main()


if __name__ == "__main__":
    main()
