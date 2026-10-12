#!/usr/bin/env python3
import argparse, json, os, shutil, subprocess, datetime, pathlib

def main():
    p = argparse.ArgumentParser(description="Freeze a BC baseline with reproducible metadata.")
    p.add_argument("--ckpt", required=True, help="Path to BC checkpoint (e.g., artifacts/bc_resnet18.pth)")
    p.add_argument("--tag", default="bc", help="Short tag for the baseline folder name")
    p.add_argument("--viewer-cmd", default="", help="The exact eval command you ran (for provenance)")
    p.add_argument("--ema", type=float, default=None)
    p.add_argument("--cmd-ema", type=float, default=None)
    p.add_argument("--assist-scale", type=float, default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    outdir = pathlib.Path(f"eye_runs/baselines/{ts}_{args.tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_src = pathlib.Path(args.ckpt)
    ckpt_dst = outdir / ckpt_src.name
    shutil.copy2(ckpt_src, ckpt_dst)

    cfg = {
        "viewer": "eval_bc_viewer.py",
        "ckpt": ckpt_dst.name,
        "model": args.model,
        "ema": args.ema,
        "cmd_ema": args.cmd_ema,
        "assist_scale": args.assist_scale,
        "command_used": args.viewer_cmd or f"mjpython eval_bc_viewer.py --ckpt {ckpt_dst.name}"
    }
    (outdir / "eval_cfg.json").write_text(json.dumps(cfg, indent=2))

    # Pin environment
    try:
        pyver = subprocess.check_output(["python", "-V"], text=True).strip()
    except Exception:
        pyver = "unknown"
    (outdir / "python_version.txt").write_text(pyver)

    try:
        reqs = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:
        reqs = ""
    (outdir / "requirements.txt").write_text(reqs)

    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "nogit"
    (outdir / "git_commit.txt").write_text(git)

    print(f"Baseline frozen at: {outdir}")

if __name__ == "__main__":
    main()
