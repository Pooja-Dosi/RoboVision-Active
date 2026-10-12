#!/usr/bin/env python3
import argparse, datetime, json, os, shutil, subprocess, sys, textwrap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="eye_runs/ball_vision_orig",
                    help="Folder that contains the successful PPO run (with eye_ppo etc.)")
    ap.add_argument("--ckpt-name", default="eye_ppo",
                    help="Checkpoint stem name inside --base (no .zip)")
    ap.add_argument("--tag", default="ball_vision_orig",
                    help="Short label to include in the baseline folder name")
    # These are today’s good eval settings; change if you later re-tune
    ap.add_argument("--discrete", action="store_true", default=True)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--step-deg", type=float, default=2.0)
    ap.add_argument("--invert-tilt", action="store_true", default=True)
    ap.add_argument("--invert-pan", action="store_true", default=False)
    ap.add_argument("--radius-frac", type=float, default=0.25)
    ap.add_argument("--visible-window", type=int, default=10)
    ap.add_argument("--reward-mode", choices=["state","vision"], default="vision")
    args = ap.parse_args()

    stamp = datetime.date.today().isoformat()
    outdir = os.path.join("eye_runs", "baselines", f"{stamp}_{args.tag}")
    os.makedirs(os.path.dirname(outdir), exist_ok=True)

    # 1) copy the whole run folder
    if not os.path.isdir(args.base):
        sys.exit(f"[error] --base not found: {args.base}")
    shutil.copytree(args.base, outdir, dirs_exist_ok=True)

    # 2) write eval_cfg.json
    cfg = {
        "discrete": bool(args.discrete),
        "frame_skip": args.frame_skip,
        "step_deg": args.step_deg,
        "invert_tilt": args.invert_tilt,
        "invert_pan": args.invert_pan,
        "radius_frac": args.radius_frac,
        "visible_window": args.visible_window,
        "reward_mode": args.reward_mode,
        "command_used": (
            "mjpython eval_eye_ppo_viewer.py "
            f"--discrete --ckpt {args.base}/{args.ckpt_name} "
            f"--frame-skip {args.frame_skip} --step-deg {args.step_deg} "
            f"{'--invert-tilt ' if args.invert_tilt else ''}"
            f"{'--invert-pan ' if args.invert_pan else ''}"
            f"--radius-frac {args.radius_frac} --visible-window {args.visible_window} "
            f"--reward-mode {args.reward_mode} --no-autoreset --pause-after-ep 1.5"
        ).strip()
    }
    with open(os.path.join(outdir, "eval_cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 3) pin environment versions
    with open(os.path.join(outdir, "python_version.txt"), "w") as f:
        f.write(sys.version + "\n")

    try:
        r = subprocess.run(["pip", "freeze"], check=False, capture_output=True, text=True)
        with open(os.path.join(outdir, "requirements.txt"), "w") as f:
            f.write(r.stdout or "")
    except Exception as e:
        with open(os.path.join(outdir, "requirements.txt"), "w") as f:
            f.write(f"# pip freeze failed: {e}\n")

    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True)
        commit = r.stdout.strip() if r.returncode == 0 else "nogit"
    except Exception:
        commit = "nogit"
    with open(os.path.join(outdir, "git_commit.txt"), "w") as f:
        f.write(commit + "\n")

    # 4) friendly summary
    print(f"=== Baseline frozen to: {outdir} ===")
    print("Contents:")
    for name in sorted(os.listdir(outdir)):
        print(" -", name)
    print("\n---- eval_cfg.json ----")
    with open(os.path.join(outdir, "eval_cfg.json")) as f:
        print(f.read())
    print("\nDone.")

if __name__ == "__main__":
    main()
