#!/usr/bin/env python3
import argparse, csv, datetime, json, os, pathlib, re, shlex, subprocess, time
from collections import Counter

# We detect outcomes straight from eval_bc_viewer.py logs.
PAT_SUCCESS = re.compile(r"\[RESULT\]\s*SUCCESS\b|\bSUCCESS\b", re.I)
PAT_TIMEOUT = re.compile(r"\bTIMEOUT\b", re.I)
PAT_FAIL    = re.compile(r"\bFAIL\b", re.I)

def detect_status(text: str) -> str:
    # Look from the end so final status wins.
    for line in reversed(text.splitlines()):
        if PAT_SUCCESS.search(line): return "SUCCESS"
        if PAT_TIMEOUT.search(line): return "TIMEOUT"
        if PAT_FAIL.search(line):    return "FAIL"
    return "UNKNOWN"

def build_base_cmd(args, eye_on: bool):
    # Compose the exact command for eval_bc_viewer.py as a token list (no zsh parsing headaches).
    tokens = [
        "mjpython", "eval_bc_viewer.py",
        "--ckpt", args.bc_ckpt,
        "--model", args.model,
        "--assist-scale", str(args.assist_scale),
        "--success-r", str(args.success_r),
        "--rim-offset", str(args.rim_offset),
        "--success-settle-steps", str(args.success_settle_steps),
        "--ball-vel-thresh", str(args.ball_vel_thresh),
        "--autoquit-sec", str(args.autoquit_sec),
    ]
    if not eye_on:
        tokens.append("--no-eye-tracking")
    return tokens

def run_trials(label, tokens, trials, outdir, env, sleep_s):
    results = []
    for t in range(trials):
        print(f"[{label} {t+1}/{trials}] running…")
        proc = subprocess.run(tokens, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = proc.stdout
        (outdir / f"{label.lower()}_{t:03d}.log").write_text(log)
        status = detect_status(log)
        print(f"  -> {status}")
        results.append({"mode": label, "trial": t, "status": status})
        time.sleep(sleep_s)
    return results

def summarize(results, label):
    c = Counter(r["status"] for r in results if r["mode"] == label)
    total = sum(c.values())
    return {
        "total": total,
        "success": c.get("SUCCESS", 0),
        "timeout": c.get("TIMEOUT", 0),
        "fail": c.get("FAIL", 0),
        "unknown": c.get("UNKNOWN", 0),
        "success_rate": (c.get("SUCCESS", 0) / total) if total else 0.0
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate BC (+ optional Eye RL) with consistent success metrics.")
    ap.add_argument("--bc-ckpt", required=True, help="Path to BC checkpoint (e.g., artifacts/bc_resnet18.pth)")
    ap.add_argument("--model", required=True, help="MuJoCo XML (e.g., panda_assets/mjx_panda.xml)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--gl", default="glfw", help="MUJOCO_GL backend (glfw, egl, osmesa)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pause between trials (s)")
    # success metric parity (use the same on both Eye ON/OFF)
    ap.add_argument("--success-r", type=float, default=0.08)
    ap.add_argument("--rim-offset", type=float, default=0.045)
    ap.add_argument("--success-settle-steps", type=int, default=120)
    ap.add_argument("--ball-vel-thresh", type=float, default=0.06)
    ap.add_argument("--autoquit-sec", type=int, default=25)
    # BC control blend
    ap.add_argument("--assist-scale", type=float, default=0.0, help="0.0=pure BC, 1.0=pure IK")
    # whether to also test with Eye OFF
    ap.add_argument("--compare-off", action="store_true", help="Also run a block with --no-eye-tracking")
    # output location
    ap.add_argument("--outdir", default="eye_runs/reports", help="Where to save logs & summary")
    args = ap.parse_args()

    # Prepare output directory
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) / f"bc_rl_eval_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Env
    env = os.environ.copy()
    env["MUJOCO_GL"] = args.gl

    # Build and run blocks
    all_results = []
    eye_on_cmd = build_base_cmd(args, eye_on=True)
    print("=== EYE ON command ===")
    print(" ", shlex.join(eye_on_cmd))
    all_results += run_trials("EYE_ON", eye_on_cmd, args.trials, outdir, env, args.sleep)

    if args.compare_off:
        eye_off_cmd = build_base_cmd(args, eye_on=False)
        print("=== EYE OFF command ===")
        print(" ", shlex.join(eye_off_cmd))
        all_results += run_trials("EYE_OFF", eye_off_cmd, args.trials, outdir, env, args.sleep)

    # Save raw results CSV
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "trial", "status"])
        w.writeheader(); w.writerows(all_results)

    # Summaries
    summary = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "trials_per_block": args.trials,
        "commands": {
            "eye_on":  shlex.join(eye_on_cmd),
            "eye_off": shlex.join(build_base_cmd(args, eye_on=False)) if args.compare_off else None
        },
        "metrics": {
            "success_r": args.success_r,
            "rim_offset": args.rim_offset,
            "success_settle_steps": args.success_settle_steps,
            "ball_vel_thresh": args.ball_vel_thresh,
            "autoquit_sec": args.autoquit_sec
        },
        "results": {
            "EYE_ON": summarize(all_results, "EYE_ON"),
            **({"EYE_OFF": summarize(all_results, "EYE_OFF")} if args.compare_off else {})
        },
        "outdir": str(outdir)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Logs & summary saved to: {outdir}")

if __name__ == "__main__":
    main()
