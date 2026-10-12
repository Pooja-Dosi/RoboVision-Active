#!/usr/bin/env python3
import argparse, json, os, re, shlex, subprocess, time, datetime, pathlib, csv
from collections import Counter

SUCCESS_PAT = re.compile(r"SUCCESS", re.I)
TIMEOUT_PAT = re.compile(r"TIMEOUT", re.I)
FAIL_PAT    = re.compile(r"FAIL", re.I)

def detect_status(text: str) -> str:
    # scan last lines first
    for line in reversed(text.splitlines()):
        if SUCCESS_PAT.search(line): return "SUCCESS"
        if TIMEOUT_PAT.search(line): return "TIMEOUT"
        if FAIL_PAT.search(line):    return "FAIL"
    return "UNKNOWN"

def replace_or_add_autoquit(cmd_str: str, secs: int) -> str:
    toks = shlex.split(cmd_str)
    if "--autoquit-sec" in toks:
        i = toks.index("--autoquit-sec")
        if i + 1 < len(toks): toks[i + 1] = str(secs)
    else:
        toks += ["--autoquit-sec", str(secs)]
    return shlex.join(toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--autoquit-sec", type=int, default=25)
    ap.add_argument("--gl", default="glfw", help="MUJOCO_GL value")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    bdir = pathlib.Path(args.baseline_dir)
    cfg  = json.loads((bdir / "eval_cfg.json").read_text())
    base_cmd = cfg["command_used"]
    run_cmd  = replace_or_add_autoquit(base_cmd, args.autoquit_sec)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = bdir / f"batch_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for t in range(args.trials):
        env = os.environ.copy()
        env["MUJOCO_GL"] = args.gl
        print(f"[{t+1}/{args.trials}] running…")
        proc = subprocess.run(run_cmd, shell=True, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout
        (outdir / f"run_{t:03d}.log").write_text(log)
        status = detect_status(log)
        results.append({"trial": t, "status": status})
        print(f"  -> {status}")
        time.sleep(args.sleep)

    # save CSV + summary
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "status"])
        w.writeheader(); w.writerows(results)

    counts = Counter(r["status"] for r in results)
    total = len(results)
    summary = {
        "total": total,
        "success": counts.get("SUCCESS", 0),
        "timeout": counts.get("TIMEOUT", 0),
        "fail": counts.get("FAIL", 0),
        "unknown": counts.get("UNKNOWN", 0),
        "success_rate": (counts.get("SUCCESS", 0) / total) if total else 0.0,
        "baseline_dir": str(bdir),
        "viewer_command": base_cmd,
        "viewer_command_used": run_cmd,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Logs: {outdir}")

if __name__ == "__main__":
    main()
