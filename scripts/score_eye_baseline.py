#!/usr/bin/env python3
import argparse, json, os, re, shlex, subprocess, time, datetime, pathlib, csv
from collections import Counter

SUCCESS_PAT = re.compile(r"SUCCESS", re.I)
TIMEOUT_PAT = re.compile(r"TIMEOUT", re.I)
FAIL_PAT    = re.compile(r"FAIL", re.I)

def detect_status(text: str) -> str:
    for line in reversed(text.splitlines()):
        if SUCCESS_PAT.search(line): return "SUCCESS"
        if TIMEOUT_PAT.search(line): return "TIMEOUT"
        if FAIL_PAT.search(line):    return "FAIL"
    return "UNKNOWN"

def replace_or_add(tokens, flag, value):
    if flag in tokens:
        i = tokens.index(flag)
        if i + 1 < len(tokens):
            tokens[i+1] = str(value)
        else:
            tokens.append(str(value))
    else:
        tokens += [flag, str(value)]
    return tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--gl", default="glfw")
    # Optional overrides to make the viewer non-blocking & deterministic:
    ap.add_argument("--pause-after-ep", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    bdir = pathlib.Path(args.baseline_dir)
    cfg  = json.loads((bdir / "eval_cfg.json").read_text())
    base_cmd = cfg["command_used"]

    # tweak command: remove pausing, ensure max steps, and vary seeds
    base = shlex.split(base_cmd)
    base = replace_or_add(base, "--pause-after-ep", args.pause_after_ep)
    base = replace_or_add(base, "--max-steps", args.max_steps)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = bdir / f"batch_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for t in range(args.trials):
        toks = list(base)
        toks = replace_or_add(toks, "--seed", args.seed_offset + t)
        cmd  = shlex.join(toks)

        env = os.environ.copy()
        env["MUJOCO_GL"] = args.gl
        print(f"[{t+1}/{args.trials}] running…")
        proc = subprocess.run(cmd, shell=True, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout
        (outdir / f"run_{t:03d}.log").write_text(log)
        status = detect_status(log)
        results.append({"trial": t, "status": status})
        print(f"  -> {status}")
        time.sleep(args.sleep)

    # write CSV + summary
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "status"])
        w.writeheader(); w.writerows(results)

    from collections import Counter
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
        "viewer_command_used": shlex.join(base),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Logs: {outdir}")

if __name__ == "__main__":
    main()
