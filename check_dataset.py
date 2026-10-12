# check_dataset.py
import numpy as np, json, glob, pathlib

roots = sorted(glob.glob("runs/*"))
if not roots:
    raise RuntimeError("No runs/* folder found. Collect episodes first.")

root = roots[-1]
eps = sorted(glob.glob(f"{root}/ep_*/episode_data.npz"))
print(f"Found episodes: {len(eps)} in {root}")

succ = 0
for epdir in sorted(pathlib.Path(root).glob("ep_*")):
    meta = epdir / "episode.json"
    if meta.exists():
        with open(meta, "r") as f:
            s = json.load(f)
        succ += int(bool(s.get("success", False)))
print(f"Success count (from episode.json): {succ}/{len(eps)}")

if eps:
    sample = np.load(eps[0])
    print("\n[Sample episode shapes]")
    for k in ["actions","qpos","qvel","ee","ball","bucket","dt"]:
        v = sample[k]
        if hasattr(v, "shape"):
            print(f"{k:>7} {v.shape}")
        else:
            print(f"{k:>7} {v}")
