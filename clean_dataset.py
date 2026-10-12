import numpy as np, json, pathlib

root = sorted(pathlib.Path("runs").glob("*"))[-1]
print("Cleaning latest run:", root)

X, Y = [], []
stride = 5

for ep in sorted(root.glob("ep_*")):
    meta = ep / "episode.json"
    if not meta.exists():
        continue
    with open(meta, "r") as f:
        s = json.load(f)
    if not s.get("success", False):
        continue

    npz = ep / "episode_data.npz"
    frames_dir = ep / "frames"
    if not npz.exists() or not frames_dir.exists():
        continue

    data = np.load(npz)
    actions = data["actions"][::stride]  # subsample to match frames
    n_frames = len(list(frames_dir.glob("*.png")))
    n = min(len(actions), n_frames)

    actions = actions[:n]
    X.extend([str(f) for f in sorted(frames_dir.glob("*.png"))[:n]])
    Y.extend(actions)

X, Y = np.array(X), np.array(Y)
print("Final dataset:", X.shape, Y.shape)

# compute normalization stats
mean, std = Y.mean(axis=0), Y.std(axis=0)
np.savez("bc_dataset_clean.npz", frames=X, actions=Y, mean=mean, std=std)
print("Saved bc_dataset_clean.npz with mean/std")
