# train_bc_resnet.py
import os, json, argparse, numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

class FramesNPZ(Dataset):
    def __init__(self, npz_path, img_size=112):
        z = np.load(npz_path, allow_pickle=True)
        self.frames = z["frames"]
        self.actions = z["actions"].astype(np.float32)
        self.mean = z["mean"].astype(np.float32)
        self.std  = z["std"].astype(np.float32)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.frames)
    def __getitem__(self, i):
        img = Image.open(self.frames[i]).convert("RGB")
        x = self.tf(img)
        y = self.actions[i]
        return x, y

class BCHead(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )
    def forward(self, f): return self.net(f)

def make_model(freeze_encoder=True):
    enc = models.resnet18(weights=None)  # no download, random init
    if freeze_encoder:
        for p in enc.parameters(): p.requires_grad = False
    feat_dim = enc.fc.in_features
    enc.fc = nn.Identity()
    head = BCHead(in_dim=feat_dim, hidden=512, out_dim=8)
    return enc, head

def train(args):
    ds = FramesNPZ(args.data, img_size=args.img_size)
    n_total = len(ds)
    n_val   = max(1024, int(0.05*n_total))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)
    tr, va = random_split(ds, [n_train, n_val], generator=g)

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, head = make_model(freeze_encoder=not args.unfreeze)
    enc, head = enc.to(device), head.to(device)

    # Predict normalized actions: y_norm = (y - mean)/std
    mean = torch.tensor(ds.mean, device=device)
    std  = torch.tensor(ds.std + 1e-8, device=device)
    loss_fn = nn.L1Loss()

    params = list(head.parameters()) + (list(enc.parameters()) if args.unfreeze else [])
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    best_va = float("inf")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        enc.train(); head.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            with torch.set_grad_enabled(True):
                f = enc(xb)
                yb_norm = (yb - mean) / std
                pred_norm = head(f)
                loss = loss_fn(pred_norm, yb_norm)
                opt.zero_grad(); loss.backward(); opt.step()
                tr_loss += loss.item() * xb.size(0)
        tr_loss /= n_train

        # val
        enc.eval(); head.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                f = enc(xb)
                yb_norm = (yb - mean) / std
                pred_norm = head(f)
                loss = loss_fn(pred_norm, yb_norm)
                va_loss += loss.item() * xb.size(0)
        va_loss /= n_val

        print(f"[epoch {epoch:03d}] train L1(norm)={tr_loss:.4f}  val L1(norm)={va_loss:.4f}")

        if va_loss < best_va:
            best_va = va_loss
            torch.save({
                "enc": enc.state_dict(),
                "head": head.state_dict(),
                "mean": ds.mean,
                "std": ds.std,
                "img_size": args.img_size,
            }, out_dir / "bc_resnet18.pth")
            with open(out_dir / "bc_config.json","w") as f:
                json.dump({"encoder":"resnet18","img_size":args.img_size,"unfreeze":bool(args.unfreeze)}, f, indent=2)

    # quick sample preds dump
    with torch.no_grad():
        xb, yb = next(iter(dl_va))
        xb, yb = xb.to(device), yb.to(device)
        f = enc(xb); pred_norm = head(f)
        pred = pred_norm * std + mean
        np.savez(out_dir / "bc_val_preview.npz",
                 gt=yb.cpu().numpy(), pred=pred.cpu().numpy())
    print(f"Saved best model + preview to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="bc_dataset_clean.npz")
    ap.add_argument("--out", type=str, default="artifacts")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", action="store_true", help="Unfreeze encoder last layers (optional)")
    args = ap.parse_args()
    train(args)
