# eval_bc_batch.py — BC batch eval with phased IK assist and proper inside-bucket drop.
import argparse, os
import numpy as np
from PIL import Image

import torch, torch.nn as nn
from torchvision import models, transforms
import mujoco

import run
from run import (
    ids_map, find_freejoint_qadr, maintain_ball_factory, set_ctrl,
    build_eye_ids, make_maintain_eye_tracking,
    READY, GRIP_OPEN, GRASP_DIST_OK,
    BALL_R, BUCKET_INNER_H, TABLE_ZTOP,
    _rand_xy, _set_body_xy, _compute_success,
    _BALL_X_RANGE, _BUCKET_X_RANGE, _Y_RANGE, _MIN_SEP,
    solve_ik,
)

# ---------- BC head (matches your training: 512 -> 256 -> 8) ----------
class BCHead(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 256),    nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )
    def forward(self, x): return self.net(x)

def make_model():
    enc = models.resnet18(weights=None)   # offline-safe
    feat_dim = enc.fc.in_features
    enc.fc = nn.Identity()
    head = BCHead(in_dim=feat_dim, hidden=512, out_dim=8)
    return enc, head

def load_bc(ckpt_path, device):
    enc, head = make_model()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc.load_state_dict(ckpt["enc"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    enc.eval().to(device); head.eval().to(device)
    mean = torch.tensor(ckpt["mean"], device=device, dtype=torch.float32)
    std  = torch.tensor(ckpt["std"],  device=device, dtype=torch.float32) + 1e-8
    img_size = int(ckpt.get("img_size", 112))
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return enc, head, mean, std, tf, img_size

# ---------- small helpers ----------
def clamp(v, lo, hi): return np.maximum(lo, np.minimum(hi, v))
def yaw_toward(src_xy, tgt_xy): return float(np.arctan2(tgt_xy[1]-src_xy[1], tgt_xy[0]-src_xy[0]))

def goals_before_grasp(data, ids, ball_qadr):
    ball = data.qpos[ball_qadr:ball_qadr+3].copy()
    hover     = ball + np.array([0.0, 0.0, 0.24])
    pre_grasp = ball + np.array([0.0, 0.0, BALL_R + 0.020])
    grasp_pos = ball + np.array([0.0, 0.0, BALL_R + 0.004])
    return hover, pre_grasp, grasp_pos, ball

def goals_after_grasp(data, ids):
    bucket = data.xpos[ids["bucket_bid"]].copy()
    # Put the "inside" target BELOW the rim so release happens inside, not above it.
    # Rim = bucket.z + BUCKET_INNER_H. Go a little below: rim - 0.5*BALL_R.
    drop_z_inside = bucket[2] + BUCKET_INNER_H - 0.5 * BALL_R
    hover_bucket  = np.array([bucket[0], bucket[1], max(drop_z_inside + 0.12, TABLE_ZTOP + 0.24)])
    inside_bucket = np.array([bucket[0], bucket[1], drop_z_inside])
    return hover_bucket, inside_bucket, bucket

def warmup_to_ready_and_yaw(model, data, ids, ball_qadr, seconds_ready=1.0, seconds_yaw=1.0):
    dt = model.opt.timestep
    # to READY
    q_now = data.qpos[ids["qadr"]].copy()
    steps = max(1, int(seconds_ready / dt))
    for k in range(steps):
        s = (k+1)/steps
        q = (1-s)*q_now + s*READY
        set_ctrl(data, ids["actids"], q, GRIP_OPEN)
        mujoco.mj_step(model, data)
    # yaw toward ball
    mujoco.mj_forward(model, data)
    base_xy = data.xpos[ids["link0_bid"]][:2].copy()
    ball_xy = data.xpos[ids["ball_bid"]][:2].copy()
    yaw = yaw_toward(base_xy, ball_xy)
    yaw_pose = READY.copy(); yaw_pose[0] = yaw
    q_now = data.qpos[ids["qadr"]].copy()
    steps = max(1, int(seconds_yaw / dt))
    for k in range(steps):
        s = (k+1)/steps
        q = (1-s)*q_now + s*yaw_pose
        set_ctrl(data, ids["actids"], q, GRIP_OPEN)
        mujoco.mj_step(model, data)

# ---------- one episode ----------
def run_one_episode(
    device, enc, head, mean, std, tf, img_size, xml_path, seed,
    randomize=True, assist_scale=0.0, max_steps=1200,
    hold_steps=5, ema=0.8, cmd_ema=0.9, dq_max=0.05, eye_settle=3,
    attach_confirm=2, attach_range=0.085, ignore_grip_for_attach=False,
    open_near_bucket_dist=0.10, release_z_margin=0.005, release_open_value=0.040,
    reattach_cooldown_steps=9999, post_drop_steps=260, post_drop_up=0.10,
    settle_sec=0.6,
    save_frames=0, save_dir="dbg_frames", log_k=0
):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)
    ids   = ids_map(model)

    rng = np.random.default_rng(seed)
    ball_qadr, ball_vadr = find_freejoint_qadr(model, "ball")
    bucket_qadr, _       = find_freejoint_qadr(model, "bucket")
    assert ball_qadr is not None and bucket_qadr is not None

    if randomize:
        while True:
            bx, by = _rand_xy(rng, _BALL_X_RANGE, _Y_RANGE)
            kx, ky = _rand_xy(rng, _BUCKET_X_RANGE, _Y_RANGE)
            if np.hypot(bx-kx, by-ky) >= _MIN_SEP: break
        _set_body_xy(data, ball_qadr, bx, by)
        _set_body_xy(data, bucket_qadr, kx, ky)
        mujoco.mj_forward(model, data)

    maintain, attach_ball, detach_ball, is_attached = maintain_ball_factory(
        model, data, ids["grip_site"], ball_qadr, ball_vadr
    )

    eye_camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    r_eye = mujoco.Renderer(model, width=img_size, height=img_size)
    eye = build_eye_ids(model)
    maintain_eye_tracking = make_maintain_eye_tracking(model, data, ids, is_attached, eye)

    warmup_to_ready_and_yaw(model, data, ids, ball_qadr, seconds_ready=1.0, seconds_yaw=1.0)
    for _ in range(int(0.2 / model.opt.timestep)):
        maintain(); maintain_eye_tracking()
        set_ctrl(data, ids["actids"], READY, GRIP_OPEN)
        mujoco.mj_step(model, data)

    jrange = ids["jrange"]
    last_q = data.qpos[ids["qadr"]].copy().astype(np.float32)
    last_g = float(GRIP_OPEN)
    ema_q = ema_g = cmd_q = None
    reattach_cd = 0
    hold_open_left = 0
    attach_ticks = 0
    saved = 0
    call_idx = 0
    phase = 0           # 0=hover, 1=pre_grasp, 2=grasp, 3=drop
    just_dropped = False

    def eye_pred():
        nonlocal saved, last_q, last_g, call_idx
        # settle pan/tilt BEFORE render
        for _ in range(max(0, int(eye_settle))):
            maintain_eye_tracking()
            set_ctrl(data, ids["actids"], last_q, last_g)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        r_eye.update_scene(data, camera=eye_camid)
        img = r_eye.render()
        if img.dtype != np.uint8: img = np.clip(img*255.0,0,255).astype(np.uint8)
        if save_frames>0 and saved<save_frames:
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(img).save(os.path.join(save_dir, f"ep0_frame_{saved:02d}.png"))
            saved += 1

        x = tf(Image.fromarray(img).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            f = enc(x); y = head(f).squeeze(0)
            y = y*std + mean
        y = y.detach().cpu().numpy().astype(np.float32)
        q_bc = clamp(y[:7], jrange[:,0], jrange[:,1])
        g_bc = float(np.clip(y[7], 0.0, 0.04))

        call_idx += 1
        if call_idx <= log_k:
            ee   = data.site_xpos[ids["grip_site"]].copy()
            ball = data.qpos[ball_qadr:ball_qadr+3].copy()
            dist = float(np.linalg.norm(ee - ball))
            dxy  = float(np.linalg.norm(ee[:2] - ball[:2]))
            dnorm = float(np.linalg.norm(q_bc - data.qpos[ids["qadr"]]))
            print(f"[POLICY] k={call_idx:02d} g_pred={g_bc:.4f} dist_ball={dist:.3f} dxy={dxy:.3f} |dq_bc|={dnorm:.3f}")
        return q_bc, g_bc

    steps = 0
    attached_ever = False
    dropped_ever  = False

    while steps < max_steps:
        maintain(); maintain_eye_tracking()

        if steps % max(1, int(hold_steps)) == 0:
            mujoco.mj_forward(model, data)
            q_bc, g_bc = eye_pred()
            if 0.0 < ema < 1.0:
                if ema_q is None: ema_q, ema_g = q_bc.copy(), g_bc
                else:
                    a = float(ema); ema_q = a*ema_q + (1-a)*q_bc; ema_g = a*ema_g + (1-a)*g_bc
                q_bc, g_bc = ema_q, float(ema_g)

            q_cur = data.qpos[ids["qadr"]].copy()
            ee     = data.site_xpos[ids["grip_site"]].copy()
            hover_b, inside_b, bucket = goals_after_grasp(data, ids)
            hover, pre_grasp, grasp_pos, ball = goals_before_grasp(data, ids, ball_qadr)

            # ----- phased IK assist -----
            assist = float(max(0.0, min(1.0, assist_scale)))
            if assist > 0.0:
                base_xy = data.xpos[ids["link0_bid"]][:2].copy()
                tgt_xy  = (bucket[:2] if is_attached() else ball[:2])
                yaw     = yaw_toward(base_xy, tgt_xy)
                yaw_pose = READY.copy(); yaw_pose[0] = yaw

                dxy = float(np.linalg.norm(ee[:2] - ball[:2]))
                dz_pre   = float(abs(ee[2] - pre_grasp[2]))

                if not is_attached():
                    if phase == 0:
                        target = hover; mode = "hover"
                        if dxy < 0.035: phase = 1
                    elif phase == 1:
                        target = pre_grasp; mode = "pre_grasp"
                        if (dz_pre < 0.012 and dxy < 0.025): phase = 2
                    else:
                        target = grasp_pos; mode = "grasp"
                else:
                    # after attach we go to the inside-bucket z (below rim)
                    if hold_open_left > 0 or just_dropped:
                        target = inside_b + np.array([0.0, 0.0, post_drop_up])
                        mode = "post_drop"
                    else:
                        target = inside_b
                        mode = "drop"
                    phase = 3

                q_ik = solve_ik(model, data, ids, ids["grip_site"],
                                q_seed=q_cur, q_pref=yaw_pose, target_pos=target)
                if call_idx <= log_k:
                    dnorm_ik = float(np.linalg.norm(q_ik - q_cur))
                    print(f"[ASSIST] phase={phase} mode={mode:<10} dxy={dxy:.3f} dz_pre={dz_pre:.3f} |dq_ik|={dnorm_ik:.3f} blend={assist:.2f}")
                q_mix = (1.0 - assist) * q_bc + assist * q_ik
            else:
                q_mix = q_bc

            # per-step change + smoothing
            speed = 0.30 if not is_attached() else 0.22
            dq = np.clip((q_mix - q_cur)*speed, -dq_max, dq_max)
            q_cmd = q_cur + dq
            cmd_q = q_cmd if 'cmd_q' not in locals() or cmd_q is None else (cmd_ema*cmd_q + (1-cmd_ema)*q_cmd)

            # attach / drop rules
            ee     = data.site_xpos[ids["grip_site"]].copy()
            ballp  = data.qpos[ball_qadr:ball_qadr+3].copy()
            ATTACH_RANGE = max(float(GRASP_DIST_OK), float(attach_range))
            dist_ball = float(np.linalg.norm(ee - ballp))

            if hold_open_left > 0:
                g_bc = max(g_bc, float(release_open_value))

            want_attach = (g_bc < 0.008 and dist_ball <= ATTACH_RANGE)
            if ignore_grip_for_attach:
                want_attach = (dist_ball <= ATTACH_RANGE)

            if reattach_cd == 0 and not is_attached():
                attach_ticks = attach_ticks + 1 if want_attach else 0
                if attach_ticks >= int(attach_confirm):
                    print(f"[GRASP] attaching at dist={dist_ball:.3f} m (thr={ATTACH_RANGE:.3f}) (g={g_bc:.4f})")
                    attach_ball(); attached_ever = True; phase = 3

            if is_attached():
                hover_b, inside_b, bucket = goals_after_grasp(data, ids)
                xy_close = np.linalg.norm(ee[:2]-bucket[:2]) < open_near_bucket_dist
                z_low    = ee[2] <= (inside_b[2] + release_z_margin)
                if xy_close and z_low:
                    print(f"[DROP] releasing near bucket (g={g_bc:.4f})")
                    detach_ball()
                    dropped_ever = True
                    reattach_cd    = int(reattach_cooldown_steps)
                    hold_open_left = int(post_drop_steps)
                    just_dropped   = True

            last_q = cmd_q.astype(np.float32)
            last_g = float(np.clip(g_bc, 0.0, 0.04))
        else:
            just_dropped = False  # only true for the step right after we drop

        set_ctrl(data, ids["actids"], last_q, last_g)
        mujoco.mj_step(model, data)

        if reattach_cd > 0: reattach_cd -= 1
        if hold_open_left > 0: hold_open_left -= 1
        steps += 1

    # let the ball settle before success check
    settle_steps = int(float(settle_sec) / model.opt.timestep)
    for _ in range(settle_steps):
        maintain(); maintain_eye_tracking()
        set_ctrl(data, ids["actids"], last_q, last_g)
        mujoco.mj_step(model, data)

    ok = _compute_success(model, data, ids, ball_vadr, goal_site="bucket_goal", xy_tol=0.045)
    try: r_eye.close()
    except Exception: pass
    return bool(ok), attached_ever, dropped_ever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/bc_resnet18.pth")
    ap.add_argument("--model", type=str, default=run.MODEL_PATH)
    ap.add_argument("-n","--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--assist-scale", type=float, default=0.0)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--save-frames", type=int, default=0)

    # controller / diagnostics
    ap.add_argument("--hold-steps", type=int, default=5)
    ap.add_argument("--dq-max", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.8)
    ap.add_argument("--cmd-ema", type=float, default=0.9)
    ap.add_argument("--eye-settle", type=int, default=3)
    ap.add_argument("--ignore-grip-for-attach", action="store_true")
    ap.add_argument("--attach-range", type=float, default=0.085)
    ap.add_argument("--log-k", type=int, default=0)

    # NEW: drop/settle controls
    ap.add_argument("--open-near-bucket-dist", type=float, default=0.10)
    ap.add_argument("--release-z-margin", type=float, default=0.005)
    ap.add_argument("--release-open-value", type=float, default=0.040)
    ap.add_argument("--reattach-cooldown", type=int, default=9999)
    ap.add_argument("--post-drop-steps", type=int, default=260)
    ap.add_argument("--post-drop-up", type=float, default=0.10)
    ap.add_argument("--settle-sec", type=float, default=0.6)

    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    enc, head, mean, std, tf, img_size = load_bc(args.ckpt, device)
    xml_path = os.path.abspath(args.model)

    ok = 0
    for i in range(args.episodes):
        s = (args.seed or 0) + i
        res, did_attach, did_drop = run_one_episode(
            device, enc, head, mean, std, tf, img_size, xml_path, s,
            randomize=not args.no_randomize, assist_scale=args.assist_scale,
            max_steps=1200,
            hold_steps=args.hold_steps, ema=args.ema, cmd_ema=args.cmd_ema, dq_max=args.dq_max,
            eye_settle=args.eye_settle,
            attach_range=args.attach_range,
            ignore_grip_for_attach=args.ignore_grip_for_attach,
            open_near_bucket_dist=args.open_near_bucket_dist,
            release_z_margin=args.release_z_margin,
            release_open_value=args.release_open_value,
            reattach_cooldown_steps=args.reattach_cooldown,
            post_drop_steps=args.post_drop_steps,
            post_drop_up=args.post_drop_up,
            settle_sec=args.settle_sec,
            save_frames=args.save_frames if i==0 else 0, save_dir="dbg_frames",
            log_k=args.log_k if i==0 else 0
        )
        tag = "SUCCESS" if res else "FAIL"
        print(f"[EVAL] ep {i:03d} -> {tag} (attach={did_attach}, drop={did_drop})")
        ok += int(res)
    print(f"[RESULT] {ok}/{args.episodes} success")

if __name__ == "__main__":
    main()
