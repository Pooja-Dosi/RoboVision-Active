# eval_bc_viewer.py
import argparse, os, time, pathlib, collections
import numpy as np
from PIL import Image

import torch, torch.nn as nn
from torchvision import models, transforms

import mujoco, mujoco.viewer

# --- import helpers & constants from your run.py ---
import run
from run import (
    ids_map, find_freejoint_qadr, maintain_ball_factory,
    build_eye_ids, make_maintain_eye_tracking, set_ctrl, step_rt,
    READY, GRIP_OPEN, GRASP_DIST_OK, solve_ik,
    BALL_R, BUCKET_INNER_H, TABLE_ZTOP
)

# --------------------------- BC model ---------------------------
class BCHead(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )
    def forward(self, x): return self.net(x)

def make_model():
    enc = models.resnet18(weights=None)  # no download needed
    feat_dim = enc.fc.in_features
    enc.fc = nn.Identity()
    head = BCHead(in_dim=feat_dim, hidden=512, out_dim=8)
    return enc, head

def load_bc(ckpt_path, device):
    enc, head = make_model()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # PyTorch 2.6
    enc.load_state_dict(ckpt["enc"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    enc.eval().to(device); head.eval().to(device)

    mean = torch.tensor(ckpt["mean"], device=device, dtype=torch.float32)
    std  = torch.tensor(ckpt["std"],  device=device, dtype=torch.float32) + 1e-8
    img_size = int(ckpt.get("img_size", 112))
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # inputs used ImageNet norm during training
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return enc, head, mean, std, tf, img_size

# --------------------------- utils ---------------------------
def clamp(v, lo, hi): return np.maximum(lo, np.minimum(hi, v))
def yaw_toward(src_xy, tgt_xy): return float(np.arctan2(tgt_xy[1]-src_xy[1], tgt_xy[0]-src_xy[0]))
def norm_xy(a, b): return float(np.linalg.norm(a[:2] - b[:2]))

def goals_before_grasp(data, ids, ball_qadr):
    ball = data.qpos[ball_qadr:ball_qadr+3].copy()
    hover     = ball + np.array([0.0, 0.0, 0.24])
    pre_grasp = ball + np.array([0.0, 0.0, BALL_R + 0.020])
    grasp_pos = ball + np.array([0.0, 0.0, BALL_R + 0.004])
    return hover, pre_grasp, grasp_pos, ball

def goals_after_grasp(data, ids):
    bucket = data.xpos[ids["bucket_bid"]].copy()
    drop_z        = bucket[2] + BUCKET_INNER_H + 0.02
    hover_bucket  = np.array([bucket[0], bucket[1], max(drop_z + 0.12, TABLE_ZTOP + 0.24)])
    inside_bucket = np.array([bucket[0], bucket[1], drop_z])
    return hover_bucket, inside_bucket, bucket

# --------------------------- success metric helpers ---------------------------
class SuccessTracker:
    def __init__(self, dt, vel_alpha=0.6):
        self.dt = float(dt)
        self.prev = None
        self.v = 0.0
        self.vel_alpha = vel_alpha
        self.pos_hist = []   # [N,3]
        self.vel_hist = []   # [N]

    def push(self, p):
        p = np.asarray(p, dtype=float)
        if self.prev is None:
            self.v = 0.0
        else:
            v_inst = np.linalg.norm((p - self.prev) / max(self.dt, 1e-6))
            self.v = self.vel_alpha * self.v + (1 - self.vel_alpha) * v_inst
        self.prev = p
        self.pos_hist.append(p)
        self.vel_hist.append(self.v)

    def success(self, bucket_xy, rim_z, r, settle_steps, vel_thresh):
        if len(self.pos_hist) < settle_steps:
            return False
        pos = np.asarray(self.pos_hist[-settle_steps:])      # [S,3]
        spd = np.asarray(self.vel_hist[-settle_steps:])      # [S]
        dxy = np.linalg.norm(pos[:, :2] - np.asarray(bucket_xy[:2]), axis=1)
        below = pos[:, 2] < rim_z
        slow  = spd < vel_thresh
        ok = (dxy < r) & below & slow
        return bool(np.all(ok))

# --------------------------- main ---------------------------
def main(args):
    # device
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    pin_ok = device.type in ["cuda", "mps"]

    enc, head, mean, std, tf, img_size = load_bc(args.ckpt, device)

    # resolve XML
    xml_path = os.path.abspath(args.model if args.model else run.MODEL_PATH)
    print(f"[MODEL] XML: {xml_path}")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)

    # load scene
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)
    ids   = ids_map(model)

    # success metric setup (bucket XY, rim height reference)
    dt = float(model.opt.timestep)
    succ = SuccessTracker(dt)
    bucket0 = data.xpos[ids["bucket_bid"]].copy()
    bucket_xy = bucket0[:2]
    rim_z = bucket0[2] + BUCKET_INNER_H + float(args.rim_offset)
    print(f"[SUCCESS-METRIC] r={args.success_r:.3f}m rim_z≈{rim_z:.3f} "
          f"settle={args.success_settle_steps} vel<{args.ball_vel_thresh:.3f} m/s")

    # sanity: eye / ball
    eye_camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    if eye_camid == -1: raise RuntimeError("Camera 'eye' not found in XML.")
    ball_qadr, ball_vadr = find_freejoint_qadr(model, "ball")
    if ball_qadr is None: raise RuntimeError("Body 'ball' must have <freejoint/>")

    # attach & eye tracking
    maintain, attach_ball, detach_ball, is_attached = maintain_ball_factory(
        model, data, ids["grip_site"], ball_qadr, ball_vadr
    )
    eye = build_eye_ids(model)
    maintain_eye_tracking = make_maintain_eye_tracking(model, data, ids, is_attached, eye)
    def maintain_all():
        maintain()
        if not args.no_eye_tracking: maintain_eye_tracking()

    # eye renderer (policy input) + optional debug frames
    r_eye = mujoco.Renderer(model, width=img_size, height=img_size)
    if args.save_eye_frames:
        out = pathlib.Path("snaps"); out.mkdir(parents=True, exist_ok=True)
        print("[SNAPS] saving eye frames to snaps/")
    frame_stride = max(1, int(args.eye_frame_stride))
    last_dump = 0

    # runtime state
    jrange = ids["jrange"]
    hold_steps = max(1, int(args.hold_steps))
    ema_pred   = float(args.ema)
    cmd_ema    = float(args.cmd_ema)
    dq_max     = float(args.dq_max)
    assist     = float(args.assist_scale)

    last_q = data.qpos[ids["qadr"]].copy().astype(np.float32)
    last_g = float(GRIP_OPEN)
    ema_q = ema_g = cmd_q = None
    dq_hist = collections.deque(maxlen=10)

    attach_ticks = 0
    reattach_cooldown = 0
    hold_open_left    = 0
    backoff_steps     = 0
    backoff_target    = None
    close_force = 0.003
    t_attach = None
    quit_at = time.time() + float(args.autoquit_sec) if args.autoquit_sec > 0 else None

    print(f"[BC] assist-scale={assist:.2f}  hold={hold_steps}  ema={ema_pred}  cmd_ema={cmd_ema}")

    with mujoco.viewer.launch_passive(model, data) as v:
        # human camera (policy always uses 'eye')
        if args.viewer_freecam:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.azimuth = float(args.freecam_az)
            v.cam.elevation = float(args.freecam_el)
            v.cam.distance = float(args.freecam_dist)
            v.cam.lookat[:] = [0.0, 0.0, 0.90]
        else:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = eye_camid

        # settle
        for _ in range(int(1.0 / model.opt.timestep)):
            if not v.is_running(): break
            maintain_all(); set_ctrl(data, ids["actids"], READY, GRIP_OPEN); step_rt(model, data, v)

        step_i = 0
        while v.is_running():
            if quit_at and time.time() >= quit_at: break
            maintain_all()

            # push ball position into success tracker each sim step
            succ.push(data.qpos[ball_qadr:ball_qadr+3])

            # optional eye frame dump
            if args.save_eye_frames and (step_i - last_dump) >= frame_stride:
                r_eye.update_scene(data, camera=eye_camid)
                img = r_eye.render()
                if img.dtype != np.uint8: img = np.clip(img*255.0,0,255).astype(np.uint8)
                Image.fromarray(img).save(f"snaps/eye_{step_i:06d}.png")
                last_dump = step_i

            # countdowns
            if reattach_cooldown > 0: reattach_cooldown -= 1
            if hold_open_left > 0:    hold_open_left    -= 1
            if backoff_steps > 0:     backoff_steps     -= 1

            if step_i % hold_steps == 0:
                mujoco.mj_forward(model, data)

                # ------------ policy inference (eye image) ------------
                r_eye.update_scene(data, camera=eye_camid)
                img = r_eye.render()
                if img.dtype != np.uint8:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                with torch.no_grad():
                    x = tf(Image.fromarray(img).convert("RGB")).unsqueeze(0).to(device, non_blocking=pin_ok)
                    f = enc(x)
                    pred_norm = head(f).squeeze(0)
                    pred = pred_norm * std + mean  # de-normalize targets

                pred_np = pred.detach().cpu().numpy().astype(np.float32)
                q_bc = clamp(pred_np[:7], jrange[:,0], jrange[:,1])
                g_bc = float(np.clip(pred_np[7], 0.0, 0.04))

                if 0.0 < ema_pred < 1.0:
                    if ema_q is None: ema_q, ema_g = q_bc.copy(), g_bc
                    else:
                        a = ema_pred; ema_q = a*ema_q + (1-a)*q_bc; ema_g = a*ema_g + (1-a)*g_bc
                    q_bc, g_bc = ema_q, float(ema_g)
                # ------------------------------------------------------

                # scene geometry
                q_cur   = data.qpos[ids["qadr"]].copy()
                base_xy = data.xpos[ids["link0_bid"]][:2].copy()
                ee      = data.site_xpos[ids["grip_site"]].copy()

                hover_b, inside_b, bucket = goals_after_grasp(data, ids)
                hover, pre_grasp, grasp_pos, ball = goals_before_grasp(data, ids, ball_qadr)

                # ----------- compute IK target for assist -----------
                if backoff_steps > 0 and backoff_target is not None:
                    target = backoff_target.copy()
                    yaw = yaw_toward(base_xy, target[:2])
                    yaw_pose = READY.copy(); yaw_pose[0] = yaw
                    q_goal = solve_ik(model, data, ids, ids["grip_site"], q_cur, yaw_pose, target)
                    speed = float(args.speed_far) * 0.9
                elif not is_attached():
                    d = float(np.linalg.norm(ee - ball))
                    if d > 0.18:
                        target, speed = pre_grasp, float(args.speed_far)
                    elif d > 0.10:
                        target, speed = pre_grasp, float(args.speed_far) * 0.8
                    else:
                        target, speed = grasp_pos, float(args.speed_near)
                        g_bc = min(g_bc, close_force)  # encourage closing near ball
                    yaw = yaw_toward(base_xy, ball[:2])
                    yaw_pose = READY.copy(); yaw_pose[0] = yaw
                    q_goal = solve_ik(model, data, ids, ids["grip_site"], q_cur, yaw_pose, target)
                else:
                    xy_err = norm_xy(ee, bucket)
                    if xy_err > float(args.xy_align_thresh):
                        target, speed = hover_b.copy(), float(args.speed_far)
                    else:
                        next_z = max(inside_b[2], ee[2] - float(args.z_step))
                        target = np.array([bucket[0], bucket[1], next_z], dtype=float)
                        speed = float(args.speed_near)
                    yaw = yaw_toward(base_xy, bucket[:2])
                    yaw_pose = READY.copy(); yaw_pose[0] = yaw
                    q_goal = solve_ik(model, data, ids, ids["grip_site"], q_cur, yaw_pose, target)
                # -----------------------------------------------------

                # ------------ blend BC and assist goal ------------
                alpha = np.clip(assist, 0.0, 1.0)  # 0=pure BC, 1=pure IK
                q_mix = (1.0 - alpha) * q_bc + alpha * q_goal
                # coax base yaw toward yaw (helps convergence)
                q_mix[0] = (1.0 - 0.4*assist) * q_mix[0] + (0.4*assist) * yaw
                # ---------------------------------------------------

                # watchdog + speed limiting + command smoothing
                dq_hist.append(float(np.linalg.norm(q_mix - q_cur)))
                dq = clamp((q_mix - q_cur) * speed, -dq_max, dq_max)
                q_target = q_cur + dq
                cmd_q = q_target if cmd_q is None else (cmd_ema*cmd_q + (1-cmd_ema)*q_target)

                # -------------- attach / release logic --------------
                dist_ball = float(np.linalg.norm(ee - data.qpos[ball_qadr:ball_qadr+3]))
                ATTACH_RANGE = max(float(GRASP_DIST_OK), float(args.attach_range))

                # keep open during lockout to avoid re-grab
                if hold_open_left > 0:
                    g_bc = max(g_bc, float(args.release_open_value))

                if reattach_cooldown > 0:
                    attach_ticks = 0  # hard block attach
                else:
                    if not is_attached():
                        if (g_bc < 0.008) and (dist_ball <= ATTACH_RANGE):
                            attach_ticks += 1
                        else:
                            attach_ticks = 0
                        if attach_ticks >= int(args.attach_confirm):
                            print(f"[GRASP] attaching at dist={dist_ball:.3f} (thr={ATTACH_RANGE:.3f})")
                            attach_ball(); t_attach = time.time()
                            g_bc = min(g_bc, close_force)

                if is_attached():
                    hover_b2, inside_b2, bucket2 = goals_after_grasp(data, ids)
                    xy_close = norm_xy(ee, bucket2) < float(args.open_near_bucket_dist)
                    z_low    = ee[2] <= inside_b2[2] + float(args.release_z_margin)
                    force_due = (args.force_release_after > 0 and t_attach is not None
                                 and (time.time() - t_attach) >= float(args.force_release_after))
                    if (xy_close and z_low) or force_due:
                        g_bc = max(g_bc, float(args.release_open_value))
                        if is_attached() and g_bc >= float(args.release_open_value):
                            print("[DROP] releasing near bucket")
                            detach_ball()
                            reattach_cooldown = int(args.reattach_cooldown)
                            hold_open_left    = int(args.hold_open_steps)
                            backoff_steps     = int(args.post_drop_steps)
                            backoff_target    = hover_b2.copy(); backoff_target[2] += float(args.post_drop_up)
                # -----------------------------------------------------

                last_q, last_g = cmd_q.astype(np.float32), float(np.clip(g_bc, 0.0, 0.04))

            set_ctrl(data, ids["actids"], last_q, last_g)
            step_rt(model, data, v)
            step_i += 1

    # --- success verdict after the viewer closes ---
    ok = succ.success(bucket_xy=bucket_xy,
                      rim_z=rim_z,
                      r=float(args.success_r),
                      settle_steps=int(args.success_settle_steps),
                      vel_thresh=float(args.ball_vel_thresh))
    print(f"[RESULT] {'SUCCESS' if ok else 'FAIL'}")

    try: r_eye.close()
    except Exception: pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/bc_resnet18.pth")
    ap.add_argument("--model", type=str, default=None, help="Override XML path (default: run.MODEL_PATH)")

    # BC cadence & filters
    ap.add_argument("--hold-steps", type=int, default=5)
    ap.add_argument("--ema", type=float, default=0.8, help="EMA on BC predictions (0..1)")
    ap.add_argument("--cmd-ema", type=float, default=0.85, help="EMA on commanded q (0..1)")

    # blending & speeds
    ap.add_argument("--assist-scale", type=float, default=0.4, help="0=pure BC, 1=pure IK")
    ap.add_argument("--dq-max", type=float, default=0.06)
    ap.add_argument("--speed-far", type=float, default=0.26)
    ap.add_argument("--speed-near", type=float, default=0.14)

    # grasp / release geometry
    ap.add_argument("--attach-confirm", type=int, default=3)
    ap.add_argument("--attach-range", type=float, default=0.07)
    ap.add_argument("--open-near-bucket-dist", type=float, default=0.20)
    ap.add_argument("--xy-align-thresh", type=float, default=0.06)
    ap.add_argument("--z-step", type=float, default=0.014)
    ap.add_argument("--release-z-margin", type=float, default=0.030)
    ap.add_argument("--release-open-value", type=float, default=0.040)
    ap.add_argument("--force-release-after", type=float, default=12.0)

    # anti re-grab & backoff
    ap.add_argument("--reattach-cooldown", type=int, default=600)
    ap.add_argument("--hold-open-steps", type=int, default=600)
    ap.add_argument("--post-drop-steps", type=int, default=260)
    ap.add_argument("--post-drop-up", type=float, default=0.10)

    # viewer / snapshots
    ap.add_argument("--viewer-freecam", action="store_true")
    ap.add_argument("--freecam-az", type=float, default=135.0)
    ap.add_argument("--freecam-el", type=float, default=-25.0)
    ap.add_argument("--freecam-dist", type=float, default=2.3)
    ap.add_argument("--autoquit-sec", type=float, default=0.0)
    ap.add_argument("--no-eye-tracking", action="store_true")
    ap.add_argument("--save-eye-frames", action="store_true")
    ap.add_argument("--eye-frame-stride", type=int, default=25)

    # --- success metric knobs ---
    ap.add_argument("--success-r", type=float, default=0.08, help="Horizontal radius for success check (m).")
    ap.add_argument("--rim-offset", type=float, default=0.0, help="Additional offset added to (bucket_z + BUCKET_INNER_H).")
    ap.add_argument("--success-settle-steps", type=int, default=120, help="Consecutive steps ball must stay in bucket.")
    ap.add_argument("--ball-vel-thresh", type=float, default=0.06, help="Speed (m/s) threshold during settle window.")

    args = ap.parse_args()
    main(args)
