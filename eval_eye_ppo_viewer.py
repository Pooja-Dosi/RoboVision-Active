# eval_eye_ppo_viewer.py
import os, time, math, argparse
import numpy as np
from PIL import Image

import mujoco, mujoco.viewer
from stable_baselines3 import PPO

# bring in your helpers
import run
from run import (
    ids_map, find_freejoint_qadr, set_ctrl, step_rt,
    READY, GRIP_OPEN, build_eye_ids, aim_eye_at_point
)

# ---------- image / vision helpers ----------
def mk_renderer(model, w, h):
    r = mujoco.Renderer(model, width=int(w), height=int(h))
    return r

def render_eye(model, data, camid, r):
    r.update_scene(data, camera=camid)
    img = r.render()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img  # H x W x 3 (RGB uint8)

def detect_red_ball_uv(img, r_min=90, margin=35):
    """
    Very simple red mask: r > r_min and r - max(g,b) > margin.
    Returns (u,v, visible_flag, z_frac) where:
      u,v: centroid in pixel coords (origin = top-left), NaN if not visible
      z_frac: sqrt(area_frac) ~= "apparent size" proxy (0..~0.4 typical)
    """
    # img: H x W x 3 uint8
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    mask = (r > r_min) & ((r - np.maximum(g, b)) > margin)

    area = float(mask.sum())
    H, W = img.shape[:2]
    if area <= 4.0:  # too small / not found
        return float("nan"), float("nan"), 0, 0.0

    ys, xs = np.nonzero(mask)
    u = float(xs.mean())  # x = column index
    v = float(ys.mean())  # y = row index
    z_frac = math.sqrt(area / float(H * W))
    return u, v, 1, z_frac

def safe_dist_to_center(u, v, W, H):
    cx, cy = W * 0.5, H * 0.5
    finite = (np.isfinite(u) and np.isfinite(v))
    if not finite:
        return float("inf"), cx, cy
    return float(math.hypot(u - cx, v - cy)), cx, cy

# ---------- action mapping ----------
def discrete_action_to_delta(action, step_rad):
    """
    5 actions: 0=NOOP, 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN
    Returns dpan, dtilt (radians).
    """
    if action == 0:
        return 0.0, 0.0
    elif action == 1:   # pan left
        return +step_rad, 0.0
    elif action == 2:   # pan right
        return -step_rad, 0.0
    elif action == 3:   # tilt up
        return 0.0, +step_rad
    elif action == 4:   # tilt down
        return 0.0, -step_rad
    else:
        return 0.0, 0.0

def clamp(val, lo, hi):
    return float(max(lo, min(hi, val)))

def get_eye_joint_ranges(model):
    pan_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "eye_pan")
    tilt_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "eye_tilt")
    if pan_jid == -1 or tilt_jid == -1:
        raise RuntimeError("eye_pan/eye_tilt joints not found in XML.")
    pan_rng  = tuple(model.jnt_range[pan_jid])
    tilt_rng = tuple(model.jnt_range[tilt_jid])
    pan_adr  = model.jnt_qposadr[pan_jid]
    tilt_adr = model.jnt_qposadr[tilt_jid]
    return pan_adr, tilt_adr, pan_rng, tilt_rng

# ---------- main ----------
def main(args):
    # load model
    xml_path = os.path.abspath(run.MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)
    ids   = ids_map(model)

    # eye
    eye_camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    if eye_camid == -1:
        raise RuntimeError("Camera 'eye' not found in model.")
    pan_adr, tilt_adr, pan_rng, tilt_rng = get_eye_joint_ranges(model)

    # ensure ball has freejoint (not strictly needed here, but we read pos sometimes)
    ball_qadr, _ball_vadr = find_freejoint_qadr(model, "ball")
    if ball_qadr is None:
        raise RuntimeError("Body 'ball' must have <freejoint/>")

    # policy
    ckpt = args.ckpt
    if not os.path.exists(ckpt) and os.path.exists(ckpt + ".zip"):
        ckpt = ckpt + ".zip"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    model_rl = PPO.load(ckpt, device="auto")

    # renderer for policy obs
    obs_size = int(args.obs_size)
    r_eye = mk_renderer(model, obs_size, obs_size)

    # viewer
    total_reward = 0.0
    inv_count = 0

    # random occasional "snap to target" (evaluation convenience)
    rng = np.random.default_rng(args.seed if args.seed is not None else 0)

    # initial settle + choose human camera
    with mujoco.viewer.launch_passive(model, data) as v:
        # freecam first (optional)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        v.cam.fixedcamid = eye_camid

        # hold arm/gripper steady in READY
        def hold_arm():
            set_ctrl(data, ids["actids"], READY, GRIP_OPEN)

        # decision loop (viewer)
        step_i = 0
        while v.is_running() and step_i < int(args.max_steps):
            # ---- observation for policy (uint8 RGB, HxWx3) ----
            img = render_eye(model, data, eye_camid, r_eye)
            obs = img  # SB3 CNN policies expect uint8 [0..255] NHWC when env is set like that

            # ---- RL policy action ----
            action, _ = model_rl.predict(obs, deterministic=True)
            if args.discrete:
                dpan, dtilt = discrete_action_to_delta(int(action), math.radians(args.step_deg))
            else:
                # continuous not used in your runs, but keep safe mapping: [-1,1] -> step
                ax, ay = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
                step = math.radians(args.step_deg)
                dpan, dtilt = float(ax) * step, float(ay) * step

            # sign flips (viewer-only)
            if args.invert_pan:  dpan  = -dpan
            if args.invert_tilt: dtilt = -dtilt

            # rare reset: point eye toward true ball position (evaluation helper)
            if rng.uniform() < float(args.reset_aim_prob):
                ball_p = data.qpos[ball_qadr:ball_qadr+3].copy()
                eye_ids = build_eye_ids(model)
                aim_eye_at_point(model, data, eye_ids, target_xyz=ball_p)

            # ---- apply action over frame_skip steps ----
            for _ in range(int(args.frame_skip)):
                # clamp joint targets
                pan_q = clamp(float(data.qpos[pan_adr]  + dpan),  pan_rng[0],  pan_rng[1])
                til_q = clamp(float(data.qpos[tilt_adr] + dtilt), tilt_rng[0], tilt_rng[1])
                data.qpos[pan_adr]  = pan_q
                data.qpos[tilt_adr] = til_q

                # step + hold arm
                hold_arm()
                step_rt(model, data, v)

            # ---- debug vision / reward ----
            img_dbg = render_eye(model, data, eye_camid, r_eye)
            u, vpx, vis, zf = detect_red_ball_uv(img_dbg, r_min=90, margin=35)

            # (viewer reward) closer-to-center and visible → larger reward
            dist, cx, cy = safe_dist_to_center(u, vpx, obs_size, obs_size)
            r_center = 0.0
            if np.isfinite(dist):
                # normalize by half-diagonal to keep in [0,~1]
                halfdiag = math.hypot(cx, cy)
                r_center = max(0.0, 1.0 - (dist / halfdiag))
            r_size = zf  # encourages bringing the ball closer/bigger
            r_step = (0.65 * r_center + 0.35 * r_size) * (1.0 if vis else 0.5)
            total_reward += float(r_step)

            # visibility streak (for scan recovery)
            if vis:
                inv_count = 0
            else:
                inv_count += 1
                # tiny pan wiggle to re-acquire after long invis streak
                if args.scan_recover > 0 and inv_count >= int(args.scan_recover):
                    wig = math.radians(4.0)
                    for sgn in (+1, -1, +1, -1):
                        data.qpos[pan_adr] = clamp(float(data.qpos[pan_adr] + sgn * wig), pan_rng[0], pan_rng[1])
                        hold_arm()
                        step_rt(model, data, v)
                    inv_count = 0

            # print debug every ~15 decisions (matches your logs style)
            if (step_i % 15) == 0:
                u_str = f"{u:.1f}" if np.isfinite(u) else "nan"
                v_str = f"{vpx:.1f}" if np.isfinite(vpx) else "nan"
                print(f"[dbg] step={step_i:06d} r={r_step:.3f} done=False u={u_str} v={v_str} zf={zf:.3f} visible={int(vis)}")

            # ---- success / timeout ----
            # success if visible and within radius_frac * obs_size of center
            thr = float(args.radius_frac) * float(obs_size)
            success = bool(vis and np.isfinite(dist) and (dist <= thr))

            if success:
                print(f"[episode 001] SUCCESS (visible={int(vis)}, u={u_str}, v={v_str}, dist={dist:.1f}, zf={zf:.3f})  pausing {args.pause_after_ep:.1f}s…")
                time.sleep(float(args.pause_after_ep))
                break

            step_i += 1

        if step_i >= int(args.max_steps):
            u_str = f"{u:.1f}" if np.isfinite(u) else "nan"
            v_str = f"{vpx:.1f}" if np.isfinite(vpx) else "nan"
            print(f"[episode 001] TIMEOUT (visible={int(vis)}, u={u_str}, v={v_str}, dist={'inf' if not np.isfinite(dist) else f'{dist:.1f}'})  pausing {args.pause_after_ep:.1f}s…")
            time.sleep(float(args.pause_after_ep))

    try:
        r_eye.close()
    except Exception:
        pass

    print(f"[viewer closed] total_reward={total_reward:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="eye_runs/ball_state/eye_ppo.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-size", type=int, default=128)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--step-deg", type=float, default=2.0)
    ap.add_argument("--reward-mode", type=str, choices=["state","vision"], default="vision")
    ap.add_argument("--discrete", action="store_true")
    ap.add_argument("--radius-frac", type=float, default=0.10, help="Success radius as fraction of obs_size.")
    ap.add_argument("--visible-window", type=int, default=12, help="(kept for compatibility; success is instant here)")
    ap.add_argument("--pause-after-ep", type=float, default=1.5)
    ap.add_argument("--no-autoreset", action="store_true")  # kept for CLI parity
    ap.add_argument("--reset-aim-prob", type=float, default=0.2, help="Chance per decision to snap aim toward true ball.")
    ap.add_argument("--debug-every", type=int, default=15)  # legacy; we print every 15 steps

    # NEW:
    ap.add_argument("--invert-tilt", action="store_true", help="Flip tilt direction at eval.")
    ap.add_argument("--invert-pan", action="store_true", help="Flip pan direction at eval.")
    ap.add_argument("--max-steps", type=int, default=300, help="Max decision steps per episode in the viewer.")
    ap.add_argument("--scan-recover", type=int, default=0, help="If >0, do a tiny pan wiggle after N invisible decisions.")

    args = ap.parse_args()
    main(args)
