import os
import time
from pathlib import Path
import datetime
import numpy as np
import mujoco
import mujoco.viewer

# ---- B-1/B-2: collection imports ----
import argparse, json, random

MODEL_PATH = "panda_assets/mjx_panda.xml"

# ---------- names / constants ----------
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ACT_NAMES  = [f"actuator{i}" for i in range(1, 9)]   # 1..7: arm, 8: gripper

TABLE_ZTOP     = 0.80
BALL_R         = 0.03
BUCKET_INNER_H = 0.11

READY       = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
GRIP_OPEN   = 0.040
GRIP_CLOSED = 0.000

# IK tuning
IK_LAM        = 0.12
IK_MAX_DQ     = 0.008
IK_POS_TOL    = 1.5e-3
IK_ROT_W      = 0.12
IK_POSTURE_W  = 0.02
IK_MAX_ITERS  = 400

# durations
MOVE_SLOW = 3.0
MOVE_FAST = 1.8

# grasp tolerance
GRASP_DIST_OK = 0.035


# ---------- utilities ----------
def ids_map(model: mujoco.MjModel):
    jids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    assert all(j != -1 for j in jids), "Panda joints not found"
    qadr   = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
    dofids = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)
    actids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACT_NAMES]
    assert all(a != -1 for a in actids), "Actuators not found"
    jrange = np.array([model.jnt_range[j] for j in jids])
    grip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    link0_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    ball_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    bucket_bid= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bucket")
    assert grip_site != -1 and link0_bid != -1 and ball_bid != -1 and bucket_bid != -1
    return {
        "qadr": qadr, "dofids": dofids, "actids": actids, "jrange": jrange,
        "grip_site": grip_site, "link0_bid": link0_bid, "ball_bid": ball_bid, "bucket_bid": bucket_bid
    }

def find_freejoint_qadr(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]; jnum = model.body_jntnum[bid]
    for k in range(jnum):
        jid = jadr + k
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    return None, None  # fixed body

def set_ctrl(data, actids, q_target7, g_target):
    for i in range(7):
        data.ctrl[actids[i]] = float(q_target7[i])
    data.ctrl[actids[7]] = float(g_target)

def step_rt(model, data, v):
    """Real-time pacing only if a viewer is open (keeps it smooth)."""
    t0 = time.time()
    mujoco.mj_step(model, data)
    if v is not None:
        v.sync()
        rem = model.opt.timestep - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

# ========= per-step eye capture (globals + helpers) =========
GRAB = None
EYE_CAMID = -1
FRAMES_DIR = None
FRAME_IDX = {"i": 0}
FRAME_STRIDE = 25   # save every N sim-steps; 500Hz / 25 ≈ 20 FPS

def init_eye_capture(model, override_dir=None, stamp=None):
    """Create runs/<stamp>/frames (interactive) or provided frames dir (collection) and a persistent eye Renderer."""
    global GRAB, EYE_CAMID, FRAMES_DIR, FRAME_IDX
    EYE_CAMID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    GRAB = mujoco.Renderer(model, width=128, height=128)
    if override_dir is None:
        stamp = stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        FRAMES_DIR = Path("runs") / stamp / "frames"
    else:
        FRAMES_DIR = Path(override_dir)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_IDX = {"i": 0, "step": 0, "saved": 0}
    print(f"[FRAMES] writing to: {FRAMES_DIR} (stride={FRAME_STRIDE})")

def grab_eye_frame(model, data):
    """Save one PNG from the 'eye' camera at the chosen stride."""
    if GRAB is None or FRAMES_DIR is None or EYE_CAMID == -1:
        return
    FRAME_IDX["step"] += 1
    if FRAME_IDX["step"] % max(1, int(FRAME_STRIDE)) != 0:
        return
    GRAB.update_scene(data, camera=EYE_CAMID)
    img = GRAB.render()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    fname = FRAMES_DIR / f"frame_{FRAME_IDX['saved']:05d}.png"
    try:
        from PIL import Image
        Image.fromarray(img).save(fname)
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(fname, img)
    FRAME_IDX["saved"] += 1
# ============================================================

def interp_joint(model, data, v, actids, q0, q1, grip, seconds, maintain=None, recorder=None):
    dt = model.opt.timestep
    steps = max(1, int(seconds / dt))
    for k in range(steps):
        s = (k + 1) / steps
        q = (1 - s) * q0 + s * q1
        if maintain: maintain()
        if recorder: recorder(q, grip)
        set_ctrl(data, actids, q, grip)
        step_rt(model, data, v)
        grab_eye_frame(model, data)

def interp_gripper(model, data, v, actids, q_hold, g0, g1, seconds, maintain=None, recorder=None):
    dt = model.opt.timestep
    steps = max(1, int(seconds / dt))
    for k in range(steps):
        s = (k + 1) / steps
        g = (1 - s) * g0 + s * g1
        if maintain: maintain()
        if recorder: recorder(q_hold, g)
        set_ctrl(data, actids, q_hold, g)
        step_rt(model, data, v)
        grab_eye_frame(model, data)

def desired_R_down_keep_yaw(R_cur):
    z_des = np.array([0, 0, -1.0])
    x_cur = R_cur[:, 0]
    x_proj = x_cur - (x_cur @ z_des) * z_des
    if np.linalg.norm(x_proj) < 1e-6: x_proj = np.array([1.0, 0.0, 0.0])
    x_des = x_proj / np.linalg.norm(x_proj)
    y_des = np.cross(z_des, x_des)
    return np.stack([x_des, y_des, z_des], axis=1)

def so3_log(R):
    return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def solve_ik(model, data, ids, site_id, q_seed, q_pref, target_pos,
             lam=IK_LAM, rot_w=IK_ROT_W, posture_w=IK_POSTURE_W,
             max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL, max_dq=IK_MAX_DQ):
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    q = q_seed.copy()
    for _ in range(max_iters):
        data.qpos[ids["qadr"]] = q; mujoco.mj_forward(model, data)
        p = data.site_xpos[site_id].copy()
        R = data.site_xmat[site_id].reshape(3,3).copy()
        e_pos = target_pos - p
        if np.linalg.norm(e_pos) < pos_tol: break
        R_des = desired_R_down_keep_yaw(R)
        e_rot = so3_log(R_des @ R.T)
        mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
        J6 = np.vstack([Jp[:, ids["dofids"]], rot_w * Jr[:, ids["dofids"]]])
        e6 = np.concatenate([e_pos, rot_w * e_rot])
        JT = J6.T; A = J6 @ JT + (lam**2) * np.eye(6)
        Jsharp = JT @ np.linalg.solve(A, np.eye(6))
        dq = Jsharp @ e6
        N = np.eye(7) - Jsharp @ J6
        dq += posture_w * (N @ (q_pref - q))
        m = float(np.max(np.abs(dq)))
        if m > max_dq: dq *= (max_dq / m)
        q = np.clip(q + dq, ids["jrange"][:,0], ids["jrange"][:,1])
    data.qpos[ids["qadr"]] = q; mujoco.mj_forward(model, data)
    return q.copy()

def maintain_ball_factory(model, data, grip_site, ball_qadr, ball_vadr, pin_until_attach=True):
    """keep ball fixed until attach; after attach, snap it to the gripper center."""
    init_p = data.qpos[ball_qadr:ball_qadr+3].copy()
    init_q = data.qpos[ball_qadr+3:ball_qadr+7].copy()

    state = {"attached": False, "pin": bool(pin_until_attach), "offset": np.zeros(3)}

    def maintain():
        if state["attached"]:
            p = data.site_xpos[grip_site].copy() + state["offset"]
            data.qpos[ball_qadr:ball_qadr+3]   = p
            data.qpos[ball_qadr+3:ball_qadr+7] = np.array([1.0, 0.0, 0.0, 0.0])
            data.qvel[ball_vadr:ball_vadr+6]   = 0.0
        elif state["pin"]:
            data.qpos[ball_qadr:ball_qadr+3]   = init_p
            data.qpos[ball_qadr+3:ball_qadr+7] = init_q
            data.qvel[ball_vadr:ball_vadr+6]   = 0.0

    def attach_now():
        mujoco.mj_forward(model, data)
        state["offset"] = np.zeros(3)  # center on gripper
        state["attached"] = True
        state["pin"] = False
        print("[GRASP] attached ball (centered)")

    def detach_now():
        state["attached"] = False
        print("[DROP] released ball")

    def is_attached():
        return state["attached"]

    return maintain, attach_now, detach_now, is_attached


# ---- B-1: randomization + episode I/O helpers ----
_BALL_X_RANGE   = (0.05, 0.45)   # meters, on table
_BUCKET_X_RANGE = (0.05, 0.45)
_Y_RANGE        = (-0.30, 0.30)
_MIN_SEP        = 0.22           # min XY separation between ball and bucket (m)  << increased

def _rand_xy(rng, xr, yr):
    return float(rng.uniform(*xr)), float(rng.uniform(*yr))

def _set_body_xy(data, qadr, x, y):
    # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
    z  = float(data.qpos[qadr + 2])
    qw = float(data.qpos[qadr + 3]); qx = float(data.qpos[qadr + 4])
    qy = float(data.qpos[qadr + 5]); qz = float(data.qpos[qadr + 6])
    data.qpos[qadr:qadr+7] = np.array([x, y, z, qw, qx, qy, qz], dtype=float)

def _mk_runs_root():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("runs") / stamp
    root.mkdir(parents=True, exist_ok=True)
    return root, stamp

def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---- B-2: success metric (circular footprint + below rim) ----
def _compute_success(model, data, ids, ball_vadr, goal_site="bucket_goal", xy_tol=0.045):
    """
    Success if the ball center is (a) below the rim and (b) within the inscribed
    circle of the bucket's inner opening. Bucket walls are at ±0.05 m with wall
    half-width ≈ 0.005 m → inner half-width ≈ 0.045 m. Use a tiny 2 mm margin.
    """
    mujoco.mj_forward(model, data)
    ball   = data.xpos[ids["ball_bid"]].copy()
    bucket = data.xpos[ids["bucket_bid"]].copy()

    # Below rim (allow half-ball clearance below the rim)
    rim_z  = bucket[2] + BUCKET_INNER_H
    z_ok   = ball[2] < (rim_z - 0.5 * BALL_R)

    # Circular footprint: inner square half-width ≈ 0.045 m → inscribed circle radius = 0.045 m
    R_OK = (0.05 - 0.005) - 0.002  # 0.043 m radius, small margin
    r_xy = float(np.linalg.norm(ball[:2] - bucket[:2]))
    in_circle = (r_xy <= R_OK)

    return in_circle and z_ok


# ---------- eye helpers ----------
def build_eye_ids(model):
    return {
        "pan_act":   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_pan_act"),
        "tilt_act":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_tilt_act"),
        "mount_bid": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,      "eye_mount"),
        "tilt_bid":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,      "eye_tilt"),
    }

def aim_eye_at_point(model, data, eye, target_xyz, use_zoom=False):
    """set pan/tilt so the eye points at target_xyz."""
    mujoco.mj_forward(model, data)
    t = np.asarray(target_xyz, float)

    # pan about base Z at the mount pivot
    pm = data.xpos[eye["mount_bid"]]
    v  = t - pm
    pan = np.arctan2(v[1], v[0])
    if eye["pan_act"] != -1:
        data.ctrl[eye["pan_act"]] = float(np.clip(pan, -np.pi, np.pi))

    # tilt about local Y at the tilt pivot
    pt = data.xpos[eye["tilt_bid"]]
    v2 = t - pt
    horiz = max(1e-6, np.linalg.norm(v2[:2]))
    tilt = np.arctan2(v2[2], horiz)
    if eye["tilt_act"] != -1:
        data.ctrl[eye["tilt_act"]] = float(np.clip(tilt, -1.57, 1.57))

def eye_joint_ids(model):
    pan_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "eye_pan")
    tilt_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "eye_tilt")
    return pan_jid, tilt_jid

def print_eye_angles(model, data, every_steps=30):
    """print pan/tilt in degrees every N sim steps to make motion obvious."""
    if not hasattr(print_eye_angles, "_k"):
        print_eye_angles._k = 0
    print_eye_angles._k += 1
    if print_eye_angles._k % max(1, int(every_steps)) != 0:
        return
    pan_jid, tilt_jid = eye_joint_ids(model)
    if pan_jid == -1 or tilt_jid == -1:
        return
    pan_qadr  = model.jnt_qposadr[pan_jid]
    tilt_qadr = model.jnt_qposadr[tilt_jid]
    pan  = float(data.qpos[pan_qadr])
    tilt = float(data.qpos[tilt_qadr])
    print(f"[EYE] pan={np.degrees(pan):6.1f}°, tilt={np.degrees(tilt):6.1f}°")

# ball-centric gaze target (with tiny occlusion-aware bias)
def gaze_target_ball(model, data, ids, is_ball_attached):
    mujoco.mj_forward(model, data)
    ee     = data.site_xpos[ids["grip_site"]]
    ball   = data.xpos[ids["ball_bid"]]
    bucket = data.xpos[ids["bucket_bid"]]
    target = ball.copy()
    # near/attached: bias slightly toward fingertips and up
    if is_ball_attached() or np.linalg.norm(ee - ball) < (BALL_R + 0.02):
        Rg = data.site_xmat[ids["grip_site"]].reshape(3,3)
        forward = -Rg[:, 2]
        target = ball + 0.05 * forward + np.array([0.0, 0.0, 0.03])
    # near bucket rim: look a tad higher
    if np.linalg.norm(ball[:2] - bucket[:2]) < 0.18 and ball[2] < (bucket[2] + 0.12):
        target = target + np.array([0.0, 0.0, 0.03])
    return target

def make_maintain_eye_tracking(model, data, ids, is_ball_attached, eye):
    def maintain_eye_tracking():
        target = gaze_target_ball(model, data, ids, is_ball_attached)
        aim_eye_at_point(model, data, eye, target_xyz=target, use_zoom=False)
    return maintain_eye_tracking


# ---------- snapshot helper (eye camera) ----------
def save_eye_snapshot(model, data, out_path, cam="eye", size=256):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = mujoco.Renderer(model, width=int(size), height=int(size))
    try:
        camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if camid == -1:
            raise RuntimeError(f"camera '{cam}' not found in model")
        r.update_scene(data, camera=camid)
        img = r.render()
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        # try PIL first, then imageio
        try:
            from PIL import Image
            Image.fromarray(img).save(out_path)
        except Exception:
            import imageio.v2 as imageio
            imageio.imwrite(out_path, img)
        print(f"[SNAP] {out_path}")
    finally:
        try:
            r.close()
        except Exception:
            pass


# ---- B-1/B-2: headless multi-episode collection (no viewer) ----
def run_one_episode_headless(ep_dir: Path, seed: int, frame_stride: int, save_frames: bool, save_npz: bool, episode_index: int, settle_sec: float = 0.6):
    global FRAME_STRIDE, GRAB, FRAMES_DIR, FRAME_IDX
    # per-episode RNG
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)
    ids   = ids_map(model)

    # freejoint adr for ball & bucket
    ball_qadr, ball_vadr   = find_freejoint_qadr(model, "ball")
    bucket_qadr, _bucket_v = find_freejoint_qadr(model, "bucket")
    assert ball_qadr is not None and bucket_qadr is not None, "Ball/Bucket must have <freejoint/>"

    # --- randomize ball/bucket XY on the table (keep z from XML) with min separation
    while True:
        bx, by = _rand_xy(rng, _BALL_X_RANGE, _Y_RANGE)
        kx, ky = _rand_xy(rng, _BUCKET_X_RANGE, _Y_RANGE)
        if np.hypot(bx - kx, by - ky) >= _MIN_SEP:
            break
    _set_body_xy(data, ball_qadr,   bx, by)
    _set_body_xy(data, bucket_qadr, kx, ky)
    mujoco.mj_forward(model, data)

    # --- recorder (adds qvel too)
    demo = {"qpos": [], "qvel": [], "ee": [], "ball": [], "bucket": [], "action": []}
    def _rec(q_target7, g_target):
        mujoco.mj_forward(model, data)
        demo["qpos"].append(data.qpos[ids["qadr"]].copy())
        demo["qvel"].append(data.qvel[ids["dofids"]].copy())
        demo["ee"].append(data.site_xpos[ids["grip_site"]].copy())
        demo["ball"].append(data.xpos[ids["ball_bid"]].copy())
        demo["bucket"].append(data.xpos[ids["bucket_bid"]].copy())
        demo["action"].append(np.concatenate([np.asarray(q_target7, float), [float(g_target)]]))

    # --- ball virtual attach
    maintain, attach_ball, detach_ball, is_ball_attached = maintain_ball_factory(
        model, data, ids["grip_site"], ball_qadr, ball_vadr
    )

    # --- eye tracking
    eye = build_eye_ids(model)
    assert eye["pan_act"] != -1 and eye["tilt_act"] != -1, "Eye actuators not found"
    maintain_eye_tracking = make_maintain_eye_tracking(model, data, ids, is_ball_attached, eye)
    def maintain_all():
        maintain()
        maintain_eye_tracking()

    # --- optional eye frames (per-episode folder)
    FRAME_STRIDE = int(frame_stride if frame_stride is not None else FRAME_STRIDE)
    if save_frames:
        frames_dir = ep_dir / "frames"
        init_eye_capture(model, override_dir=frames_dir)
    else:
        GRAB = None; FRAMES_DIR = None; FRAME_IDX = {"i": 0, "step": 0, "saved": 0}

    # ===== replicate scripted sequence (v=None so headless) =====
    # 1) READY (open)
    q_now = data.qpos[ids["qadr"]].copy()
    interp_joint(model, data, None, ids["actids"], q_now, READY, GRIP_OPEN, seconds=3.0,
                 maintain=maintain_all, recorder=_rec)

    # 2) yaw joint1 toward ball
    mujoco.mj_forward(model, data)
    base_xy  = data.xpos[ids["link0_bid"]][:2].copy()
    ball_xy  = data.xpos[ids["ball_bid"]][:2].copy()
    yaw      = float(np.arctan2(ball_xy[1]-base_xy[1], ball_xy[0]-base_xy[0]))
    yaw_pose = READY.copy(); yaw_pose[0] = yaw
    interp_joint(model, data, None, ids["actids"], data.qpos[ids["qadr"]].copy(), yaw_pose, GRIP_OPEN,
                 seconds=2.0, maintain=maintain_all, recorder=_rec)

    # 3) IK waypoints
    mujoco.mj_forward(model, data)
    ball_p   = data.qpos[ball_qadr:ball_qadr+3].copy()
    bucket_p = data.xpos[ids["bucket_bid"]].copy()

    hover_ball    = ball_p + np.array([0.0, 0.0, 0.24])
    grasp_pos     = ball_p + np.array([0.0, 0.0, BALL_R + 0.004])
    # NEW: pre-grasp 2 cm above the ball for a vertical final approach
    pre_grasp     = ball_p + np.array([0.0, 0.0, BALL_R + 0.020])
    lift_pos      = ball_p + np.array([0.0, 0.0, 0.30])

    drop_z        = bucket_p[2] + BUCKET_INNER_H + 0.02
    hover_bucket  = np.array([bucket_p[0], bucket_p[1], max(drop_z + 0.12, TABLE_ZTOP + 0.24)])
    inside_bucket = np.array([bucket_p[0], bucket_p[1], drop_z])

    q_seed       = data.qpos[ids["qadr"]].copy()
    q_hover_ba   = solve_ik(model, data, ids, ids["grip_site"], q_seed,        yaw_pose, hover_ball)
    q_pre_grasp  = solve_ik(model, data, ids, ids["grip_site"], q_hover_ba,    yaw_pose, pre_grasp)
    q_grasp      = solve_ik(model, data, ids, ids["grip_site"], q_pre_grasp,   yaw_pose, grasp_pos)
    q_lift       = solve_ik(model, data, ids, ids["grip_site"], q_grasp,       yaw_pose, lift_pos)
    q_hover_bkt  = solve_ik(model, data, ids, ids["grip_site"], q_lift,        yaw_pose, hover_bucket)
    q_inside_b   = solve_ik(model, data, ids, ids["grip_site"], q_hover_bkt,   yaw_pose, inside_bucket)

    # 4) execute path (no snapshots in headless)
    interp_joint(model, data, None, ids["actids"], data.qpos[ids["qadr"]].copy(), q_hover_ba,  GRIP_OPEN,
                 seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)
    # NEW: go to pre-grasp first, then final short drop to grasp
    interp_joint(model, data, None, ids["actids"], q_hover_ba,  q_pre_grasp, GRIP_OPEN,
                 seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)
    interp_joint(model, data, None, ids["actids"], q_pre_grasp, q_grasp,     GRIP_OPEN,
                 seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)

    mujoco.mj_forward(model, data)
    ee   = data.site_xpos[ids["grip_site"]].copy()
    dist = np.linalg.norm(ee - data.qpos[ball_qadr:ball_qadr+3])
    if dist > GRASP_DIST_OK:
        print(f"[GRASP] dist {dist:.3f} > {GRASP_DIST_OK:.3f} m (attaching anyway).")
    attach_ball()
    interp_gripper(model, data, None, ids["actids"], q_grasp, GRIP_OPEN, GRIP_CLOSED,
                   seconds=0.8, maintain=maintain_all, recorder=_rec)
    time.sleep(0.15)

    interp_joint(model, data, None, ids["actids"], q_grasp,     q_lift,      GRIP_CLOSED,
                 seconds=MOVE_FAST, maintain=maintain_all, recorder=_rec)
    interp_joint(model, data, None, ids["actids"], q_lift,      q_hover_bkt, GRIP_CLOSED,
                 seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)
    interp_joint(model, data, None, ids["actids"], q_hover_bkt, q_inside_b,  GRIP_CLOSED,
                 seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)

    detach_ball()
    interp_gripper(model, data, None, ids["actids"], q_inside_b, GRIP_CLOSED, GRIP_OPEN,
                   seconds=0.8, maintain=maintain_all, recorder=_rec)

    # back out & return to ready
    interp_joint(model, data, None, ids["actids"], q_inside_b,  q_hover_bkt, GRIP_OPEN,
                 seconds=MOVE_FAST, maintain=maintain_all, recorder=_rec)
    interp_joint(model, data, None, ids["actids"], q_hover_bkt, yaw_pose,     GRIP_OPEN,
                 seconds=MOVE_SLOW,  maintain=maintain_all, recorder=_rec)
    interp_joint(model, data, None, ids["actids"], yaw_pose,    READY,        GRIP_OPEN,
                 seconds=MOVE_SLOW,  maintain=maintain_all, recorder=_rec)

    # --- settle briefly, then compute success (B-2) ---
    settle_steps = int(float(settle_sec) / model.opt.timestep)
    for _ in range(settle_steps):
        if maintain_all:  # keep eye tracking; arm holds READY
            maintain_all()
        set_ctrl(data, ids["actids"], READY, GRIP_OPEN)
        step_rt(model, data, None)

    success = _compute_success(model, data, ids, ball_vadr, goal_site="bucket_goal", xy_tol=0.045)

    ep_summary = {
        "seed": int(seed),
        "episode_index": int(episode_index),
        "ball_xy":   [float(bx), float(by)],
        "bucket_xy": [float(kx), float(ky)],
        "success": bool(success),
    }
    _write_json(ep_dir / "episode.json", ep_summary)

    if save_npz:
        np.savez(
            ep_dir / "episode_data.npz",
            actions=np.asarray(demo["action"]),
            qpos=np.asarray(demo["qpos"]),
            qvel=np.asarray(demo["qvel"]),
            ee=np.asarray(demo["ee"]),
            ball=np.asarray(demo["ball"]),
            bucket=np.asarray(demo["bucket"]),
            dt=np.float32(model.opt.timestep),
            success=np.bool_(success),
            seed=np.int64(seed),
            episode_index=np.int64(episode_index),
        )

    # cleanup renderer if opened for frames
    if GRAB is not None:
        try: GRAB.close()
        except Exception: pass
        GRAB = None; FRAMES_DIR = None; FRAME_IDX = {"i": 0, "step": 0, "saved": 0}

    return success


def collect_episodes(N, base_seed, frame_stride, save_frames, save_npz, settle_sec=0.6):
    runs_root, stamp = _mk_runs_root()
    if base_seed is None:
        base_seed = int(time.time())

    print(f"[COLLECT] runs root: {runs_root}")
    print(f"[COLLECT] episodes: {N}, base_seed: {base_seed}, frame_stride: {frame_stride or FRAME_STRIDE}, "
          f"save_frames={save_frames}, save_npz={save_npz}, settle_sec={settle_sec}")

    success_count = 0
    for ep in range(N):
        ep_dir = runs_root / f"ep_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_seed = base_seed + ep
        print(f"[COLLECT] Episode {ep:03d}  seed={ep_seed}")
        ok = run_one_episode_headless(
            ep_dir=ep_dir,
            seed=ep_seed,
            frame_stride=(frame_stride if frame_stride is not None else FRAME_STRIDE),
            save_frames=save_frames,
            save_npz=save_npz,
            episode_index=ep,
            settle_sec=settle_sec,
        )
        success_count += int(bool(ok))

    print(f"Done. Episodes written to: {runs_root}  Success: {success_count}/{N}")


# ---------- main (interactive viewer path unchanged, no pan/tilt spam) ----------
def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model); mujoco.mj_forward(model, data)
    ids   = ids_map(model)
    ball_qadr, ball_vadr = find_freejoint_qadr(model, "ball")
    assert ball_qadr is not None, "Ball must have a <freejoint/>"

    # init per-step eye capture
    init_eye_capture(model)

    # simple demo recorder
    demo = {"qpos": [], "ee": [], "ball": [], "bucket": [], "action": []}
    def _rec(q_target7, g_target):
        mujoco.mj_forward(model, data)
        demo["qpos"].append(data.qpos[ids["qadr"]].copy())
        demo["ee"].append(data.site_xpos[ids["grip_site"]].copy())
        demo["ball"].append(data.xpos[ids["ball_bid"]].copy())
        demo["bucket"].append(data.xpos[ids["bucket_bid"]].copy())
        demo["action"].append(np.concatenate([np.asarray(q_target7, float), [float(g_target)]]))

    # ball “attach” virtual constraint
    maintain, attach_ball, detach_ball, is_ball_attached = maintain_ball_factory(
        model, data, ids["grip_site"], ball_qadr, ball_vadr
    )
    # eye tracking
    eye = build_eye_ids(model)
    assert eye["pan_act"] != -1 and eye["tilt_act"] != -1, "Eye actuators not found"
    maintain_eye_tracking = make_maintain_eye_tracking(model, data, ids, is_ball_attached, eye)

    # combine maintainers (silenced pan/tilt prints)
    def maintain_all():
        maintain()
        maintain_eye_tracking()

    with mujoco.viewer.launch_passive(model, data) as v:
        # short third-person preview, then switch to eye POV
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0.0, 0.0, 0.90]
        v.cam.distance = 2.2
        v.cam.azimuth = 120
        v.cam.elevation = -20
        t_end = time.time() + 1.5
        while v.is_running() and time.time() < t_end:
            maintain_eye_tracking()
            set_ctrl(data, ids["actids"], READY, GRIP_OPEN)
            step_rt(model, data, v)

        camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
        if camid != -1:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = camid

        # ----- 1) move to READY (open) -----
        q_now = data.qpos[ids["qadr"]].copy()
        interp_joint(model, data, v, ids["actids"], q_now, READY, GRIP_OPEN, seconds=3.0,
                     maintain=maintain_all, recorder=_rec)

        # ----- 2) yaw joint1 toward ball -----
        mujoco.mj_forward(model, data)
        base_xy  = data.xpos[ids["link0_bid"]][:2].copy()
        ball_xy  = data.xpos[ids["ball_bid"]][:2].copy()
        yaw      = float(np.arctan2(ball_xy[1]-base_xy[1], ball_xy[0]-base_xy[0]))
        yaw_pose = READY.copy(); yaw_pose[0] = yaw
        interp_joint(model, data, v, ids["actids"], data.qpos[ids["qadr"]].copy(), yaw_pose, GRIP_OPEN,
                     seconds=2.0, maintain=maintain_all, recorder=_rec)

        # ----- 3) IK waypoints -----
        mujoco.mj_forward(model, data)
        ball_p   = data.qpos[ball_qadr:ball_qadr+3].copy()
        bucket_p = data.xpos[ids["bucket_bid"]].copy()

        hover_ball    = ball_p + np.array([0.0, 0.0, 0.24])
        grasp_pos     = ball_p + np.array([0.0, 0.0, BALL_R + 0.004])
        # NEW: pre-grasp 2 cm above the ball for a vertical final approach
        pre_grasp     = ball_p + np.array([0.0, 0.0, BALL_R + 0.020])
        lift_pos      = ball_p + np.array([0.0, 0.0, 0.30])

        drop_z        = bucket_p[2] + BUCKET_INNER_H + 0.02
        hover_bucket  = np.array([bucket_p[0], bucket_p[1], max(drop_z + 0.12, TABLE_ZTOP + 0.24)])
        inside_bucket = np.array([bucket_p[0], bucket_p[1], drop_z])

        q_seed       = data.qpos[ids["qadr"]].copy()
        q_hover_ba   = solve_ik(model, data, ids, ids["grip_site"], q_seed,        yaw_pose, hover_ball)
        q_pre_grasp  = solve_ik(model, data, ids, ids["grip_site"], q_hover_ba,    yaw_pose, pre_grasp)
        q_grasp      = solve_ik(model, data, ids, ids["grip_site"], q_pre_grasp,   yaw_pose, grasp_pos)
        q_lift       = solve_ik(model, data, ids, ids["grip_site"], q_grasp,       yaw_pose, lift_pos)
        q_hover_bkt  = solve_ik(model, data, ids, ids["grip_site"], q_lift,        yaw_pose, hover_bucket)
        q_inside_b   = solve_ik(model, data, ids, ids["grip_site"], q_hover_bkt,   yaw_pose, inside_bucket)

        # ----- 4) execute path + snapshots -----
        # to hover over ball
        interp_joint(model, data, v, ids["actids"], data.qpos[ids["qadr"]].copy(), q_hover_ba,  GRIP_OPEN,
                     seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)
        save_eye_snapshot(model, data, "snaps/01_hover_ball.png")

        # NEW: pre-grasp first, then final short drop to grasp
        interp_joint(model, data, v, ids["actids"], q_hover_ba,  q_pre_grasp, GRIP_OPEN,
                     seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)
        interp_joint(model, data, v, ids["actids"], q_pre_grasp, q_grasp,     GRIP_OPEN,
                     seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)

        # attach & close
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ids["grip_site"]].copy()
        dist = np.linalg.norm(ee - data.qpos[ball_qadr:ball_qadr+3])
        if dist > GRASP_DIST_OK:
            print(f"[GRASP] dist {dist:.3f} > {GRASP_DIST_OK:.3f} m (attaching anyway).")
        attach_ball()
        interp_gripper(model, data, v, ids["actids"], q_grasp, GRIP_OPEN, GRIP_CLOSED,
                       seconds=0.8, maintain=maintain_all, recorder=_rec)
        time.sleep(0.15)

        # lift (snapshot after lift)
        interp_joint(model, data, v, ids["actids"], q_grasp, q_lift, GRIP_CLOSED,
                     seconds=MOVE_FAST, maintain=maintain_all, recorder=_rec)
        save_eye_snapshot(model, data, "snaps/02_lift.png")

        # to hover bucket
        interp_joint(model, data, v, ids["actids"], q_lift, q_hover_bkt, GRIP_CLOSED,
                     seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)

        # inside bucket (snapshot before drop)
        interp_joint(model, data, v, ids["actids"], q_hover_bkt, q_inside_b, GRIP_CLOSED,
                     seconds=MOVE_SLOW, maintain=maintain_all, recorder=_rec)
        save_eye_snapshot(model, data, "snaps/03_pre_drop.png")

        # drop
        detach_ball()
        interp_gripper(model, data, v, ids["actids"], q_inside_b, GRIP_CLOSED, GRIP_OPEN,
                       seconds=0.8, maintain=maintain_all, recorder=_rec)

        # back out & return to ready
        interp_joint(model, data, v, ids["actids"], q_inside_b,  q_hover_bkt, GRIP_OPEN,
                     seconds=MOVE_FAST, maintain=maintain_all, recorder=_rec)
        interp_joint(model, data, v, ids["actids"], q_hover_bkt, yaw_pose,     GRIP_OPEN,
                     seconds=MOVE_SLOW,  maintain=maintain_all, recorder=_rec)
        interp_joint(model, data, v, ids["actids"], yaw_pose,    READY,        GRIP_OPEN,
                     seconds=MOVE_SLOW,  maintain=maintain_all, recorder=_rec)

        # save a tiny demo npz for debugging
        os.makedirs("demos", exist_ok=True)
        np.savez(
            "demos/demo1.npz",
            qpos=np.asarray(demo["qpos"]),
            ee=np.asarray(demo["ee"]),
            ball=np.asarray(demo["ball"]),
            bucket=np.asarray(demo["bucket"]),
            action=np.asarray(demo["action"]),
            dt=np.float32(model.opt.timestep),
        )
        print("Saved demo to demos/demo1.npz")
        print("Done. Close the window to exit.")

        # idle: keep tracking; no frame saving here (prevents extra frames)
        while v.is_running():
            maintain_all()
            set_ctrl(data, ids["actids"], READY, GRIP_OPEN)
            step_rt(model, data, v)

    # cleanup renderer
    if GRAB is not None:
        try: GRAB.close()
        except Exception: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", type=int, default=0, help="Run N headless episodes (no viewer).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (base).")
    parser.add_argument("--frame-stride", type=int, default=None, help="Override FRAME_STRIDE for collection (default=25).")
    parser.add_argument("--save-frames", action="store_true", help="Save eye frames per-step (respect stride).")
    parser.add_argument("--save-npz", action="store_true", help="Save episode_data.npz per episode.")
    parser.add_argument("--settle-sec", type=float, default=0.6,
                        help="Seconds to settle after drop before success check (default 0.6).")
    args = parser.parse_args()

    if args.collect and args.collect > 0:
        collect_episodes(
            N=int(args.collect),
            base_seed=args.seed,
            frame_stride=args.frame_stride,
            save_frames=bool(args.save_frames),
            save_npz=bool(args.save_npz),
            settle_sec=float(args.settle_sec),
        )
    else:
        main()
