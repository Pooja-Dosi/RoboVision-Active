# eye_env_viewer_demo.py
# Watch the EyeEnv in MuJoCo's viewer with nicer lighting and flexible camera setup.

import time, argparse
import mujoco, mujoco.viewer

from eye_env import EyeEnv, EyeEnvConfig

def _set_attr_safe(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False

def apply_pretty_lighting(model,
                          ambient=0.45, diffuse=0.85,
                          head_ambient=0.40, head_diffuse=0.80, head_spec=0.10):
    """
    Softer scene lighting without editing the XML.
    Works across MuJoCo Python versions by probing attributes safely.
    """
    # Global lighting lives under model.vis.global or model.vis.global_
    vis = model.vis
    global_block = getattr(vis, "global", None)
    if global_block is None:
        global_block = getattr(vis, "global_", None)

    if global_block is not None:
        _set_attr_safe(global_block, "ambient",  float(ambient))
        _set_attr_safe(global_block, "diffuse",  float(diffuse))
        # specular may not exist in some versions — set if present
        _set_attr_safe(global_block, "specular", 0.10)

    # Headlight (camera-facing light)
    if hasattr(vis, "headlight"):
        _set_attr_safe(vis.headlight, "ambient",  float(head_ambient))
        _set_attr_safe(vis.headlight, "diffuse",  float(head_diffuse))
        _set_attr_safe(vis.headlight, "specular", float(head_spec))

def set_freecam(v, lookat=(0.0, 0.0, 0.90), dist=2.2, az=135, el=-25):
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = lookat
    v.cam.distance = dist
    v.cam.azimuth = az
    v.cam.elevation = el

def main(args):
    # Build EyeEnv with your chosen render size (purely cosmetic for viewer)
    cfg = EyeEnvConfig(
        reward_mode="state",
        debug_dump_every=0,
        reset_aim_at_target_prob=0.9,  # start roughly aimed at target so you can see it
        obs_size=args.obs_size,
        frame_skip=args.frame_skip,
    )
    env = EyeEnv(cfg)
    obs, info = env.reset(seed=args.seed)

    model, data = env.model, env.data
    eye_camid = env.eye_camid

    # Nicer lighting (no XML changes)
    apply_pretty_lighting(
        model,
        ambient=args.ambient,
        diffuse=args.diffuse,
        head_ambient=args.head_ambient,
        head_diffuse=args.head_diffuse,
        head_spec=args.head_spec,
    )

    # Action cadence (how long to hold a sampled action)
    action_hold = max(1, args.action_hold)
    step_i = 0
    total_r = 0.0

    # Optional timed switch to the eye camera (seconds)
    switch_after_simsteps = None
    if args.switch_to_eye_after > 0:
        # ~60 viewer frames per second
        switch_after_simsteps = int(args.switch_to_eye_after * 60)

    with mujoco.viewer.launch_passive(model, data) as v:
        # Camera choice
        if args.freecam_first:
            set_freecam(v, dist=args.freecam_dist, az=args.freecam_az, el=args.freecam_el)
        else:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = eye_camid

        t_last = time.time()
        while v.is_running():
            # Switch to eye camera after some time (if requested)
            if switch_after_simsteps is not None and step_i == switch_after_simsteps:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                v.cam.fixedcamid = eye_camid

            # Every 'action_hold' viewer frames, advance one env step with a random action
            if step_i % action_hold == 0:
                act = env.action_space.sample()
                obs, r, term, trunc, _ = env.step(act)
                total_r += r
                if term or trunc:
                    obs, _ = env.reset()

            # Keep viewer smooth in real time
            v.sync()

            # Gentle pacing (target ~60 FPS)
            now = time.time()
            delay = max(0.0, (1.0 / 60.0) - (now - t_last))
            if delay > 0:
                time.sleep(delay)
            t_last = now
            step_i += 1

    env.close()
    print(f"[viewer closed] steps={step_i}  total_reward={total_r:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-size", type=int, default=256, help="Eye render resolution (cosmetic for viewing).")
    ap.add_argument("--frame-skip", type=int, default=5, help="mj_steps per env step.")
    ap.add_argument("--action-hold", type=int, default=4, help="Hold each sampled action for N viewer frames.")
    ap.add_argument("--freecam-first", action="store_true", help="Start in free camera instead of 'eye'.")
    ap.add_argument("--freecam-az", type=float, default=135.0)
    ap.add_argument("--freecam-el", type=float, default=-25.0)
    ap.add_argument("--freecam-dist", type=float, default=2.3)
    ap.add_argument("--switch-to-eye-after", type=float, default=2.0,
                    help="Seconds before auto-switching to the eye camera (0=never).")

    # Lighting knobs (safe, no XML edits)
    ap.add_argument("--ambient", type=float, default=0.45)
    ap.add_argument("--diffuse", type=float, default=0.85)
    ap.add_argument("--head-ambient", type=float, default=0.40)
    ap.add_argument("--head-diffuse", type=float, default=0.80)
    ap.add_argument("--head-spec", type=float, default=0.10)

    args = ap.parse_args()
    main(args)
