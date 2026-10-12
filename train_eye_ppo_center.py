# train_eye_ppo_center.py
import os, math, argparse
import numpy as np
from typing import Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

from eye_env import EyeEnv, EyeEnvConfig

# ---------- projection ----------
def project_uv(model, data, camid: int, body_bid: int, size: int) -> Tuple[float,float,float]:
    cpos = data.cam_xpos[camid].copy()
    cmat = data.cam_xmat[camid].reshape(3,3).copy()
    x_c, y_c, z_c = cmat[:,0], cmat[:,1], cmat[:,2]  # camera looks along -Z
    tgt = data.xpos[body_bid].copy()
    d = tgt - cpos
    x_cam = float(np.dot(x_c, d))
    y_cam = float(np.dot(y_c, d))
    z_fwd = float(-np.dot(z_c, d))
    if z_fwd <= 1e-6:
        return float("nan"), float("nan"), z_fwd
    fovy = float(model.cam_fovy[camid])
    fy = 0.5 * size / math.tan(math.radians(fovy) * 0.5)
    fx = fy
    cx = cy = 0.5 * size
    u = fx * (x_cam / z_fwd) + cx
    v = fy * (y_cam / z_fwd) + cy
    return u, v, z_fwd

# ---------- reward-shaping wrapper ----------
class EyeRewardWrapper(EyeEnv):
    """
    Rewards:
      - progress toward image center even if OFF-SCREEN (when zf>0)
      - extra vertical progress (helps re-enter from top/bottom edges)
      - center 'well' bonus (stabilize at center)
    Penalties:
      - off-screen window
      - behind camera
      - small time penalty
    """
    def __init__(self, base_cfg: EyeEnvConfig,
                 prog_beta=0.8, vert_beta=0.25,
                 center_w=1.0,
                 off_pen=0.15, back_pen=0.35, time_pen=0.0015,
                 dist_clip_mult=1.5):
        super().__init__(base_cfg)
        self._camid = self.eye_camid
        self._tgtid = self._target_bid
        self._size  = self.cfg.obs_size
        self._radius_px = float(self.cfg.success_radius_frac * self._size)
        self._dist_clip = float(dist_clip_mult * self._size)

        self._prev_dist = None
        self._prev_verr = None
        self._prog_beta  = float(prog_beta)
        self._vert_beta  = float(vert_beta)
        self._center_w   = float(center_w)
        self._off_pen    = float(off_pen)
        self._back_pen   = float(back_pen)
        self._time_pen   = float(time_pen)

    def _center_geom(self):
        u, v, zf = project_uv(self.model, self.data, self._camid, self._tgtid, self._size)
        cx = cy = 0.5 * self._size

        if not (zf > 1e-6) or math.isnan(u) or math.isnan(v):
            # Behind camera or invalid → very large distance, undefined vertical error
            return self._dist_clip, None, False, float(zf if not math.isnan(zf) else -1.0)

        # True distance to center even if off-screen (keeps gradient)
        dist = math.hypot(float(u) - cx, float(v) - cy)
        dist = min(dist, self._dist_clip)
        visible = (0 <= u < self._size and 0 <= v < self._size)

        # signed vertical error: negative if above center, positive if below
        verr = float(v - cy)
        return float(dist), verr, bool(visible), float(zf)

    def reset(self, *, seed: Optional[int]=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        dist, verr, _, _ = self._center_geom()
        self._prev_dist = dist
        self._prev_verr = verr
        return obs, info

    def step(self, action):
        obs, base_r, term, trunc, info = super().step(action)

        dist, verr, visible, zf = self._center_geom()
        shaped = float(base_r) - self._time_pen

        # center progress
        if self._prev_dist is not None:
            shaped += self._prog_beta * (self._prev_dist - dist)
        self._prev_dist = dist

        # vertical progress (helps re-enter from v just outside the frame)
        if (self._prev_verr is not None) and (verr is not None):
            shaped += self._vert_beta * (abs(self._prev_verr) - abs(verr))
        self._prev_verr = verr

        # penalties
        if not visible:
            shaped -= self._off_pen
        if zf <= 0.0:
            shaped -= self._back_pen

        # center well bonus (linear inside success radius)
        if self._radius_px > 1e-6 and zf > 0:
            dnorm = dist / self._radius_px  # 1 at the rim
            if dnorm < 1.0:
                shaped += self._center_w * (1.0 - dnorm)

        return obs, float(shaped), term, trunc, info

# ---------- builders ----------
def make_env(seed, obs_size, frame_skip, step_deg, target,
             discrete, reward_mode, reset_prob,
             prog_beta, vert_beta, center_w,
             off_pen, back_pen, time_pen, dist_clip_mult):
    def _f():
        cfg = EyeEnvConfig(
            obs_size=obs_size,
            frame_skip=frame_skip,
            step_deg=step_deg,
            target=target,
            discrete_actions=discrete,
            reward_mode=reward_mode,
            reset_aim_at_target_prob=reset_prob,
            require_visible=True,      # success gate stays strict
            visible_window=12,
            success_radius_frac=0.10,
        )
        env = EyeRewardWrapper(
            cfg,
            prog_beta=prog_beta, vert_beta=vert_beta, center_w=center_w,
            off_pen=off_pen, back_pen=back_pen, time_pen=time_pen, dist_clip_mult=dist_clip_mult
        )
        env.reset(seed=seed)
        return env
    return _f

def main(args):
    os.makedirs(args.out, exist_ok=True)
    reset_prob = args.reset_prob_start

    def env_fn():
        return make_env(args.seed, args.obs_size, args.frame_skip, args.step_deg, args.target,
                        args.discrete, args.reward_mode, reset_prob,
                        args.prog_beta, args.vert_beta, args.center_w,
                        args.off_pen, args.back_pen, args.time_pen, args.dist_clip_mult)()

    venv = DummyVecEnv([env_fn])
    venv = VecTransposeImage(venv)

    if args.init_from and os.path.exists(args.init_from):
        print(f"[INIT] loading: {args.init_from}")
        model = PPO.load(args.init_from, env=venv, device="auto")
    else:
        model = PPO(
            "CnnPolicy", venv,
            learning_rate=3e-4, n_steps=2048, batch_size=256,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            verbose=1, device="auto",
        )

    ckpt = CheckpointCallback(save_freq=20000, save_path=args.out, name_prefix="eye_ppo_ckpt")
    pb   = ProgressBarCallback()

    total = int(args.total_timesteps)
    stages = max(1, int(args.anneal_stages))
    per_stage = max(1, total // stages)

    for s in range(stages):
        frac = s / max(1, stages - 1)
        reset_prob = float(args.reset_prob_start * (1 - frac) + args.reset_prob_end * frac)
        print(f"[stage {s+1}/{stages}] reset_aim_at_target_prob={reset_prob:.2f}")
        model.learn(total_timesteps=per_stage, callback=[ckpt, pb], progress_bar=True)

    out_path = os.path.join(args.out, "eye_ppo.zip")
    model.save(out_path)
    print(f"Saved policy to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="eye_runs/ball_state_center")
    ap.add_argument("--total-timesteps", type=int, default=300000)
    ap.add_argument("--init-from", type=str, default="", help="Resume from an existing PPO zip (optional).")

    # training cadence (match eval)
    ap.add_argument("--obs-size", type=int, default=128)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--step-deg", type=float, default=3.0)

    ap.add_argument("--target", type=str, default="ball", choices=["ball","bucket"])
    ap.add_argument("--reward-mode", type=str, default="state", choices=["state","vision"])
    ap.add_argument("--discrete", action="store_true", default=True)

    # curriculum
    ap.add_argument("--anneal-stages", type=int, default=4)
    ap.add_argument("--reset-prob-start", type=float, default=0.7)
    ap.add_argument("--reset-prob-end", type=float, default=0.2)

    # shaping coeffs
    ap.add_argument("--prog-beta", type=float, default=0.8)
    ap.add_argument("--vert-beta", type=float, default=0.25)
    ap.add_argument("--center-w", type=float, default=1.0)
    ap.add_argument("--off-pen", type=float, default=0.15)
    ap.add_argument("--back-pen", type=float, default=0.35)
    ap.add_argument("--time-pen", type=float, default=0.0015)
    ap.add_argument("--dist-clip-mult", type=float, default=1.5)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
