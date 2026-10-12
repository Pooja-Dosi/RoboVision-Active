# train_eye_ppo.py
import os, math, argparse
import numpy as np
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

from eye_env import EyeEnv, EyeEnvConfig

# ---------- geometry helpers (copy of what we used in viewer) ----------
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
    fx = fy; cx = cy = 0.5 * size
    u = fx * (x_cam / z_fwd) + cx
    v = fy * (y_cam / z_fwd) + cy
    return u, v, z_fwd

# ---------- reward shaping wrapper ----------
class EyeRewardWrapper(EyeEnv):
    def __init__(self, base_cfg: EyeEnvConfig,
                 prog_beta=0.6, off_pen=0.05, back_pen=0.10, time_pen=0.0):
        super().__init__(base_cfg)
        self._prev_dist = None
        self._prog_beta = float(prog_beta)
        self._off_pen   = float(off_pen)
        self._back_pen  = float(back_pen)
        self._time_pen  = float(time_pen)
        # cache ids
        self._camid = self.eye_camid
        self._tgtid = self._target_bid
        self._size  = self.cfg.obs_size
        self._maxd  = 0.5 * math.hypot(self._size, self._size)

    def _center_dist(self) -> Tuple[float, bool, float]:
        # returns (distance_to_center, visible, z_forward)
        u, v, zf = project_uv(self.model, self.data, self._camid, self._tgtid, self._size)
        visible = (0 <= u < self._size and 0 <= v < self._size and zf > 1e-6)
        if not visible:
            return float(self._maxd), False, zf
        dist = math.hypot(u - 0.5*self._size, v - 0.5*self._size)
        return float(dist), True, zf

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        dist, _, _ = self._center_dist()
        self._prev_dist = dist
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        # progress term
        dist, visible, zf = self._center_dist()
        prog = self._prog_beta * (self._prev_dist - dist)
        self._prev_dist = dist
        shaped = r + prog
        # penalties
        if not visible:
            shaped -= self._off_pen
        if zf <= 0.0:
            shaped -= self._back_pen
        if self._time_pen > 0.0:
            shaped -= self._time_pen
        return obs, float(shaped), term, trunc, info

# ---------- factory ----------
def make_env(seed, obs_size, frame_skip, target,
             discrete, reward_mode, step_deg,
             reset_prob, prog_beta, off_pen, back_pen, time_pen):
    def _f():
        cfg = EyeEnvConfig(
            obs_size=obs_size,
            frame_skip=frame_skip,
            target=target,
            discrete_actions=discrete,
            reward_mode=reward_mode,
            step_deg=step_deg,
            reset_aim_at_target_prob=reset_prob,
            # keep termination strict (occlusion-aware) even in training:
            require_visible=True,
            visible_window=10,
            success_radius_frac=0.10,
        )
        env = EyeRewardWrapper(cfg, prog_beta=prog_beta, off_pen=off_pen, back_pen=back_pen, time_pen=time_pen)
        env.reset(seed=seed)
        return env
    return _f

# ---------- main ----------
def main(args):
    os.makedirs(args.out, exist_ok=True)

    # curriculum on reset probability (start easy → harder)
    reset_prob = args.reset_prob_start

    def env_fn():
        return make_env(args.seed, args.obs_size, args.frame_skip, args.target,
                        args.discrete, args.reward_mode, args.step_deg,
                        reset_prob, args.prog_beta, args.off_pen, args.back_pen, args.time_pen)()

    venv = DummyVecEnv([env_fn])
    venv = VecTransposeImage(venv)

    # PPO config (solid defaults for discrete image control)
    model = PPO(
        "CnnPolicy", venv,
        learning_rate=3e-4, n_steps=2048, batch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        verbose=1, device="auto",
    )

    # checkpointing
    ckpt = CheckpointCallback(save_freq=20000, save_path=args.out, name_prefix="eye_ppo_ckpt")
    pb   = ProgressBarCallback()

    # staged training with reset-prob anneal
    total = int(args.total_timesteps)
    stages = max(1, int(args.anneal_stages))
    per_stage = total // stages
    timesteps_done = 0
    for s in range(stages):
        # linearly anneal reset_aim_at_target_prob toward args.reset_prob_end
        frac = s / max(1, stages - 1)
        reset_prob = float(args.reset_prob_start * (1 - frac) + args.reset_prob_end * frac)
        print(f"[stage {s+1}/{stages}] reset_aim_at_target_prob={reset_prob:.2f}")
        model.learn(total_timesteps=per_stage, callback=[ckpt, pb], progress_bar=True)
        timesteps_done += per_stage

    # final save
    out_path = os.path.join(args.out, "eye_ppo.zip")
    model.save(out_path)
    print(f"Saved policy to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="eye_runs/ball_state_shaped")
    ap.add_argument("--total-timesteps", type=int, default=500000)

    # env settings
    ap.add_argument("--obs-size", type=int, default=128)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--step-deg", type=float, default=3.0)
    ap.add_argument("--target", type=str, default="ball", choices=["ball","bucket"])
    ap.add_argument("--reward-mode", type=str, default="state", choices=["state","vision"])
    ap.add_argument("--discrete", action="store_true", default=True)

    # curriculum
    ap.add_argument("--anneal-stages", type=int, default=5)
    ap.add_argument("--reset-prob-start", type=float, default=1.0)
    ap.add_argument("--reset-prob-end", type=float, default=0.2)

    # reward-shaping coeffs
    ap.add_argument("--prog-beta", type=float, default=0.6)
    ap.add_argument("--off-pen", type=float, default=0.05)
    ap.add_argument("--back-pen", type=float, default=0.10)
    ap.add_argument("--time-pen", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
