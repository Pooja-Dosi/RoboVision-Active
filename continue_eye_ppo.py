# continue_eye_ppo.py
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from eye_env import EyeEnv, EyeEnvConfig

def make_env(obs_size, frame_skip, step_deg, target, discrete, reward_mode, reset_aim_prob, seed):
    def _f():
        cfg = EyeEnvConfig(
            obs_size=obs_size,
            frame_skip=frame_skip,
            step_deg=step_deg,
            target=target,
            discrete_actions=discrete,
            reward_mode=reward_mode,
            # key bit: keep the ball in view to get a learning signal
            reset_aim_at_target_prob=reset_aim_prob,
            require_visible=True,
            visible_window=12,
            success_radius_frac=0.10,
        )
        env = EyeEnv(cfg)
        env.reset(seed=seed)
        return env
    return _f

def run(args):
    # env must match eval cadence; we’ll use smaller steps to reduce overshoot
    env = DummyVecEnv([make_env(args.obs_size, args.frame_skip, args.step_deg,
                                args.target, args.discrete, args.reward_mode,
                                args.reset_aim_prob, args.seed)])
    env = VecTransposeImage(env)

    print(f"[RESUME] loading {args.ckpt}")
    model = PPO.load(args.ckpt, env=env, device="auto")
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    print(f"[SAVE] {args.out}")
    model.save(args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path WITHOUT .zip or with .zip")
    ap.add_argument("--out",  type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=150000)

    # cadence (match eval; you can tighten to reduce overshoot)
    ap.add_argument("--obs-size", type=int, default=128)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--step-deg", type=float, default=2.0)

    # env knobs
    ap.add_argument("--target", type=str, default="ball", choices=["ball","bucket"])
    ap.add_argument("--reward-mode", type=str, default="vision", choices=["vision","state"])
    ap.add_argument("--discrete", action="store_true", default=True)
    ap.add_argument("--reset-aim-prob", type=float, default=1.0)  # start easy
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)
