# eye_env.py
# MuJoCo Eye-only environment for pan/tilt control.
# - State-based reward projects target into the real MuJoCo eye camera.
# - Optional occlusion-aware success: requires the target to be actually visible
#   (via a small color mask window around the projected pixel).
# - Vision reward also available as a fallback/debug option.
#
# macOS note: interactive viewer scripts must run via `mjpython`, but this file
# can be imported/used under plain python for training/headless eval.

import math, os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym
    gym = gym

import mujoco

# ---------- defaults / constants ----------
DEFAULT_MODEL = "panda_assets/mjx_panda.xml"
OBS_SIZE     = 128
FRAME_SKIP   = 5
MAX_STEPS    = 300
STEP_DEG     = 6.0

PAN_LIMIT   = (-math.pi, math.pi)
TILT_LIMIT  = (-math.pi/2, math.pi/2)

TARGET_CHOICES = ("ball", "bucket")


# ---------- config ----------
@dataclass
class EyeEnvConfig:
    model_path: str = DEFAULT_MODEL
    obs_size: int = OBS_SIZE
    frame_skip: int = FRAME_SKIP
    max_steps: int = MAX_STEPS
    step_deg: float = STEP_DEG
    discrete_actions: bool = True         # 9-way discrete (recommended)
    target: str = "ball"                  # "ball" or "bucket"
    reward_mode: str = "state"            # "state" (recommended) or "vision"
    success_radius_frac: float = 0.12     # success if within this fraction of half-diagonal
    debug_dump_every: int = 0             # save PNG every N steps (0=off)
    reset_aim_at_target_prob: float = 0.75

    # NEW: require that the target is actually visible near the projected pixel to call SUCCESS.
    require_visible: bool = False
    visible_window: int = 10              # pixel radius around (u,v) for the visibility check


# ---------- env ----------
class EyeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(500 / FRAME_SKIP)}

    def __init__(self, config: Optional[EyeEnvConfig] = None):
        super().__init__()
        self.cfg = config or EyeEnvConfig()

        # load model/data
        xml = os.path.abspath(self.cfg.model_path)
        if not os.path.exists(xml):
            raise FileNotFoundError(f"XML not found: {xml}")
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # IDs: eye joints/acts/camera
        self.eye_pan_act  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_pan_act")
        self.eye_tilt_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "eye_tilt_act")
        self.eye_pan_jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    "eye_pan")
        self.eye_tilt_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    "eye_tilt")
        self.eye_camid    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,   "eye")
        if min(self.eye_pan_act, self.eye_tilt_act, self.eye_pan_jid, self.eye_tilt_jid, self.eye_camid) < 0:
            raise RuntimeError("Eye actuators/joints/camera not found in XML (eye_pan/_tilt + camera 'eye').")

        self.pan_qadr  = self.model.jnt_qposadr[self.eye_pan_jid]
        self.tilt_qadr = self.model.jnt_qposadr[self.eye_tilt_jid]

        # target body id
        if self.cfg.target not in TARGET_CHOICES:
            raise ValueError(f"target must be one of {TARGET_CHOICES}")
        self._target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.target)
        if self._target_bid < 0:
            raise RuntimeError(f"Target body '{self.cfg.target}' not found in XML.")

        # renderer (eye camera)
        self.r_eye = mujoco.Renderer(self.model, width=self.cfg.obs_size, height=self.cfg.obs_size)

        # action/observation spaces
        if self.cfg.discrete_actions:
            # 9 actions: stay, +/- pan, +/- tilt, 4 diagonals
            self._dirs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
            self.action_space = gym.spaces.Discrete(len(self._dirs))
        else:
            lim = math.radians(self.cfg.step_deg)
            self.action_space = gym.spaces.Box(
                low=np.array([-lim, -lim], dtype=np.float32),
                high=np.array([+lim, +lim], dtype=np.float32),
                dtype=np.float32
            )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.cfg.obs_size, self.cfg.obs_size, 3), dtype=np.uint8
        )

        # runtime
        self._rng = np.random.RandomState(42)
        self._step = 0
        self._step_rad = math.radians(self.cfg.step_deg)
        self._pan_range, self._tilt_range = PAN_LIMIT, TILT_LIMIT

        # camera intrinsics (vertical fov; square render so fx == fy)
        self._fovy = float(self.model.cam_fovy[self.eye_camid])
        self._fy = 0.5 * self.cfg.obs_size / math.tan(math.radians(self._fovy) * 0.5)
        self._fx = self._fy
        self._cx = self.cfg.obs_size * 0.5
        self._cy = self.cfg.obs_size * 0.5

        self._dump_idx = 0

    # ----- gym api -----
    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(None if seed is None else seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self._step = 0
        mujoco.mj_forward(self.model, self.data)

        # start near the target sometimes (smoke-test friendly)
        pan0  = float(self._rng.uniform(-0.6, 0.6))
        tilt0 = float(self._rng.uniform(-0.25, 0.25))
        if self._rng.rand() < self.cfg.reset_aim_at_target_prob:
            pan_a, tilt_a = self._angles_to_body(self._target_bid)
            if not np.isnan(pan_a):
                pan0  = pan_a  + float(self._rng.uniform(-0.10, 0.10))
                tilt0 = tilt_a + float(self._rng.uniform(-0.06, 0.06))

        self._set_eye_angles(pan0, tilt0)
        mujoco.mj_forward(self.model, self.data)
        obs = self._render_eye()
        return obs, {"pan_deg": float(np.degrees(pan0)), "tilt_deg": float(np.degrees(tilt0))}

    def step(self, action):
        pan, tilt = self._get_eye_angles()

        if self.cfg.discrete_actions:
            dx, dy = self._dirs[int(action)]
            dpan, dtilt = self._step_rad * dx, self._step_rad * dy
        else:
            dpan, dtilt = np.asarray(action, dtype=np.float32)
            dpan  = float(np.clip(dpan,  -self._step_rad, self._step_rad))
            dtilt = float(np.clip(dtilt, -self._step_rad, self._step_rad))

        pan  = float(np.clip(pan + dpan,  self._pan_range[0],  self._pan_range[1]))
        tilt = float(np.clip(tilt + dtilt, self._tilt_range[0], self._tilt_range[1]))

        self.data.ctrl[self.eye_pan_act]  = pan
        self.data.ctrl[self.eye_tilt_act] = tilt

        # advance physics
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step += 1
        obs = self._render_eye()

        # compute reward/success
        if self.cfg.reward_mode == "state":
            reward, success = self._reward_state_projection()
        else:
            reward, success = self._reward_vision(obs)

        term  = bool(success)                            # success ends episode
        trunc = bool(self._step >= self.cfg.max_steps and not term)

        # optional debug frame dump
        if self.cfg.debug_dump_every and (self._step % self.cfg.debug_dump_every == 0):
            os.makedirs("snaps", exist_ok=True)
            from PIL import Image, ImageDraw
            im = Image.fromarray(obs)
            draw = ImageDraw.Draw(im)
            draw.ellipse((self._cx - 5, self._cy - 5, self._cx + 5, self._cy + 5),
                         outline=(255, 255, 0), width=2)
            im.save(f"snaps/eye_{self._dump_idx:06d}.png")
            self._dump_idx += 1

        info = {"pan_deg": float(np.degrees(pan)), "tilt_deg": float(np.degrees(tilt))}
        info["success"] = bool(term)
        return obs, float(reward), term, trunc, info

    def render(self):
        return self._render_eye()

    def close(self):
        try:
            self.r_eye.close()
        except Exception:
            pass

    # ----- helpers -----
    def _get_eye_angles(self) -> Tuple[float, float]:
        return float(self.data.qpos[self.pan_qadr]), float(self.data.qpos[self.tilt_qadr])

    def _set_eye_angles(self, pan: float, tilt: float):
        self.data.qpos[self.pan_qadr]  = float(np.clip(pan,  *self._pan_range))
        self.data.qpos[self.tilt_qadr] = float(np.clip(tilt, *self._tilt_range))
        mujoco.mj_forward(self.model, self.data)
        # keep actuators in sync (position servos)
        self.data.ctrl[self.eye_pan_act]  = float(self.data.qpos[self.pan_qadr])
        self.data.ctrl[self.eye_tilt_act] = float(self.data.qpos[self.tilt_qadr])

    def _render_eye(self) -> np.ndarray:
        self.r_eye.update_scene(self.data, camera=self.eye_camid)
        img = self.r_eye.render()
        return (np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img)

    # ---- occlusion-aware state reward ----
    def _reward_state_projection(self) -> Tuple[float, bool]:
        """
        Projects the target's world position into the eye image using MuJoCo's actual
        camera pose (cam_xpos/cam_xmat) and vertical fov. Optionally requires the
        target-colored pixels to be visible near the projected pixel to declare SUCCESS.
        """
        mujoco.mj_forward(self.model, self.data)

        # camera pose
        cpos = self.data.cam_xpos[self.eye_camid].copy()
        cmat = self.data.cam_xmat[self.eye_camid].reshape(3, 3).copy()
        # interpret columns as camera axes in world; MuJoCo cameras look along -Z
        x_c, y_c, z_c = cmat[:, 0], cmat[:, 1], cmat[:, 2]

        tgt_w = self.data.xpos[self._target_bid].copy()
        d = tgt_w - cpos  # world vector from cam to target

        x_cam = float(np.dot(x_c, d))
        y_cam = float(np.dot(y_c, d))
        z_fwd = float(-np.dot(z_c, d))  # depth along viewing direction (-Z)

        if z_fwd <= 1e-6:
            return 0.0, False

        u = self._fx * (x_cam / z_fwd) + self._cx
        v = self._fy * (y_cam / z_fwd) + self._cy
        if not (0 <= u < self.cfg.obs_size and 0 <= v < self.cfg.obs_size):
            return 0.0, False

        # radial reward toward image center
        dist  = math.hypot(u - self._cx, v - self._cy)
        max_d = 0.5 * math.hypot(self.cfg.obs_size, self.cfg.obs_size)
        r = max(0.0, 1.0 - dist / max_d)
        success = dist <= self.cfg.success_radius_frac * max_d

        # gate success on actual visibility if requested
        if success and self.cfg.require_visible:
            img = self._render_eye()
            if not self._visible_near_uv(img, u, v, int(self.cfg.visible_window)):
                success = False

        return r, bool(success)

    def _visible_near_uv(self, img: np.ndarray, u: float, v: float, rad: int) -> bool:
        """
        Simple color-based visibility check near projected (u,v).
        Returns True if a patch around (u,v) contains target-colored pixels.
        """
        h, w, _ = img.shape
        x0 = max(0, int(u) - rad); x1 = min(w, int(u) + rad + 1)
        y0 = max(0, int(v) - rad); y1 = min(h, int(v) + rad + 1)
        if x0 >= x1 or y0 >= y1:
            return False
        patch = img[y0:y1, x0:x1, :].astype(np.float32)
        r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
        s = r + g + b + 1e-3
        rn, gn, bn = r / s, g / s, b / s
        if self.cfg.target == "ball":          # red ball
            mask = (rn > 0.45) & (gn < 0.35) & (bn < 0.35)
        else:                                   # blue bucket
            mask = (bn > 0.42) & (gn < 0.40) & (rn < 0.40)
        return bool(np.any(mask))

    # ---- optional pure-vision reward ----
    def _reward_vision(self, img: np.ndarray) -> Tuple[float, bool]:
        h, w, _ = img.shape
        cx, cy = w * 0.5, h * 0.5
        r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
        s = r + g + b + 1e-3
        rn, gn, bn = r / s, g / s, b / s

        if self.cfg.target == "ball":
            mask = (rn > 0.45) & (gn < 0.35) & (bn < 0.35)
        else:
            mask = (bn > 0.42) & (gn < 0.40) & (rn < 0.40)

        if not np.any(mask):
            return 0.0, False

        ys, xs = np.nonzero(mask)
        x_mean, y_mean = float(xs.mean()), float(ys.mean())
        dist  = math.hypot(x_mean - cx, y_mean - cy)
        max_d = 0.5 * math.hypot(w, h)
        r = max(0.0, 1.0 - dist / max_d)
        success = dist <= self.cfg.success_radius_frac * max_d
        return r, bool(success)

    # ---- utility: approximate pan/tilt toward a body (for reset) ----
    def _angles_to_body(self, body_id: int) -> Tuple[float, float]:
        mujoco.mj_forward(self.model, self.data)
        cpos = self.data.cam_xpos[self.eye_camid].copy()
        tgt  = self.data.xpos[body_id].copy()
        v = tgt - cpos
        pan  = math.atan2(v[1], v[0])
        horiz = max(1e-6, math.hypot(v[0], v[1]))
        tilt = math.atan2(v[2], horiz)
        return pan, tilt


# ---------- smoke test ----------
if __name__ == "__main__":
    # Force success to require visibility so the terminal condition matches what you see.
    env = EyeEnv(EyeEnvConfig(
        reward_mode="state",
        debug_dump_every=25,
        reset_aim_at_target_prob=0.9,
        require_visible=True,
        visible_window=10,
    ))
    obs, info = env.reset(seed=0)
    total = 0.0
    term = trunc = False
    for t in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"Smoke test: steps={t+1}, total_reward={total:.3f}, term={term}, trunc={trunc}")
    env.close()
