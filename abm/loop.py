"""
abm/loop.py — Main A↔B training loop.

run_abm_loop(condition, device, max_steps, seed) → metrics dict

Three conditions:
  "autonomous" — AutonomousSystemM (plateau-triggered switching)
  "fixed"      — FixedSystemM (switch every K steps)
  "ppo_only"   — Raw-pixel PPO baseline (no LeWM, no mode switching)
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium
from minigrid.wrappers import RGBImgObsWrapper

from .lewm import LeWM, ReplayBuffer
from .ppo import PPO, PPOAgent, RolloutBuffer
from .meta_controller import AutonomousSystemM, FixedSystemM, Mode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LATENT_DIM  = 128
N_ACTIONS   = 7
IMG_H = IMG_W = 48      # MiniGrid-DoorKey-6x6 with tile_size=8

LEWM_LR     = 3e-4
LEWM_BATCH  = 64
LEWM_WARMUP = 200       # fill buffer before training LeWM

PPO_LR          = 2.5e-4
PPO_ROLLOUT     = 2_048  # steps per PPO update — larger batches, more stable gradients
EVAL_INTERVAL   = 5_000
EVAL_EPISODES   = 50
SUCCESS_REWARD  = 1.0   # DoorKey gives 1.0 on success minus step penalty

FIXED_SWITCH_EVERY   = 10_000   # longer phases → more PPO updates per ACT block


# ---------------------------------------------------------------------------
# Reward shaping wrapper
# ---------------------------------------------------------------------------

class ShapedRewardWrapper(gymnasium.Wrapper):
    """
    Adds small intermediate rewards to DoorKey's sparse signal:
      +0.1 first time agent picks up the key
      +0.1 first time agent opens the door
    Terminal reward (+1.0 on goal) is unchanged.
    """
    KEY_BONUS  = 0.1
    DOOR_BONUS = 0.1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._has_key     = False
        self._door_opened = False
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        inner = self.env.unwrapped

        if not self._has_key and inner.carrying is not None:
            reward += self.KEY_BONUS
            self._has_key = True

        if not self._door_opened:
            for obj in inner.grid.grid:
                if obj is not None and obj.type == "door" and obj.is_open:
                    reward += self.DOOR_BONUS
                    self._door_opened = True
                    break

        return obs, reward, term, trunc, info


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0):
    env = gymnasium.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = ShapedRewardWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=8)
    env.reset(seed=seed)
    return env


def _obs_to_tensor(obs_dict, device: str) -> torch.Tensor:
    """Convert MiniGrid RGBImgObsWrapper obs dict → (1, 3, H, W) float32."""
    img = obs_dict["image"]                          # (H, W, 3) uint8
    x   = torch.from_numpy(img).float() / 255.0     # (H, W, 3) [0,1]
    return x.permute(2, 0, 1).unsqueeze(0).to(device)   # (1, 3, H, W)


def evaluate(env_fn, encoder, device: str, n_eps: int = EVAL_EPISODES) -> float:
    """Run n_eps episodes with greedy policy; return success rate."""
    successes = 0
    for ep in range(n_eps):
        env  = env_fn(seed=1000 + ep)
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                z = encoder(obs_to_chw(obs, device))
            # Greedy: argmax over actor logits — but we only have encoder here.
            # Actual eval policy is passed in separately; see run_abm_loop().
            break  # placeholder — overridden by caller
        env.close()
    return successes / n_eps


def obs_to_chw(obs_dict, device: str) -> torch.Tensor:
    """(H,W,3) uint8 dict entry → (1,3,H,W) float32 on device."""
    img = obs_dict["image"]
    x   = torch.from_numpy(img.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


def eval_agent(agent: PPOAgent, encoder, device: str, seed_offset: int = 1000,
               n_eps: int = EVAL_EPISODES, frozen: bool = True) -> float:
    """Evaluate PPO agent — returns success rate."""
    successes = 0
    for ep in range(n_eps):
        env  = make_env(seed=seed_offset + ep)
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        while not done and ep_steps < 300:
            with torch.no_grad():
                z = encoder(obs)
                action, _, _, _ = agent.get_action_and_value(z)
            obs, r, term, trunc, _ = env.step(action.item())
            done = term or trunc
            if term and r > 0:
                successes += 1
            ep_steps += 1
        env.close()
    return successes / n_eps


# ---------------------------------------------------------------------------
# Core loops for each condition
# ---------------------------------------------------------------------------

def _observe_phase(
    env,
    lewm: LeWM,
    opt_lewm: optim.Optimizer,
    buf: ReplayBuffer,
    device: str,
    n_steps: int,
    obs,
) -> tuple:
    """
    Run n_steps of OBSERVE mode:
      - random/ε-greedy exploration
      - store transitions in replay buffer
      - train LeWM every 4 steps once buffer is warm

    Returns: (final_obs, ssl_loss_history)
    """
    ssl_losses = []

    for _ in range(n_steps):
        # Random exploration
        action = env.action_space.sample()
        obs_np = obs["image"]
        next_obs_dict, _, term, trunc, _ = env.step(action)
        next_obs_np = next_obs_dict["image"]

        buf.push(obs_np, action, next_obs_np)
        obs = next_obs_dict

        if term or trunc:
            obs, _ = env.reset()

        # Train LeWM
        if len(buf) >= LEWM_WARMUP and len(buf) % 4 == 0:
            obs_t, acts, obs_next = buf.sample(LEWM_BATCH, device)
            opt_lewm.zero_grad()
            loss, info = lewm.loss(obs_t, acts, obs_next)
            loss.backward()
            opt_lewm.step()
            ssl_losses.append(info["loss_total"])

    return obs, ssl_losses


def _act_phase(
    env,
    agent: PPOAgent,
    ppo: PPO,
    encoder,
    buf_ppo: RolloutBuffer,
    device: str,
    n_steps: int,
    obs,
) -> tuple:
    """
    Run n_steps of ACT mode:
      - encoder is frozen
      - collect PPO rollout
      - update PPO every PPO_ROLLOUT steps

    Returns: (final_obs, episode_rewards, ppo_metrics_list)
    """
    ep_rewards  = []
    ep_ret      = 0.0
    ppo_metrics = []
    last_done   = False
    last_z      = None

    for _ in range(n_steps):
        with torch.no_grad():
            z = encoder(obs_to_chw(obs, device))
            action, log_prob, _, value = agent.get_action_and_value(z)

        last_z = z.squeeze(0)
        next_obs_dict, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc
        ep_ret += reward

        buf_ppo.add(
            z.squeeze(0).detach(),
            action.squeeze(0),
            log_prob.squeeze(0),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(float(done)),
            value.squeeze(0),
        )

        obs = next_obs_dict
        last_done = done

        if done:
            ep_rewards.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if buf_ppo.is_full:
            with torch.no_grad():
                lz = encoder(obs_to_chw(obs, device)).squeeze(0)
            metrics = ppo.update(buf_ppo, lz, last_done)
            ppo_metrics.append(metrics)

    return obs, ep_rewards, ppo_metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_abm_loop(
    condition: str,
    device:    str  = "mps",
    max_steps: int  = 100_000,
    seed:      int  = 42,
    log_every: int  = 1_000,
) -> Dict:
    """
    Run a single condition of the A-B-M experiment.

    Parameters
    ----------
    condition : "autonomous" | "fixed" | "ppo_only"
    device    : "mps" | "cpu"
    max_steps : total environment steps
    seed      : random seed
    log_every : print log every N steps

    Returns
    -------
    metrics dict with keys:
      env_steps, success_rate, ssl_loss, mode, mode_switches, wall_time_s
      steps_to_80pct (int or None)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"[{condition.upper()}] Starting — device={device}, max_steps={max_steps}")
    t0 = time.time()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # faster fixed-size convolutions

    # ---- Setup ----
    env = make_env(seed=seed)
    obs, _ = env.reset()

    lewm    = LeWM(latent_dim=LATENT_DIM, n_actions=N_ACTIONS).to(device)
    agent   = PPOAgent(latent_dim=LATENT_DIM if condition != "ppo_only" else IMG_H * IMG_W * 3,
                       n_actions=N_ACTIONS).to(device)
    ppo     = PPO(agent, lr=PPO_LR)
    buf_lew = ReplayBuffer(capacity=10_000)
    buf_ppo = RolloutBuffer(PPO_ROLLOUT, LATENT_DIM, device)

    opt_lewm = optim.Adam(lewm.parameters(), lr=LEWM_LR)

    # System M
    if condition == "autonomous":
        sysm = AutonomousSystemM(
            obs_plateau_steps=8_000,
            act_plateau_steps=20_000,
            plateau_threshold=0.01,
        )
    elif condition == "fixed":
        sysm = FixedSystemM(switch_every=FIXED_SWITCH_EVERY)
    else:
        sysm = None   # ppo_only

    # ---- Metric tracking ----
    metrics: Dict[str, List] = {
        "env_step":     [],
        "success_rate": [],
        "ssl_loss_ewa": [],
        "mode":         [],
        "wall_time_s":  [],
    }
    ssl_ewa      = None
    mode_str     = "OBSERVE" if condition != "ppo_only" else "ACT"
    steps_to_80  = None
    ep_rew_buf   = []   # recent episode rewards for mode-switch signal
    env_step     = 0

    # For ppo_only: use raw pixel MLP (flattened 48×48×3 = 6912-dim state)
    if condition == "ppo_only":
        flat_dim = IMG_H * IMG_W * 3
        agent    = PPOAgent(latent_dim=flat_dim, n_actions=N_ACTIONS).to(device)
        ppo      = PPO(agent, lr=PPO_LR)
        buf_ppo  = RolloutBuffer(PPO_ROLLOUT, flat_dim, device)

        def raw_encode(obs_dict):
            img = obs_dict["image"]
            x   = torch.from_numpy(img.astype(np.float32) / 255.0)
            return x.flatten().unsqueeze(0).to(device)

        encoder = raw_encode
    else:
        encoder = lambda obs_dict: lewm.encode(obs_to_chw(obs_dict, device))

    # ---- Main loop ----
    last_done  = False
    ep_ret     = 0.0

    while env_step < max_steps:

        # ── Determine current mode ──────────────────────────────────────────
        if condition == "ppo_only":
            current_mode = Mode.ACT
        elif condition == "autonomous":
            current_mode = sysm.mode
        else:  # fixed
            current_mode = sysm.step(env_step)

        mode_str = current_mode.name if current_mode else "ACT"

        # ── One environment step ────────────────────────────────────────────
        if current_mode == Mode.OBSERVE:
            # Random exploration + LeWM training
            action  = env.action_space.sample()
            obs_np  = obs["image"]
            nobs, _, term, trunc, _ = env.step(action)
            buf_lew.push(obs_np, action, nobs["image"])
            obs      = nobs
            env_step += 1

            if term or trunc:
                obs, _ = env.reset()

            # Train LeWM
            ssl_loss_val = None
            if len(buf_lew) >= LEWM_WARMUP and env_step % 4 == 0:
                obs_t, acts, obs_next = buf_lew.sample(LEWM_BATCH, device)
                opt_lewm.zero_grad()
                loss, info = lewm.loss(obs_t, acts, obs_next)
                loss.backward()
                opt_lewm.step()
                ssl_loss_val = info["loss_total"]
                ssl_ewa = ssl_loss_val if ssl_ewa is None else 0.95 * ssl_ewa + 0.05 * ssl_loss_val

            # Notify autonomous System M
            if condition == "autonomous" and ssl_loss_val is not None:
                sysm.observe_step(ssl_loss_val, env_step)

        else:
            # ACT mode — PPO step
            with torch.no_grad():
                z = encoder(obs)
            action, log_prob, _, value = agent.get_action_and_value(z)

            nobs, reward, term, trunc, _ = env.step(action.item())
            done   = term or trunc
            ep_ret += reward
            env_step += 1

            z_flat = z.squeeze(0) if z.ndim > 1 else z
            buf_ppo.add(
                z_flat.detach(),
                action.squeeze(0) if action.ndim > 0 else action,
                log_prob.squeeze(0) if log_prob.ndim > 0 else log_prob,
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(float(done)),
                value.squeeze(0) if value.ndim > 0 else value,
            )

            obs = nobs
            last_done = done

            if done:
                ep_rew_buf.append(ep_ret)
                if condition == "autonomous":
                    sysm.act_step(ep_ret, None, env_step)
                ep_ret = 0.0
                obs, _ = env.reset()

            if buf_ppo.is_full:
                with torch.no_grad():
                    lz = encoder(obs)
                lz_flat = lz.squeeze(0) if lz.ndim > 1 else lz
                ppo.update(buf_ppo, lz_flat, last_done)

        # ── Periodic evaluation ─────────────────────────────────────────────
        if env_step % EVAL_INTERVAL == 0:
            # Freeze LeWM encoder during eval (doesn't apply to ppo_only lambda)
            if condition != "ppo_only":
                for p in lewm.encoder.parameters():
                    p.requires_grad_(False)

            sr = eval_agent(agent, encoder, device, seed_offset=9000 + env_step,
                            n_eps=EVAL_EPISODES)

            if condition != "ppo_only":
                for p in lewm.encoder.parameters():
                    p.requires_grad_(True)

            # Autonomous: pass success_rate to System M
            if condition == "autonomous":
                sysm.act_step(None, sr, env_step)

            n_sw    = sysm.n_switches() if sysm else 0
            elapsed = time.time() - t0

            logger.info(
                f"[{condition.upper()}] step={env_step:6d} | mode={mode_str:7s} | "
                f"success={sr:.1%} | ssl_ewa={ssl_ewa or 0.0:.4f} | "
                f"switches={n_sw} | {elapsed:.0f}s elapsed"
            )

            metrics["env_step"].append(env_step)
            metrics["success_rate"].append(sr)
            metrics["ssl_loss_ewa"].append(ssl_ewa or 0.0)
            metrics["mode"].append(mode_str)
            metrics["wall_time_s"].append(elapsed)

            if steps_to_80 is None and sr >= 0.80:
                steps_to_80 = env_step
                logger.info(f"[{condition.upper()}] *** 80% success reached at step {env_step} ***")

        # Early stop if solved
        if sysm is not None and sysm.is_solved:
            logger.info(f"[{condition.upper()}] Solved! Stopping at step {env_step}.")
            break

    env.close()
    elapsed_total = time.time() - t0

    # Save checkpoint so episodes can be replayed later
    ckpt_dir = Path("results/abm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if condition == "ppo_only":
        ckpt = {
            "condition": condition,
            "agent":     agent.state_dict(),
            "flat_dim":  IMG_H * IMG_W * 3,
            "env_step":  env_step,
        }
    else:
        ckpt = {
            "condition":  condition,
            "lewm":       lewm.state_dict(),
            "agent":      agent.state_dict(),
            "latent_dim": LATENT_DIM,
            "n_actions":  N_ACTIONS,
            "env_step":   env_step,
        }
    ckpt_path = ckpt_dir / f"checkpoint_{condition}.pt"
    torch.save(ckpt, ckpt_path)
    logger.info(f"[{condition.upper()}] Checkpoint saved → {ckpt_path}")

    return {
        "condition":      condition,
        "env_steps":      metrics["env_step"],
        "success_rate":   metrics["success_rate"],
        "ssl_loss_ewa":   metrics["ssl_loss_ewa"],
        "mode":           metrics["mode"],
        "wall_time_s":    metrics["wall_time_s"],
        "steps_to_80pct": steps_to_80,
        "n_switches":     sysm.n_switches() if sysm else 0,
        "switch_log":     sysm.switch_log if sysm else [],
        "total_time_s":   elapsed_total,
    }
