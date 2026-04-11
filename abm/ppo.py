"""
abm/ppo.py — Minimal self-contained PPO for discrete action spaces.

Used as System B in the A-B-M loop.  Takes pre-computed latent
embeddings (from a frozen LeWM encoder) as state — no CNN inside.

Based on CleanRL PPO (vwxyzjn/cleanrl) condensed to ~200 lines.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Policy + Value networks
# ---------------------------------------------------------------------------

class PPOAgent(nn.Module):
    """
    Separate actor / critic networks operating on latent features.

    Input:  z (B, latent_dim)
    Actor:  → logits (B, n_actions)
    Critic: → value  (B, 1)
    """

    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Orthogonal init (recommended for PPO)
        for layer in [*self.actor, *self.critic]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def get_action_and_value(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        logits = self.actor(z)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)

    def get_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a single PPO rollout of fixed length."""

    def __init__(self, n_steps: int, latent_dim: int, device: str):
        self.n_steps    = n_steps
        self.latent_dim = latent_dim
        self.device     = device
        self._ptr       = 0
        self._full      = False

        self.latents  = torch.zeros(n_steps, latent_dim, device=device)
        self.actions  = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.rewards  = torch.zeros(n_steps, device=device)
        self.dones    = torch.zeros(n_steps, device=device)
        self.values   = torch.zeros(n_steps, device=device)

    def add(self, z, action, log_prob, reward, done, value):
        i = self._ptr % self.n_steps
        self.latents[i]   = z
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full

    def reset(self):
        self._ptr  = 0
        self._full = False

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma:  float = 0.99,
        gae_lam: float = 0.95,
    ) -> tuple:
        """Generalised Advantage Estimation."""
        adv  = torch.zeros_like(self.rewards)
        last = torch.zeros(1, device=self.device)

        for t in reversed(range(self.n_steps)):
            next_val   = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done  = self.dones[t]
            delta      = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            adv[t]     = last = delta + gamma * gae_lam * (1 - next_done) * last

        returns = adv + self.values
        return adv, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

class PPO:
    """
    Proximal Policy Optimisation trainer.

    Typical usage in ACT mode:
        ppo = PPO(agent, lr=2.5e-4)
        while not rollout.is_full:
            z = encoder(obs)
            a, lp, _, v = agent.get_action_and_value(z)
            rollout.add(z, a, lp, reward, done, v)
        ppo.update(rollout, last_z)
    """

    CLIP_EPS   = 0.2
    ENT_COEF   = 0.01
    VF_COEF    = 0.5
    MAX_GRAD   = 0.5
    N_EPOCHS   = 4
    MINI_BATCH = 64

    def __init__(self, agent: PPOAgent, lr: float = 2.5e-4):
        self.agent = agent
        self.opt   = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    @torch.no_grad()
    def _last_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.agent.get_value(z)

    def update(
        self,
        buf:       RolloutBuffer,
        last_z:    torch.Tensor,
        last_done: bool,
    ) -> dict:
        last_val = self._last_value(last_z) * (1 - float(last_done))
        adv, returns = buf.compute_gae(last_val)

        # Normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten for mini-batch sampling
        b_lat   = buf.latents.detach()
        b_act   = buf.actions.detach()
        b_lp    = buf.log_probs.detach()
        b_ret   = returns.detach()
        b_adv   = adv.detach()

        n       = buf.n_steps
        indices = np.arange(n)
        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(self.N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, n, self.MINI_BATCH):
                mb = indices[start: start + self.MINI_BATCH]

                _, new_lp, ent, new_val = self.agent.get_action_and_value(
                    b_lat[mb], b_act[mb]
                )
                ratio = (new_lp - b_lp[mb]).exp()

                pg_loss1 = -b_adv[mb] * ratio
                pg_loss2 = -b_adv[mb] * ratio.clamp(1 - self.CLIP_EPS, 1 + self.CLIP_EPS)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = 0.5 * F.mse_loss(new_val, b_ret[mb])
                ent_loss = ent.mean()

                loss = pg_loss + self.VF_COEF * vf_loss - self.ENT_COEF * ent_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.MAX_GRAD)
                self.opt.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        buf.reset()
        return {
            "pg_loss":  np.mean(pg_losses),
            "vf_loss":  np.mean(vf_losses),
            "ent_loss": np.mean(ent_losses),
        }


# fix missing import inside the class method
import torch.nn.functional as F  # noqa: E402 (already imported via nn but explicit is safer)
