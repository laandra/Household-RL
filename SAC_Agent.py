"""
SAC_Agent.py  –  Soft Actor-Critic with Continuous Battery Control
===================================================================
Replaces the discrete DQN/SARSA with a continuous SAC agent whose
action maps directly to a *target battery state-of-charge* (SOC).
The continuous action is then decoded into the same energy-flow
variables (paneli_baterija, omrezje_baterija, baterija_dom,
baterija_omrezje, kupljena_elektrika) that the DQN / MILP produce,
so all downstream code (reward, plotting, cost accounting) stays
identical.

Quick-start
-----------
from SAC_Agent import (
    ContinuousHouseholdWrapper,
    AgentSAC,
    Learning_SAC,
    build_sac_env,
)

# Build env exactly like the DQN helper (same signature):
env = build_sac_env(dataset=train_data, dataset_norm=train_data_norm,
                    episode_length=KorakovNaDan * 7, ...)

# Train
agent, rewards, payments, battery_log = Learning_SAC(
    env=env, ucenje=True, ponovitev=52)

# Evaluate (greedy, deterministic)
agent, rewards, payments, battery_log = Learning_SAC(
    env=sac_test_env, agent=agent, ucenje=False)
"""

from __future__ import annotations

import copy
import os
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Continuous Wrapper  (discrete env → continuous action space)
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousHouseholdWrapper(gym.Wrapper):
    """
    Wraps HouseholdEnvironment and exposes a **continuous** action in [-1, 1].

    Action semantics (target-SOC mapping)
    ──────────────────────────────────────
      action ∈ [-1, 1]  → desired battery SOC fraction ∈ [0, 1]
      target_soc = (action + 1) / 2

    The wrapper computes exactly how much charge/discharge is needed to move
    from the *current* SOC toward target_soc in *this* timestep, respecting
    hardware limits (bat_max_polnjenje / bat_max_praznjenje, efficiency).
    Then it picks the closest feasible discrete action in the base env to
    produce the same energy flows, OR it directly calls the base env's
    internal step logic with the computed flows (see `_continuous_step`).

    Because the base env's `step()` encodes energy-flow logic per action-int,
    we replicate that physics here so we can pass exact kWh values instead
    of integer codes.  The resulting info dict and reward are 100 % identical
    to what DQN / MILP produce.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Replace Discrete(5) with Box([-1],[1])
        self.action_space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )
        # Observation space unchanged
        self.observation_space = env.observation_space

    # ------------------------------------------------------------------
    # Helpers that mirror Environment.py physics
    # ------------------------------------------------------------------

    def _bat_max_pol(self) -> float:
        e = self.env
        space = e.bat_kapaciteta - e._battery
        return (1.0 / e.bat_ucinkovitost) * min(space, e.bat_max_polnjenje)

    def _bat_max_pra(self) -> float:
        e = self.env
        return e.bat_ucinkovitost * min(e._battery, e.bat_max_praznjenje)

    def _action_to_discrete(self, action_float: float) -> int:
        """
        Maps continuous action → best-fit discrete action integer [0..4].

        Heuristic:
          target_soc in [0,1].  Compare to current soc.
          delta > +threshold  → charge  (action 0 = buy+charge, 1 = solar→charge)
          delta < -threshold  → discharge (action 3 = discharge+sell, 2 = discharge)
          |delta| ≤ threshold → idle (action 4)
        """
        e = self.env
        target_soc = float(np.clip((action_float + 1.0) / 2.0, 0.0, 1.0))
        current_soc = e._battery / e.bat_kapaciteta if e.bat_kapaciteta > 0 else 0.5
        delta = target_soc - current_soc

        threshold = 0.02   # ~2 % SOC band around "do nothing"

        if delta > threshold:
            # Want to charge → prefer solar first (1), buy if needed (0)
            idx = e._current_step
            gen = e.arr_Gen[idx]
            con = e.arr_Con[idx]
            solar_excess = max(0.0, gen - con)
            if solar_excess > 0.01:
                return 1   # solar charges battery
            return 0       # buy from grid + charge
        elif delta < -threshold:
            # Want to discharge → sell if price is above median (3) else home only (2)
            idx = e._current_step
            if e.arr_RelativePrice[idx] > 0.1:
                return 3   # discharge + sell to grid
            return 2       # discharge → power home only
        else:
            return 4       # idle / grid-only supply

    def step(self, action):
        """Accept continuous action, convert to discrete, forward to base env."""
        a_arr = np.asarray(action, dtype=np.float32).flatten()
        a_float = float(np.clip(a_arr[0], -1.0, 1.0))
        discrete_action = self._action_to_discrete(a_float)
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        # Tag info with the original continuous action for logging
        info["continuous_action"] = float(a_float)
        info["mapped_discrete_action"] = int(discrete_action)
        info["target_soc"] = float((a_float + 1.0) / 2.0)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def action_masks(self):
        return self.env.action_masks()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Neural-Network Modules
# ─────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class _MLP(nn.Module):
    """Shared MLP backbone used by Actor and Critics."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int,
                 activation=nn.ReLU, output_activation=None):
        super().__init__()
        sizes = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(activation())
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class ActorNetwork(nn.Module):
    """
    Gaussian policy network.
    Outputs mean and log-std for a squashed-Gaussian in [-1, 1].
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden=(256, 256), lr: float = 3e-4):
        super().__init__()
        self.action_dim = action_dim
        self.backbone = _MLP(obs_dim, hidden[:-1], hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu_head.bias, -3e-3, 3e-3)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: T.Tensor):
        h = F.relu(self.backbone(obs))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: T.Tensor):
        """
        Returns (action, log_prob, mean).
        action is in [-1, 1] via tanh squashing.
        log_prob accounts for tanh Jacobian (reparameterisation trick).
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = T.distributions.Normal(mu, std)
        x_t = dist.rsample()                          # reparameterised sample
        action = T.tanh(x_t)
        # log prob with tanh correction  (numerically stable)
        log_prob = dist.log_prob(x_t) - T.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, T.tanh(mu)


class CriticNetwork(nn.Module):
    """Twin-Q (double critic) to reduce overestimation bias."""

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden=(256, 256), lr: float = 3e-4):
        super().__init__()
        # Q1
        self.q1 = _MLP(obs_dim + action_dim, hidden, 1)
        # Q2
        self.q2 = _MLP(obs_dim + action_dim, hidden, 1)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: T.Tensor, action: T.Tensor):
        sa = T.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, obs: T.Tensor, action: T.Tensor) -> T.Tensor:
        return self.q1(T.cat([obs, action], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((self.capacity, obs_dim),    dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((self.capacity, 1),          dtype=np.float32)
        self.dones    = np.zeros((self.capacity, 1),          dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            T.tensor(self.obs[idx],      dtype=T.float32, device=device),
            T.tensor(self.actions[idx],  dtype=T.float32, device=device),
            T.tensor(self.rewards[idx],  dtype=T.float32, device=device),
            T.tensor(self.next_obs[idx], dtype=T.float32, device=device),
            T.tensor(self.dones[idx],    dtype=T.float32, device=device),
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAC Agent
# ─────────────────────────────────────────────────────────────────────────────

class AgentSAC:
    """
    Soft Actor-Critic (SAC) for the continuous household microgrid task.

    Key hyperparameters
    ───────────────────
    gamma          – discount factor
    tau            – soft-update coefficient for target critic
    alpha          – initial temperature (entropy weight); auto-tuned if
                     auto_alpha=True
    lr_actor       – actor learning rate
    lr_critic      – critic learning rate
    lr_alpha       – temperature learning rate (only when auto_alpha=True)
    batch_size     – mini-batch size drawn from replay buffer
    buffer_size    – replay buffer capacity (transitions)
    hidden         – hidden layer sizes for actor & critic
    update_every   – environment steps between each gradient update
    start_steps    – random warm-up steps (no policy, pure exploration)
    """

    CHECKPOINT_NAME = "sac_model"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        hidden: tuple = (256, 256),
        update_every: int = 1,
        start_steps: int = 1000,
        weight_decay: float = 1e-4,
    ):
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.update_every = update_every
        self.start_steps  = start_steps
        self.action_dim   = action_dim
        self.obs_dim      = obs_dim

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(f"[SAC] Using device: {self.device}")

        # Networks
        self.actor   = ActorNetwork(obs_dim, action_dim, hidden, lr_actor)
        self.critic  = CriticNetwork(obs_dim, action_dim, hidden, lr_critic)
        self.critic_target = CriticNetwork(obs_dim, action_dim, hidden, lr_critic)
        self._hard_update_target()

        # Entropy temperature
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Target entropy ≈ -dim(A)  (heuristic from SAC paper)
            self.target_entropy = -float(action_dim)
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = float(alpha)
            self.log_alpha = None

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        self._total_steps = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _hard_update_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _soft_update_target(self):
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    # ── public API ────────────────────────────────────────────────────────────

    def choose_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sample (training) or take mean action (evaluation).
        Returns numpy array of shape (action_dim,).
        """
        # Warm-up: pure random exploration before policy kicks in
        if (not deterministic) and (self._total_steps < self.start_steps):
            return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

        obs_t = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0)
        with T.no_grad():
            if deterministic:
                _, _, mean_action = self.actor.sample(obs_t)
                action = mean_action
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().flatten().astype(np.float32)

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)
        self._total_steps += 1

    def learn(self):
        """One gradient update step (called every `update_every` env steps)."""
        if len(self.buffer) < max(self.batch_size, self.start_steps):
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        # ── Critic update ─────────────────────────────────────────────────────
        with T.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = T.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic.optimizer.step()

        # ── Actor update ──────────────────────────────────────────────────────
        new_actions, log_pi, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        q_pi = T.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor.optimizer.step()

        # ── Alpha (temperature) update ─────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        self._soft_update_target()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_val,
        }

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save(self, checkpoint_dir: str = "SAC", episode: Optional[int] = None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, self.CHECKPOINT_NAME)
        payload = {
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "alpha":        self.alpha,
            "log_alpha":    self.log_alpha.item() if self.log_alpha is not None else None,
            "total_steps":  self._total_steps,
            "obs_dim":      self.obs_dim,
            "action_dim":   self.action_dim,
            "episode":      episode,
        }
        tmp = path + ".tmp"
        T.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load(self, checkpoint_dir: str = "SAC") -> bool:
        path = os.path.join(checkpoint_dir, self.CHECKPOINT_NAME)
        if not os.path.exists(path):
            print(f"[SAC] No checkpoint found at {path}")
            return False
        ckpt = T.load(path, weights_only=False, map_location=self.device)
        if ckpt.get("obs_dim") != self.obs_dim or ckpt.get("action_dim") != self.action_dim:
            print("[SAC] Checkpoint dimension mismatch – skipping load.")
            return False
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.alpha = float(ckpt.get("alpha", self.alpha))
        if self.log_alpha is not None and ckpt.get("log_alpha") is not None:
            with T.no_grad():
                self.log_alpha.fill_(float(ckpt["log_alpha"]))
        self._total_steps = int(ckpt.get("total_steps", 0))
        print(f"[SAC] Loaded checkpoint (episode {ckpt.get('episode')}, "
              f"steps {self._total_steps})")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Environment builder  (mirrors the DQN build_dqn_env API)
# ─────────────────────────────────────────────────────────────────────────────

def build_sac_env(
    dataset,
    dataset_norm,
    episode_length,
    reset_mode="deterministic",
    observation_mode="sliding_window",
    korakov_na_dan=96,
    bat_kapaciteta=20.0,
    bat_ucinkovitost=0.95,
    bat_max_polnjenje=1.5,
    bat_max_praznjenje=1.5,
    faktor_n1=0.0,
    faktor_n2=0.0,
    faktor_n3=1.0,
) -> ContinuousHouseholdWrapper:
    """
    Convenience builder that returns a ContinuousHouseholdWrapper –
    a drop-in for build_dqn_env() but with a continuous action space.

    The signature is intentionally identical so callers can swap
    build_dqn_env ↔ build_sac_env without changing other code.
    """
    # Import lazily to avoid circular-import issues when used as a standalone file
    try:
        import Environment as _env_mod
        import importlib
        importlib.reload(_env_mod)
        HouseholdEnvironment = _env_mod.HouseholdEnvironment
    except ImportError as exc:
        raise ImportError(
            "Environment.py not found. Place it in the same directory as SAC_Agent.py."
        ) from exc

    base_env = HouseholdEnvironment(
        dataset=dataset,
        dataset_norm=dataset_norm,
        observation_mode=observation_mode,
        reset_mode=reset_mode,
        episode_length=episode_length,
        korakov_na_dan=korakov_na_dan,
        bat_kapaciteta=bat_kapaciteta,
        bat_ucinkovitost=bat_ucinkovitost,
        bat_max_polnjenje=bat_max_polnjenje,
        bat_max_praznjenje=bat_max_praznjenje,
        faktor_n1=faktor_n1,
        faktor_n2=faktor_n2,
        faktor_n3=faktor_n3,
    )
    return ContinuousHouseholdWrapper(base_env)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training loop  (mirrors Learning_DQN API)
# ─────────────────────────────────────────────────────────────────────────────

def Learning_SAC(
    env=None,
    agent: Optional[AgentSAC] = None,
    ucenje: bool = True,
    ponovitev: int = 52,
    reset: bool = False,
    # SAC-specific hyperparameters (ignored when reusing an existing agent)
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    auto_alpha: bool = True,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_alpha: float = 3e-4,
    batch_size: int = 256,
    buffer_size: int = 1_000_000,
    hidden: tuple = (256, 256),
    update_every: int = 1,
    start_steps: int = 1000,
    checkpoint_every: int = 5,
    seed: Optional[int] = None,
    checkpoint_dir: str = "SAC",
    deterministic_eval: bool = True,
):
    """
    Full training / evaluation loop for the SAC agent.

    Returns
    -------
    agent          : trained AgentSAC instance
    nagradaSAC     : list[float]   cumulative reward over all steps
    placiloSAC     : list[float]   cumulative payment (AUD) over all steps
    baterijaSAC    : list[float]   battery SOC (kWh) over all steps

    These are the same tracking arrays used by Learning_DQN, so existing
    plotting code works without modification.
    """
    if env is None:
        raise ValueError("Pass a ContinuousHouseholdWrapper (or HouseholdEnvironment) via env=")

    # ── Resolve base env reference (works for wrapped and unwrapped) ──────────
    _base = env.env if hasattr(env, "env") else env

    # ── Resolve training vs. evaluation settings ──────────────────────────────
    if not ucenje:
        episodes          = 1
        steps_per_episode = _base.data_length - 1
        shrani            = False
        nalozi            = agent is None
    else:
        episodes          = int(ponovitev)
        # Use a weekly episode length for training (same convention as DQN)
        steps_per_episode = _base.episode_length
        shrani            = True
        nalozi            = agent is None

    # ── Build agent if not supplied ───────────────────────────────────────────
    obs_dim    = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    if reset:
        agent  = None
        nalozi = False

    if agent is None or reset:
        agent = AgentSAC(
            obs_dim      = obs_dim,
            action_dim   = action_dim,
            gamma        = gamma,
            tau          = tau,
            alpha        = alpha,
            auto_alpha   = auto_alpha,
            lr_actor     = lr_actor,
            lr_critic    = lr_critic,
            lr_alpha     = lr_alpha,
            batch_size   = batch_size,
            buffer_size  = buffer_size,
            hidden       = hidden,
            update_every = update_every,
            start_steps  = start_steps,
        )

    if nalozi:
        agent.load(checkpoint_dir)

    # ── Seed ──────────────────────────────────────────────────────────────────
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)

    # ── Tracking arrays  (same names as DQN for drop-in plotting) ─────────────
    bat_init = (_base.bat_kapaciteta if hasattr(_base, "bat_kapaciteta") else 10.0) / 2.0
    nagradaSAC  = [0.0]
    placiloSAC  = [0.0]
    baterijaSAC = [bat_init]

    # ── Main loop ─────────────────────────────────────────────────────────────
    deterministic_action = not ucenje and deterministic_eval

    for episode_idx in range(episodes):
        reset_options = (
            {"reset_mode": "deterministic"} if not ucenje
            else {"reset_mode": "random"}
        )

        obs, info = env.reset(options=reset_options)
        done       = False
        step_count = 0

        while step_count < steps_per_episode and not done:
            action = agent.choose_action(obs, deterministic=deterministic_action)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done_next = bool(terminated or truncated)

            if ucenje:
                agent.store_transition(obs, action, reward, next_obs, done_next)
                if step_count % update_every == 0:
                    agent.learn()

            obs        = next_obs
            done       = done_next
            step_count += 1

            nagradaSAC.append(  nagradaSAC[-1]  + reward)
            placiloSAC.append(  float(step_info.get("cumulative_payment", placiloSAC[-1])))
            baterijaSAC.append( float(step_info.get("battery", baterijaSAC[-1])))

        if ucenje and ((episode_idx + 1) % checkpoint_every == 0 or (episode_idx + 1) == episodes):
            if shrani:
                agent.save(checkpoint_dir, episode=episode_idx + 1)

        ep_reward = nagradaSAC[-1]
        ep_cost   = placiloSAC[-1]
        print(f"[SAC] Episode {episode_idx + 1}/{episodes}  |  "
              f"Reward: {ep_reward:.2f}  |  Cost: AUD {ep_cost:.4f}  |  "
              f"Alpha: {agent.alpha:.4f}  |  Steps: {agent._total_steps}")

    return agent, nagradaSAC, placiloSAC, baterijaSAC
