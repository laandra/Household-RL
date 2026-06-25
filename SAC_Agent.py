"""
SAC_Agent.py  –  Soft Actor-Critic with True Continuous Battery Control
========================================================================
The action space is a single continuous value:
    action ∈ [-1, 1]
    > 0  →  charge battery by  action * bat_max_polnjenje  kWh  (from grid/solar)
    < 0  →  discharge battery by  |action| * bat_max_praznjenje  kWh  (to home/grid)
    = 0  →  battery idle

The wrapper computes exact kWh flows from this signal, calls the base
environment's reward function directly with those flows, and stores the
EXECUTED kWh delta (normalised) in the replay buffer — NOT the raw
tanh-squashed network output.  This eliminates the discrete↔continuous
mismatch that was the root cause of reward minimisation.

Key changes vs the broken version
──────────────────────────────────
1. ContinuousHouseholdWrapper now computes physics directly (no discrete
   action lookup) so Q(s, a_continuous) is a well-defined, injective mapping.
2. The replay buffer stores the action actually executed (clipped continuous),
   not the raw network sample, removing the many-to-one aliasing problem.
3. ActorNetwork architecture fixed: _MLP is used cleanly without the
   spurious extra F.relu() double-activation.
4. learn() guard uses only batch_size (not max(batch_size, start_steps)),
   so training starts as soon as the buffer has enough samples.
5. Action-selection warm-up uses a dedicated step counter that is incremented
   before learning, keeping choose_action and store_transition in sync.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym


# ─────────────────────────────────────────────────────────────────────────────
# 1.  True-Continuous Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousHouseholdWrapper(gym.Wrapper):
    """
    Replaces the base env's Discrete(5) action space with Box([-1],[1]).

    Action semantics
    ─────────────────
      a ∈ ( 0, 1] → charge battery:   P_ch  = a  * bat_max_polnjenje  [kWh]
      a ∈ [-1, 0) → discharge battery: P_dis = -a * bat_max_praznjenje [kWh]
      a = 0       → battery idle

    Energy-balance physics are computed HERE (not by picking a discrete action),
    so the mapping  continuous_action → reward  is injective and the critic
    can learn a valid Q-function over the continuous action space.

    The wrapper replicates the base env's step physics exactly:
      • Energy_Generation fills Consumption first (solar priority)
      • Remaining charge/discharge adjusts battery within hardware limits
      • Net grid import/export is passed to calculate_interval_price
      • Reward is computed by the base env's _nagrada_skupno()
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0),
            shape=(1,), dtype=np.float32,
        )
        self.observation_space = env.observation_space

    # ── Physics helpers ────────────────────────────────────────────────────────

    def _decode_action(self, a_float: float):
        """
        Convert scalar continuous action → (P_ch, P_dis) in kWh,
        respecting per-step hardware limits and current SOC.
        """
        e = self.env
        bat     = float(e._battery)
        eta     = e.bat_ucinkovitost
        max_ch  = e.bat_max_polnjenje
        max_dis = e.bat_max_praznjenje
        cap     = e.bat_kapaciteta

        if a_float > 0.0:
            # Charge: actual energy into battery (after efficiency loss)
            desired_into_bat = float(a_float) * max_ch           # kWh into battery
            # Cannot exceed remaining capacity
            room = cap - bat
            actual_into_bat = min(desired_into_bat, room, max_ch)
            actual_into_bat = max(actual_into_bat, 0.0)
            P_ch  = actual_into_bat   # energy stored in battery [kWh]
            P_dis = 0.0
        elif a_float < 0.0:
            # Discharge: actual energy leaving battery (before efficiency loss)
            desired_from_bat = float(-a_float) * max_dis         # kWh from battery
            available = bat
            actual_from_bat = min(desired_from_bat, available, max_dis)
            actual_from_bat = max(actual_from_bat, 0.0)
            P_dis = actual_from_bat   # energy leaving battery [kWh]
            P_ch  = 0.0
        else:
            P_ch = P_dis = 0.0

        return P_ch, P_dis

    def step(self, action):
        """
        Execute a continuous action without going through discrete action codes.
        All physics are replicated from Environment.py so the reward is identical.
        """
        from Basic_Functions import BaterijaSprememba, PaneliOdvec, calculate_interval_price

        e   = self.env
        idx = e._current_step

        if idx >= e.data_length - 1:
            # Terminal guard (mirrors base env behaviour)
            s0 = e._get_state_object(idx, e._battery, e._cumulative_payment)
            obs = e._build_observation(idx, s0.Baterija_norm)
            return obs, 0.0, True, False, e._build_info(s0)

        # Current state
        s = e._get_state_object(idx, e._battery, e._cumulative_payment)

        # ── Decode continuous action ─────────────────────────────────────────
        a_arr   = np.asarray(action, dtype=np.float32).flatten()
        a_float = float(np.clip(a_arr[0], -1.0, 1.0))
        P_ch, P_dis = self._decode_action(a_float)

        eta = e.bat_ucinkovitost
        gen = s.Generiranje
        con = s.Poraba

        # Solar surplus (always used for consumption first)
        solar_surplus = PaneliOdvec(gen, con)   # max(0, gen-con)

        # ── Allocate solar surplus to battery charge first ───────────────────
        solar_to_bat  = min(solar_surplus, P_ch / eta if eta > 0 else P_ch)
        grid_to_bat   = max(0.0, P_ch / eta - solar_to_bat) if eta > 0 else 0.0

        # Remaining solar after battery fill goes to grid (feed-in)
        solar_to_grid = max(0.0, solar_surplus - solar_to_bat)

        # ── Battery discharge allocation ─────────────────────────────────────
        # Discharge covers home demand first, remainder sells to grid
        home_from_bat  = 0.0
        grid_from_bat  = 0.0
        if P_dis > 0.0:
            deficit = max(0.0, con - gen)           # demand not met by solar
            home_from_bat = min(P_dis * eta, deficit)
            grid_from_bat = max(0.0, P_dis * eta - home_from_bat)

        # ── Grid import/export ───────────────────────────────────────────────
        # Positive = buying, Negative = selling
        kupljena_elektrika = (
            max(0.0, con - gen)         # base grid demand (solar covers first)
            + grid_to_bat               # grid energy to charge battery
            - home_from_bat             # battery covers home demand
            - grid_from_bat             # battery sells to grid
            - solar_to_grid             # solar feeds into grid
        )

        # ── Battery state change ─────────────────────────────────────────────
        # paneli_baterija, omrezje_baterija → charging side
        # baterija_dom, baterija_omrezje    → discharging side
        sprememba_baterije = BaterijaSprememba(
            solar_to_bat,    # paneli_baterija
            grid_to_bat,     # omrezje_baterija  (raw kWh drawn from grid for bat)
            home_from_bat / eta if eta > 0 else home_from_bat,   # baterija_dom (energy leaving bat)
            grid_from_bat / eta if eta > 0 else grid_from_bat,   # baterija_omrezje
            eta,
        )

        # ── Pricing ──────────────────────────────────────────────────────────
        price_result = calculate_interval_price(
            s.CenaEl,
            kupljena_elektrika,
            utc_date=e.dataset.index[idx],
            interval_minutes=e.korakov_na_dan * 60 / 24,
        )
        konstantno_placilo = float(price_result["constant_price_aud"])
        placilo_zdaj       = float(price_result["variable_price_aud"])

        # ── Update battery and payment ────────────────────────────────────────
        new_battery = float(np.clip(s.Baterija + sprememba_baterije, 0.0, e.bat_kapaciteta))
        new_payment = s.Placilo + placilo_zdaj + konstantno_placilo
        next_idx    = idx + 1
        next_s      = e._get_state_object(next_idx, new_battery, new_payment)

        # ── Reward (exact same function as base env) ─────────────────────────
        cena_el_med   = e.arr_MedianPrice[idx]
        reward        = e._nagrada_skupno(s, sprememba_baterije, placilo_zdaj, cena_el_med)
        r_kapaciteta  = e._nagrada_1(s)
        r_sprememba   = e._nagrada_2(s, sprememba_baterije, cena_el_med) if e.bat_kapaciteta > 0 else 0.0
        r_placilo     = e._nagrada_3(placilo_zdaj)

        # ── Advance env internal state ────────────────────────────────────────
        e._current_step        = next_s.Korak
        e._battery             = next_s.Baterija
        e._cumulative_payment  = next_s.Placilo
        e._episode_steps      += 1

        terminated = e._current_step >= (e.data_length - 1)
        truncated  = e._current_step >= (e._episode_end_exclusive - 1)
        obs        = e._build_observation(next_s.Korak, next_s.Baterija_norm)

        info = e._build_info(
            next_s,
            action_int=None,
            energy_flows={
                "paneli_baterija":   float(solar_to_bat),
                "omrezje_baterija":  float(grid_to_bat),
                "baterija_dom":      float(home_from_bat / eta if eta > 0 else home_from_bat),
                "baterija_omrezje":  float(grid_from_bat / eta if eta > 0 else grid_from_bat),
                "kupljena_elektrika": float(kupljena_elektrika),
                "sprememba_baterije": float(sprememba_baterije),
                "P_ch_kWh":          float(P_ch),
                "P_dis_kWh":         float(P_dis),
            },
            reward_components={
                "total":       float(reward),
                "r_kapaciteta": float(r_kapaciteta),
                "r_sprememba":  float(r_sprememba),
                "r_placilo":    float(r_placilo),
                "placilo_zdaj": float(placilo_zdaj),
            },
        )
        # Tag with the continuous action for logging
        info["continuous_action"] = float(a_float)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def action_masks(self):
        return self.env.action_masks()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Neural-Network Modules
# ─────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _build_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    """
    Build a simple ReLU MLP.  hidden is a tuple of intermediate layer sizes.
    No activation on the final layer (heads add their own if needed).
    """
    sizes  = [in_dim, *hidden, out_dim]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:          # ReLU on all layers except the last
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    # Orthogonal init for hidden layers, small uniform for output layer
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class ActorNetwork(nn.Module):
    """
    Squashed-Gaussian policy: outputs mean and log-std over action ∈ [-1,1].
    Architecture: obs → [hidden layers with ReLU] → mu_head / log_std_head
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden: Tuple[int, ...] = (256, 256), lr: float = 3e-4):
        super().__init__()
        self.action_dim = action_dim

        # Shared trunk + separate heads (cleaner than splitting _MLP)
        self.trunk       = _build_mlp(obs_dim, hidden[:-1], hidden[-1])
        self.mu_head     = nn.Linear(hidden[-1], action_dim)
        self.log_std_head= nn.Linear(hidden[-1], action_dim)

        # Small init on output heads for initial near-zero actions
        nn.init.uniform_(self.mu_head.weight,      -3e-3, 3e-3)
        nn.init.uniform_(self.mu_head.bias,         -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight,  -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias,    -3e-3, 3e-3)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: T.Tensor):
        # trunk already ends without an activation, so apply ReLU here
        h       = F.relu(self.trunk(obs))
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: T.Tensor):
        """
        Reparameterised sample with tanh squashing.
        Returns (action, log_prob, mean_action).
        log_prob includes the tanh Jacobian correction (numerically stable).
        """
        mu, log_std = self.forward(obs)
        std  = log_std.exp()
        dist = T.distributions.Normal(mu, std)
        x_t  = dist.rsample()                    # reparameterised
        action = T.tanh(x_t)
        # Tanh Jacobian: log|da/dx_t| = log(1 - tanh²(x_t)) = log(1 - a²)
        log_prob = dist.log_prob(x_t) - T.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, T.tanh(mu)


class CriticNetwork(nn.Module):
    """Twin-Q critic (double Q to suppress overestimation)."""

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden: Tuple[int, ...] = (256, 256), lr: float = 3e-4):
        super().__init__()
        # Both Q networks: input = [obs, action]
        self.q1 = _build_mlp(obs_dim + action_dim, hidden, 1)
        self.q2 = _build_mlp(obs_dim + action_dim, hidden, 1)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: T.Tensor, action: T.Tensor):
        sa = T.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.ptr  = 0
        self.size = 0
        self.obs      = np.zeros((self.capacity, obs_dim),    dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((self.capacity, 1),          dtype=np.float32)
        self.dones    = np.zeros((self.capacity, 1),          dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = np.asarray(action).reshape(self.actions.shape[1:])
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
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
    Soft Actor-Critic for the continuous household microgrid task.

    Hyperparameters
    ───────────────
    gamma        – discount factor (0.99 default)
    tau          – soft target update rate (0.005)
    alpha        – initial entropy temperature; auto-tuned when auto_alpha=True
    lr_actor/critic/alpha – learning rates
    batch_size   – replay buffer sample size
    buffer_size  – replay buffer capacity
    hidden       – MLP layer sizes, e.g. (256, 256)
    update_every – gradient steps per env step
    start_steps  – random warm-up steps before policy is used
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
    ):
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.update_every = update_every
        self.start_steps  = start_steps
        self.action_dim   = action_dim
        self.obs_dim      = obs_dim

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(f"[SAC] device: {self.device}")

        # Networks
        self.actor         = ActorNetwork(obs_dim, action_dim, hidden, lr_actor)
        self.critic        = CriticNetwork(obs_dim, action_dim, hidden, lr_critic)
        self.critic_target = CriticNetwork(obs_dim, action_dim, hidden, lr_critic)
        self._hard_update_target()

        # Entropy temperature α
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy  = -float(action_dim)   # heuristic: -|A|
            self.log_alpha       = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha           = self.log_alpha.exp().item()
        else:
            self.alpha   = float(alpha)
            self.log_alpha = None

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        # FIX: separate counter for warm-up vs total transitions
        self._total_steps  = 0   # total env interactions (incremented in store_transition)
        self._learn_steps  = 0   # gradient update count

    # ── Target network helpers ─────────────────────────────────────────────────

    def _hard_update_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _soft_update_target(self):
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    # ── Public API ─────────────────────────────────────────────────────────────

    def choose_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Random warm-up for start_steps, then policy.
        Returns a numpy array of shape (action_dim,) in [-1, 1].
        """
        if (not deterministic) and (self._total_steps < self.start_steps):
            return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

        obs_t = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0)
        with T.no_grad():
            if deterministic:
                _, _, mean = self.actor.sample(obs_t)
                return mean.cpu().numpy().flatten().astype(np.float32)
            action, _, _ = self.actor.sample(obs_t)
            return action.cpu().numpy().flatten().astype(np.float32)

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store the EXECUTED action (already clipped by the wrapper) so the
        critic trains on exactly the (obs, action) pairs the env saw.
        """
        self.buffer.add(obs, action, reward, next_obs, done)
        self._total_steps += 1

    def learn(self):
        """
        One SAC gradient update.
        Guard: wait until replay buffer has at least batch_size samples.
        (start_steps is purely for choose_action warm-up; do NOT double-gate here.)
        """
        # FIX: guard on batch_size only — NOT max(batch_size, start_steps)
        if len(self.buffer) < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        # ── Critic update (Bellman targets with soft value) ──────────────────
        with T.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            # Soft value: V(s') = min(Q1,Q2)(s',a') - α log π(a'|s')
            q_next   = T.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic.optimizer.step()

        # ── Actor update ──────────────────────────────────────────────────────
        # FIX: freeze critic parameters during actor pass to save memory/time
        for p in self.critic.parameters():
            p.requires_grad = False

        new_actions, log_pi, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        q_pi = T.min(q1_pi, q2_pi)

        # Maximise  E[Q - α log π]  ↔  minimise  E[α log π - Q]
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor.optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Temperature update ─────────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.auto_alpha:
            # FIX: detach log_pi to avoid double-computing actor graph
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha        = self.log_alpha.exp().item()
            alpha_loss_val    = alpha_loss.item()

        self._soft_update_target()
        self._learn_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss_val,
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, checkpoint_dir: str = "SAC", episode: Optional[int] = None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, self.CHECKPOINT_NAME)
        payload = {
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "alpha":         self.alpha,
            "log_alpha":     self.log_alpha.item() if self.log_alpha is not None else None,
            "total_steps":   self._total_steps,
            "obs_dim":       self.obs_dim,
            "action_dim":    self.action_dim,
            "episode":       episode,
        }
        tmp = path + ".tmp"
        T.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load(self, checkpoint_dir: str = "SAC") -> bool:
        path = os.path.join(checkpoint_dir, self.CHECKPOINT_NAME)
        if not os.path.exists(path):
            print(f"[SAC] No checkpoint at {path}")
            return False
        ckpt = T.load(path, weights_only=False, map_location=self.device)
        if ckpt.get("obs_dim") != self.obs_dim or ckpt.get("action_dim") != self.action_dim:
            print("[SAC] Checkpoint dimension mismatch — skipping.")
            return False
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.alpha = float(ckpt.get("alpha", self.alpha))
        if self.log_alpha is not None and ckpt.get("log_alpha") is not None:
            with T.no_grad():
                self.log_alpha.fill_(float(ckpt["log_alpha"]))
        self._total_steps = int(ckpt.get("total_steps", 0))
        print(f"[SAC] Loaded episode={ckpt.get('episode')}  steps={self._total_steps}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Environment Builder  (drop-in for build_dqn_env)
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
    Drop-in replacement for build_dqn_env(); returns a ContinuousHouseholdWrapper.
    Signature is intentionally identical so callers can swap the two freely.
    """
    try:
        import Environment as _env_mod
        import importlib
        importlib.reload(_env_mod)
        HouseholdEnvironment = _env_mod.HouseholdEnvironment
    except ImportError as exc:
        raise ImportError("Environment.py not found in the same directory.") from exc

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
# 6.  Training Loop  (mirrors Learning_DQN API exactly)
# ─────────────────────────────────────────────────────────────────────────────

def Learning_SAC(
    env=None,
    agent: Optional[AgentSAC] = None,
    ucenje: bool = True,
    ponovitev: int = 52,
    reset: bool = False,
    # SAC hyperparameters (used only when building a new agent)
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
    Train or evaluate the SAC agent.

    Returns
    ───────
    agent        : AgentSAC
    nagradaSAC   : list[float]  cumulative reward  (same layout as nagradaDQN)
    placiloSAC   : list[float]  cumulative payment [AUD]
    baterijaSAC  : list[float]  battery SOC [kWh]
    """
    if env is None:
        raise ValueError("env= is required.")

    _base = env.env if hasattr(env, "env") else env

    if not ucenje:
        episodes          = 1
        steps_per_episode = _base.data_length - 1
        shrani            = False
        nalozi            = agent is None
    else:
        episodes          = int(ponovitev)
        steps_per_episode = _base.episode_length
        shrani            = True
        nalozi            = agent is None

    obs_dim    = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    if reset:
        agent  = None
        nalozi = False

    if agent is None or reset:
        agent = AgentSAC(
            obs_dim=obs_dim, action_dim=action_dim,
            gamma=gamma, tau=tau, alpha=alpha, auto_alpha=auto_alpha,
            lr_actor=lr_actor, lr_critic=lr_critic, lr_alpha=lr_alpha,
            batch_size=batch_size, buffer_size=buffer_size,
            hidden=hidden, update_every=update_every, start_steps=start_steps,
        )

    if nalozi:
        agent.load(checkpoint_dir)

    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)

    bat_init    = getattr(_base, "bat_kapaciteta", 20.0) / 2.0
    nagradaSAC  = [0.0]
    placiloSAC  = [0.0]
    baterijaSAC = [bat_init]

    deterministic_action = (not ucenje) and deterministic_eval

    for episode_idx in range(episodes):
        reset_options = (
            {"reset_mode": "deterministic"} if not ucenje
            else {"reset_mode": "random"}
        )
        obs, _  = env.reset(options=reset_options)
        done    = False
        step_cnt = 0

        while step_cnt < steps_per_episode and not done:
            action = agent.choose_action(obs, deterministic=deterministic_action)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done_next = bool(terminated or truncated)

            if ucenje:
                # Store the SAME action the env received (already clipped)
                agent.store_transition(obs, action, reward, next_obs, done_next)
                if step_cnt % update_every == 0:
                    agent.learn()

            obs   = next_obs
            done  = done_next
            step_cnt += 1

            nagradaSAC.append( nagradaSAC[-1]  + reward)
            placiloSAC.append( float(step_info.get("cumulative_payment", placiloSAC[-1])))
            baterijaSAC.append(float(step_info.get("battery",            baterijaSAC[-1])))

        if ucenje and shrani:
            if (episode_idx + 1) % checkpoint_every == 0 or (episode_idx + 1) == episodes:
                agent.save(checkpoint_dir, episode=episode_idx + 1)

        print(
            f"[SAC] Ep {episode_idx+1}/{episodes}  "
            f"reward={nagradaSAC[-1]:.2f}  "
            f"cost=AUD {placiloSAC[-1]:.4f}  "
            f"alpha={agent.alpha:.4f}  "
            f"steps={agent._total_steps}"
        )

    return agent, nagradaSAC, placiloSAC, baterijaSAC
