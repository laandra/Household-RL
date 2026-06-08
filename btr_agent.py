"""
Beyond the Rainbow (BTR) Agent Implementation
Core agent for training with IQN and C51 network architectures.
Includes PER (Prioritized Experience Replay) and training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from collections import deque
import heapq


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration (Noisy Networks)."""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Register buffers for noise (not trainable parameters)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        
        self.reset_parameters()
        self.sample_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def sample_noise(self):
        """Sample noise for epsilon (re-parameterization trick)."""
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class IQNNetwork(nn.Module):
    """Implicit Quantile Network (IQN) for distributional RL."""
    
    def __init__(
        self,
        state_shape: int,
        n_actions: int,
        hidden_size: int = 512,
        n_hidden_layers: int = 2,
        n_quantiles: int = 200,
        use_noisy: bool = True,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.use_noisy = use_noisy
        
        # Embedding layer for taus (quantile levels)
        self.tau_embedding_dim = 64
        self.tau_embedding = nn.Linear(self.tau_embedding_dim, hidden_size)
        
        # State encoder
        layers = []
        in_size = state_shape
        for i in range(n_hidden_layers):
            layer = nn.Linear(in_size, hidden_size)
            if use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        
        # Dueling Q-heads: V and A
        self.fc_v = nn.Linear(hidden_size, 1)
        self.fc_a = nn.Linear(hidden_size, n_actions)
        
        # IQN quantile networks
        if use_noisy:
            self.fc_q = NoisyLinear(hidden_size, n_actions)
        else:
            self.fc_q = nn.Linear(hidden_size, n_actions)
    
    def encode_tau(self, taus: torch.Tensor) -> torch.Tensor:
        """Encode tau values (quantile levels) as embeddings."""
        # taus shape: (batch_size, n_tau) or (n_tau,)
        # Compute tau embedding: sin(pi * k * tau) for k = 1..tau_embedding_dim
        if taus.dim() == 1:
            taus = taus.unsqueeze(0)
        
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        
        # Compute sin(pi * k * tau)
        tau_embed = np.pi * torch.arange(1, self.tau_embedding_dim + 1, device=taus.device).float()
        tau_embed = tau_embed.unsqueeze(0).unsqueeze(0) * taus.unsqueeze(-1)  # (1, batch, n_tau, embedding_dim)
        tau_embed = torch.sin(tau_embed)
        
        # Linear projection
        tau_embed = self.tau_embedding(tau_embed)  # (batch_size, n_tau, hidden_size)
        return tau_embed
    
    def forward(self, state: torch.Tensor, taus: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_shape)
            taus: Quantile levels (batch_size, n_tau) or (n_tau,). If None, sample uniformly.
            
        Returns:
            Tuple of (q_values, taus) where q_values shape is (batch_size, n_tau, n_actions)
        """
        batch_size = state.shape[0]
        
        if taus is None:
            # Sample taus uniformly
            taus = torch.rand(batch_size, self.n_quantiles, device=state.device)
        
        # Encode state
        encoded = self.encoder(state)  # (batch_size, hidden_size)
        
        # Encode taus
        tau_embed = self.encode_tau(taus)  # (batch_size, n_tau, hidden_size)
        
        # Combine state and tau embeddings element-wise
        encoded_exp = encoded.unsqueeze(1)  # (batch_size, 1, hidden_size)
        combined = encoded_exp * tau_embed  # (batch_size, n_tau, hidden_size)
        combined = F.relu(combined)
        
        # Compute Q-values for each tau
        q_values = self.fc_q(combined)  # (batch_size, n_tau, n_actions)
        
        return q_values, taus
    
    def sample_noise(self):
        """Resample noise in noisy layers."""
        if self.use_noisy and hasattr(self.fc_q, "sample_noise"):
            self.fc_q.sample_noise()


class C51Network(nn.Module):
    """Categorical DQN (C51) for distributional RL."""
    
    def __init__(
        self,
        state_shape: int,
        n_actions: int,
        hidden_size: int = 512,
        n_hidden_layers: int = 2,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_noisy: bool = True,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = n_atoms
        self.use_noisy = use_noisy
        
        # Compute atom values
        self.register_buffer("atom_values", torch.linspace(v_min, v_max, n_atoms))
        
        # State encoder
        layers = []
        in_size = state_shape
        for i in range(n_hidden_layers):
            layer = nn.Linear(in_size, hidden_size)
            if use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        
        # Output: categorical distribution over atoms for each action
        if use_noisy:
            self.fc_out = NoisyLinear(hidden_size, n_actions * n_atoms)
        else:
            self.fc_out = nn.Linear(hidden_size, n_actions * n_atoms)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_shape)
            
        Returns:
            Tuple of (distributions, atom_values) where distributions is (batch_size, n_actions, n_atoms)
        """
        batch_size = state.shape[0]
        
        # Encode state
        encoded = self.encoder(state)  # (batch_size, hidden_size)
        
        # Compute categorical distribution
        logits = self.fc_out(encoded)  # (batch_size, n_actions * n_atoms)
        logits = logits.view(batch_size, self.n_actions, self.n_atoms)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs, self.atom_values
    
    def sample_noise(self):
        """Resample noise in noisy layers."""
        if self.use_noisy and hasattr(self.fc_out, "sample_noise"):
            self.fc_out.sample_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) with SumTree-like priority sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize PER buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Increment beta towards 1.0 each sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.idx = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[Tuple, np.ndarray, np.ndarray]:
        """
        Sample batch according to priorities.
        
        Returns:
            Tuple of (transitions, indices, weights) where weights are importance sampling corrections
        """
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        priorities = np.array(list(self.priorities))
        priorities = np.power(priorities, self.alpha)
        probs = priorities / priorities.sum()
        
        # Sample indices according to probabilities
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = np.power(len(self.buffer) * probs[indices], -self.beta)
        weights /= weights.max()  # Normalize to [0, 1]
        
        transitions = [self.buffer[i] for i in indices]
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-5):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = (np.abs(error) + epsilon) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class BTRAgent:
    """
    Beyond the Rainbow Agent.
    Trains IQN or C51 network with PER and exploration strategies.
    """
    
    def __init__(
        self,
        state_shape: int,
        n_actions: int,
        algorithm: str = "IQN",
        device: str = "cpu",
        **config
    ):
        """
        Initialize BTR agent.
        
        Args:
            state_shape: Size of state vector
            n_actions: Number of discrete actions
            algorithm: "IQN" or "C51"
            device: "cpu" or "cuda"
            **config: Algorithm-specific hyperparameters (from btr_config_builder)
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.algorithm = algorithm
        self.device = torch.device(device)
        self.config = config
        
        # Extract key params
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 128)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.001)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)
        self.use_noisy_nets = config.get("use_noisy_nets", True)
        self.use_spectral_norm = config.get("use_spectral_norm", True)
        
        # Exploration
        self.exploration_fraction = config.get("exploration_fraction", 0.1)
        self.exploration_initial_eps = config.get("exploration_initial_eps", 1.0)
        self.exploration_final_eps = config.get("exploration_final_eps", 0.05)
        self.epsilon = self.exploration_initial_eps
        self.total_steps = 0
        self.total_training_steps_max = 1000000  # Will be set during training
        
        # PER
        self.use_per = config.get("use_per", True)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.get("buffer_size", 100000),
            alpha=config.get("per_alpha", 0.6),
            beta=config.get("per_beta", 0.4),
            beta_increment=config.get("per_beta_increment", 0.001),
        )
        
        # Create networks
        self._create_networks()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.target_network.parameters()),
            lr=self.learning_rate
        )
        
        self.learning_starts = config.get("learning_starts", 1000)
        self.train_freq = config.get("train_freq", 4)
        self.gradient_steps = config.get("gradient_steps", 1)
        self.target_update_freq = config.get("target_update_freq", 10000)
        self.step_counter = 0
        
        # Tracking
        self.total_loss = 0.0
        self.update_counter = 0
    
    def _create_networks(self):
        """Create online and target networks."""
        if self.algorithm == "IQN":
            net_class = IQNNetwork
            net_kwargs = {
                "state_shape": self.state_shape,
                "n_actions": self.n_actions,
                "hidden_size": self.config.get("hidden_size", 512),
                "n_hidden_layers": self.config.get("n_hidden_layers", 2),
                "n_quantiles": self.config.get("n_quantiles", 200),
                "use_noisy": self.use_noisy_nets,
                "use_spectral_norm": self.use_spectral_norm,
            }
        elif self.algorithm == "C51":
            net_class = C51Network
            net_kwargs = {
                "state_shape": self.state_shape,
                "n_actions": self.n_actions,
                "hidden_size": self.config.get("hidden_size", 512),
                "n_hidden_layers": self.config.get("n_hidden_layers", 2),
                "n_atoms": self.config.get("n_atoms", 51),
                "v_min": self.config.get("v_min", -10.0),
                "v_max": self.config.get("v_max", 10.0),
                "use_noisy": self.use_noisy_nets,
                "use_spectral_norm": self.use_spectral_norm,
            }
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.network = net_class(**net_kwargs).to(self.device)
        self.target_network = net_class(**net_kwargs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy (with noisy nets or traditional epsilon).
        
        Args:
            state: State observation (numpy array, float32)
            training: If True, use exploration; if False, deterministic
            
        Returns:
            Action integer (0 to n_actions-1)
        """
        if not training or np.random.rand() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                
                if self.algorithm == "IQN":
                    # For IQN, evaluate at tau=1.0 (deterministic)
                    taus = torch.ones(1, 1, device=self.device)
                    q_values, _ = self.network(state_tensor, taus)
                    q_mean = q_values.mean(dim=1).squeeze(0)  # Average over quantiles: (batch, n_tau, n_actions) -> (n_actions,)
                else:  # C51
                    probs, atom_values = self.network(state_tensor)
                    q_mean = (probs * atom_values.unsqueeze(0).unsqueeze(0)).sum(dim=-1).squeeze(0)  # (1, n_actions) -> (n_actions,)
                
                action = q_mean.argmax(dim=-1).item()
        else:
            # Random action
            action = np.random.randint(self.n_actions)
        
        # Update epsilon
        if training:
            self.epsilon = self._get_epsilon(self.total_steps, self.total_training_steps_max)
        
        return action
    
    def _get_epsilon(self, step: int, total_steps: int) -> float:
        """Compute epsilon for epsilon-greedy schedule."""
        fraction = min(step / (self.exploration_fraction * total_steps), 1.0)
        return self.exploration_final_eps + (self.exploration_initial_eps - self.exploration_final_eps) * (1.0 - fraction)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def learn_step(self) -> Optional[float]:
        """
        Perform single learning step (gradient update).
        
        Returns:
            Loss value or None if not enough data
        """
        if len(self.replay_buffer) < self.learning_starts:
            return None
        
        # Sample batch
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size, device=self.device.type)
        
        # Prepare batch tensors
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        # Compute loss
        if self.algorithm == "IQN":
            loss = self._compute_iqn_loss(states, actions, rewards, next_states, dones)
        else:  # C51
            loss = self._compute_c51_loss(states, actions, rewards, next_states, dones)
        
        weighted_loss = (loss * weights).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update priorities
        td_errors = loss.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Soft update target network
        self._soft_update()
        
        self.total_loss += weighted_loss.item()
        self.update_counter += 1
        
        return weighted_loss.item()
    
    def _compute_iqn_loss(self, states, actions, rewards, next_states, dones):
        """Compute IQN loss (quantile Huber loss)."""
        # Sample taus for current and next states
        n_tau = 32
        taus = torch.rand(states.shape[0], n_tau, device=self.device)
        
        # Current Q-values
        q_values, _ = self.network(states, taus)
        q_selected = q_values[range(len(actions)), :, actions]  # (batch, n_tau)
        
        # Next Q-values (no-grad)
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states, taus)
            next_q_max = next_q_values.max(dim=-1)[0]  # (batch, n_tau)
            targets = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * next_q_max
        
        # Quantile Huber loss
        td_error = targets - q_selected
        kappa = self.config.get("kappa", 1.0)
        huber_loss = torch.where(td_error.abs() <= kappa, 0.5 * td_error ** 2, kappa * (td_error.abs() - 0.5 * kappa))
        loss = huber_loss.mean(dim=-1)  # Average over quantiles
        
        return loss
    
    def _compute_c51_loss(self, states, actions, rewards, next_states, dones):
        """Compute C51 loss (categorical cross-entropy)."""
        # Current distribution
        probs, atom_values = self.network(states)
        selected_probs = probs[range(len(actions)), actions, :]  # (batch, n_atoms)
        
        # Next distribution
        with torch.no_grad():
            next_probs, next_atom_values = self.target_network(next_states)
            next_q_values = (next_probs * next_atom_values.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=-1)
            next_selected_probs = next_probs[range(len(next_actions)), next_actions, :]  # (batch, n_atoms)
            
            # Project atoms
            v_min = self.config.get("v_min", -10.0)
            v_max = self.config.get("v_max", 10.0)
            delta_z = (v_max - v_min) / (len(atom_values) - 1)
            
            target_atoms = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * atom_values.unsqueeze(0)
            target_atoms = torch.clamp(target_atoms, v_min, v_max)
            
            # Project distribution
            b_j = (target_atoms - v_min) / delta_z
            l = b_j.floor().long()
            u = b_j.ceil().long()
            ml = next_selected_probs * (u.float() - b_j)
            mu = next_selected_probs * (b_j - l.float())
            
            target_probs = torch.zeros_like(selected_probs)
            for j in range(len(atom_values)):
                target_probs[:, l[:, j].clamp(0, len(atom_values) - 1)] += ml[:, j]
                target_probs[:, u[:, j].clamp(0, len(atom_values) - 1)] += mu[:, j]
        
        # Cross-entropy loss
        loss = -(target_probs * torch.log(selected_probs + 1e-8)).sum(dim=-1)
        
        return loss
    
    def _soft_update(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def reset_noise(self):
        """Resample noise in noisy layers."""
        if self.use_noisy_nets:
            self.network.sample_noise()
            self.target_network.sample_noise()
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.exploration_initial_eps)
