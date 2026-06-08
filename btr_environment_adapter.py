"""
Beyond the Rainbow (BTR) Environment Adapter
Wraps HouseholdEnvironment to be compatible with BTR training loop.
Handles state shape conversion, PyTorch tensor conversion, and validation.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import torch


class BTREnvironmentAdapter:
    """
    Adapter between HouseholdEnvironment (Gymnasium-based) and BTR's PyTorch training loop.
    
    BTR expects:
    - States as float32 numpy arrays (converted to torch.FloatTensor for networks)
    - Actions as discrete integers (0 to n_actions-1)
    - Rewards as floats
    - Dones as booleans
    - Info dict with metadata
    
    HouseholdEnvironment provides these natively via Gymnasium API.
    This adapter mainly validates shapes and optionally converts to PyTorch tensors.
    """
    
    def __init__(self, env, device: str = "cpu", use_torch: bool = True):
        """
        Initialize adapter.
        
        Args:
            env: HouseholdEnvironment instance
            device: PyTorch device ("cpu" or "cuda")
            use_torch: If True, convert states to torch.FloatTensor; else keep as numpy
        """
        self.env = env
        self.device = torch.device(device)
        self.use_torch = use_torch
        
        # Validate environment has required attributes
        self._validate_env()
        
        # Cache state and action dimensions
        self.state_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        
    def _validate_env(self):
        """Validate that wrapped environment has required Gymnasium attributes."""
        required_attrs = ["observation_space", "action_space", "reset", "step"]
        for attr in required_attrs:
            if not hasattr(self.env, attr):
                raise AttributeError(f"Environment missing required attribute: {attr}")
        
        if not hasattr(self.env.observation_space, "shape"):
            raise ValueError("observation_space must have 'shape' attribute")
        if not hasattr(self.env.action_space, "n"):
            raise ValueError("action_space must have 'n' attribute (discrete)")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and return initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (state, info) where state is float32 numpy array
        """
        obs, info = self.env.reset(seed=seed)
        
        # Validate state shape
        self._validate_state(obs)
        
        return self._convert_state(obs), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return transition.
        
        Args:
            action: Discrete action (0 to n_actions-1)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
            where next_state is float32 numpy array
        """
        # Validate action
        if not isinstance(action, (int, np.integer)):
            action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action {action}. Expected 0 to {self.n_actions-1}")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Validate outputs
        self._validate_state(obs)
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        
        return self._convert_state(obs), reward, terminated, truncated, info
    
    def _validate_state(self, state: np.ndarray):
        """Validate state shape and dtype."""
        if not isinstance(state, np.ndarray):
            raise TypeError(f"State must be numpy array, got {type(state)}")
        if state.dtype != np.float32:
            raise TypeError(f"State dtype must be float32, got {state.dtype}")
        if state.shape != self.state_shape:
            raise ValueError(
                f"State shape mismatch. Expected {self.state_shape}, got {state.shape}"
            )
    
    def _convert_state(self, state: np.ndarray) -> np.ndarray:
        """Convert state to PyTorch tensor if enabled, else return numpy."""
        if self.use_torch:
            # Convert to torch tensor and move to device
            tensor = torch.from_numpy(state).float().to(self.device)
            return tensor
        return state
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get shape of state for network initialization."""
        return self.state_shape
    
    def get_n_actions(self) -> int:
        """Get number of discrete actions."""
        return self.n_actions
    
    def get_observation_mode(self) -> str:
        """Get observation mode (compact or sliding_window)."""
        if hasattr(self.env, "observation_mode"):
            return self.env.observation_mode
        return "unknown"
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        return self.env.seed(seed)
    
    def render(self):
        """Render environment (if supported)."""
        if hasattr(self.env, "render"):
            return self.env.render()
        return None
    
    def close(self):
        """Close environment and release resources."""
        if hasattr(self.env, "close"):
            self.env.close()


class BTREnvironmentBatch:
    """
    Handles a batch of parallel environments (if needed for efficiency).
    For now, provides single-environment interface; can be extended for vectorization.
    """
    
    def __init__(self, envs: list, device: str = "cpu", use_torch: bool = True):
        """
        Initialize batch of environments.
        
        Args:
            envs: List of HouseholdEnvironment instances
            device: PyTorch device
            use_torch: Convert states to torch tensors
        """
        self.adapters = [
            BTREnvironmentAdapter(env, device=device, use_torch=use_torch)
            for env in envs
        ]
        self.n_envs = len(self.adapters)
        self.device = torch.device(device)
        self.use_torch = False  # Always return numpy arrays; conversion happens in agent methods
        
    def reset(self, seed: Optional[int] = None):
        """Reset all environments."""
        states, infos = [], []
        for i, adapter in enumerate(self.adapters):
            seed_i = seed + i if seed is not None else None
            state, info = adapter.reset(seed=seed_i)
            states.append(state)
            infos.append(info)
        return np.array(states), infos
    
    def step(self, actions: np.ndarray):
        """Execute actions in all environments in parallel."""
        results = [adapter.step(int(action)) for adapter, action in zip(self.adapters, actions)]
        
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminateds = np.array([r[2] for r in results])
        truncateds = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        
        return states, rewards, terminateds, truncateds, infos
    
    def close_all(self):
        """Close all environments."""
        for adapter in self.adapters:
            adapter.close()
