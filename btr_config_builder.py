"""
Beyond the Rainbow (BTR) Algorithm Configuration and Hyperparameter Space Builder
Provides predefined configs and Optuna search spaces for IQN and C51 variants.
Supports both Optuna optimization and grid search.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


@dataclass
class BTRAlgorithmConfig:
    """Configuration for a single BTR algorithm variant."""
    name: str
    variant: str  # "IQN" or "C51"
    default_params: Dict[str, Any]
    optuna_space: Optional[Dict[str, Callable]] = None
    grid_search_params: Optional[List[Dict[str, Any]]] = None
    description: str = ""

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters for training."""
        return self.default_params.copy()


def build_iqn_config(enable_optuna: bool = False) -> BTRAlgorithmConfig:
    """
    Build Implicit Quantile Network (IQN) configuration for BTR.
    
    IQN: Distributional RL with implicit quantile functions for flexible value distributions.
    Better for environments with complex reward structures (like energy management).
    """
    default_params = {
        # Core learning parameters
        "learning_rate": 1e-4,
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "gamma": 0.99,
        "tau": 0.001,  # soft update for target network
        
        # IQN-specific
        "n_quantiles": 200,  # Number of quantile samples for implicit distribution
        "n_quantiles_samples": 32,  # Samples for computing loss
        "kappa": 1.0,  # Huber loss parameter
        
        # Network architecture
        "hidden_size": 512 * 1,
        "n_hidden_layers": 2,
        "activation": "relu",
        
        # Exploration
        "use_noisy_nets": True,  # Noisy networks for exploration
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        
        # Stability
        "use_spectral_norm": True,  # Spectral normalization for stability
        "use_munchausen": True,  # Munchausen addon for enhanced rewards
        "max_grad_norm": 10.0,
        "gradient_clip": True,
        
        # Training dynamics
        "target_update_freq": 10000,
        "train_freq": 4,
        "gradient_steps": 1,
        
        # Replay buffer
        "use_per": True,  # Prioritized Experience Replay
        "per_alpha": 0.6,  # How much prioritization
        "per_beta": 0.4,  # How much importance sampling correction
        "per_beta_increment": 0.001,
    }
    
    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("iqn_lr", 5e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("iqn_bs", [64, 128, 256]),
            "gamma": lambda trial: trial.suggest_float("iqn_gamma", 0.95, 0.9999, log=True),
            "n_quantiles": lambda trial: trial.suggest_categorical("iqn_n_quant", [64, 128, 200, 256]),
            "n_quantiles_samples": lambda trial: trial.suggest_categorical("iqn_n_quant_samples", [16, 32, 64]),
            "hidden_size": lambda trial: trial.suggest_categorical("iqn_hidden", [256 * 1, 512 * 1, 1024 * 1]),
            "exploration_fraction": lambda trial: trial.suggest_float("iqn_exp_frac", 0.05, 0.3),
            "exploration_final_eps": lambda trial: trial.suggest_float("iqn_eps_final", 0.01, 0.1),
            "max_grad_norm": lambda trial: trial.suggest_float("iqn_grad_norm", 5.0, 20.0),
            "per_alpha": lambda trial: trial.suggest_float("iqn_per_alpha", 0.3, 0.8),
        }
    
    grid_search_params = [
        # Baseline
        default_params,
        # Larger batch size, lower learning rate
        {**default_params, "batch_size": 256, "learning_rate": 5e-5},
        # More quantiles for finer distribution
        {**default_params, "n_quantiles": 256, "n_quantiles_samples": 64},
        # Less exploration
        {**default_params, "exploration_final_eps": 0.01},
        # More PER prioritization
        {**default_params, "per_alpha": 0.8},
    ]
    
    return BTRAlgorithmConfig(
        name="BTR-IQN",
        variant="IQN",
        default_params=default_params,
        optuna_space=optuna_space,
        grid_search_params=grid_search_params,
        description="Implicit Quantile Network - Distributional RL with flexible value distributions",
    )


def build_c51_config(enable_optuna: bool = False) -> BTRAlgorithmConfig:
    """
    Build Categorical DQN (C51) configuration for BTR.
    
    C51: Distributional RL with categorical atoms over fixed support.
    Simpler than IQN, often faster to train.
    """
    default_params = {
        # Core learning parameters
        "learning_rate": 1e-4,
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "gamma": 0.99,
        "tau": 0.001,  # soft update for target network
        
        # C51-specific
        "n_atoms": 51,  # Number of atoms for categorical distribution
        "v_min": -10.0,  # Minimum value support
        "v_max": 10.0,  # Maximum value support
        
        # Network architecture
        "hidden_size": 512 * 1,
        "n_hidden_layers": 2,
        "activation": "relu",
        
        # Exploration
        "use_noisy_nets": True,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        
        # Stability
        "use_spectral_norm": True,
        "use_munchausen": True,
        "max_grad_norm": 10.0,
        "gradient_clip": True,
        
        # Training dynamics
        "target_update_freq": 10000,
        "train_freq": 4,
        "gradient_steps": 1,
        
        # Replay buffer
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "per_beta_increment": 0.001,
    }
    
    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("c51_lr", 5e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("c51_bs", [64, 128, 256]),
            "gamma": lambda trial: trial.suggest_float("c51_gamma", 0.95, 0.9999, log=True),
            "n_atoms": lambda trial: trial.suggest_categorical("c51_n_atoms", [31, 51, 81]),
            "v_min": lambda trial: trial.suggest_float("c51_v_min", -20.0, -5.0),
            "v_max": lambda trial: trial.suggest_float("c51_v_max", 5.0, 20.0),
            "hidden_size": lambda trial: trial.suggest_categorical("c51_hidden", [256 * 1, 512 * 1, 1024 * 1]),
            "exploration_fraction": lambda trial: trial.suggest_float("c51_exp_frac", 0.05, 0.3),
            "exploration_final_eps": lambda trial: trial.suggest_float("c51_eps_final", 0.01, 0.1),
            "max_grad_norm": lambda trial: trial.suggest_float("c51_grad_norm", 5.0, 20.0),
        }
    
    grid_search_params = [
        # Baseline
        default_params,
        # Larger batch size, lower learning rate
        {**default_params, "batch_size": 256, "learning_rate": 5e-5},
        # More atoms for finer distribution
        {**default_params, "n_atoms": 81},
        # Wider support for value range
        {**default_params, "v_min": -20.0, "v_max": 20.0},
        # Less exploration
        {**default_params, "exploration_final_eps": 0.01},
    ]
    
    return BTRAlgorithmConfig(
        name="BTR-C51",
        variant="C51",
        default_params=default_params,
        optuna_space=optuna_space,
        grid_search_params=grid_search_params,
        description="Categorical DQN - Distributional RL with fixed atom distribution",
    )


def get_btr_algorithm_config(
    algorithm_name: str,
    enable_optuna: bool = False
) -> BTRAlgorithmConfig:
    """
    Retrieve BTR algorithm configuration by name.
    
    Args:
        algorithm_name: "IQN" or "C51"
        enable_optuna: If True, include Optuna search space in config
        
    Returns:
        BTRAlgorithmConfig instance
    """
    configs = {
        "IQN": build_iqn_config(enable_optuna),
        "C51": build_c51_config(enable_optuna),
    }
    
    if algorithm_name not in configs:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Supported: {list(configs.keys())}"
        )
    
    return configs[algorithm_name]


def get_all_btr_algorithms() -> List[str]:
    """Return list of supported BTR algorithm names."""
    return ["IQN", "C51"]


def suggest_hyperparams_optuna(
    trial,
    algorithm_name: str
) -> Dict[str, Any]:
    """
    Suggest hyperparameters for an algorithm using Optuna trial.
    
    Args:
        trial: Optuna trial object
        algorithm_name: "IQN" or "C51"
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    config = get_btr_algorithm_config(algorithm_name, enable_optuna=True)
    
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed. Install with: pip install optuna")
    if config.optuna_space is None:
        raise ValueError(f"Optuna optimization not available for {algorithm_name}")
    
    suggested = {}
    for param_name, suggester_fn in config.optuna_space.items():
        suggested[param_name] = suggester_fn(trial)
    
    return suggested


def get_grid_search_params(algorithm_name: str) -> List[Dict[str, Any]]:
    """
    Get predefined parameter grid for algorithm.
    
    Args:
        algorithm_name: "IQN" or "C51"
        
    Returns:
        List of parameter dictionaries for grid search
    """
    config = get_btr_algorithm_config(algorithm_name, enable_optuna=False)
    
    if config.grid_search_params is None:
        # Fallback: return only default params
        return [config.get_params()]
    
    return config.grid_search_params
