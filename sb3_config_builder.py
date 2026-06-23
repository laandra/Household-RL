"""
SB3 Algorithm Configuration and Hyperparameter Space Builder
Provides predefined configs and Optuna search spaces for each algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional
import optuna


@dataclass
class SB3AlgorithmConfig:
    """Configuration for a single SB3 algorithm."""
    name: str
    algorithm_class: str  # "DQN", "PPO", "A2C", etc.
    default_params: Dict[str, Any]
    optuna_space: Optional[Dict[str, Callable]] = None
    description: str = ""

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters for training."""
        return self.default_params.copy()


def build_dqn_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build DQN configuration."""
    default_params = {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 0.001,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 10000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10.0,
    }

    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("dqn_lr", 1e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("dqn_batch_size", [32, 64, 128]),
            "gamma": lambda trial: trial.suggest_float("dqn_gamma", 0.95, 0.9999, log=True),
            "exploration_fraction": lambda trial: trial.suggest_float("dqn_exp_frac", 0.05, 0.3),
            "exploration_initial_eps": lambda trial: trial.suggest_float("dqn_eps_init", 0.8, 1.0),
            "exploration_final_eps": lambda trial: trial.suggest_float("dqn_eps_final", 0.01, 0.1),
            "tau": lambda trial: trial.suggest_float("dqn_tau", 0.0001, 0.01, log=True),
            "max_grad_norm": lambda trial: trial.suggest_float("dqn_grad_norm", 5.0, 20.0),
        }

    return SB3AlgorithmConfig(
        name="DQN",
        algorithm_class="DQN",
        default_params=default_params,
        optuna_space=optuna_space,
        description="Deep Q-Network with experience replay",
    )


def build_ppo_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build PPO configuration."""
    default_params = {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("ppo_batch_size", [32, 64, 128]),
            "n_epochs": lambda trial: trial.suggest_categorical("ppo_n_epochs", [5, 10, 20]),
            "gamma": lambda trial: trial.suggest_float("ppo_gamma", 0.95, 0.9999, log=True),
            "gae_lambda": lambda trial: trial.suggest_float("ppo_gae_lambda", 0.8, 0.99),
            "clip_range": lambda trial: trial.suggest_float("ppo_clip", 0.1, 0.4),
            "ent_coef": lambda trial: trial.suggest_float("ppo_ent", 0.0, 0.01),
            "vf_coef": lambda trial: trial.suggest_float("ppo_vf", 0.3, 0.7),
        }

    return SB3AlgorithmConfig(
        name="PPO",
        algorithm_class="PPO",
        default_params=default_params,
        optuna_space=optuna_space,
        description="Proximal Policy Optimization",
    )


def build_a2c_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build A2C configuration."""
    default_params = {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "rms_prop_eps": 1e-5,
        "use_rms_prop": True,
        "use_sde": False,
    }

    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("a2c_lr", 1e-5, 1e-3, log=True),
            "n_steps": lambda trial: trial.suggest_categorical("a2c_n_steps", [5, 8, 16]),
            "gamma": lambda trial: trial.suggest_float("a2c_gamma", 0.95, 0.9999, log=True),
            "ent_coef": lambda trial: trial.suggest_float("a2c_ent", 0.0, 0.01),
            "vf_coef": lambda trial: trial.suggest_float("a2c_vf", 0.3, 0.7),
            "use_sde": lambda trial: trial.suggest_categorical("a2c_sde", [False, True]),
        }

    return SB3AlgorithmConfig(
        name="A2C",
        algorithm_class="A2C",
        default_params=default_params,
        optuna_space=optuna_space,
        description="Advantage Actor-Critic",
    )


def build_qr_dqn_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build QR-DQN configuration (from sb3-contrib)."""
    default_params = {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 0.001,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 10000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10.0,
        "n_quantiles": 200,
    }

    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("qrdqn_lr", 1e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("qrdqn_batch_size", [32, 64, 128]),
            "gamma": lambda trial: trial.suggest_float("qrdqn_gamma", 0.95, 0.9999, log=True),
            "n_quantiles": lambda trial: trial.suggest_categorical("qrdqn_n_quantiles", [50, 100, 200]),
        }

    return SB3AlgorithmConfig(
        name="QR_DQN",
        algorithm_class="QR_DQN",
        default_params=default_params,
        optuna_space=optuna_space,
        description="Quantile Regression DQN (distributional RL)",
    )


def build_maskable_ppo_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build Maskable PPO configuration (from sb3-contrib)."""
    default_params = {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("mppo_lr", 1e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("mppo_batch_size", [32, 64, 128]),
            "n_epochs": lambda trial: trial.suggest_categorical("mppo_n_epochs", [5, 10, 20]),
            "gamma": lambda trial: trial.suggest_float("mppo_gamma", 0.95, 0.9999, log=True),
            "clip_range": lambda trial: trial.suggest_float("mppo_clip", 0.1, 0.4),
        }

    return SB3AlgorithmConfig(
        name="MASKABLE_PPO",
        algorithm_class="MaskablePPO",
        default_params=default_params,
        optuna_space=optuna_space,
        description="Maskable PPO with action masking (sb3-contrib)",
    )


def build_recurrent_ppo_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build Recurrent PPO configuration (from sb3-contrib)."""
    default_params = {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "lstm_hidden_size": 256 * 1,
        "n_lstm_layers": 1,
    }

    optuna_space = None
    if enable_optuna:
        optuna_space = {
            "learning_rate": lambda trial: trial.suggest_float("rppo_lr", 1e-5, 1e-3, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("rppo_batch_size", [32, 64]),
            "n_epochs": lambda trial: trial.suggest_categorical("rppo_n_epochs", [5, 10]),
            "gamma": lambda trial: trial.suggest_float("rppo_gamma", 0.95, 0.9999, log=True),
            "lstm_hidden_size": lambda trial: trial.suggest_categorical("rppo_lstm_size", [128 * 1, 256 * 1, 512 * 1]),
        }

    return SB3AlgorithmConfig(
        name="RECURRENT_PPO",
        algorithm_class="RecurrentPPO",
        default_params=default_params,
        optuna_space=optuna_space,
        description="Recurrent PPO with LSTM layers (sb3-contrib)",
    )


def get_algorithm_config(algorithm_name: str, enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Get configuration for a specific algorithm.
    
    Args:
        algorithm_name: One of "DQN", "PPO", "A2C", "QR_DQN", "MASKABLE_PPO", "RECURRENT_PPO"
        enable_optuna: Whether to include Optuna search spaces
        
    Returns:
        SB3AlgorithmConfig instance
        
    Raises:
        ValueError: If algorithm_name is not recognized
    """
    config_builders = {
        "DQN": build_dqn_config,
        "PPO": build_ppo_config,
        "A2C": build_a2c_config,
        "QR_DQN": build_qr_dqn_config,
        "MASKABLE_PPO": build_maskable_ppo_config,
        "RECURRENT_PPO": build_recurrent_ppo_config,
    }

    if algorithm_name not in config_builders:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {', '.join(config_builders.keys())}"
        )

    return config_builders[algorithm_name](enable_optuna=enable_optuna)


def get_all_algorithm_names() -> list:
    """Get list of all supported algorithm names."""
    return ["DQN", "PPO", "A2C", "QR_DQN", "MASKABLE_PPO", "RECURRENT_PPO"]
