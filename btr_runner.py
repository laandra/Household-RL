"""
BTR Training Orchestration and Multi-Seed Benchmarking
Handles single-seed training, evaluation, and multi-seed aggregation for BTR algorithms.
Mirrors SB3Trainer structure for consistency.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import traceback
import torch

from btr_environment_adapter import BTREnvironmentAdapter
from btr_agent import BTRAgent
from btr_config_builder import (
    get_btr_algorithm_config,
    suggest_hyperparams_optuna,
    get_grid_search_params,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_env_metadata(env) -> Dict[str, Any]:
    """Extract comparable episode metadata from environment instances."""
    episode_length_steps = _safe_int(getattr(env, "episode_length", 0), default=0)
    korakov_na_dan = _safe_int(getattr(env, "korakov_na_dan", 0), default=0)
    episode_days = (
        float(episode_length_steps) / float(korakov_na_dan)
        if episode_length_steps > 0 and korakov_na_dan > 0
        else None
    )

    return {
        "episode_length_steps": episode_length_steps,
        "korakov_na_dan": korakov_na_dan,
        "episode_days": episode_days,
        "reset_mode": str(getattr(env, "reset_mode", "unknown")),
        "observation_mode": str(getattr(env, "observation_mode", "unknown")),
        "data_length_steps": _safe_int(getattr(env, "data_length", 0), default=0),
    }


def _run_deterministic_horizon_eval_btr(agent, env_orig, device: str = "cpu") -> Dict[str, Any]:
    """Run one deterministic rollout over full dataset and extract full + first-7-day totals."""
    prev_reset_mode = getattr(env_orig, "reset_mode", None)
    prev_episode_length = getattr(env_orig, "episode_length", None)

    data_length = _safe_int(getattr(env_orig, "data_length", 0), default=0)
    korakov_na_dan = _safe_int(getattr(env_orig, "korakov_na_dan", 0), default=0)
    seven_day_steps = 7 * korakov_na_dan if korakov_na_dan > 0 else 0

    total_reward = 0.0
    total_price = 0.0
    total_steps = 0
    reward_7day = None
    price_7day = None

    try:
        if hasattr(env_orig, "reset_mode"):
            env_orig.reset_mode = "deterministic"
        if data_length > 0 and hasattr(env_orig, "episode_length"):
            env_orig.episode_length = data_length

        eval_env = BTREnvironmentAdapter(env_orig, device=device, use_torch=False)
        obs, _ = eval_env.reset()
        done = False

        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            total_steps += 1
            if "cumulative_payment" in info:
                total_price = float(info["cumulative_payment"])

            if reward_7day is None and seven_day_steps > 0 and total_steps >= seven_day_steps:
                reward_7day = float(total_reward)
                price_7day = float(total_price)

        if reward_7day is None:
            reward_7day = float(total_reward)
            price_7day = float(total_price)

        return {
            "total_reward": float(total_reward),
            "total_price": float(total_price),
            "total_steps": int(total_steps),
            "reward_7day": float(reward_7day),
            "price_7day": float(price_7day),
            "seven_day_steps": int(seven_day_steps),
            "seven_day_truncated": bool(seven_day_steps <= 0 or total_steps < seven_day_steps),
            "korakov_na_dan": int(korakov_na_dan),
        }
    finally:
        if prev_reset_mode is not None and hasattr(env_orig, "reset_mode"):
            env_orig.reset_mode = prev_reset_mode
        if prev_episode_length is not None and hasattr(env_orig, "episode_length"):
            env_orig.episode_length = prev_episode_length


class BTRTrainer:
    """Single-algorithm, multi-seed trainer for BTR agents."""

    def __init__(
        self,
        algorithm_name: str,
        env,
        eval_env,
        total_timesteps: int = 60000,
        n_eval_episodes: int = 10,
        eval_freq: int = 5000,
        seed: int = None,
        device: str = "cpu",
        verbose: int = 0,
    ):
        """
        Initialize trainer.
        
        Args:
            algorithm_name: "IQN" or "C51"
            env: Training environment (HouseholdEnvironment)
            eval_env: Evaluation environment
            total_timesteps: Total training timesteps
            n_eval_episodes: Episodes per evaluation
            eval_freq: Evaluate every N timesteps
            seed: Random seed
            device: PyTorch device ("cpu" or "cuda")
            verbose: Verbosity level
        """
        self.algorithm_name = algorithm_name
        self.env_orig = env
        self.eval_env_orig = eval_env
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.seed = seed
        self.device = device
        self.verbose = verbose
        
        # Wrap environments
        self.env = BTREnvironmentAdapter(env, device=device, use_torch=False)
        self.eval_env = BTREnvironmentAdapter(eval_env, device=device, use_torch=False)
        
        self.agent = None
        self.eval_rewards = []
        self.eval_prices = []
        self.timestamps = []
        self.training_steps_count = 0

    def train(self, hyperparams: Optional[Dict[str, Any]] = None, use_optuna: bool = False, n_optuna_trials: int = 3) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            hyperparams: Override default hyperparameters
            use_optuna: If True, use Optuna optimization
            n_optuna_trials: Number of Optuna trials
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Set random seeds
            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
            
            # Get config
            config = get_btr_algorithm_config(self.algorithm_name, enable_optuna=False)
            params = config.get_params()
            if hyperparams:
                params.update(hyperparams)
            
            # Optuna optimization
            if use_optuna and OPTUNA_AVAILABLE:
                params = self._optuna_optimize(n_optuna_trials, params)
                if self.verbose > 0:
                    print(f"[{self.algorithm_name}@seed={self.seed}] Optuna best params: {params}")
            
            # Create agent
            self.agent = BTRAgent(
                state_shape=self.env.get_state_shape()[0],
                n_actions=self.env.get_n_actions(),
                algorithm=self.algorithm_name,
                device=self.device,
                **params
            )
            self.agent.total_training_steps_max = self.total_timesteps
            
            # Training loop
            self.eval_rewards = []
            self.eval_prices = []
            self.timestamps = []
            self.training_steps_count = 0
            
            state, _ = self.env.reset(seed=self.seed)
            episode_reward = 0.0
            episode_price = 0.0
            
            while self.training_steps_count < self.total_timesteps:
                # Select action
                self.agent.total_steps = self.training_steps_count
                action = self.agent.select_action(state, training=True)
                
                # Step environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Learn
                if self.training_steps_count % self.agent.train_freq == 0:
                    for _ in range(self.agent.gradient_steps):
                        self.agent.learn_step()
                
                # Reset noise
                self.agent.reset_noise()
                
                # Track episode metrics
                episode_reward += reward
                if "cumulative_payment" in info:
                    episode_price = float(info["cumulative_payment"])
                
                self.training_steps_count += 1
                
                # Reset if done
                if done:
                    state, _ = self.env.reset(seed=self.seed)
                    episode_reward = 0.0
                    episode_price = 0.0
                else:
                    state = next_state
                
                # Periodic evaluation
                if self.training_steps_count % self.eval_freq == 0:
                    episode_rewards, episode_prices = self._evaluate()
                    self.eval_rewards.append(episode_rewards)
                    self.eval_prices.append(episode_prices)
                    self.timestamps.append(self.training_steps_count)
                    
                    if self.verbose > 0:
                        print(
                            f"[{self.algorithm_name}@seed={self.seed}] "
                            f"Timesteps: {self.training_steps_count}/{self.total_timesteps} | "
                            f"Reward: {episode_rewards:.2f} | Price: {episode_prices:.2f}"
                        )
            
            # Compute final metrics
            final_metrics = self._compute_final_metrics(params)
            return final_metrics

        except Exception as e:
            print(f"Error training {self.algorithm_name}: {repr(e)}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "price_mean": 0.0,
                "price_std": 0.0,
            }

    def _optuna_optimize(self, n_trials: int, default_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Optuna optimization to find best hyperparameters."""
        if not OPTUNA_AVAILABLE:
            return default_params
        
        def objective(trial):
            # Suggest hyperparameters
            params = suggest_hyperparams_optuna(trial, self.algorithm_name)
            
            # Train briefly
            agent = BTRAgent(
                state_shape=self.env.get_state_shape()[0],
                n_actions=self.env.get_n_actions(),
                algorithm=self.algorithm_name,
                device=self.device,
                **params
            )
            
            # Mini training loop (limited timesteps for speed)
            mini_steps = min(10000, self.total_timesteps // 10)
            agent.total_training_steps_max = mini_steps
            
            state, _ = self.env.reset(seed=self.seed)
            rewards = []
            
            for step in range(mini_steps):
                agent.total_steps = step
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                if step >= agent.learning_starts:
                    agent.learn_step()
                agent.reset_noise()
                
                rewards.append(reward)
                
                if done:
                    state, _ = self.env.reset(seed=self.seed)
                else:
                    state = next_state
            
            # Return mean reward (to be maximized, but we negate for minimization if needed)
            return np.mean(rewards)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=self.verbose > 0)
        
        best_params = default_params.copy()
        best_params.update(study.best_params)
        return best_params

    def _evaluate(self) -> Tuple[float, float]:
        """
        Evaluate agent and extract reward and price metrics.
        
        Returns:
            Tuple of (mean_reward, mean_price)
        """
        episode_rewards = []
        episode_prices = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed)
            episode_reward = 0.0
            episode_price = 0.0
            done = False
            
            while not done:
                action = self.agent.select_action(obs, training=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if "cumulative_payment" in info:
                    episode_price = float(info["cumulative_payment"])
            
            episode_rewards.append(episode_reward)
            episode_prices.append(episode_price)
        
        mean_reward = np.mean(episode_rewards)
        mean_price = np.mean(episode_prices)
        
        return mean_reward, mean_price

    def _compute_final_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final training metrics."""
        rewards = np.array(self.eval_rewards)
        prices = np.array(self.eval_prices)

        return {
            "success": True,
            "algorithm": self.algorithm_name,
            "seed": self.seed,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "price_mean": float(np.mean(prices)),
            "price_std": float(np.std(prices)),
            "total_timesteps": self.total_timesteps,
            "n_eval_episodes": self.n_eval_episodes,
            "hyperparams": params,
        }

    def save_agent(self, path: str) -> None:
        """Save trained agent checkpoint."""
        if self.agent is not None:
            self.agent.save_checkpoint(path)

    def load_agent(self, path: str) -> None:
        """Load trained agent checkpoint."""
        if self.agent is None:
            self.agent = BTRAgent(
                state_shape=self.env.get_state_shape()[0],
                n_actions=self.env.get_n_actions(),
                algorithm=self.algorithm_name,
                device=self.device,
            )
        self.agent.load_checkpoint(path)


def run_btr_benchmark(
    train_env,
    eval_env,
    test_env,
    algorithms: List[str] = None,
    total_timesteps: int = 60000,
    n_seeds: int = 3,
    n_eval_episodes: int = 10,
    eval_freq: int = 5000,
    output_dir: str = "BTR",
    device: str = "cpu",
    use_optuna: bool = False,
    n_optuna_trials: int = 3,
    verbose: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Run BTR benchmark across multiple algorithms and seeds.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment (same config as train, different data)
        test_env: Test environment (for final evaluation)
        algorithms: List of algorithm names to benchmark ("IQN", "C51")
        total_timesteps: Total training timesteps per seed
        n_seeds: Number of seeds to train
        n_eval_episodes: Episodes per evaluation
        eval_freq: Evaluate every N timesteps during training
        output_dir: Directory to save results and models
        device: PyTorch device ("cpu" or "cuda")
        use_optuna: Enable Optuna hyperparameter optimization
        n_optuna_trials: Number of Optuna trials
        verbose: Verbosity level
        
    Returns:
        Dictionary mapping algorithm_name -> {reward_mean, reward_std, price_mean, price_std}
    """
    if algorithms is None:
        algorithms = ["IQN", "C51"]
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    results = {}
    all_metrics = []
    eval_meta = _extract_env_metadata(eval_env)
    test_meta = _extract_env_metadata(test_env if test_env is not None else eval_env)
    
    for algo in algorithms:
        algo_rewards = []
        algo_prices = []
        algo_rewards_7day = []
        algo_prices_7day = []
        algo_models = []
        
        for seed in range(n_seeds):
            if verbose > 0:
                print(f"\n{'='*60}")
                print(f"Training {algo} - Seed {seed+1}/{n_seeds}")
                print(f"{'='*60}")
            
            trainer = BTRTrainer(
                algorithm_name=algo,
                env=train_env,
                eval_env=eval_env,
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                seed=seed,
                device=device,
                verbose=verbose,
            )
            
            metrics = trainer.train(use_optuna=use_optuna, n_optuna_trials=n_optuna_trials)
            
            if metrics["success"]:
                final_eval_env = test_env if test_env is not None else eval_env
                final_horizon = _run_deterministic_horizon_eval_btr(
                    trainer.agent,
                    final_eval_env,
                    device=device,
                )

                algo_rewards.append(float(final_horizon["total_reward"]))
                algo_prices.append(float(final_horizon["total_price"]))
                algo_rewards_7day.append(float(final_horizon["reward_7day"]))
                algo_prices_7day.append(float(final_horizon["price_7day"]))

                data_steps = _safe_int(test_meta.get("data_length_steps"), default=0)
                data_days = (
                    float(data_steps) / float(final_horizon["korakov_na_dan"])
                    if data_steps > 0 and final_horizon["korakov_na_dan"] > 0
                    else None
                )
                price_mean_eur_per_day = (
                    float(final_horizon["total_price"]) / float(data_days)
                    if data_days
                    else None
                )
                
                # Save model
                model_path = os.path.join(models_dir, f"{algo}_seed{seed}.pt")
                trainer.save_agent(model_path)
                algo_models.append(model_path)
                
                # Save metrics
                all_metrics.append({
                    "algorithm": algo,
                    "seed": seed,
                    "reward_mean": float(final_horizon["total_reward"]),
                    "reward_7day": float(final_horizon["reward_7day"]),
                    "reward_mean_eval_checkpoints": float(metrics["reward_mean"]),
                    "reward_std_eval_checkpoints": float(metrics["reward_std"]),
                    "reward_std": float(metrics["reward_std"]),
                    "price_mean": float(final_horizon["total_price"]),
                    "price_7day": float(final_horizon["price_7day"]),
                    "price_mean_eval_checkpoints": float(metrics["price_mean"]),
                    "price_std_eval_checkpoints": float(metrics["price_std"]),
                    "price_std": float(metrics["price_std"]),
                    "price_mean_eur_per_day": price_mean_eur_per_day,
                    "total_timesteps": int(total_timesteps),
                    "n_eval_episodes": int(n_eval_episodes),
                    "final_eval_steps": int(final_horizon["total_steps"]),
                    "final_eval_data_length_steps": int(data_steps),
                    "final_eval_days": data_days,
                    "seven_day_steps": int(final_horizon["seven_day_steps"]),
                    "seven_day_truncated": bool(final_horizon["seven_day_truncated"]),
                    "final_eval_source": "test_env" if test_env is not None else "eval_env_fallback",
                    "final_eval_semantics": "deterministic full-horizon rollout cumulative totals",
                    "episode_length_steps": int(eval_meta["episode_length_steps"]),
                    "episode_days": eval_meta["episode_days"],
                    "korakov_na_dan": int(eval_meta["korakov_na_dan"]),
                    "reset_mode": eval_meta["reset_mode"],
                    "observation_mode": eval_meta["observation_mode"],
                    "model_path": model_path,
                })
        
        if algo_rewards:
            reward_std_across_seeds = float(np.std(algo_rewards))
            price_std_across_seeds = float(np.std(algo_prices))
            reward_7day_mean = float(np.mean(algo_rewards_7day))
            price_7day_mean = float(np.mean(algo_prices_7day))
            summary_data_days = (
                float(test_meta["data_length_steps"]) / float(test_meta["korakov_na_dan"])
                if test_meta["data_length_steps"] > 0 and test_meta["korakov_na_dan"] > 0
                else None
            )
            price_mean_eur_per_day = (
                float(np.mean(algo_prices)) / float(summary_data_days)
                if summary_data_days
                else None
            )

            results[algo] = {
                "reward_mean": float(np.mean(algo_rewards)),
                "reward_std": reward_std_across_seeds,
                "reward_std_across_seeds": reward_std_across_seeds,
                "reward_7day": reward_7day_mean,
                "price_mean": float(np.mean(algo_prices)),
                "price_std": price_std_across_seeds,
                "price_std_across_seeds": price_std_across_seeds,
                "price_7day": price_7day_mean,
                "price_mean_eur_per_day": price_mean_eur_per_day,
                "episode_length_steps": int(eval_meta["episode_length_steps"]),
                "episode_days": eval_meta["episode_days"],
                "korakov_na_dan": int(eval_meta["korakov_na_dan"]),
                "reset_mode": eval_meta["reset_mode"],
                "observation_mode": eval_meta["observation_mode"],
                "n_eval_episodes": int(n_eval_episodes),
                "total_timesteps": int(total_timesteps),
                "metric_semantics": {
                    "price_mean": "EUR total cumulative_payment over deterministic full-horizon test rollout",
                    "price_7day": "EUR cumulative_payment over first 7 days from deterministic test start",
                    "price_std": "across-seed std of deterministic full-horizon price_mean",
                    "reward_std": "across-seed std of deterministic full-horizon reward_mean",
                },
                "n_seeds": n_seeds,
            }
        else:
            results[algo] = {
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "price_mean": 0.0,
                "price_std": 0.0,
                "error": "Training failed",
            }
    
    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_csv = os.path.join(output_dir, "btr_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    summary_json = os.path.join(output_dir, "btr_summary.json")
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose > 0:
        print(f"\n{'='*60}")
        print("BTR Benchmark Summary")
        print(f"{'='*60}")
        for algo, metrics in results.items():
            print(f"\n{algo}:")
            print(f"  Reward: {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
            print(f"  Price:  {metrics['price_mean']:.2f} ± {metrics['price_std']:.2f}")
            print(f"  Quick 7D Price: {metrics['price_7day']:.2f}")
        print(f"\nResults saved to: {output_dir}")
    
    return results
