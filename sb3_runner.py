"""
SB3 Training Orchestration and Benchmarking
Handles multi-seed training, evaluation, and result aggregation for SB3 algorithms.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import traceback

from stable_baselines3 import DQN, PPO, A2C

try:
    from sb3_contrib import QRDQN, MaskablePPO, RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    QRDQN, MaskablePPO, RecurrentPPO = None, None, None

from sb3_config_builder import get_algorithm_config


class SB3Trainer:
    """Single-algorithm, multi-seed trainer for SB3 agents."""

    def __init__(
        self,
        algorithm_name: str,
        env,
        eval_env,
        total_timesteps: int = 60000,
        n_eval_episodes: int = 10,
        eval_freq: int = 5000,
        seed: int = None,
        verbose: int = 0,
    ):
        """Initialize trainer.
        
        Args:
            algorithm_name: e.g., "DQN", "PPO", etc.
            env: Training environment
            eval_env: Evaluation environment
            total_timesteps: Total training timesteps
            n_eval_episodes: Episodes per evaluation
            eval_freq: Evaluate every N timesteps
            seed: Random seed
            verbose: Verbosity level
        """
        self.algorithm_name = algorithm_name
        self.env = env
        self.eval_env = eval_env
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.seed = seed
        self.verbose = verbose
        self.model = None
        self.eval_rewards = []
        self.eval_prices = []
        self.timestamps = []

    def _get_algorithm_class(self, algorithm_name: str):
        """Get algorithm class from name."""
        algorithm_map = {
            "DQN": DQN,
            "PPO": PPO,
            "A2C": A2C,
            "QR_DQN": QRDQN,
            "MASKABLE_PPO": MaskablePPO,
            "RECURRENT_PPO": RecurrentPPO,
        }
        return algorithm_map.get(algorithm_name)

    def _check_algorithm_availability(self, algorithm_name: str) -> bool:
        """Check if algorithm is available."""
        if algorithm_name in ["DQN", "PPO", "A2C"]:
            return True
        if algorithm_name in ["QR_DQN", "MASKABLE_PPO", "RECURRENT_PPO"]:
            return SB3_CONTRIB_AVAILABLE
        return False

    def _prepare_model_kwargs(self, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Translate generic config params into algorithm-specific SB3 kwargs."""
        params = params.copy()
        policy_name = "MlpPolicy"

        if self.algorithm_name == "QR_DQN":
            n_quantiles = int(params.pop("n_quantiles", 200))
            params.pop("top_quantiles_to_drop_per_net", None)
            policy_kwargs = params.pop("policy_kwargs", {})
            policy_kwargs.update({"n_quantiles": n_quantiles})
            params["policy_kwargs"] = policy_kwargs

        if self.algorithm_name == "RECURRENT_PPO":
            policy_name = "MlpLstmPolicy"
            lstm_hidden_size = int(params.pop("lstm_hidden_size", 256))
            n_lstm_layers = int(params.pop("n_lstm_layers", 1))
            policy_kwargs = params.pop("policy_kwargs", {})
            policy_kwargs.update(
                {
                    "lstm_hidden_size": lstm_hidden_size,
                    "n_lstm_layers": n_lstm_layers,
                }
            )
            params["policy_kwargs"] = policy_kwargs

        return policy_name, params

    def train(self, hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train the agent.
        
        Args:
            hyperparams: Override default hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Check algorithm availability
            if not self._check_algorithm_availability(self.algorithm_name):
                raise RuntimeError(
                    f"{self.algorithm_name} requires sb3-contrib which is not installed"
                )

            # Get config and hyperparameters
            config = get_algorithm_config(self.algorithm_name, enable_optuna=False)
            params = config.get_params()
            if hyperparams:
                params.update(hyperparams)

            # Get algorithm class
            algo_class = self._get_algorithm_class(self.algorithm_name)
            if algo_class is None:
                raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

            # Create model
            policy_name, params = self._prepare_model_kwargs(params)

            self.model = algo_class(
                policy_name,
                self.env,
                seed=self.seed,
                verbose=self.verbose,
                **params,
            )

            # Training loop with periodic evaluation
            self.eval_rewards = []
            self.eval_prices = []
            self.timestamps = []

            timesteps_done = 0
            while timesteps_done < self.total_timesteps:
                remaining = self.total_timesteps - timesteps_done
                train_steps = min(self.eval_freq, remaining)

                self.model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
                timesteps_done += train_steps

                # Evaluate
                episode_rewards, episode_prices = self._evaluate()
                self.eval_rewards.append(episode_rewards)
                self.eval_prices.append(episode_prices)
                self.timestamps.append(timesteps_done)

                if self.verbose > 0:
                    print(
                        f"[{self.algorithm_name}@seed={self.seed}] "
                        f"Timesteps: {timesteps_done}/{self.total_timesteps} | "
                        f"Reward: {episode_rewards:.2f} | Price: {episode_prices:.2f}"
                    )

            # Compute final metrics
            final_metrics = self._compute_final_metrics()
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

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate model and extract reward and price metrics.
        
        Returns:
            Tuple of (mean_reward, mean_price)
        """
        episode_rewards = []
        episode_prices = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            episode_price = 0.0
            done = False
            recurrent_state = None
            episode_start = np.array([True], dtype=bool)
            
            while not done:
                if self.algorithm_name == "RECURRENT_PPO":
                    action, recurrent_state = self.model.predict(
                        obs,
                        state=recurrent_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if "cumulative_payment" in info:
                    episode_price = float(info["cumulative_payment"])

                episode_start = np.array([done], dtype=bool)
            
            episode_rewards.append(episode_reward)
            episode_prices.append(episode_price)
        
        mean_reward = np.mean(episode_rewards)
        mean_price = np.mean(episode_prices)
        
        return mean_reward, mean_price


    def _compute_final_metrics(self) -> Dict[str, Any]:
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
        }

    def save_model(self, path: str) -> None:
        """Save trained model."""
        if self.model is not None:
            self.model.save(path)

    def load_model(self, path: str) -> None:
        """Load trained model."""
        algo_class = self._get_algorithm_class(self.algorithm_name)
        self.model = algo_class.load(path, env=self.env)


def run_sb3_benchmark(
    train_env,
    eval_env,
    test_env,
    algorithms: List[str] = None,
    total_timesteps: int = 60000,
    n_seeds: int = 3,
    n_eval_episodes: int = 10,
    output_dir: str = "SB3",
    verbose: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Run SB3 benchmark across multiple algorithms and seeds.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment (same config as train, different data)
        test_env: Test environment (for final evaluation)
        algorithms: List of algorithm names to benchmark
        total_timesteps: Total training timesteps per seed
        n_seeds: Number of seeds to train
        n_eval_episodes: Episodes per evaluation
        output_dir: Directory to save results and models
        verbose: Verbosity level
        
    Returns:
        Dictionary mapping algorithm_name -> {reward_mean, reward_std, price_mean, price_std}
    """
    if algorithms is None:
        algorithms = ["DQN", "PPO", "A2C", "QR_DQN", "MASKABLE_PPO", "RECURRENT_PPO"]

    # Filter out unavailable algorithms
    available_algorithms = []
    for algo in algorithms:
        trainer = SB3Trainer(algo, train_env, eval_env)
        if trainer._check_algorithm_availability(algo):
            available_algorithms.append(algo)
        else:
            print(f"Warning: {algo} not available (requires sb3-contrib), skipping")

    algorithms = available_algorithms

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Storage for results
    all_results = {algo: {"rewards": [], "prices": []} for algo in algorithms}
    summary = {}

    print(f"\n{'='*60}")
    print(f"SB3 Benchmark: {len(algorithms)} algorithms × {n_seeds} seeds")
    print(f"{'='*60}\n")

    seeds = [11 + 11 * idx for idx in range(n_seeds)]

    for algorithm in algorithms:
        print(f"\n{'─'*60}")
        print(f"Training {algorithm}")
        print(f"{'─'*60}")

        algo_dir = os.path.join(output_dir, algorithm)
        os.makedirs(algo_dir, exist_ok=True)

        algo_rewards = []
        algo_prices = []

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)

            trainer = SB3Trainer(
                algorithm,
                train_env,
                eval_env,
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes,
                seed=seed,
                verbose=0,
            )

            metrics = trainer.train()

            if metrics.get("success", False):
                algo_rewards.append(metrics["reward_mean"])
                algo_prices.append(metrics["price_mean"])

                # Save best model per seed
                seed_model_path = os.path.join(algo_dir, f"model_seed_{seed}")
                trainer.save_model(seed_model_path)

                print(
                    f"✓ R={metrics['reward_mean']:.2f}±{metrics['reward_std']:.2f} | "
                    f"P={metrics['price_mean']:.2f}±{metrics['price_std']:.2f}"
                )
            else:
                print(f"✗ Error: {metrics.get('error', 'Unknown')}")

        # Aggregate results across seeds
        if algo_rewards:
            summary[algorithm] = {
                "reward_mean": float(np.mean(algo_rewards)),
                "reward_std": float(np.std(algo_rewards)),
                "price_mean": float(np.mean(algo_prices)),
                "price_std": float(np.std(algo_prices)),
                "n_seeds": len(algo_rewards),
            }

            all_results[algorithm]["rewards"] = algo_rewards
            all_results[algorithm]["prices"] = algo_prices

            print(
                f"\n  Summary {algorithm}:\n"
                f"    Reward: {summary[algorithm]['reward_mean']:.2f} ± {summary[algorithm]['reward_std']:.2f}\n"
                f"    Price:  {summary[algorithm]['price_mean']:.2f} ± {summary[algorithm]['price_std']:.2f}"
            )
        else:
            print(f"\n  ✗ No successful runs for {algorithm}")

    # Save summary to file
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete. Results saved to {output_dir}/")
    print(f"{'='*60}\n")

    return summary


def create_comparison_table(
    sb3_results: Dict[str, Dict[str, float]],
    baseline_results: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Create comparison table for SB3 algorithms vs baseline.
    
    Args:
        sb3_results: Output from run_sb3_benchmark()
        baseline_results: Optional baseline (e.g., custom DQN) results dict
                         with keys: reward_mean, reward_std, price_mean, price_std
        
    Returns:
        DataFrame with all algorithms and metrics
    """
    rows = []

    # Add baseline if provided
    if baseline_results:
        rows.append({
            "Algorithm": "Custom_DQN_Baseline",
            "Reward_Mean": baseline_results.get("reward_mean", 0.0),
            "Reward_Std": baseline_results.get("reward_std", 0.0),
            "Price_Mean": baseline_results.get("price_mean", 0.0),
            "Price_Std": baseline_results.get("price_std", 0.0),
        })

    # Add SB3 results
    for algo_name, metrics in sb3_results.items():
        rows.append({
            "Algorithm": algo_name,
            "Reward_Mean": metrics.get("reward_mean", 0.0),
            "Reward_Std": metrics.get("reward_std", 0.0),
            "Price_Mean": metrics.get("price_mean", 0.0),
            "Price_Std": metrics.get("price_std", 0.0),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Price_Mean")
