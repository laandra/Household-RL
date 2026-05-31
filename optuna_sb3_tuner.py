"""
Optuna-based Hyperparameter Tuning for SB3 Agents
Supports per-algorithm tuning with pruning and early stopping.
"""

import os
import json
import optuna
from optuna.pruners import MedianPruner
from typing import Dict, Any, Optional, Callable
import traceback

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

try:
    from sb3_contrib import QRDQN, MaskablePPO, RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False
    QRDQN, MaskablePPO, RecurrentPPO = None, None, None

from sb3_config_builder import get_algorithm_config


class SB3Tuner:
    """Optuna-based hyperparameter tuner for SB3 algorithms."""

    def __init__(
        self,
        algorithm_name: str,
        env,
        eval_env,
        n_trials: int = 20,
        study_name: Optional[str] = None,
        storage_dir: str = "optuna_studies",
        verbose: int = 0,
    ):
        """Initialize tuner.
        
        Args:
            algorithm_name: Algorithm to tune (e.g., "DQN", "PPO")
            env: Training environment
            eval_env: Evaluation environment
            n_trials: Number of Optuna trials
            study_name: Custom study name (default: {algo_name}_tuning)
            storage_dir: Directory to store Optuna study database
            verbose: Verbosity level
        """
        self.algorithm_name = algorithm_name
        self.env = env
        self.eval_env = eval_env
        self.n_trials = n_trials
        self.verbose = verbose
        self.study_name = study_name or f"{algorithm_name}_tuning"
        self.storage_dir = storage_dir

        os.makedirs(storage_dir, exist_ok=True)

        self.config = get_algorithm_config(algorithm_name, enable_optuna=True)
        self.best_params = None
        self.study = None

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

    def _prepare_model_kwargs(self, params: Dict[str, Any]):
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

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for a single trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score to optimize (negative mean price for minimization)
        """
        try:
            # Check availability
            if not self._check_algorithm_availability(self.algorithm_name):
                raise RuntimeError(
                    f"{self.algorithm_name} requires sb3-contrib which is not installed"
                )

            # Get base params
            base_params = self.config.get_params()

            # Suggest hyperparameters
            if self.config.optuna_space:
                for param_name, suggest_fn in self.config.optuna_space.items():
                    try:
                        suggested_value = suggest_fn(trial)
                        base_params[param_name] = suggested_value
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Warning: Could not suggest {param_name}: {e}")

            # Get algorithm class
            algo_class = self._get_algorithm_class(self.algorithm_name)
            if algo_class is None:
                raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

            # Create and train model
            policy_name, base_params = self._prepare_model_kwargs(base_params)

            model = algo_class(
                policy_name,
                self.env,
                seed=trial.number,
                verbose=0,
                **base_params,
            )

            # Train for a limited number of steps
            # (typical: 10k-50k for tuning, full training happens after tuning)
            model.learn(total_timesteps=20000)

            # Evaluate
            mean_reward, _ = evaluate_policy(
                model, self.eval_env, n_eval_episodes=5, deterministic=True
            )

            # Report intermediate result for pruning
            trial.report(mean_reward, step=0)

            # Check if pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Return negative mean price as objective (for minimization)
            # Fallback: use negative reward if price not available
            return mean_reward

        except optuna.TrialPruned:
            raise
        except Exception as e:
            if self.verbose > 0:
                print(f"Trial failed: {repr(e)}")
                print(traceback.format_exc())
            # Return very low score to penalize failures
            return -1e6

    def tune(self, n_jobs: int = 1) -> Dict[str, Any]:
        """Run Optuna tuning study.
        
        Args:
            n_jobs: Number of parallel jobs (default: 1 for notebook stability)
            
        Returns:
            Dictionary with best parameters and study info
        """
        try:
            # Create study
            sampler = optuna.samplers.TPESampler(seed=42)
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
                interval_steps=1,
            )

            storage_path = os.path.join(self.storage_dir, f"{self.study_name}.db")
            storage_url = f"sqlite:///{storage_path.replace(chr(92), '/')}"  # Windows path fix

            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=storage_url,
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
                load_if_exists=True,
            )

            print(f"\n{'='*60}")
            print(f"Tuning {self.algorithm_name}: {self.n_trials} trials")
            print(f"{'='*60}\n")

            # Optimize
            self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=n_jobs, show_progress_bar=True)

            # Extract best trial
            best_trial = self.study.best_trial

            print(f"\n{'─'*60}")
            print(f"Tuning Complete!")
            print(f"{'─'*60}")
            print(f"Best trial value: {best_trial.value:.4f}")
            print(f"Best parameters:")
            for param, value in best_trial.params.items():
                print(f"  {param}: {value}")
            print()

            self.best_params = best_trial.params

            result = {
                "success": True,
                "algorithm": self.algorithm_name,
                "best_trial_number": best_trial.number,
                "best_trial_value": best_trial.value,
                "best_params": best_trial.params,
                "n_trials": len(self.study.trials),
                "n_complete_trials": len(
                    [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                ),
            }

            return result

        except Exception as e:
            print(f"Tuning failed: {repr(e)}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "algorithm": self.algorithm_name,
            }

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found during tuning."""
        if self.best_params:
            return self.best_params
        if self.study and self.study.best_trial:
            self.best_params = self.study.best_trial.params
            return self.best_params
        return {}

    def save_study(self, output_dir: str = ".") -> str:
        """Save study summary to JSON.
        
        Args:
            output_dir: Directory to save summary
            
        Returns:
            Path to saved file
        """
        if not self.study:
            return None

        summary = {
            "algorithm": self.algorithm_name,
            "study_name": self.study_name,
            "best_trial_number": self.study.best_trial.number,
            "best_value": self.study.best_trial.value,
            "best_params": self.study.best_trial.params,
            "n_trials": len(self.study.trials),
            "completed_trials": len(
                [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
        }

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"optuna_{self.algorithm_name}_summary.json")

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return output_path


def run_optimization_for_algorithms(
    algorithms: list,
    env,
    eval_env,
    n_trials: int = 20,
    output_dir: str = "SB3",
    verbose: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Run Optuna tuning for multiple algorithms.
    
    Args:
        algorithms: List of algorithm names to tune
        env: Training environment
        eval_env: Evaluation environment
        n_trials: Number of Optuna trials per algorithm
        output_dir: Directory to save results
        verbose: Verbosity level
        
    Returns:
        Dictionary mapping algorithm_name -> best_params
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"SB3 Hyperparameter Optimization: {len(algorithms)} algorithms")
    print(f"{'='*80}\n")

    for algorithm in algorithms:
        print(f"\n{'─'*80}")
        print(f"Optimizing {algorithm}")
        print(f"{'─'*80}\n")

        tuner = SB3Tuner(
            algorithm,
            env,
            eval_env,
            n_trials=n_trials,
            storage_dir=os.path.join(output_dir, "optuna"),
            verbose=verbose,
        )

        tune_result = tuner.tune(n_jobs=1)
        results[algorithm] = tune_result

        if tune_result.get("success"):
            tuner.save_study(os.path.join(output_dir, "optuna_summaries"))
            print(f"✓ {algorithm} optimization complete")
        else:
            print(f"✗ {algorithm} optimization failed: {tune_result.get('error')}")

    return results
