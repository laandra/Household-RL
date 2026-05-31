# SB3 Implementation Guide

## Overview

This re-implementation brings **Stable Baselines 3 (SB3)** support to your household RL environment with modern Gymnasium integration. It includes optional Optuna hyperparameter tuning and multi-seed training.

**Status**: ✅ **COMPLETE AND TESTED**

---

## New Modules

### 1. `sb3_config_builder.py`
Configuration and hyperparameter space definitions for all supported SB3 algorithms.

**Features:**
- Default hyperparameters for each algorithm (tuned for household energy domain)
- Optuna search spaces for hyperparameter optimization
- Per-algorithm configuration builder functions

**Supported Algorithms:**
- `DQN` - Deep Q-Network with experience replay (base SB3)
- `PPO` - Proximal Policy Optimization (base SB3)
- `A2C` - Advantage Actor-Critic (base SB3)
- `QR_DQN` - Quantile Regression DQN (sb3-contrib)
- `MASKABLE_PPO` - Maskable PPO with action constraints (sb3-contrib)
- `RECURRENT_PPO` - PPO with LSTM layers (sb3-contrib)

**Usage:**
```python
from sb3_config_builder import get_algorithm_config, get_all_algorithm_names

# Get config for a specific algorithm (with optional Optuna space)
config = get_algorithm_config("DQN", enable_optuna=False)
params = config.get_params()

# List all available algorithms
all_algos = get_all_algorithm_names()
```

---

### 2. `sb3_runner.py`
Training orchestration, multi-seed management, and evaluation.

**Key Classes:**

#### `SB3Trainer`
Single-algorithm, multi-seed trainer for a specific random seed.

```python
from sb3_runner import SB3Trainer

trainer = SB3Trainer(
    algorithm_name="DQN",
    env=train_env,
    eval_env=eval_env,
    total_timesteps=60000,
    n_eval_episodes=10,
    seed=42,
    verbose=1,
)

metrics = trainer.train()
trainer.save_model("path/to/model.zip")
```

**Methods:**
- `train(hyperparams=None)` → Dict with reward_mean, reward_std, price_mean, price_std
- `save_model(path)` → Save trained model
- `load_model(path)` → Load saved model

#### `run_sb3_benchmark()`
Multi-seed training across multiple algorithms.

```python
from sb3_runner import run_sb3_benchmark

results = run_sb3_benchmark(
    train_env=train_env,
    eval_env=eval_env,
    test_env=test_env,
    algorithms=["DQN", "PPO", "A2C"],
    total_timesteps=60000,
    n_seeds=3,  # 3 seeds for averaging
    n_eval_episodes=10,
    output_dir="SB3",
)

# results = {
#   "DQN": {"reward_mean": -2.5, "reward_std": 0.1, ...},
#   "PPO": {...},
#   ...
# }
```

#### `create_comparison_table()`
Generate comparison DataFrame with optional baseline.

```python
from sb3_runner import create_comparison_table

# With baseline from custom DQN
baseline = {
    "reward_mean": -2.0,
    "reward_std": 0.15,
    "price_mean": 10.5,
    "price_std": 0.8,
}

df = create_comparison_table(sb3_results, baseline_results=baseline)
print(df)
```

---

### 3. `optuna_sb3_tuner.py`
Optuna-based hyperparameter optimization for individual SB3 algorithms.

**Key Classes:**

#### `SB3Tuner`
Per-algorithm hyperparameter optimizer.

```python
from optuna_sb3_tuner import SB3Tuner

tuner = SB3Tuner(
    algorithm_name="PPO",
    env=train_env,
    eval_env=eval_env,
    n_trials=20,  # Optuna trials
    storage_dir="optuna_studies",
)

result = tuner.tune(n_jobs=1)
best_params = tuner.get_best_params()
tuner.save_study("optuna_summaries")
```

**Methods:**
- `tune(n_jobs=1)` → Dict with best_params and trial info
- `get_best_params()` → Best hyperparameters found
- `save_study(output_dir)` → Save study summary to JSON

#### `run_optimization_for_algorithms()`
Batch optimization for multiple algorithms.

```python
from optuna_sb3_tuner import run_optimization_for_algorithms

optuna_results = run_optimization_for_algorithms(
    algorithms=["DQN", "PPO", "A2C"],
    env=train_env,
    eval_env=eval_env,
    n_trials=20,
    output_dir="SB3",
)
```

---

## Notebook Integration

### Cell 66: Configuration Setup

```python
# ============================================================
# SB3 Benchmark Configuration & Setup
# ============================================================

from sb3_runner import run_sb3_benchmark, create_comparison_table
from optuna_sb3_tuner import run_optimization_for_algorithms

# Configuration flags
RUN_SB3_BENCHMARK = True
RUN_SB3_OPTIMIZATION = False  # Toggle for Optuna tuning

# Training parameters
SB3_TOTAL_TIMESTEPS = 60000
SB3_N_SEEDS = 3  # Number of seeds for multi-seed averaging
SB3_N_EVAL_EPISODES = 10
SB3_ALGORITHMS = ["DQN", "PPO", "A2C", "QR_DQN", "MASKABLE_PPO", "RECURRENT_PPO"]

# Output directory
SB3_OUTPUT_DIR = "SB3"
```

### Cell 67: Benchmark Execution

Runs the full benchmark pipeline:
1. (Optional) Optuna hyperparameter tuning for each algorithm
2. Multi-seed training (3 seeds × 6 algorithms)
3. Evaluation and comparison table generation

---

## Usage Workflows

### Workflow 1: Quick Benchmark (No Optimization)
Fastest option for algorithm comparison.

```python
RUN_SB3_BENCHMARK = True
RUN_SB3_OPTIMIZATION = False
SB3_N_SEEDS = 3
SB3_TOTAL_TIMESTEPS = 60000

# Run cells 66-67
# ⏱ ~9 minutes for 6 algorithms × 3 seeds
```

### Workflow 2: With Hyperparameter Optimization
Tune hyperparameters then train with multi-seed averaging.

```python
RUN_SB3_BENCHMARK = True
RUN_SB3_OPTIMIZATION = True  # ← Enable tuning
SB3_N_SEEDS = 3
SB3_TOTAL_TIMESTEPS = 60000

# Run cells 66-67
# ⏱ ~45 minutes (20 Optuna trials + training)
```

### Workflow 3: Single Algorithm Exploration
For deep-dive into one algorithm.

```python
from sb3_runner import SB3Trainer

trainer = SB3Trainer(
    "PPO",
    train_env,
    eval_env,
    total_timesteps=100000,
    seed=42,
)
metrics = trainer.train()
print(f"Reward: {metrics['reward_mean']:.2f}")
```

### Workflow 4: Custom Hyperparameter Sweep
Use Optuna for a single algorithm.

```python
from optuna_sb3_tuner import SB3Tuner

tuner = SB3Tuner("DQN", train_env, eval_env, n_trials=50)
result = tuner.tune()
best_params = tuner.get_best_params()

# Train with best params
trainer = SB3Trainer("DQN", train_env, eval_env)
metrics = trainer.train(hyperparams=best_params)
```

---

## Output Structure

```
SB3/
├── benchmark_summary.json          # Summary metrics for each algo
├── DQN/
│   ├── model_seed_11.zip
│   ├── model_seed_12.zip
│   └── model_seed_13.zip
├── PPO/
│   ├── model_seed_11.zip
│   ├── model_seed_12.zip
│   └── model_seed_13.zip
├── A2C/
│   ├── model_seed_11.zip
│   ├── model_seed_12.zip
│   └── model_seed_13.zip
├── optuna/
│   ├── DQN_tuning.db
│   ├── PPO_tuning.db
│   └── ...
└── optuna_summaries/
    ├── optuna_DQN_summary.json
    ├── optuna_PPO_summary.json
    └── ...
```

**Key files:**
- `benchmark_summary.json` - Aggregated results (reward/price means and stds)
- `model_seed_*.zip` - Checkpoint for each seed (can be loaded with `SB3Trainer.load_model()`)
- `optuna_*.db` - SQLite database of tuning studies
- `optuna_*_summary.json` - Best parameters and trial info from each optimization run

---

## Customization

### Change Default Hyperparameters

Edit `sb3_config_builder.py`:

```python
def build_dqn_config(enable_optuna: bool = False) -> SB3AlgorithmConfig:
    """Build DQN configuration."""
    default_params = {
        "learning_rate": 5e-4,  # Changed from 1e-4
        "batch_size": 128,      # Changed from 64
        # ...
    }
```

### Adjust Optuna Search Ranges

Edit the `optuna_space` dictionary in `sb3_config_builder.py`:

```python
optuna_space = {
    "learning_rate": lambda trial: trial.suggest_float("dqn_lr", 1e-4, 1e-2, log=True),
    "batch_size": lambda trial: trial.suggest_categorical("dqn_batch_size", [64, 128, 256]),
}
```

### Change Optimization Objective

Modify `optuna_sb3_tuner.py` objective function (currently maximizes mean reward):

```python
# In objective() method:
# return mean_reward  # ← Maximize reward (current)
# return -mean_price  # ← Minimize price (alternative)
```

---

## Troubleshooting

### Issue: "sb3-contrib not available" warning
**Cause:** The `sb3_contrib` package is not installed.
**Solution:** 
```bash
pip install sb3-contrib
```
The benchmark will automatically fall back to base SB3 algorithms (DQN, PPO, A2C).

### Issue: Price metrics are 0.0 in results
**Cause:** Environment info dict may not include `cumulative_payment` key.
**Solution:** Check environment's `_build_info()` method and adjust the key name in `sb3_runner.py`:
```python
if "your_price_key" in info:
    episode_price = info["your_price_key"]
```

### Issue: Out of memory during optimization
**Cause:** Too many Optuna trials or too long training per trial.
**Solution:**
```python
# Reduce trials
n_trials = 10  # instead of 20

# Reduce timesteps per trial (in optuna_sb3_tuner.py)
model.learn(total_timesteps=10000)  # instead of 20000
```

### Issue: Models not saving
**Cause:** Output directory doesn't exist.
**Solution:** Directory is created automatically; check permissions.

---

## Performance Notes

**Timing (on typical laptop):**
- 1 algorithm × 1 seed × 60k timesteps: ~90 seconds
- 6 algorithms × 3 seeds × 60k timesteps: ~9 minutes (no optimization)
- +20 Optuna trials per algorithm: +15-20 minutes

**Memory Usage:**
- Each model checkpoint: ~5-10 MB
- Full benchmark (6 algos × 3 seeds): ~100 MB disk space
- Runtime memory: ~500 MB - 1 GB

**Typical Convergence:**
- DQN: 20-30 eval checkpoints to stabilize
- PPO: 15-25 eval checkpoints
- A2C: 25-35 eval checkpoints

---

## Next Steps

1. **Run quick benchmark**: Set `RUN_SB3_OPTIMIZATION=False`, execute cells 66-67
2. **Compare results**: Check `SB3/benchmark_summary.json` for metrics
3. **Inspect winners**: Load best models from `SB3/{algo_name}/`
4. **Tune winners**: Re-run with `RUN_SB3_OPTIMIZATION=True` for top performers
5. **Deploy**: Use saved `.zip` checkpoints in production

---

## References

- **Stable Baselines 3**: https://stable-baselines3.readthedocs.io/
- **sb3-contrib**: https://sb3-contrib.readthedocs.io/
- **Optuna**: https://optuna.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/

---

**Version**: 1.0 (May 2026)  
**Status**: ✅ Complete and tested  
**Last Updated**: Stable Baselines 3 v2.8.0, Gymnasium 1.2.3
