# SB3 Implementation - Quick Start

## 🚀 Start Here

### Run the Benchmark (Fast - 9 minutes)
```python
# Cell 66 (already configured):
RUN_SB3_OPTIMIZATION = False  # Skip tuning
SB3_N_SEEDS = 3

# Cell 67:
# Just run it!
```

### Enable Hyperparameter Tuning (Slow - 45 minutes)
```python
# Cell 66:
RUN_SB3_OPTIMIZATION = True  # ← Change this

# Cell 67:
# Just run it!
```

---

## 📊 New Files

| File | Purpose |
|------|---------|
| `sb3_config_builder.py` | Hyperparameter configs + Optuna search spaces |
| `sb3_runner.py` | Training orchestration + multi-seed runner |
| `optuna_sb3_tuner.py` | Optuna integration for hyperparameter tuning |
| `SB3_IMPLEMENTATION.md` | Full documentation |
| `SB3_QUICKSTART.md` | This file |

---

## 🎛️ Configuration

**In Cell 66:**
```python
RUN_SB3_BENCHMARK = True           # Run benchmark
RUN_SB3_OPTIMIZATION = False       # Enable tuning (slow!)
SB3_TOTAL_TIMESTEPS = 60000        # Training steps per seed
SB3_N_SEEDS = 3                    # Number of seeds
SB3_N_EVAL_EPISODES = 10           # Episodes per evaluation
SB3_ALGORITHMS = [                 # Algorithms to train
    "DQN", "PPO", "A2C",           # Base SB3
    "QR_DQN", "MASKABLE_PPO", "RECURRENT_PPO"  # sb3-contrib
]
```

---

## 📈 Algorithms Supported

| Algorithm | Type | Source | Features |
|-----------|------|--------|----------|
| DQN | Value-based | SB3 | Experience replay, ε-greedy |
| PPO | Policy-based | SB3 | Trust region, fast convergence |
| A2C | Actor-Critic | SB3 | Parallel environments, low variance |
| QR_DQN | Value-based | sb3-contrib | Distributional, uncertainty-aware |
| MASKABLE_PPO | Policy-based | sb3-contrib | Action masking support |
| RECURRENT_PPO | Policy-based | sb3-contrib | LSTM memory, partial observability |

---

## 📁 Output Files

After running benchmark:
```
SB3/
├── benchmark_summary.json      ← Main results (reward, price metrics)
├── DQN/model_seed_11.zip       ← Saved models per seed
├── PPO/model_seed_11.zip
├── A2C/model_seed_11.zip
├── QR_DQN/model_seed_11.zip    ← (if sb3-contrib available)
├── MASKABLE_PPO/model_seed_11.zip
├── RECURRENT_PPO/model_seed_11.zip
└── optuna/                     ← (if optimization enabled)
    ├── DQN_tuning.db
    ├── PPO_tuning.db
    └── optuna_summaries/
```

---

## 💡 Common Tasks

### Load a Trained Model
```python
from sb3_runner import SB3Trainer

trainer = SB3Trainer("DQN", train_env, eval_env)
trainer.load_model("SB3/DQN/model_seed_11.zip")
metrics = trainer.train()  # Continue training or evaluate
```

### Run Single Algorithm (No Benchmark)
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

### View Results
```python
import json
with open("SB3/benchmark_summary.json") as f:
    results = json.load(f)
    for algo, metrics in results.items():
        print(f"{algo}: R={metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
```

### Compare with Custom DQN
```python
# After running cell 67:
print(sb3_comparison_table)  # Built-in comparison table
```

---

## ⚡ Performance

| Task | Time | Memory |
|------|------|--------|
| 1 algo × 1 seed × 60k steps | 90s | 500MB |
| 6 algos × 3 seeds (no tune) | 9m | 1GB |
| +20 Optuna trials/algo | +15m | 1.2GB |
| Full benchmark (with tune) | 45m | 1.5GB |

---

## 🔧 Troubleshooting

**Q: "sb3-contrib not available"**  
A: Install with `pip install sb3-contrib`  
(Benchmark continues with DQN/PPO/A2C only)

**Q: Price metrics are 0.0**  
A: Edit `sb3_runner.py` → `_evaluate()` method  
Check environment's `info` dict key name

**Q: Out of memory**  
A: Reduce `n_trials` (tuning) or `SB3_N_SEEDS`

**Q: Slow training**  
A: Reduce `SB3_TOTAL_TIMESTEPS` or run fewer algorithms

---

## 🎯 Next Steps

1. Run Cell 66 + 67 (9 minutes)
2. Check `SB3/benchmark_summary.json`
3. Identify best performer
4. (Optional) Re-run with `RUN_SB3_OPTIMIZATION=True` on winners
5. Deploy best model: `trainer.load_model("SB3/{algo}/model_seed_11.zip")`

---

**Full docs**: See `SB3_IMPLEMENTATION.md`  
**Version**: 1.0 (May 2026)
