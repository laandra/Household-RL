# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -125.6062 | 0.5796 | 0.5796 | 401.9052 | 0.3736 | 0.3736 | 3.3124 | -0.5833 | 401.5494 | 402.2944 | 0.745 |
| IQN | 3 | -125.7905 | 0.6521 | 0.6521 | 402.0531 | 0.4362 | 0.4362 | 3.3136 | -0.4354 | 401.5494 | 402.305 | 0.7556 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -125.6062 | 0.5796 | 0.5796 | 401.9052 | 0.3736 | 0.3736 | 3.3124 | -0.5833 | 401.5494 | 402.2944 | 0.745 | 1 |
| IQN | 3 | -125.7905 | 0.6521 | 0.6521 | 402.0531 | 0.4362 | 0.4362 | 3.3136 | -0.4354 | 401.5494 | 402.305 | 0.7556 | 2 |
| SB3_MASKABLE_PPO | 1 | -549.4547 |  |  | 879.1276 |  |  | 3.6228 | 85.3368 | 879.1276 | 879.1276 | 0 | 3 |
| SB3_DQN | 1 | -550.8422 |  |  | 881.3475 |  |  | 3.6319 | 85.3606 | 881.3475 | 881.3475 | 0 | 4 |
| SB3_QR_DQN | 1 | -551.0707 |  |  | 881.7131 |  |  | 3.6334 | 85.3606 | 881.7131 | 881.7131 | 0 | 5 |
| SB3_RECURRENT_PPO | 1 | -551.0716 |  |  | 881.7145 |  |  | 3.6334 | 85.3606 | 881.7145 | 881.7145 | 0 | 6 |
| SB3_PPO | 1 | -558.693 |  |  | 893.9089 |  |  | 3.6837 | 85.4024 | 893.9089 | 893.9089 | 0 | 7 |
| SB3_A2C | 1 | -561.5536 |  |  | 898.4858 |  |  | 3.7026 | 85.3606 | 898.4858 | 898.4858 | 0 | 8 |

## Best Agent

- **Algorithm**: IQN
- **Seed**: 1
- **Price Mean**: 401.55
- **Quick 7D Price**: -0.94
- **Price Std**: 0.789524881666722
- **Reward Mean**: -125.04
- **Model Path**: BTR/models/IQN_seed1.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -125.79 ± 0.65 (min: -126.17, max: -125.04)
- **Price**: 402.05 ± 0.44 (min: 401.55, max: 402.30)

### C51

- **Seeds**: 3
- **Reward**: -125.61 ± 0.58 (min: -126.20, max: -125.04)
- **Price**: 401.91 ± 0.37 (min: 401.55, max: 402.29)

