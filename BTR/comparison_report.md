# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -275.3745 | 1.1323 | 1.1323 | 883.452 | 0.8478 | 0.8478 | 3.6406 | 87.2984 | 882.6955 | 884.3684 | 1.6729 |
| IQN | 3 | -276.2372 | 0.3621 | 0.3621 | 884.2927 | 0.9649 | 0.9649 | 3.6441 | 87.9402 | 883.2922 | 885.2175 | 1.9253 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_MASKABLE_PPO | 1 | -549.4547 |  |  | 879.1276 |  |  | 3.6228 | 85.3368 | 879.1276 | 879.1276 | 0 | 1 |
| SB3_DQN | 1 | -550.8422 |  |  | 881.3475 |  |  | 3.6319 | 85.3606 | 881.3475 | 881.3475 | 0 | 2 |
| SB3_QR_DQN | 1 | -551.0707 |  |  | 881.7131 |  |  | 3.6334 | 85.3606 | 881.7131 | 881.7131 | 0 | 3 |
| SB3_RECURRENT_PPO | 1 | -551.0716 |  |  | 881.7145 |  |  | 3.6334 | 85.3606 | 881.7145 | 881.7145 | 0 | 4 |
| C51 | 3 | -275.3745 | 1.1323 | 1.1323 | 883.452 | 0.8478 | 0.8478 | 3.6406 | 87.2984 | 882.6955 | 884.3684 | 1.6729 | 5 |
| IQN | 3 | -276.2372 | 0.3621 | 0.3621 | 884.2927 | 0.9649 | 0.9649 | 3.6441 | 87.9402 | 883.2922 | 885.2175 | 1.9253 | 6 |
| SB3_PPO | 1 | -558.693 |  |  | 893.9089 |  |  | 3.6837 | 85.4024 | 893.9089 | 893.9089 | 0 | 7 |
| SB3_A2C | 1 | -561.5536 |  |  | 898.4858 |  |  | 3.7026 | 85.3606 | 898.4858 | 898.4858 | 0 | 8 |

## Best Agent

- **Algorithm**: C51
- **Seed**: 1
- **Price Mean**: 882.70
- **Quick 7D Price**: 86.94
- **Price Std**: 89.3147729863687
- **Reward Mean**: -274.07
- **Model Path**: BTR/models/C51_seed1.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -276.24 ± 0.36 (min: -276.66, max: -276.03)
- **Price**: 884.29 ± 0.96 (min: 883.29, max: 885.22)

### C51

- **Seeds**: 3
- **Reward**: -275.37 ± 1.13 (min: -276.03, max: -274.07)
- **Price**: 883.45 ± 0.85 (min: 882.70, max: 884.37)

