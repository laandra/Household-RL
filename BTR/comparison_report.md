# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -45.6821 | 138.7209 | 138.7209 | 384.6975 | 29.455 | 29.455 | 3.1706 | -1.7317 | 350.6927 | 402.2944 | 51.6017 |
| IQN | 3 | -125.492 | 0.7549 | 0.7549 | 401.7176 | 0.6675 | 0.6675 | 3.3109 | -0.7709 | 400.9864 | 402.2944 | 1.308 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -45.6821 | 138.7209 | 138.7209 | 384.6975 | 29.455 | 29.455 | 3.1706 | -1.7317 | 350.6927 | 402.2944 | 51.6017 | 1 |
| IQN | 3 | -125.492 | 0.7549 | 0.7549 | 401.7176 | 0.6675 | 0.6675 | 3.3109 | -0.7709 | 400.9864 | 402.2944 | 1.308 | 2 |
| SB3_MASKABLE_PPO | 1 | -549.4547 |  |  | 879.1276 |  |  | 3.6228 | 85.3368 | 879.1276 | 879.1276 | 0 | 3 |
| SB3_DQN | 1 | -550.8422 |  |  | 881.3475 |  |  | 3.6319 | 85.3606 | 881.3475 | 881.3475 | 0 | 4 |
| SB3_QR_DQN | 1 | -551.0707 |  |  | 881.7131 |  |  | 3.6334 | 85.3606 | 881.7131 | 881.7131 | 0 | 5 |
| SB3_RECURRENT_PPO | 1 | -551.0716 |  |  | 881.7145 |  |  | 3.6334 | 85.3606 | 881.7145 | 881.7145 | 0 | 6 |
| SB3_PPO | 1 | -558.693 |  |  | 893.9089 |  |  | 3.6837 | 85.4024 | 893.9089 | 893.9089 | 0 | 7 |
| SB3_A2C | 1 | -561.5536 |  |  | 898.4858 |  |  | 3.7026 | 85.3606 | 898.4858 | 898.4858 | 0 | 8 |

## Best Agent

- **Algorithm**: C51
- **Seed**: 0
- **Price Mean**: 350.69
- **Quick 7D Price**: -4.38
- **Price Std**: 20.48565202679565
- **Reward Mean**: 114.50
- **Model Path**: BTR/models/C51_seed0.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -125.49 ± 0.75 (min: -126.20, max: -124.69)
- **Price**: 401.72 ± 0.67 (min: 400.99, max: 402.29)

### C51

- **Seeds**: 3
- **Reward**: -45.68 ± 138.72 (min: -126.20, max: 114.50)
- **Price**: 384.70 ± 29.46 (min: 350.69, max: 402.29)

