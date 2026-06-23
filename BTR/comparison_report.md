# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -14.011 | 1.7386 | 1.7386 | 47.9446 | 0.1778 | 0.1778 | 0.2625 | 8.5864 | 47.7393 | 48.0472 | 0.3079 |
| IQN | 3 | -15.1202 | 0.3106 | 0.3106 | 48.1815 | 0.2898 | 0.2898 | 0.2638 | 8.8183 | 47.8469 | 48.3488 | 0.5019 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_QR_DQN | 1 | -29.9182 |  |  | 47.8691 |  |  | 0.131 | 4.0846 | 47.8691 | 47.8691 | 0 | 1 |
| C51 | 3 | -14.011 | 1.7386 | 1.7386 | 47.9446 | 0.1778 | 0.1778 | 0.2625 | 8.5864 | 47.7393 | 48.0472 | 0.3079 | 2 |
| IQN | 3 | -15.1202 | 0.3106 | 0.3106 | 48.1815 | 0.2898 | 0.2898 | 0.2638 | 8.8183 | 47.8469 | 48.3488 | 0.5019 | 3 |
| SB3_MASKABLE_PPO | 1 | -30.6373 |  |  | 49.0197 |  |  | 0.1342 | 4.0951 | 49.0197 | 49.0197 | 0 | 4 |
| SB3_PPO | 1 | -30.8451 |  |  | 49.3521 |  |  | 0.1351 | 4.701 | 49.3521 | 49.3521 | 0 | 5 |
| SB3_RECURRENT_PPO | 1 | -31.6672 |  |  | 50.6675 |  |  | 0.1387 | 4.1594 | 50.6675 | 50.6675 | 0 | 6 |
| SB3_A2C | 1 | -33.4277 |  |  | 53.4844 |  |  | 0.1464 | 4.2284 | 53.4844 | 53.4844 | 0 | 7 |
| SB3_DQN | 1 | -51.6931 |  |  | 82.7089 |  |  | 0.2264 | 4.8496 | 82.7089 | 82.7089 | 0 | 8 |

## Best Agent

- **Algorithm**: C51
- **Seed**: 2
- **Price Mean**: 47.74
- **Quick 7D Price**: 8.39
- **Price Std**: 0.1594336453267206
- **Reward Mean**: -12.00
- **Model Path**: BTR/models/C51_seed2.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -15.12 ± 0.31 (min: -15.30, max: -14.76)
- **Price**: 48.18 ± 0.29 (min: 47.85, max: 48.35)

### C51

- **Seeds**: 3
- **Reward**: -14.01 ± 1.74 (min: -15.01, max: -12.00)
- **Price**: 47.94 ± 0.18 (min: 47.74, max: 48.05)

