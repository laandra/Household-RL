# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -109.6052 | 0.5347 | 0.5347 | 5999.6076 | 0.7842 | 0.7842 | 32.8446 | 225.8478 | 5999.1126 | 6000.5117 | 1.3991 |
| IQN | 3 | -110.7343 | 1.1298 | 1.1298 | 6000.0911 | 0.7731 | 0.7731 | 32.8472 | 226.3314 | 5999.1984 | 6000.5374 | 1.339 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_MASKABLE_PPO | 1 | -214.6216 |  |  | 3167.0166 |  |  | 8.6688 | 56.1644 | 3167.0166 | 3167.0166 | 0 | 1 |
| SB3_PPO | 1 | -218.4054 |  |  | 3173.0707 |  |  | 8.6854 | 56.1936 | 3173.0707 | 3173.0707 | 0 | 2 |
| SB3_QR_DQN | 1 | -220.2826 |  |  | 3176.0743 |  |  | 8.6936 | 55.3527 | 3176.0743 | 3176.0743 | 0 | 3 |
| SB3_DQN | 1 | -220.7991 |  |  | 3176.9006 |  |  | 8.6959 | 56.179 | 3176.9006 | 3176.9006 | 0 | 4 |
| SB3_RECURRENT_PPO | 1 | -220.8083 |  |  | 3176.9154 |  |  | 8.6959 | 56.1938 | 3176.9154 | 3176.9154 | 0 | 5 |
| SB3_A2C | 1 | -221.5102 |  |  | 3178.0384 |  |  | 8.699 | 57.3168 | 3178.0384 | 3178.0384 | 0 | 6 |
| C51 | 3 | -109.6052 | 0.5347 | 0.5347 | 5999.6076 | 0.7842 | 0.7842 | 32.8446 | 225.8478 | 5999.1126 | 6000.5117 | 1.3991 | 7 |
| IQN | 3 | -110.7343 | 1.1298 | 1.1298 | 6000.0911 | 0.7731 | 0.7731 | 32.8472 | 226.3314 | 5999.1984 | 6000.5374 | 1.339 | 8 |

## Best Agent

- **Algorithm**: C51
- **Seed**: 0
- **Price Mean**: 5999.11
- **Quick 7D Price**: 225.35
- **Price Std**: 2.9587795146021563
- **Reward Mean**: -109.18
- **Model Path**: BTR/models/C51_seed0.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -110.73 ± 1.13 (min: -111.39, max: -109.43)
- **Price**: 6000.09 ± 0.77 (min: 5999.20, max: 6000.54)

### C51

- **Seeds**: 3
- **Reward**: -109.61 ± 0.53 (min: -110.21, max: -109.18)
- **Price**: 5999.61 ± 0.78 (min: 5999.11, max: 6000.51)

