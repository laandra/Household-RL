# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -16.0426 | 1.5866 | 1.5866 | 53.6754 | 0.3828 | 0.3828 | 0.2938 | 1.245 | 53.4281 | 54.1163 | 0.6882 |
| IQN | 3 | -16.9556 | 0.8881 | 0.8881 | 53.8029 | 0.3585 | 0.3585 | 0.2945 | 1.3703 | 53.4281 | 54.1426 | 0.7144 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_QR_DQN | 1 | -26.9574 |  |  | 43.1318 |  |  | 0.1181 | -0.7896 | 43.1318 | 43.1318 | 0 | 1 |
| SB3_MASKABLE_PPO | 1 | -26.9602 |  |  | 43.1363 |  |  | 0.1181 | -0.7476 | 43.1363 | 43.1363 | 0 | 2 |
| SB3_PPO | 1 | -27.3261 |  |  | 43.7218 |  |  | 0.1197 | -0.4781 | 43.7218 | 43.7218 | 0 | 3 |
| SB3_A2C | 1 | -27.3596 |  |  | 43.7754 |  |  | 0.1198 | -0.146 | 43.7754 | 43.7754 | 0 | 4 |
| SB3_RECURRENT_PPO | 1 | -28.1809 |  |  | 45.0894 |  |  | 0.1234 | -0.7508 | 45.0894 | 45.0894 | 0 | 5 |
| C51 | 3 | -16.0426 | 1.5866 | 1.5866 | 53.6754 | 0.3828 | 0.3828 | 0.2938 | 1.245 | 53.4281 | 54.1163 | 0.6882 | 6 |
| IQN | 3 | -16.9556 | 0.8881 | 0.8881 | 53.8029 | 0.3585 | 0.3585 | 0.2945 | 1.3703 | 53.4281 | 54.1426 | 0.7144 | 7 |
| SB3_DQN | 1 | -51.9413 |  |  | 83.1061 |  |  | 0.2275 | 1.0788 | 83.1061 | 83.1061 | 0 | 8 |

## Best Agent

- **Algorithm**: IQN
- **Seed**: 2
- **Price Mean**: 53.43
- **Quick 7D Price**: 1.00
- **Price Std**: 0.0
- **Reward Mean**: -16.14
- **Model Path**: BTR/models/IQN_seed2.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -16.96 ± 0.89 (min: -17.90, max: -16.14)
- **Price**: 53.80 ± 0.36 (min: 53.43, max: 54.14)

### C51

- **Seeds**: 3
- **Reward**: -16.04 ± 1.59 (min: -17.58, max: -14.41)
- **Price**: 53.68 ± 0.38 (min: 53.43, max: 54.12)

