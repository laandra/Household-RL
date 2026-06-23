# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -276.0645 | 0.5733 | 0.5733 | 883.7664 | 1.8281 | 1.8281 | 3.6419 | 87.4138 | 881.7131 | 885.2175 | 3.5044 |
| IQN | 3 | -276.2737 | 0.661 | 0.661 | 884.0494 | 2.0232 | 2.0232 | 3.6431 | 87.6968 | 881.7131 | 885.2175 | 3.5044 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_QR_DQN | 1 | -275.6527 |  |  | 881.8021 |  |  | 3.6338 | 85.4496 | 881.8021 | 881.8021 | 0 | 1 |
| C51 | 3 | -276.0645 | 0.5733 | 0.5733 | 883.7664 | 1.8281 | 1.8281 | 3.6419 | 87.4138 | 881.7131 | 885.2175 | 3.5044 | 2 |
| IQN | 3 | -276.2737 | 0.661 | 0.661 | 884.0494 | 2.0232 | 2.0232 | 3.6431 | 87.6968 | 881.7131 | 885.2175 | 3.5044 | 3 |
| SB3_DQN | 1 | -122.4436 |  |  | 885.9221 |  |  | 3.6508 | 89.92 | 885.9221 | 885.9221 | 0 | 4 |
| SB3_PPO | 1 | -215.9305 |  |  | 897.7102 |  |  | 3.6994 | 90.6688 | 897.7102 | 897.7102 | 0 | 5 |
| SB3_MASKABLE_PPO | 1 | -137.2006 |  |  | 908.5214 |  |  | 3.7439 | 89.0666 | 908.5214 | 908.5214 | 0 | 6 |
| SB3_A2C | 1 | -256.5433 |  |  | 943.8561 |  |  | 3.8895 | 87.2052 | 943.8561 | 943.8561 | 0 | 7 |
| SB3_RECURRENT_PPO | 1 | -297.1491 |  |  | 985.4864 |  |  | 4.0611 | 87.5808 | 985.4864 | 985.4864 | 0 | 8 |

## Best Agent

- **Algorithm**: IQN
- **Seed**: 1
- **Price Mean**: 881.71
- **Quick 7D Price**: 85.36
- **Price Std**: 0.5884771017780681
- **Reward Mean**: -275.51
- **Model Path**: BTR/models/IQN_seed1.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -276.27 ± 0.66 (min: -276.66, max: -275.51)
- **Price**: 884.05 ± 2.02 (min: 881.71, max: 885.22)

### C51

- **Seeds**: 3
- **Reward**: -276.06 ± 0.57 (min: -276.66, max: -275.51)
- **Price**: 883.77 ± 1.83 (min: 881.71, max: 885.22)

