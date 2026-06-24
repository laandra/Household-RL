# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IQN | 3 | -111.2402 | 0.2546 | 0.2546 | 6000.9916 | 0.6484 | 0.6484 | 32.8521 | 227.1549 | 6000.6172 | 6001.7403 | 1.123 |
| C51 | 3 | -96.8398 | 23.5864 | 23.5864 | 6003.3324 | 5.9712 | 5.9712 | 32.865 | 226.791 | 5999.1232 | 6010.1664 | 11.0432 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_DQN | 1 | -215.3985 |  |  | 3168.2987 |  |  | 8.6724 | 55.8779 | 3168.2987 | 3168.2987 | 0 | 1 |
| SB3_QR_DQN | 1 | -217.8441 |  |  | 3172.2116 |  |  | 8.6831 | 56.1803 | 3172.2116 | 3172.2116 | 0 | 2 |
| SB3_RECURRENT_PPO | 1 | -220.7874 |  |  | 3176.9209 |  |  | 8.696 | 56.1591 | 3176.9209 | 3176.9209 | 0 | 3 |
| SB3_MASKABLE_PPO | 1 | -222.1054 |  |  | 3179.0298 |  |  | 8.7017 | 58.2852 | 3179.0298 | 3179.0298 | 0 | 4 |
| SB3_PPO | 1 | -222.152 |  |  | 3179.1042 |  |  | 8.7019 | 58.9646 | 3179.1042 | 3179.1042 | 0 | 5 |
| SB3_A2C | 1 | -232.5447 |  |  | 3195.7326 |  |  | 8.7474 | 74.8092 | 3195.7326 | 3195.7326 | 0 | 6 |
| IQN | 3 | -111.2402 | 0.2546 | 0.2546 | 6000.9916 | 0.6484 | 0.6484 | 32.8521 | 227.1549 | 6000.6172 | 6001.7403 | 1.123 | 7 |
| C51 | 3 | -96.8398 | 23.5864 | 23.5864 | 6003.3324 | 5.9712 | 5.9712 | 32.865 | 226.791 | 5999.1232 | 6010.1664 | 11.0432 | 8 |

## Best Agent

- **Algorithm**: C51
- **Seed**: 0
- **Price Mean**: 5999.12
- **Quick 7D Price**: 225.29
- **Price Std**: 0.7017850000006547
- **Reward Mean**: -109.84
- **Model Path**: BTR/models/C51_seed0.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -111.24 ± 0.25 (min: -111.39, max: -110.95)
- **Price**: 6000.99 ± 0.65 (min: 6000.62, max: 6001.74)

### C51

- **Seeds**: 3
- **Reward**: -96.84 ± 23.59 (min: -111.07, max: -69.61)
- **Price**: 6003.33 ± 5.97 (min: 5999.12, max: 6010.17)

