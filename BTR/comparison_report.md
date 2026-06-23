# BTR Algorithm Comparison Report

## Within-BTR Comparison (IQN vs C51)

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C51 | 3 | -275.5017 | 0.0076 | 0.0076 | 881.6162 | 0.084 | 0.084 | 3.633 | 85.2636 | 881.5677 | 881.7131 | 0.1455 |
| IQN | 3 | -275.6829 | 0.2985 | 0.2985 | 882.5982 | 1.533 | 1.533 | 3.6371 | 86.2457 | 881.7131 | 884.3684 | 2.6553 |

## BTR vs SB3 Comparison

| algorithm | n_seeds | reward_mean | reward_std | reward_std_across_seeds | price_mean | price_std | price_std_across_seeds | price_mean_eur_per_day | price_7day | best_price | worst_price | price_range | price_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB3_MASKABLE_PPO | 1 | -549.4547 |  |  | 879.1276 |  |  | 3.6228 | 85.3368 | 879.1276 | 879.1276 | 0 | 1 |
| SB3_DQN | 1 | -550.8422 |  |  | 881.3475 |  |  | 3.6319 | 85.3606 | 881.3475 | 881.3475 | 0 | 2 |
| C51 | 3 | -275.5017 | 0.0076 | 0.0076 | 881.6162 | 0.084 | 0.084 | 3.633 | 85.2636 | 881.5677 | 881.7131 | 0.1455 | 3 |
| SB3_QR_DQN | 1 | -551.0707 |  |  | 881.7131 |  |  | 3.6334 | 85.3606 | 881.7131 | 881.7131 | 0 | 4 |
| SB3_RECURRENT_PPO | 1 | -551.0716 |  |  | 881.7145 |  |  | 3.6334 | 85.3606 | 881.7145 | 881.7145 | 0 | 5 |
| IQN | 3 | -275.6829 | 0.2985 | 0.2985 | 882.5982 | 1.533 | 1.533 | 3.6371 | 86.2457 | 881.7131 | 884.3684 | 2.6553 | 6 |
| SB3_PPO | 1 | -558.693 |  |  | 893.9089 |  |  | 3.6837 | 85.4024 | 893.9089 | 893.9089 | 0 | 7 |
| SB3_A2C | 1 | -561.5536 |  |  | 898.4858 |  |  | 3.7026 | 85.3606 | 898.4858 | 898.4858 | 0 | 8 |

## Best Agent

- **Algorithm**: C51
- **Seed**: 0
- **Price Mean**: 881.57
- **Quick 7D Price**: 85.22
- **Price Std**: 1.7041731958478186
- **Reward Mean**: -275.50
- **Model Path**: BTR/models/C51_seed0.pt

## Detailed Statistics

### IQN

- **Seeds**: 3
- **Reward**: -275.68 ± 0.30 (min: -276.03, max: -275.51)
- **Price**: 882.60 ± 1.53 (min: 881.71, max: 884.37)

### C51

- **Seeds**: 3
- **Reward**: -275.50 ± 0.01 (min: -275.51, max: -275.50)
- **Price**: 881.62 ± 0.08 (min: 881.57, max: 881.71)

