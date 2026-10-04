# Trading performance evaluation

Fixed default heuristic calibration; 120-session training, 20-session nonoverlapping test windows; last 252 sessions held out; LONG SWING; monthly block bootstrap, 2000 resamples, seed 20261002

Data: 2021-10-04 00:00:00 to 2026-10-01 00:00:00; holdout from 2025-09-24 00:00:00.

| Run | Trades | Mean net R | Profit factor | Monthly block 95% CI |
|---|---:|---:|---:|---|
| development | 116 | -0.2319 | 0.633 | [-0.458486549583705, 0.004271113042039043] |
| holdout | 34 | 0.183 | 1.405 | [-0.3401132638888889, 0.6858088235294117] |
| holdout_2x_slippage | 34 | 0.1493 | 1.32 | [-0.40543073529411766, 0.6453438194444441] |
| holdout_3x_slippage | 34 | 0.1156 | 1.24 | [-0.42834710937500003, 0.625064705882353] |

Limitations:

- Current constituents introduce survivorship bias
- Subset selected by sector rotation, not point-in-time membership
- Adjusted research prices, not broker executable quotes
- Simulator selects at fold boundaries, not a daily live portfolio replay
- R statistics are not INR NAV returns; no proven live trading edge
- Final holdout is a retrospective protocol, not a prospectively registered experiment

Conclusion: Insufficient evidence of positive net expectancy.
