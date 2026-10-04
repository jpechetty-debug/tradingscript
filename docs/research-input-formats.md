# As-of inputs for structural swing research

Use `python scripts/research_structural_edge.py` with the saved snapshot. Optional `--earnings path.csv` and `--membership path.csv` audit supplied histories. The current run uses neither, because the snapshot has no event observations or historical membership feed. These adapters do not certify completeness or turn the 60-stock snapshot into a point-in-time Nifty 500 backtest.

## Earnings observations

Required CSV header:

```csv
ticker,event_date,known_at
```

Use exchange security symbols with the same `.NS` suffix as the price data. `event_date` is an exchange-session date; `known_at` is a timezone-aware timestamp when that scheduled earnings date was actually known. Historical signal-close time is 15:30 IST in this analysis. An event published later cannot affect an earlier decision. The five-session hypothesis includes an event on the entry session or in the next five exchange sessions.

An absent row means `NO_KNOWN_EVENT_COVERAGE_UNVERIFIED`, not permission to trade. Include coverage provenance, historical schedule revisions/cancellations and the exchange session calendar before claiming an earnings-avoidance effect. This basic adapter only recognizes supplied known scheduled dates; it does not reconstruct revised event histories. Off-session events are not silently mapped to a trading session.

## Membership observations

Required CSV header:

```csv
ticker,effective_date,action,known_at
```

`action` is `ADD` or `REMOVE`. Include a complete initial membership baseline followed by every dated change. The adapter applies only events known at the historical decision time and effective by the contemplated entry session. Duplicate ambiguous events are rejected. It reports members without price bars, including omitted delisted instruments; silently discarding these would preserve selection bias.

The official [Nifty equity-index methodology](https://www.niftyindices.com/Methodology/Method_NIFTY_Equity_Indices.pdf) describes scheduled reviews and additional constituent changes. Today's constituent list cannot stand in for that dated history. Full reconstruction also requires security identifier/rename mappings, delisted prices, corporate actions and historical sector classifications. The adapter supplies an audit boundary; this dataset still lacks those inputs.

## Sector observations

The current research uses leave-one-out median 63-session peer RS within the liquid available subset, requiring at least two peers. Missing peers stay unknown. It does not claim an official sector-index EMA200 test. [Moskowitz and Grinblatt's industry-momentum research](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00146) gives a rationale for studying group strength, but does not establish an edge for these NSE rules.

## Future protocol and signals

[swing-hypotheses-v1.json](swing-hypotheses-v1.json) freezes the primary benchmark-EMA200 hypothesis, secondary cohorts and exit recipes for future observations beginning 2026-10-05. The fixed evaluation end is 2027-10-02, independent of outcomes. Sparse groups remain inconclusive. This is a local protocol, not external registration or a promise of statistical power.

New paper signals retain the protocol-file SHA-256, parameter fingerprint and `research_context`: liquid-universe RS ranks and denominators, leave-one-out peer RS, EMA50/200 breadth and denominators, distance above EMA200, ATR20 percentile/expansion and Bollinger-bandwidth expansion. Repeated scans preserve the first observed values. Earnings status remains unknown without the as-of feed. Changing the protocol requires a new version; prior journal rows are never relabeled.

The alternate exit engine operates only in historical research. The authenticated trade API still accepts full fills/exits; scale-outs in this research do not enable broker partial-fill or partial-close support. Prospective shadow-policy comparisons must keep independent books, faithful observed execution times and actual/modelled costs. No retrospective or backdated trade counts as prospective evidence.
