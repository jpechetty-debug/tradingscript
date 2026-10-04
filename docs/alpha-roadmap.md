# Four-phase alpha roadmap

All four phases have executable components in a new research version. CodeGraph was used to locate the existing RS, sector, breadth, replay, PEAD and allocation components. The roadmap extends these instead of replacing them. Existing live selection and the frozen V1/V2 protocols remain unchanged; no performance gain is claimed.

## Run

From the repository root, using trusted local pickle price snapshots:

```powershell
# Missing feeds remain unavailable; export decisions and research packets.
.venv\Scripts\python.exe scripts\research_alpha.py

# Phase 1 ablation, then all phases with genuinely observed feeds.
.venv\Scripts\python.exe scripts\research_alpha.py --phase 1 --feeds research_data/v3 --snapshot research_data/prices.pkl --out artifacts/alpha_phase1
.venv\Scripts\python.exe scripts\research_alpha.py --feeds research_data/v3 --snapshot research_data/prices.pkl --out artifacts/alpha_scan

# Cash-constrained daily replay plus 10,000 stress paths.
.venv\Scripts\python.exe scripts\research_alpha.py --feeds research_data/v3 --snapshot research_data/prices.pkl --mode replay --start 2024-01-01 --out artifacts/alpha_replay

# 2015-2020 training -> 2021 test, 2016-2021 -> 2022, etc.
# A 5-year window starting in 2015 first tests 2020; use --train-years 6 for first test in 2021.
.venv\Scripts\python.exe scripts\research_alpha.py --feeds research_data/v3 --snapshot research_data/prices.pkl --mode walk-forward --start 2015-01-01 --train-years 6 --meta --out artifacts/alpha_walk_forward

# Optional grounded AI draft. Sends the supplied candidate/source packets externally.
# Configure OPENAI_API_KEY privately; choose an accessible model supporting Structured Outputs.
.venv\Scripts\python.exe scripts\research_alpha.py --feeds research_data/v3 --snapshot research_data/prices.pkl --documents research_data/documents.json --ai-model YOUR_MODEL_ID --out artifacts/alpha_ai
```

`--phase 1` uses RS, sector and price structure. Phase 2 adds PEAD, earnings growth and instrument flow. Phase 3 adds adaptive ADX/regime gates. Phase 4 provides the optional ML gate and AI research agent, with constrained portfolio allocation in every phase. `--meta` requires Phase 4 scan or walk-forward; replay refuses it to prevent fitting on the evaluation period. Insufficient model evidence produces unavailable predictions and rejects candidates when the meta gate is enabled.

Default scan outputs: `report.json`, `candidates.json`, `decisions.jsonl`, `research.json`. Replays also export `trades.csv`, `closed_trades.json`, `equity.csv`, `execution_decisions.csv` and `validation.json`. Walk-forward exports one subdirectory per annual fold, training sample/purge diagnostics and aggregate stitched returns. Each test book starts flat and liquidates at its boundary; stitched returns are descriptive, not a continuous held portfolio. Boundary-forced DATA_END labels are excluded from meta training in every mode. Too-short calendar history reports unavailable validation. Snapshot, feed, protocol and optional training/document/held-book hashes are retained.

## Inputs

Use the manifest, indices, sectors, surveillance, earnings and delivery formats from [swing-v2.md](swing-v2.md). Every feed requires `published_at,ingested_at,source`. Timestamps include an offset; publication must precede ingestion. Data only becomes available at ingestion. Local timestamps require provider audit. The benchmark snapshot must match the declared official index feed.

Two additional typed CSVs are supported:

| File | Required additional columns | Units/meaning |
|---|---|---|
| `fundamentals.csv` | `ticker,period_end,eps_growth,revenue_growth,guidance_raised` | Growth as fractions (0.25 means 25%), on consistent reporting/share/currency bases. Guidance is explicit true/false. Unknown guidance cannot be a false or true observation. |
| `flows.csv` | `ticker,session,fii_net_inr,dii_net_inr` | Instrument-specific signed net purchases in INR, each exchange session. National daily FII/DII aggregates cannot be assigned to every stock. If no provider supplies this evidence, the full flow factor stays unavailable. |

Earnings surprise, fundamentals, instrument flows and delivery are separate evidence. Earnings surprise and fundamental growth must describe the same reporting period. Delivery is an accumulation proxy and does not identify institutions. Missing optional V3 feeds do not affect V2; missing required V3 factors reject its Phase 2+ candidates. Latest known revisions supersede earlier values only after their ingestion time. Session/period observations cannot be published before their date.

Source documents are a JSON array:

```json
[
  {
    "ticker": "EXAMPLE",
    "topic": "quarterly_results",
    "published_at": "2026-10-01T17:00:00+05:30",
    "ingested_at": "2026-10-01T17:10:00+05:30",
    "source": "provider document identifier or source URL",
    "text": "Observed source text, retained with its provenance."
  }
]
```

Topics: `financials`, `quarterly_results`, `promoter_holding`, `institutional_holding`, `sector_tailwinds`, `management_commentary`. Future documents are excluded at the candidate's observation time. Without `--ai-model`, the agent writes evidence packets and missing-topic lists locally. With it, the [Responses API](https://developers.openai.com/api/docs/guides/text) produces drafts using [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs). Citation IDs and exact quotes are validated locally; that does not establish that a paraphrase follows from the quote. Semantic review remains necessary. Refusals, incomplete responses and fabricated citations fail validation. Tests use mocked responses and make no paid calls.

`--held-book` accepts JSON rows with `ticker,sector,shares,price,risk_inr`. Price is the current mark, not the original entry. NAV comes from configured CAPITAL_INR; scan allocations are indicative and exclude fees. Duplicate/invalid holdings are rejected. The allocator preserves held quantities and considers their exposure/risk before selecting new entries. Replay instead uses actual modeled cash, marks, costs and the simulated held book.

`--training-trades` with Phase 4 `--meta` scan accepts a JSON array of closed records with `exit_date,r_multiple,research_context`. Context must retain the original `observed_at`, `alpha_factors`, `breadth_sma50`, and `adaptive_regime`; later reconstructed feature values are invalid evidence. Labels beyond the embargoed cutoff are excluded. Never use trades from the evaluated period for training.

## Interpretation

The frozen [protocol](alpha-roadmap-protocol.json) defines score recipes and thresholds. RS uses a log-return difference, avoiding division by a zero or negative benchmark return. SMA50 breadth discloses its available-universe denominator and coverage. It is not certified exchange-wide breadth. Correlation requires at least 30 paired daily observations and rejects missing history. The new-entry heat/sector limits may be exceeded later by price changes; the engine does not retroactively sell the held book.

The Monte Carlo engine bootstraps net trade R in contiguous circular blocks and adds configurable modeled shocks. It uses fixed-fractional sequential wealth, includes initial equity in drawdown, and defines ruin as touching a chosen fraction of initial equity. It does not replay overlapping positions or daily covariance. Its VaR and ruin estimates depend on supplied outcomes and assumptions; zero historical trades yields unavailable statistics.

Compare each phase with matched prices, dates, costs and coverage. Preserve all variants and their rejections. Historical folds, research scores, training-model probabilities and AI narratives do not establish live readiness. Genuine point-in-time earnings/flows, historical membership including delistings, and independent future observations remain necessary for performance claims.

## Verification — 2026-10-04

The final Windows/Python 3.12 run passed **766 tests**, including 31 new roadmap cases, with **93.24% core coverage**. Repository-wide Ruff and `git diff --check` passed. Strict Mypy found only the two existing `fields(slice_obj)` argument-type errors at `core/config.py:648` and `:682`; those lines match the unchanged Git baseline. No new module typing errors were reported.

Local scan, short replay, unavailable-history reporting, and a real calendar train/test fold with meta training completed. The existing saved snapshot has no required dated alpha feeds; it generated zero qualified candidates, zero validation trades and insufficient model training evidence. A separate explicitly synthetic ten-outcome test completed the default 10,000 stress paths. These runs validate software behavior and do not establish returns. AI adapter tests use mocked responses; no paid API call was made.

Reproduction outputs are under `artifacts/alpha_roadmap_validation/`: `scan/report.json`, `replay/validation.json`, `short_history/report.json`, `walk_forward/report.json`, and `synthetic_stress.json`. These generated files are ignored by Git. Each report retains the protocol hash used at its execution time; earlier reports retain their earlier hashes rather than being relabeled after edits.
