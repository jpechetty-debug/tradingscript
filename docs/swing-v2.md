# Swing V2: independent alpha experiments

V2 is implemented for local research and prospective paper observation. Sector leadership and PEAD are separate models; COMBINED is a secondary experiment. Existing V1 deployment behavior and its prospective protocol are preserved. No positive expectancy or live execution readiness is claimed.

## Run

Use the project Python runtime, from the repository root:

```powershell
# Existing snapshot, no fabricated feeds: reports unavailable inputs and zero candidates.
.venv\Scripts\python.exe scripts\research_v2.py --model SECTOR
.venv\Scripts\python.exe scripts\research_v2.py --model PEAD --out artifacts/v2_pead

# Supply genuine feeds and a matching benchmark price snapshot.
.venv\Scripts\python.exe scripts\research_v2.py --feeds research_data/v2 --snapshot research_data/prices.pkl --model SECTOR
.venv\Scripts\python.exe scripts\research_v2.py --feeds research_data/v2 --snapshot research_data/prices.pkl --model PEAD --mode replay --start 2024-01-01 --out artifacts/v2_pead_replay

# Individual ablations; use different output directories to preserve comparisons.
.venv\Scripts\python.exe scripts\research_v2.py --feeds research_data/v2 --snapshot research_data/prices.pkl --mode replay --gate market --out artifacts/v2_market
.venv\Scripts\python.exe scripts\research_v2.py --feeds research_data/v2 --snapshot research_data/prices.pkl --mode replay --model PEAD --exit CHANDELIER --out artifacts/v2_pead_chandelier

# Today's completed scan after 18:00 IST: append-only independent V2 observations.
.venv\Scripts\python.exe scripts\research_v2.py --feeds research_data/v2 --snapshot research_data/prices.pkl --model SECTOR --journal-db state/v2_sector.db
```

Only use trusted local pickle snapshots. Each run exports its protocol/settings/input hashes, per-stock rejection reasons and feature context. Replays additionally export trades, execution decisions and daily cash/NAV with matched benchmark statistics. Gates (`market`, `breadth`, `vix`, `delivery`) are disabled by default and should be tested individually. Neither research rank nor earnings surprise is a calibrated win probability. Numeric zero in the legacy replay probability column is an **unused sentinel**; paper events use null with UNAVAILABLE status.

The observation journal records zero-candidate scans and first-observed signals, with current system timestamps. It refuses historical/future scans as prospective records. No orders are placed; candidates are not automatically selected or filled. App fill/close support remains the existing explicit paper workflow; before any V2 fill, check current surveillance, sector mapping, bounds, cash and the held-book limits. The historical replay implements these entry checks; the application does not automatically run V2 or its guards.

## Input directory

`manifest.json` declares a fixed complete rank universe, not just sectors present in a selected stock subset:

```json
{
  "benchmark_index": "NIFTY500",
  "benchmark_ticker": "NIFTY500",
  "vix_index": "INDIA_VIX",
  "sector_universe": ["NIFTY_AUTO", "NIFTY_BANK", "NIFTY_IT", "NIFTY_PHARMA"]
}
```

This four-sector example is illustrative, not a recommended NSE sector universe. Freeze a consistently defined actual universe before research. The trusted price snapshot maps `benchmark_ticker` to benchmark OHLCV and each stock to its `.NS` OHLCV. Matching index-feed closes are checked to 0.1%; Nifty 50 cannot silently stand in for Nifty 500. Benchmark membership/stock breadth still requires a complete point-in-time price universe to remove survivorship bias.

Every CSV has `published_at,ingested_at,source`. Use explicit timezone offsets, e.g. `2026-10-01T17:00:00+05:30`. Ingestion must follow publication. Availability uses ingestion time; later revisions cannot leak into earlier observations. Calendar dates below use `YYYY-MM-DD`, without timezone or time of day. Identical key/ingestion-time revisions are rejected. Sources should identify the provider/file or stable source URL, with originals retained externally.

| CSV | Additional required columns | Meaning |
|---|---|---|
| `indices.csv` | `index,session,close` | Official index daily closes including benchmark, every declared sector and optional India VIX. Complete aligned lookback required for ranks; any missing sector makes ranks unavailable. |
| `sectors.csv` | `ticker,effective_date,valid_until,sector_index` | Dated stock-to-index mapping; intervals inclusive. Latest known effective mapping supersedes older mapping. |
| `surveillance.csv` | `ticker,effective_date,valid_until,state` | Explicit `CLEAR`, `ASM`, `GSM`, `BOTH`, `UNKNOWN`. Absence/expiry is unavailable. CLEAR must certify checked ASM and GSM coverage, not absence in a partial downloaded list. |
| `earnings.csv` | `ticker,period_end,announced_at,expected_at,expectation_ingested_at,actual_eps,expected_eps,expectation_kind` | Actual/expected EPS on the same accounting/share/currency basis. `CONSENSUS` or `SEASONAL`. Both expectation times must precede announcement; actual publication follows announcement. Seasonal growth/forecasts cannot qualify consensus PEAD. |
| `delivery.csv` | `ticker,session,delivery_qty,total_qty,turnover` | Quantities in shares, turnover INR. Complete latest 25-session window; fraction/quantity/turnover compared with prior 20 sessions. |

Tickers normalize to uppercase without `.NS` in feeds. Index identifiers must match the manifest exactly. EPS surprise is `(actual-expected)/abs(expected)`; zero expectations are unscorable. Earnings announcements after close first contribute to the next completed session; revisions retain their own publication/ingestion timestamps. Missing feeds, expired intervals, missing history and invalid values never become passing signals.

Surveillance is checked at the 18:00 scan and at 09:15 IST before next-session replay entries. A newly published restriction blocks entry; information ingested after opening cannot be retroactively used at opening. Freeze/verify real observation timing before changing execution assumptions. Predicting ASM thresholds is not implemented.

## Promotion requirements

Follow `swing-v2-protocol.json`. Require verified data coverage, point-in-time constituents including delisted prices, independent future evidence and realistic execution/cost assumptions before promoting a model. Compare separate sector/PEAD variants first, then incremental combinations. Current-snapshot backtests remain exploratory. Price-band/auction feasibility, licensed ingestion, accounting-basis validation, brokerage reconciliation and partial live fills remain outside this research implementation.
