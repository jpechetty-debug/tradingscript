# Standalone LP/PEAD replay correction — 20 October 2026

The standalone `scripts/swing_research2.py` portfolio loop now uses an explicit chronological cash book in `core/scheduled_replay.py`. The core swing/alpha replay engines and application strategy defaults were not changed.

## Execution rules

1. Lock sizing NAV from the previous session's closing book, initially 1.0.
2. Execute previously held positions' scheduled opening exits and release their proceeds, net of exit fees.
3. Process the opening entry batch in descending priority, with deterministic ticker/setup ties. Enforce available cash, position count, single-position-per-ticker, initial risk and position notional caps.
4. Process intraday exits and closing exits, including positions opened in this session. Later proceeds and position slots cannot fund the earlier opening batch.
5. Mark remaining positions using only closes observed by this date. Missing closes carry the latest observed mark; a missing entry-day close retains the entry mark.

Gap stops have phase `OPEN`, low-price stop touches have phase `INTRADAY`, and the original SMA/time exits retain their modeled `CLOSE` phase. Opening exits funding opening entries is an explicit modeling assumption; auction order and cash settlement are not established by daily OHLC.

The replay calendar includes observed stock sessions as well as benchmark sessions. This snapshot has five stock dates absent from the benchmark, all within each evaluated portfolio's calendar. One LP exit is on 15 January 2026, absent from the benchmark. It is now processed instead of leaving the position open indefinitely. Reports list missing benchmark dates; benchmark marks carry forward and the hedge assumes zero benchmark return on these dates. This observed calendar is not certified exchange-calendar evidence.

Fractional quantities, the original setup definitions and per-setup overlap prefilter remain. The prefilter can omit later opportunities even when a portfolio rejected an earlier one; this remains an explicit limitation of this separate experiment. There is no automatic strategy activation.

## Regression evidence

- A same-day stop from 100 to 90 at 0.5% account risk now ends at NAV **0.995** with zero fees, rather than 1.0.
- Changing today's closing price cannot change today's opening quantity.
- Later exit prices/dates cannot change the earlier book.
- Closing/intraday sale proceeds cannot fund opening orders; an actual opening exit can.
- An entry-day stop cannot fund another order in the same opening batch.
- A position exiting at the close still occupies its opening slot.
- Entry/exit fees reconcile to realized cash and NAV; empty opportunities retain cash.
- Missing closes, benchmark calendar gaps, invalid chronology and empty-result CLI reports are covered.
- Both LP-only and supplied-earnings CLI workflows export ledgers; the latter uses explicitly synthetic earnings in tests only.

The CLI is import-safe and works as a direct script or module. Reports include input/source SHA-256 hashes, execution assumptions, equity, fills and rejected-decision CSVs. Per-trade `R` is now net of the same proportional entry/exit fees used by the book; `gross_R` remains separate. Summary statistics include first-session P&L and the capital entering each development/holdout segment.

## Matched local reproduction

Price snapshot: `artifacts/performance_validation/historical.pkl`, 61 series, 4 October 2021 through 1 October 2026. SHA-256: `698d4232d6f6e164deeb2ed87e094e373ec53204c26b28a64168bc0a6ecfcf23`.

| Local input variant | LP opportunities | Same-day exits | Old loop final NAV | Corrected final NAV | Corrected total return |
|---|---:|---:|---:|---:|---:|
| Snapshot universe | 218 | 35 | 1.330799 | 1.026177 | +2.6177% |
| Supplied membership filter | 161 | 31 | 1.112946 | 1.007618 | +0.7618% |

All opportunity identity, signal/entry/exit dates, planned prices/stops and priorities were checked for exact equality against the archived original source. Both corrected runs have matching entry/exit counts, no remaining positions, and nonnegative cash. Minimum cash is 0.620938 and 0.701064 respectively.

The old and corrected portfolio metrics use a common accounting method for comparison. Original-calendar legacy equity files are retained, and marks are carried onto additional stock-session dates only for comparison. The original source is archived without edits under `artifacts/standalone_replay_fix/2026-10-20/baseline_source.py`.

These are matched reruns using the available saved price snapshot, not a reproduction of unavailable original download inputs. The earnings snapshot used by the earlier LP/PEAD experiment is absent, so **PEAD has not been reproduced**. No earnings observations were invented. Synthetic test earnings are not included in historical reports. The membership CSV is partial/stale and these results do not establish complete point-in-time coverage or a positive trading edge.

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\swing_research2.py artifacts\performance_validation\historical.pkl --lp-only --out artifacts\standalone_replay_fix\2026-10-20\corrected
.venv\Scripts\python.exe scripts\swing_research2.py artifacts\performance_validation\historical.pkl --lp-only --membership artifacts\universe\nifty500_members.csv --out artifacts\standalone_replay_fix\2026-10-20\corrected_membership
.venv\Scripts\python.exe artifacts\standalone_replay_fix\2026-10-20\reproduce.py
```

The last command uses the archived original source to verify unchanged opportunities and write `comparison.json` plus legacy curves. Generated evidence is local and ignored by Git; this document records its principal results. Preserve the original snapshot and archived source to rerun that comparison.

With a genuine earnings snapshot, the original positional interface remains supported:

```powershell
.venv\Scripts\python.exe scripts\swing_research2.py PRICES.pkl EARNINGS.pkl MEMBERS.csv --out artifacts\standalone_research\with_earnings
```

## Checks

The final focused suite covers the new chronological book and CLI plus existing swing replay, leader-pullback and alpha-roadmap tests: **76 passed**. Ruff and strict Mypy pass on the changed production modules; `git diff --check` passes. Repository-wide checks still report 23 Ruff findings in other files and three Mypy errors in `scripts/swing_dl_earn.py` and `core/config.py`. This is not a current full-suite or full-core-coverage claim. The earlier API/journal integration stall is outside this change.
