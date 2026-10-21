# Universe and dated-feed audit — 20 October 2026

Priority 2 is implemented as a read-only, offline audit in `core/coverage_audit.py` and `scripts/audit_research_coverage.py`. It measures the supplied evidence and reports unavailable intervals. It does not download replacement data, repair membership history, infer missing observations or change trading policy.

## Saved-input findings

The audit covers **1,240 observed dates from 4 October 2021 through 1 October 2026**, using the union of stock and benchmark dates. This is an observed calendar, not a certified exchange calendar. The price snapshot contains 60 stock series and one benchmark series.

| Finding | Evidence |
|---|---|
| Legacy membership | 591 intervals, 578 distinct symbols; no publication, ingestion or source columns |
| Stale membership | 1 April 2022–1 October 2026; 1,117 audited dates |
| Overlapping intervals | `EQUITASBNK` has open-ended intervals starting 30 October 2020 and 29 October 2021; overlap is active on 1,222 audited dates |
| Daily valid-bar coverage | **4.4088%–7.6152%** of the declared member set; this denominator is not certified point-in-time membership |
| Latest audited date | 499 declared members; 22 have valid bars, 461 have no series, and 16 have a series but no bar on 1 October 2026 |
| Missing series over the period | 516 distinct declared symbols have a no-series interval; this does not establish why their prices are missing |
| Benchmark gaps | 15 January, 1 May, 28 May, 26 June and 14 September 2026 |
| Dated feeds | Indices, sectors, surveillance, earnings, delivery, fundamentals and instrument flows are all absent from the supplied feed inventory |
| Saved alpha provenance | Both saved scan/replay reports match this price snapshot and record an empty feed-hash dictionary; the saved scan reports zero qualified candidates |

The supplied membership CSV remains untouched. It contains the overlap above, so the audit preserves a unique declared-symbol count for diagnosis while flagging the integrity failure separately. It does not silently deduplicate intervals into trustworthy history.

The `2022-03-31` supported-through boundary comes from the reviewed builder's four-snapshot definition in `scripts/swing_build_members.py`. The CLI records that source file's hash and labels the date a **declared upper bound**, not proof of ingestion, publication, complete membership, correct symbol mapping or coverage before that date. No boundary is inferred from open-ended rows. Without this explicit argument, the audit reports unavailable legacy timing and does not invent a freshness cutoff.

## Explicit unavailable intervals

| Component | Start | End | Audited dates | Reason |
|---|---|---|---:|---|
| Membership | 2021-10-04 | 2021-10-28 | 18 | Legacy observation timing unavailable |
| Membership | 2021-10-29 | 2022-03-31 | 105 | Overlapping membership intervals |
| Membership | 2022-04-01 | 2026-10-01 | 1,117 | Beyond the declared supported-through boundary |
| Membership integrity: EQUITASBNK | 2021-10-29 | 2026-10-01 | 1,222 | Overlapping open-ended intervals, including throughout the stale interval |
| Price observation timing | 2021-10-04 | 2026-10-01 | 1,240 | Per-observation publication/ingestion and revision evidence unavailable |
| Each of the seven dated-feed categories | 2021-10-04 | 2026-10-01 | 1,240 | No feed supplied |

Per-symbol missing-series, missing-bar, invalid-bar and insufficient-lookback intervals are exported separately in `unavailable_intervals.csv`. Interval bounds are inclusive and summarize adjacent **audited sessions**, including intervening weekends; they do not assert observations for unaudited calendar days.

## Audit behavior

- **Membership:** supports legacy `[start,end)` intervals or dated `ADD`/`REMOVE` events. Optional timing requires all of `published_at,ingested_at,source`. Dated rows enter the diagnostic set only at ingestion. A future revision cannot alter an earlier known row set. Neither format establishes a complete initial baseline and uninterrupted change history by itself.
- **Prices:** validates OHLCV finiteness, positive OHLC, nonnegative volume and consistent high/low bounds. Missing series, absent session bars and invalid OHLCV have distinct counts/reasons. Coverage never forward-fills prices. The default 200-session complete-bar lookback uses prior history even when `--start` narrows the output period.
- **Ingestion:** retains raw timestamp claims, normalized UTC values, source identifiers, logical CSV record numbers, delay in seconds and validation errors. Naive timestamps, ingestion before publication, missing provenance and conflicting same-time revisions remain unavailable. File modification time is never used as evidence of historical knowledge.
- **Dated feeds:** reuses the application's `AlphaFeed` schemas and validation. An invalid supplied file is quarantined as a whole; its rows remain visible in the ingestion audit. Window coverage requires every as-of observation: 20 sessions for flows, 25 for delivery, 64 for sector indices by default and 200 for the declared benchmark. A declared VIX index uses 127 sessions. Expired intervals do not resurrect an older clearance or mapping. Earnings must have usable observed consensus and remain within the existing 20-session freshness limit; fundamentals use the existing 150-day period-age limit.
- **Restrictions:** a known ASM/GSM restriction counts as an observed surveillance record, not permission to trade. `UNKNOWN` is unavailable. Feed coverage does not evaluate all joint strategy conditions, including earnings/fundamentals period matching or benchmark-price agreement; those remain in the research engine.
- **Completeness:** price ingestion/revision evidence, verified initial membership/continuity, rename mappings and delisted histories are not supplied. Overall status remains `UNAVAILABLE_POINT_IN_TIME_COMPLETENESS`. Valid local timestamps would still require provider verification.

## Reproduce and inspect

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\audit_research_coverage.py --membership-supported-through 2022-03-31 --boundary-source scripts\swing_build_members.py --saved-run artifacts\alpha_roadmap_validation\scan\report.json --saved-run artifacts\alpha_roadmap_validation\replay\report.json --out artifacts\coverage_audit\2026-10-20
```

Outputs under `artifacts/coverage_audit/2026-10-20/`:

- `report.json`: inventory, source/code hashes, assumptions, summary and saved-run provenance comparisons.
- `report.md`: human-readable status summary.
- `daily_coverage.csv`: membership/price counts, denominator basis, integrity and knowledge status for every audited date.
- `feed_coverage.csv`: per-date, per-feed record coverage, required entity counts and missing/stale/expired/window reasons.
- `ingestion_audit.csv`: row-level timing claims and errors, plus explicit absence markers for missing feeds and price timing.
- `unavailable_intervals.csv`: compressed gaps by component, entity and reason.

To audit later genuine inputs, supply `--snapshot PRICES.pkl --membership MEMBERS.csv --feeds FEED_DIRECTORY`. Feed CSVs and `manifest.json` follow [the V2 schemas](swing-v2.md) and [the alpha extensions](alpha-roadmap.md). Omit the old supported-through declaration when auditing a different membership history; provide a justified boundary for that specific source instead. Only trusted local pickle snapshots should be used.

`--sessions SESSIONS.csv` accepts an explicit `session` column instead of the observed-date union. Its source hash is retained, but the audit does not certify that the supplied calendar is complete. `--start`, `--end`, `--price-lookback`, `--sector-lookback` and `--observation-hour` make the audit basis explicit. By default observations are evaluated at 18:00 IST.

The saved price and membership hashes remain:

```text
prices:     698d4232d6f6e164deeb2ed87e094e373ec53204c26b28a64168bc0a6ecfcf23
membership: f5e868ab79b6da3ecfeafdcfdc1d395eaeb32a45b0ce1f7dc10bdb10d2544e04
```

Generated audit artifacts are ignored by Git. This document records the findings; preserve the hashed source inputs to reproduce them. No missing earnings, institutional flow, surveillance or price observations were manufactured.

## Verification

**109 focused tests passed**, covering the new audit plus existing alpha-roadmap, V2, structural and standalone replay behavior. Coverage of the new audit module is **93.94%**; this is not full-project coverage. Tests cover delayed ingestion, future revisions, stale/overlapping membership, missing/delisted-price proxies, no price forward fill, preserved warm-up history, invalid timestamps, expired revisions, unavailable feed windows, deterministic exports and rejection of file modification times as knowledge evidence.

Ruff and strict Mypy pass for the new modules and Ruff passes for the tests. Existing application/strategy behavior is unchanged. Earlier unrelated repository quality findings and API integration limitations remain outside this audit change.
