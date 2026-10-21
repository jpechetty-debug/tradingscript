# Daily paper-trading reconciliation — 21 October 2026

The daily report reads the existing `paper_events` journal and exports observed signals, explicit fill/exit links, cost gaps, timing exceptions and arithmetic checks. It does not register trades or change the journal, strategy or execution book.

## Run

```powershell
.venv\Scripts\python.exe scripts/reconcile_paper_trading.py
.venv\Scripts\python.exe scripts/reconcile_paper_trading.py --date 2026-10-21
.venv\Scripts\python.exe scripts/reconcile_paper_trading.py --date 2026-10-02 --as-of 2026-10-02T18:00:00+05:30 --db state/sovereign_state.db --out artifacts/paper_reconciliation/2026-10-02
```

The default date is today in IST. The database and output root follow `RuntimePaths.discover()`; `--db` and `--out` override them. Output defaults to `artifacts/paper_reconciliation/YYYY-MM-DD/`:

- `report.md`: daily summary, exception counts, newly observed signals and paper lifecycles observed that day or still open.
- `report.json`: complete structured results, original evidence identifiers and source digest.
- `signals.csv`, `scans.csv`, `trades.csv`, `exceptions.csv`, `market_dates.csv`: all included history through the observation cutoff, including outstanding items and earlier closed lifecycles.

The current-day report is partial through the run time. A historical report stops at that day's IST end; `--as-of` can choose an earlier timezone-aware cutoff. Later-recorded events cannot enter a historical report, even when their supplied trade timestamp is old. Daily counts use the journal's observation day in IST; supplied signal/fill/exit dates are retained separately. Run the command for each desired day as part of the existing daily workflow; this change does not install a scheduler.

## What is reconciled

Signals retain first-observed timestamps, market-date labels, strategy IDs, original parameter fingerprints and protocol hashes. Both the V1/V2 shape (`strategy_version`, `signal_time`) and leader-pullback shape (`strategy_id`, `signal_date`) are supported. Missing hashes stay unavailable and are reported; the report never reconstructs a historical fingerprint from today's settings. Selection/watchlist flags remain separate from actual fills. A signal without a matching fill remains **NO_LINKED_FILL**, with no assumed position, return, missed fill or expiry.

Fills link only through `signal_event_id`, with instrument/strategy identity checks; exits link through `trade_id` and their supplied signal ID. The report flags missing/broken links, mismatched identity, orphan exits, multiple fills/exits and repeated use of a signal. Ambiguous full-fill/full-exit chains are not silently collapsed or used to compute P&L. `valid_signal_link` means the explicit link identifies the same instrument and strategy; timing is assessed separately.

Earlier signals and fills remain available to reconcile today's exits. Outstanding open paper trades and signals without linked fills are counted through the cutoff. Scan observations, including zero-signal scans, are retained separately. Multiple observations of a market date are classified as all-zero, mixed zero/nonzero, signals reported or counts unavailable. A day without a scan is not called an exchange no-trade day. Existing scans lack a scan-to-signal ID, so scan counts are not treated as counts of unique new signals.

Explicit replay and executed-source records are excluded from paper results. Other unsupported event types are disclosed in scope diagnostics. The report checks relationships within the journal; it does not reconcile against broker statements or reconstruct historical versions of mutable `executed_trades` / `open_positions` tables.

## Cost and timing exceptions

Entry/exit fees and slippage are checked separately, preserving explicit zeros. Missing or invalid amounts and unknown cost bases remain visible. `OBSERVED` and `MODELED` cost bases are carried through. Exit `costs` is the reported total round-trip charge; entry/exit fee and slippage annotations are **not subtracted again**. Missing total costs prevent net P&L calculation. A reported total can support an arithmetic comparison even when its components are incomplete, but those gaps still require review. The report checks recorded net P&L and net R against fill prices, quantity, initial stop and the total charge. These checks are not an aggregate performance or validated-edge claim.

Timing is recomputed from timezone-aware supplied times and the journal's immutable `recorded_at`, rather than trusting previously stored audit booleans. Exceptions include:

- Entry/exit timestamps missing a timezone or otherwise invalid.
- Signal observation after entry, signal journal order after fill, exit time before entry, and exit journal order before fill.
- Future-dated entry/exit times relative to their observation timestamps.
- Recording delays above `--max-recording-delay-seconds` (default **900 seconds**). Actual delays are retained even below the threshold.
- Invalid/future signal or scan date labels and signals observed on a later calendar date than their market label. Calendar-day lag is descriptive; it does not infer exchange sessions or an exact candle publication time.

No grace interval hides negative recording delays. Passing timing checks or retaining a protocol hash does not prove independently collected prospective evidence.

Malformed payloads, duplicate identities and unplaceable journal timestamps are explicit integrity exceptions. Rows whose observation time cannot be parsed cannot enter state reconstruction; their diagnostic is marked with an unassigned observation date. Missing databases/tables fail explicitly rather than appearing to be empty valid journals.

## Read-only behavior and evidence

The reader opens SQLite with `mode=ro`, enables `query_only`, and reads all journal rows within one transaction. It does not instantiate the schema-migrating database wrapper, initialize a missing database, update a lifecycle or append to an export. Source digests cover included raw rows through the cutoff; they are not hashes of a market-data snapshot. Output targets resolving to the source database or its SQLite sidecars are rejected.

**54 focused tests passed**, including 41 reconciliation tests plus existing database and leader-pullback tests. The reconciliation module has **97.92% statement coverage** in that run. Tests cover a real `PaperLedger` / `TradeLifecycle` chain, more than 1,000 records, source-byte/journal preservation, cutoff stability, IST boundaries, short/long arithmetic, unknown versus zero costs, retrospective/future timestamps, invalid/ambiguous links, populated/empty exports and CLI execution. The existing pinned Ruff and strict Mypy commands pass across **78 source files**, without exclusions or suppressions. No full cross-platform CI or HTTP integration claim is made.

The configured local journal was empty when the report was run on 21 October 2026. Its report correctly says **NO_NEW_OBSERVATIONS**, with no scans, signals, fills or exits through the recorded cutoff. This is absence of journal observations, not a claim that a market scan ran and found no opportunities. The generated report is under `artifacts/paper_reconciliation/2026-10-21/`; validation evidence is under `artifacts/paper_reconciliation_validation/2026-10-21/`. Generated artifacts are ignored by Git.
