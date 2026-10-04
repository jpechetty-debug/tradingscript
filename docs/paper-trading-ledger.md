# Prospective paper journal

The authoritative journal is the `paper_events` table in the configured state database. It starts recording when future scans and explicit paper fill/exit events occur. Historical replay never writes to this journal. The initialized `state/paper_ledger.jsonl` export is empty; this is infrastructure for gathering evidence, not evidence already gathered.

## Daily workflow

1. Run `.venv\Scripts\python.exe run.py --swing-rules --no-telegram` after the exchange close. A `SCAN` event records the completed market date, regime and candidate/selection counts, including zero-signal observations. Regime confirmation counts distinct completed dates rather than repeated scans. New swing signals append `SIGNAL` events; repeated observations preserve the first signal snapshot.
2. Read authenticated `GET /api/paper/events?after=0&limit=1000` with `X-API-Key`. Continue from `next_after` to paginate. Retain the signal's `event_id`, strategy version, bound, shares and factor snapshot. Watchlist and unselected signals are identified explicitly.
3. If a next-session paper fill is actually observed under the plan, record it through `POST /api/trades/fills` with `source=SIMULATED`. Supply a stable `trade_id`, the signal's `strategy_id` and `signal_event_id`, actual modeled entry price, timezone-aware `entry_ts`, original stop/target, quantity, composite, probability and factors. Use the observed fill quantity and price, rather than assuming the recommendation was filled. A linked event must match the instrument and strategy. Fill records audit entry-bound compliance; they do not submit an order or enforce a complete pre-order risk engine.
4. Record the full paper exit through `POST /api/trades/{trade_id}/close`, with timezone-aware `exit_ts`, `exit_price`, `exit_reason` and total round-trip `costs`. Existing daily-candle paper stop/target monitoring can also record modeled exits. Time exits require an explicit observed paper exit. Net P&L subtracts `costs` once.
5. Run `.venv\Scripts\python.exe scripts/export_paper_ledger.py` daily to append new journal rows to `state/paper_ledger.jsonl`. The export validates the full prior prefix and does not overwrite or silently repair history. A lock prevents concurrent exporters; an interrupted exporter may leave its `.lock` file for the operator to inspect.

The scan command can use `--watch` for repeated local scans while running. There is no new background scheduler in this change. Repeated scans generate separate operational observations but do not manufacture additional confirmation bars or rewrite signal snapshots.

## Event fields and timing

Every row has an increasing `sequence`, stable `event_id`, `event_type`, application-generated UTC `recorded_at`, and JSON `payload`. SQLite triggers reject updates, deletes and replacements. First-observed signal identity includes ticker, daily candle label, strategy version and a parameter fingerprint. Factor values and factor weights are frozen in that signal snapshot.

`SIGNAL` payloads preserve `strategy_version`, `signal_time`, `entry_bound`, entry reference, stop, target, holding limit, proposed shares, composite, heuristic probability/status, factors, regime and selection/watchlist flags. Signal time is the daily candle label; `recorded_at` is when the application actually observed and stored it.

New signals additionally retain `research_context`: liquid-universe RS percentiles and denominators, leave-one-out sector-peer RS, EMA50/200 breadth, distance above EMA200 and volatility expansion. The signal stores the SHA-256 of the [future hypothesis protocol](swing-hypotheses-v1.json); repeated observations preserve the original features. These measurements do not add new entry gates. The historical alternate-exit simulator does not enable partial fills or partial closes in the application API.

Paper fills/exits preserve the linked signal ID and supplied event time, fill, stop/target, fees, slippage, modeled/observed cost basis and exit reason. `fees_inr` and `slippage_inr` are optional nonnegative INR amounts; omitted amounts are unknown, not assumed zero. Set `cost_basis` to `OBSERVED`, `MODELED` or `UNKNOWN`. For an exit, `costs` remains the total round-trip amount, including any slippage that the operator intends to charge to modeled P&L. Fee/slippage fields annotate that total and are not subtracted again.

Fill audit fields include `signal_precedes_entry`, `entry_recording_delay_seconds`, `future_dated_event` and `within_entry_bounds`. Exit delay is also retained. A row supplied later with an old trade timestamp is still recorded now. These facts make delayed or retrospective submissions visible; they do not automatically certify a record as genuine out-of-sample evidence. Assess prospective evidence using signals observed before the fill, faithful event timing and a strategy frozen before outcomes were known. Future-dated or backdated events must not be presented as prospective fills.

Paper fills and exits are committed in the same transaction as the execution ledger and open-position book. Identical retries append nothing. Conflicting events are rejected, and a journaling or book-write failure rolls the transaction back. Paper outcomes remain excluded from live executed-trade calibration.

## Review after sufficient future observations

Use the same setup, regime, distribution, exposure and benchmark metrics as the historical research report. Include no-trade days, rejected opportunities, fill delays, cost assumptions and operating failures. Three to six months is a collection period, not a guarantee of an adequate sample or positive expectancy. Reserve a new chronological evaluation period and keep point-in-time universe/event-data limitations visible.
