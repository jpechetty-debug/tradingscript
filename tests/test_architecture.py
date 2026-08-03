"""
tests/test_architecture.py
===========================
Architecture and characterization tests for SystemConfig decoupling.
"""

from __future__ import annotations

import ast
from pathlib import Path
from core.config import (
    AppSettings,
    ExecutionCostSettings,
    MarketDataSettings,
    PortfolioSettings,
    RegimeSettings,
    ScoringRuntime,
    SignalSettings,
    SystemConfig,
)


def test_system_config_adapters_preserve_values():
    """Verify that SystemConfig slice adapters extract exact configuration values."""
    cfg = SystemConfig(
        REGIME_CONFIRM_BARS=4,
        SUPER_PERIOD=14,
        PORTFOLIO_SIZE=8,
        SLIPPAGE_BPS=12,
        PLATT_A=-3.5,
    )

    md = cfg.as_market_data()
    assert isinstance(md, MarketDataSettings)
    assert md.daily_period == cfg.DAILY_PERIOD

    rg = cfg.as_regime()
    assert isinstance(rg, RegimeSettings)
    assert rg.regime_confirm_bars == 4

    sig = cfg.as_signal()
    assert isinstance(sig, SignalSettings)
    assert sig.super_period == 14

    port = cfg.as_portfolio()
    assert isinstance(port, PortfolioSettings)
    assert port.portfolio_size == 8

    cost = cfg.as_execution_cost()
    assert isinstance(cost, ExecutionCostSettings)
    assert cost.slippage_bps == 12

    rt = cfg.as_scoring_runtime()
    assert isinstance(rt, ScoringRuntime)
    assert rt.platt_a == -3.5

    app = cfg.as_app_settings()
    assert isinstance(app, AppSettings)
    assert app.regime.regime_confirm_bars == 4
    assert app.signal.super_period == 14
    assert app.portfolio.portfolio_size == 8


def test_leaf_modules_no_global_config_import():
    """Ensure leaf modules in core/ do not import global CONFIG singleton."""
    core_dir = Path(__file__).parent.parent / "core"
    leaf_files = [
        core_dir / "data_provider.py",
        core_dir / "regime.py",
        core_dir / "portfolio.py",
        core_dir / "factors.py",
        core_dir / "indicators.py",
    ]

    for file_path in leaf_files:
        if not file_path.exists():
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=file_path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == ".config" or node.module == "core.config":
                    imported_names = [alias.name for alias in node.names]
                    assert "CONFIG" not in imported_names, (
                        f"Leaf module {file_path.name} imports global CONFIG singleton!"
                    )
