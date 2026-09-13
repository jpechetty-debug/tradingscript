import numpy as np
from core.factors import FactorScores, DEFAULT_WEIGHTS
from core.scorer import apply_cohort_factor_ranking, FACTOR_NAMES


def _make_dummy_factors(
    trend: float = 0.5,
    momentum: float = 0.5,
    volume: float = 0.5,
    volatility: float = 0.5,
    rs: float = 0.5,
    breakout: float = 0.5,
    quality: float = 0.5,
    weights: dict | None = None,
) -> FactorScores:
    w = weights or dict(DEFAULT_WEIGHTS)
    comp = (
        trend * w.get("trend", 1 / 7)
        + momentum * w.get("momentum", 1 / 7)
        + volume * w.get("volume", 1 / 7)
        + volatility * w.get("volatility", 1 / 7)
        + rs * w.get("rs", 1 / 7)
        + breakout * w.get("breakout", 1 / 7)
        + quality * w.get("quality", 1 / 7)
    )
    return FactorScores(
        trend=trend,
        momentum=momentum,
        volume=volume,
        volatility=volatility,
        rs=rs,
        breakout=breakout,
        quality=quality,
        composite=round(comp, 4),
        ic_weights=w,
    )


class DummyCandidate:
    def __init__(self, ticker: str, direction: str, factors: FactorScores):
        self.ticker = ticker
        self.direction = direction
        self.factors = factors
        self.composite = factors.composite


class TestCohortFactorRanking:
    def test_cohort_ranking_basic_n15(self):
        # Create 15 candidates with strictly increasing trend scores: 0.1, 0.15, ..., 0.8
        candidates = []
        for i in range(15):
            trend_val = 0.10 + i * 0.05
            fs = _make_dummy_factors(trend=trend_val, momentum=0.5)
            candidates.append(DummyCandidate(f"TICK_{i}", "LONG", fs))

        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10)

        assert len(ranked) == 15
        # The lowest score candidate (i=0) has rank 1 -> rank_pct = 0.0
        # raw = 0.10, blended = 0.60 * 0.10 + 0.40 * 0.0 = 0.06
        assert abs(ranked[0].factors.trend - 0.06) < 1e-3

        # The highest score candidate (i=14) has rank 15 -> rank_pct = 1.0
        # raw = 0.80, blended = 0.60 * 0.80 + 0.40 * 1.0 = 0.48 + 0.40 = 0.88
        assert abs(ranked[14].factors.trend - 0.88) < 1e-3

        # The middle candidate (i=7) has rank 8 -> rank_pct = 7/14 = 0.50
        # raw = 0.10 + 7 * 0.05 = 0.45
        # blended = 0.60 * 0.45 + 0.40 * 0.50 = 0.27 + 0.20 = 0.47
        assert abs(ranked[7].factors.trend - 0.47) < 1e-3

    def test_cohort_ranking_tie_handling(self):
        # 10 candidates where 4 have score 0.20, 2 have score 0.50, and 4 have score 0.80
        candidates = []
        for i in range(4):
            candidates.append(DummyCandidate(f"LOW_{i}", "LONG", _make_dummy_factors(trend=0.20)))
        for i in range(2):
            candidates.append(DummyCandidate(f"MID_{i}", "LONG", _make_dummy_factors(trend=0.50)))
        for i in range(4):
            candidates.append(DummyCandidate(f"HIGH_{i}", "LONG", _make_dummy_factors(trend=0.80)))

        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10)

        # Ranks for ties:
        # 4 lowest values occupy ranks 1, 2, 3, 4 -> average rank = 2.5
        # rank_pct = (2.5 - 1.0) / 9.0 = 1.5 / 9.0 = 1/6 ~= 0.1667
        # blended = 0.60 * 0.20 + 0.40 * (1/6) = 0.12 + 0.0667 = 0.1867
        for c in ranked[:4]:
            assert abs(c.factors.trend - 0.1867) < 1e-3

        # 2 mid values occupy ranks 5, 6 -> average rank = 5.5
        # rank_pct = (5.5 - 1.0) / 9.0 = 4.5 / 9.0 = 0.50
        # blended = 0.60 * 0.50 + 0.40 * 0.50 = 0.50
        for c in ranked[4:6]:
            assert abs(c.factors.trend - 0.50) < 1e-3

        # 4 high values occupy ranks 7, 8, 9, 10 -> average rank = 8.5
        # rank_pct = (8.5 - 1.0) / 9.0 = 7.5 / 9.0 = 5/6 ~= 0.8333
        # blended = 0.60 * 0.80 + 0.40 * (5/6) = 0.48 + 0.3333 = 0.8133
        for c in ranked[6:]:
            assert abs(c.factors.trend - 0.8133) < 1e-3

    def test_cohort_fallback_when_below_min_obs(self):
        # 8 candidates (< 10) -> fallback to 100% raw scores
        candidates = []
        for i in range(8):
            candidates.append(DummyCandidate(f"TICK_{i}", "LONG", _make_dummy_factors(trend=0.30 + i * 0.05)))

        original_trends = [c.factors.trend for c in candidates]
        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10)

        # Scores should be unchanged
        for c, orig in zip(ranked, original_trends):
            assert c.factors.trend == orig

    def test_independent_directional_cohorts(self):
        # 12 LONG candidates and 5 SHORT candidates
        candidates = []
        for i in range(12):
            candidates.append(DummyCandidate(f"LONG_{i}", "LONG", _make_dummy_factors(volume=0.20 + i * 0.05)))
        for i in range(5):
            candidates.append(DummyCandidate(f"SHORT_{i}", "SHORT", _make_dummy_factors(volume=0.50)))

        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10)

        long_ranked = [c for c in ranked if c.direction == "LONG"]
        short_ranked = [c for c in ranked if c.direction == "SHORT"]

        # LONG cohort (N=12 >= 10) should be ranked and modified
        assert long_ranked[0].factors.volume != 0.20
        assert abs(long_ranked[0].factors.volume - (0.60 * 0.20 + 0.40 * 0.0)) < 1e-3

        # SHORT cohort (N=5 < 10) should remain completely unmodified
        for c in short_ranked:
            assert c.factors.volume == 0.50

    def test_composite_and_bounds_preservation(self):
        # Test that all blended factor scores strictly reside in [0, 1]
        np.random.seed(42)
        candidates = []
        for i in range(25):
            f_vals = {f: float(np.random.uniform(0.0, 1.0)) for f in FACTOR_NAMES}
            candidates.append(DummyCandidate(f"TICK_{i}", "LONG", _make_dummy_factors(**f_vals)))

        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10)

        for c in ranked:
            for f in FACTOR_NAMES:
                val = getattr(c.factors, f)
                assert 0.0 <= val <= 1.0
            # Composite should match weighted sum of blended factors
            w = c.factors.ic_weights
            expected_comp = sum(w[f] * getattr(c.factors, f) for f in FACTOR_NAMES)
            assert abs(c.factors.composite - expected_comp) < 1e-3
            assert abs(c.composite - expected_comp) < 1e-3

    def test_dict_candidate_support(self):
        candidates = [
            {"ticker": f"TICK_{i}", "direction": "LONG", "factors": _make_dummy_factors(momentum=i * 0.08), "composite": 0.5}
            for i in range(12)
        ]
        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10)
        assert len(ranked) == 12
        assert isinstance(ranked[0]["factors"], FactorScores)
        assert abs(ranked[0]["factors"].momentum - 0.0) < 1e-3
        assert abs(ranked[-1]["factors"].momentum - (0.60 * (11 * 0.08) + 0.40 * 1.0)) < 1e-3
