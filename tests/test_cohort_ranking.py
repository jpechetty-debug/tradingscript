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
    def test_cohort_ranking_smooth_ramp_and_c0_continuity(self):
        # 9 candidates -> weight 0 (below min_obs)
        cands_9 = [
            DummyCandidate(f"T_{i}", "LONG", _make_dummy_factors(trend=0.55))
            for i in range(9)
        ]
        res_9 = apply_cohort_factor_ranking(cands_9, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        assert res_9[0].factors.trend == 0.55

        # 10 candidates -> at N=10, ramp is 0.0 (C0 continuity, no cliff!)
        cands_10 = [
            DummyCandidate(f"T_{i}", "LONG", _make_dummy_factors(trend=0.55 if i < 9 else 0.50))
            for i in range(10)
        ]
        res_10 = apply_cohort_factor_ranking(cands_10, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        # Should be unchanged at N=10 because ramp starts at 0
        assert res_10[0].factors.trend == 0.55

        # 20 candidates -> ramp is (20-10)/(30-10) = 0.50 -> effective weight = 0.20
        cands_20 = [
            DummyCandidate(f"T_{i}", "LONG", _make_dummy_factors(trend=0.50 + i * 0.01))
            for i in range(20)
        ]
        res_20 = apply_cohort_factor_ranking(cands_20, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        # Top candidate raw = 0.69, rank_p = 1.0. With mean >= 0.50, tape_mult = 1.0
        # b_val = 0.69 + 0.20 * (1.0 - 0.69) = 0.69 + 0.062 = 0.752
        assert abs(res_20[-1].factors.trend - (0.69 + 0.20 * (1.0 - 0.69))) < 1e-3

        # 30 candidates -> ramp is 1.0 -> full weight 0.40
        cands_30 = [
            DummyCandidate(f"T_{i}", "LONG", _make_dummy_factors(trend=0.50 + i * 0.01))
            for i in range(30)
        ]
        res_30 = apply_cohort_factor_ranking(cands_30, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        raw_top = 0.50 + 29 * 0.01
        expected = raw_top + 0.40 * (1.0 - raw_top)
        assert abs(res_30[-1].factors.trend - expected) < 1e-3

    def test_cohort_ranking_bad_tape_anchor(self):
        # In a bad tape, all candidates have raw factor = 0.30
        cands = [
            DummyCandidate(f"T_{i}", "LONG", _make_dummy_factors(trend=0.30))
            for i in range(30)
        ]
        ranked = apply_cohort_factor_ranking(cands, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        # With identical raw scores, ranks are tied, relative rank adjustment is 0
        assert all(c.factors.trend == 0.30 for c in ranked)

        # Now suppose one candidate is slightly better (0.35) while rest are 0.30
        cands[0] = DummyCandidate("T_BEST", "LONG", _make_dummy_factors(trend=0.35))
        ranked2 = apply_cohort_factor_ranking(cands, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        # Check that top candidate is NOT inflated over 0.52 gate because tape mean is ~0.30
        assert ranked2[0].factors.trend < 0.52

    def test_cohort_ranking_step_compat(self):
        # When cohort_full_obs == cohort_min_obs, acts as step function
        candidates = []
        for i in range(15):
            trend_val = 0.10 + i * 0.05
            fs = _make_dummy_factors(trend=trend_val, momentum=0.5)
            candidates.append(DummyCandidate(f"TICK_{i}", "LONG", fs))

        ranked = apply_cohort_factor_ranking(
            candidates, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=10
        )
        assert len(ranked) == 15
        # With full weight 0.40:
        # For middle candidate (i=7): raw=0.45, rank_pct=0.50, mean=0.45, tape_mult=0.45/0.50=0.9
        # b_val = 0.45 + 0.40 * (0.50 - 0.45) * 0.9 = 0.45 + 0.018 = 0.468
        assert 0.40 <= ranked[7].factors.trend <= 0.50

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
        # 30 LONG candidates and 5 SHORT candidates
        candidates = []
        for i in range(30):
            candidates.append(DummyCandidate(f"LONG_{i}", "LONG", _make_dummy_factors(volume=0.50 + i * 0.01)))
        for i in range(5):
            candidates.append(DummyCandidate(f"SHORT_{i}", "SHORT", _make_dummy_factors(volume=0.50)))

        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)

        long_ranked = [c for c in ranked if c.direction == "LONG"]
        short_ranked = [c for c in ranked if c.direction == "SHORT"]

        # LONG cohort (N=30 >= 10) should be ranked and modified
        assert long_ranked[0].factors.volume != 0.50

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
            for i in range(30)
        ]
        ranked = apply_cohort_factor_ranking(candidates, cohort_rank_weight=0.40, cohort_min_obs=10, cohort_full_obs=30)
        assert len(ranked) == 30
        assert isinstance(ranked[0]["factors"], FactorScores)
        assert 0.0 <= ranked[0]["factors"].momentum <= 1.0
        assert 0.0 <= ranked[-1]["factors"].momentum <= 1.0
