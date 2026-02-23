"""
Tests for Monte Carlo portfolio simulation (Task 6.4).

Tests cover:
- Core simulation engine (geometry, statistics, correlations)
- Edge cases (empty portfolio, insufficient data, single position)
- VaR/CVaR calculations
- Cholesky decomposition and PD matrix correction
- Reproducibility with seed
"""
import pytest
import pandas as pd
import numpy as np
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import portfolio


# ── Helpers ──────────────────────────────────────────────────────────

def _make_price_df(n: int = 300, start_price: float = 100.0,
                   daily_return: float = 1.001, noise: float = 0.01,
                   seed: int = 42) -> pd.DataFrame:
    """Create a synthetic price DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2023-01-02', periods=n, freq='B')
    returns = daily_return + rng.normal(0, noise, n)
    returns = np.clip(returns, 0.93, 1.07)
    close = start_price * np.cumprod(returns)
    high = close * 1.005
    low = close * 0.995
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame({
        'Open': (close + low) / 2,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


def _make_correlated_dfs(corr: float = 0.8, n: int = 300,
                         seed: int = 42) -> tuple:
    """Create two correlated price DataFrames."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2023-01-02', periods=n, freq='B')
    
    # Correlated normal returns
    z1 = rng.normal(0, 0.015, n)
    z2 = corr * z1 + np.sqrt(1 - corr**2) * rng.normal(0, 0.015, n)
    
    close1 = 100 * np.cumprod(1 + z1 + 0.0003)
    close2 = 150 * np.cumprod(1 + z2 + 0.0003)
    
    def _build(close):
        return pd.DataFrame({
            'Open': close * 0.999,
            'High': close * 1.005,
            'Low': close * 0.995,
            'Close': close,
            'Volume': np.full(n, 500_000.0)
        }, index=dates)
    
    return _build(close1), _build(close2)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def portfolio_file(tmp_path, monkeypatch):
    """Set up a temporary portfolio file with 2 positions."""
    pf_path = str(tmp_path / "portfolio.json")
    hist_path = str(tmp_path / "history.json")
    
    pf_data = {
        "positions": {
            "TICKER_A": {
                "quantity": 100,
                "avg_price": 100.0,
                "buy_date": "2023-06-01",
                "stop_loss": 92.0,
                "trailing_high": 120.0
            },
            "TICKER_B": {
                "quantity": 50,
                "avg_price": 150.0,
                "buy_date": "2023-06-01",
                "stop_loss": 138.0,
                "trailing_high": 170.0
            }
        },
        "created": "2023-06-01T00:00:00",
        "last_updated": "2024-01-01T00:00:00",
        "version": "1.0"
    }
    
    with open(pf_path, 'w') as f:
        json.dump(pf_data, f)
    with open(hist_path, 'w') as f:
        json.dump([], f)
    
    monkeypatch.setattr(portfolio, '_get_portfolio_path', lambda: pf_path)
    monkeypatch.setattr(portfolio, '_get_history_path', lambda: hist_path)
    
    return pf_data


@pytest.fixture
def df_dict_2_tickers():
    """Two correlated price DataFrames."""
    df_a, df_b = _make_correlated_dfs(corr=0.6, n=300, seed=42)
    return {'TICKER_A': df_a, 'TICKER_B': df_b}


@pytest.fixture
def single_position_file(tmp_path, monkeypatch):
    """Portfolio with a single position."""
    pf_path = str(tmp_path / "portfolio.json")
    hist_path = str(tmp_path / "history.json")
    
    pf_data = {
        "positions": {
            "SOLO": {
                "quantity": 200,
                "avg_price": 80.0,
                "buy_date": "2023-03-01",
                "stop_loss": 73.6,
                "trailing_high": 95.0
            }
        },
        "created": "2023-03-01T00:00:00",
        "last_updated": "2024-01-01T00:00:00",
        "version": "1.0"
    }
    
    with open(pf_path, 'w') as f:
        json.dump(pf_data, f)
    with open(hist_path, 'w') as f:
        json.dump([], f)
    
    monkeypatch.setattr(portfolio, '_get_portfolio_path', lambda: pf_path)
    monkeypatch.setattr(portfolio, '_get_history_path', lambda: hist_path)
    
    return pf_data


# =====================================================================
# TEST CLASS: Core Simulation
# =====================================================================

class TestMonteCarloCore:
    """Core simulation functionality."""
    
    def test_returns_dict_with_expected_keys(self, portfolio_file, df_dict_2_tickers):
        """Simulation result should contain all expected keys."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=42
        )
        assert result is not None
        
        expected_keys = [
            'startverdi', 'n_simuleringer', 'n_dager', 'n_posisjoner',
            'tickers', 'vekter',
            'pct_5', 'pct_10', 'pct_25', 'pct_50', 'pct_75', 'pct_90', 'pct_95',
            'median_sluttverdi', 'worst_case_5', 'best_case_95',
            'median_avkastning_pct', 'worst_avkastning_pct', 'best_avkastning_pct',
            'var_95', 'var_95_pct', 'cvar_95', 'cvar_95_pct',
            'max_drawdown_median_pct',
            'prob_tap_pct', 'prob_gevinst_10_pct', 'prob_tap_20_pct',
            'sharpe_estimate', 'avg_korrelasjon', 'korrelasjon_matrise'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_percentile_curves_correct_length(self, portfolio_file, df_dict_2_tickers):
        """Percentile curves should have length n_dager + 1."""
        n_dager = 63
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=n_dager, seed=42
        )
        assert len(result['pct_50']) == n_dager + 1
        assert len(result['pct_5']) == n_dager + 1
        assert len(result['pct_95']) == n_dager + 1
    
    def test_percentile_ordering(self, portfolio_file, df_dict_2_tickers):
        """At any time point, percentiles should be properly ordered: 5 < 25 < 50 < 75 < 95."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=2000, n_dager=126, seed=42
        )
        for i in range(len(result['pct_50'])):
            assert result['pct_5'][i] <= result['pct_25'][i], f"Day {i}: pct5 > pct25"
            assert result['pct_25'][i] <= result['pct_50'][i], f"Day {i}: pct25 > pct50"
            assert result['pct_50'][i] <= result['pct_75'][i], f"Day {i}: pct50 > pct75"
            assert result['pct_75'][i] <= result['pct_95'][i], f"Day {i}: pct75 > pct95"
    
    def test_startverdi_matches_day_zero(self, portfolio_file, df_dict_2_tickers):
        """All percentile curves should start at the same portfolio value."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=42
        )
        start = result['startverdi']
        assert result['pct_5'][0] == pytest.approx(start, rel=1e-6)
        assert result['pct_50'][0] == pytest.approx(start, rel=1e-6)
        assert result['pct_95'][0] == pytest.approx(start, rel=1e-6)
    
    def test_startverdi_positive(self, portfolio_file, df_dict_2_tickers):
        """Portfolio start value should be positive."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=42
        )
        assert result['startverdi'] > 0
    
    def test_reproducibility_with_seed(self, portfolio_file, df_dict_2_tickers):
        """Same seed should produce identical results."""
        r1 = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=123
        )
        r2 = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=123
        )
        assert r1['median_sluttverdi'] == r2['median_sluttverdi']
        assert r1['var_95'] == r2['var_95']
        assert r1['pct_50'] == r2['pct_50']
    
    def test_different_seeds_different_results(self, portfolio_file, df_dict_2_tickers):
        """Different seeds should produce different results."""
        r1 = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=1000, n_dager=63, seed=42
        )
        r2 = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=1000, n_dager=63, seed=99
        )
        # Very unlikely to be exactly equal
        assert r1['median_sluttverdi'] != r2['median_sluttverdi']


# =====================================================================
# TEST CLASS: VaR and Risk Measures
# =====================================================================

class TestVaRMeasures:
    """Value-at-Risk and risk metric tests."""
    
    def test_var_95_positive(self, portfolio_file, df_dict_2_tickers):
        """VaR should be a positive loss amount."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=2000, n_dager=252, seed=42
        )
        assert result['var_95'] > 0
        assert result['var_95_pct'] > 0
    
    def test_cvar_geq_var(self, portfolio_file, df_dict_2_tickers):
        """CVaR (Expected Shortfall) should be >= VaR."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=5000, n_dager=252, seed=42
        )
        assert result['cvar_95'] >= result['var_95'] - 0.01  # Tiny tolerance
    
    def test_var_reasonable_range(self, portfolio_file, df_dict_2_tickers):
        """VaR 95% for equity portfolio should typically be 10-50% for 1 year."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=5000, n_dager=252, seed=42
        )
        # Very generous bounds — just sanity check
        assert 1.0 < result['var_95_pct'] < 80.0
    
    def test_prob_tap_between_0_and_100(self, portfolio_file, df_dict_2_tickers):
        """Probabilities should be between 0 and 100."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=2000, n_dager=252, seed=42
        )
        assert 0 <= result['prob_tap_pct'] <= 100
        assert 0 <= result['prob_gevinst_10_pct'] <= 100
        assert 0 <= result['prob_tap_20_pct'] <= 100
    
    def test_worst_case_below_best_case(self, portfolio_file, df_dict_2_tickers):
        """Worst case (5th percentile) should be below best case (95th)."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=2000, n_dager=252, seed=42
        )
        assert result['worst_case_5'] < result['best_case_95']
        assert result['worst_avkastning_pct'] < result['best_avkastning_pct']


# =====================================================================
# TEST CLASS: Correlations
# =====================================================================

class TestCorrelations:
    """Tests for correlation-aware simulation."""
    
    def test_correlation_matrix_shape(self, portfolio_file, df_dict_2_tickers):
        """Correlation matrix should be NxN where N = number of positions."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=42
        )
        corr = result['korrelasjon_matrise']
        n = len(corr['tickers'])
        assert len(corr['matrise']) == n
        assert all(len(row) == n for row in corr['matrise'])
    
    def test_correlation_diagonal_is_one(self, portfolio_file, df_dict_2_tickers):
        """Diagonal of correlation matrix should be 1.0."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=42
        )
        corr = result['korrelasjon_matrise']['matrise']
        for i in range(len(corr)):
            assert corr[i][i] == pytest.approx(1.0, abs=0.01)
    
    def test_avg_correlation_reasonable(self, portfolio_file, df_dict_2_tickers):
        """Average correlation should be between -1 and 1."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=500, n_dager=63, seed=42
        )
        assert -1.0 <= result['avg_korrelasjon'] <= 1.0
    
    def test_correlated_assets_detected(self, portfolio_file):
        """Highly correlated assets should show high avg correlation."""
        df_a, df_b = _make_correlated_dfs(corr=0.9, n=300, seed=42)
        result = portfolio.monte_carlo_portefolje(
            {'TICKER_A': df_a, 'TICKER_B': df_b},
            n_simuleringer=500, n_dager=63, seed=42
        )
        # With 0.9 input correlation, measured should be notably positive
        assert result['avg_korrelasjon'] > 0.3


# =====================================================================
# TEST CLASS: Single Position
# =====================================================================

class TestSinglePosition:
    """Tests with single-position portfolio."""
    
    def test_single_position_works(self, single_position_file):
        """Monte Carlo should work with a single position."""
        df = _make_price_df(n=300, start_price=80, seed=42)
        result = portfolio.monte_carlo_portefolje(
            {'SOLO': df}, n_simuleringer=500, n_dager=63, seed=42
        )
        assert result is not None
        assert result['n_posisjoner'] == 1
    
    def test_single_position_no_correlation(self, single_position_file):
        """Single position should have avg correlation of 0."""
        df = _make_price_df(n=300, start_price=80, seed=42)
        result = portfolio.monte_carlo_portefolje(
            {'SOLO': df}, n_simuleringer=500, n_dager=63, seed=42
        )
        assert result['avg_korrelasjon'] == pytest.approx(0.0, abs=0.01)
    
    def test_single_position_weights_100pct(self, single_position_file):
        """Single position should have 100% weight."""
        df = _make_price_df(n=300, start_price=80, seed=42)
        result = portfolio.monte_carlo_portefolje(
            {'SOLO': df}, n_simuleringer=500, n_dager=63, seed=42
        )
        assert result['vekter']['SOLO'] == pytest.approx(100.0, abs=0.1)


# =====================================================================
# TEST CLASS: Edge Cases
# =====================================================================

class TestEdgeCases:
    """Edge case handling."""
    
    def test_empty_portfolio_returns_none(self, tmp_path, monkeypatch):
        """Empty portfolio should return None."""
        pf_path = str(tmp_path / "portfolio.json")
        hist_path = str(tmp_path / "history.json")
        
        with open(pf_path, 'w') as f:
            json.dump({"positions": {}, "version": "1.0"}, f)
        with open(hist_path, 'w') as f:
            json.dump([], f)
        
        monkeypatch.setattr(portfolio, '_get_portfolio_path', lambda: pf_path)
        monkeypatch.setattr(portfolio, '_get_history_path', lambda: hist_path)
        
        result = portfolio.monte_carlo_portefolje({}, n_simuleringer=100, n_dager=63)
        assert result is None
    
    def test_insufficient_data_returns_none(self, portfolio_file):
        """Positions with < 60 days of data should be skipped → None if all skip."""
        short_df = _make_price_df(n=30, start_price=100, seed=42)
        result = portfolio.monte_carlo_portefolje(
            {'TICKER_A': short_df, 'TICKER_B': short_df},
            n_simuleringer=100, n_dager=63
        )
        assert result is None
    
    def test_missing_ticker_data_partial(self, portfolio_file):
        """If one ticker has data and one doesn't, should still run with 1 position."""
        df_a = _make_price_df(n=300, start_price=100, seed=42)
        df_b = _make_price_df(n=20, start_price=150, seed=42)  # Too short
        
        result = portfolio.monte_carlo_portefolje(
            {'TICKER_A': df_a, 'TICKER_B': df_b},
            n_simuleringer=500, n_dager=63, seed=42
        )
        assert result is not None
        assert result['n_posisjoner'] == 1
        assert 'TICKER_A' in result['tickers']
    
    def test_empty_df_dict(self, portfolio_file):
        """Empty df_dict should return None."""
        result = portfolio.monte_carlo_portefolje({}, n_simuleringer=100, n_dager=63)
        assert result is None
    
    def test_n_dager_matches_output(self, portfolio_file, df_dict_2_tickers):
        """Output n_dager should match input."""
        for n in [63, 126, 252]:
            result = portfolio.monte_carlo_portefolje(
                df_dict_2_tickers, n_simuleringer=200, n_dager=n, seed=42
            )
            assert result['n_dager'] == n
            assert len(result['pct_50']) == n + 1


# =====================================================================
# TEST CLASS: Numerical Stability
# =====================================================================

class TestNumericalStability:
    """Tests for numerical robustness."""
    
    def test_nearest_positive_definite(self):
        """_nearest_positive_definite should fix a non-PD matrix."""
        # Slightly negative eigenvalue
        A = np.array([[1.0, 0.9, 0.9],
                       [0.9, 1.0, 0.9],
                       [0.9, 0.9, 1.0]])
        # This one is PD already, test with a forced non-PD
        non_pd = A.copy()
        non_pd[0, 0] = 0.5  # Break PD property
        non_pd[1, 1] = 0.5
        non_pd[2, 2] = 0.5
        
        result = portfolio._nearest_positive_definite(non_pd)
        assert portfolio._is_positive_definite(result)
    
    def test_is_positive_definite_identity(self):
        """Identity matrix should be positive definite."""
        assert portfolio._is_positive_definite(np.eye(3))
    
    def test_is_positive_definite_negative(self):
        """Matrix with negative eigenvalue should not be PD."""
        A = np.array([[1.0, 2.0],
                       [2.0, 1.0]])  # Eigenvalues: 3, -1
        assert not portfolio._is_positive_definite(A)
    
    def test_large_simulation_no_crash(self, portfolio_file, df_dict_2_tickers):
        """Larger simulation should run without numerical errors."""
        result = portfolio.monte_carlo_portefolje(
            df_dict_2_tickers, n_simuleringer=10000, n_dager=252, seed=42
        )
        assert result is not None
        # No NaN/Inf in percentile curves
        for key in ['pct_5', 'pct_50', 'pct_95']:
            assert all(np.isfinite(v) for v in result[key]), f"Non-finite values in {key}"
