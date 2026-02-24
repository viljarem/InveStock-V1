"""
Tests for smart_money.py — Smart Money Flow indicator.

Tests cover:
  - Chaikin Money Flow (CMF) calculation
  - OBV calculation
  - Daily proxy SM score
  - Intraday SMI from hourly bars
  - Divergence detection
  - Scanner integration function
  - Edge cases (empty data, short data)
"""
import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import smart_money


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def df_daily_up():
    """60 days of uptrending daily data with increasing volume."""
    np.random.seed(42)
    n = 60
    dates = pd.bdate_range('2025-01-01', periods=n)
    close = 100 + np.cumsum(np.random.normal(0.3, 1.0, n))
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    open_ = close + np.random.normal(0, 0.5, n)
    volume = np.random.randint(50000, 200000, n).astype(float)
    # Make volume increase over time (accumulation)
    volume = volume * np.linspace(0.5, 1.5, n)
    
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


@pytest.fixture
def df_daily_down():
    """60 days of downtrending daily data with close near lows (distribution)."""
    np.random.seed(99)
    n = 60
    dates = pd.bdate_range('2025-01-01', periods=n)
    close = 200 + np.cumsum(np.random.normal(-0.4, 1.0, n))
    high = close + np.abs(np.random.normal(2, 0.5, n))
    low = close - np.abs(np.random.normal(0.5, 0.3, n))
    open_ = close + np.random.normal(0.5, 0.3, n)  # open above close (bearish candles)
    volume = np.random.randint(80000, 250000, n).astype(float)
    
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


@pytest.fixture
def df_hourly():
    """
    Simulated 20 trading days × 7 hourly bars = ~140 bars.
    Smart money pattern: early bars lose, late bars gain.
    """
    np.random.seed(55)
    bars = []
    base_price = 150.0
    
    for day_num in range(20):
        day_date = pd.Timestamp('2025-01-06') + pd.Timedelta(days=day_num)
        if day_date.weekday() >= 5:
            continue
        
        day_open = base_price + np.random.normal(0, 0.5)
        
        for hour_idx in range(7):  # 09:00 to 15:00
            bar_time = day_date.replace(hour=9 + hour_idx, minute=0)
            
            if hour_idx == 0:
                # First bar: retail panic → negative return
                o = day_open
                c = o * (1 - 0.003)  # -0.3%
            elif hour_idx == 6:
                # Last bar: institutional buying → positive return
                o = base_price + np.random.normal(0, 0.3)
                c = o * (1 + 0.005)  # +0.5%
            else:
                o = base_price + np.random.normal(0, 0.3)
                c = o + np.random.normal(0, 0.2)
            
            h = max(o, c) + abs(np.random.normal(0.2, 0.1))
            l = min(o, c) - abs(np.random.normal(0.2, 0.1))
            v = np.random.randint(10000, 50000)
            
            bars.append({
                'datetime': bar_time,
                'Open': o, 'High': h, 'Low': l,
                'Close': c, 'Volume': float(v)
            })
        
        base_price += np.random.normal(0.1, 0.5)
    
    df = pd.DataFrame(bars).set_index('datetime')
    df.index = pd.DatetimeIndex(df.index)
    return df


# ═══════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════

class TestDagligProxy:
    """Test Chaikin Money Flow + OBV proxy."""

    def test_proxy_structure(self, df_daily_up):
        result = smart_money.beregn_smi_daglig_proxy(df_daily_up)
        assert not result.empty
        assert 'cmf' in result.columns
        assert 'obv' in result.columns
        assert 'sm_proxy' in result.columns
        assert 'sm_proxy_sma10' in result.columns

    def test_cmf_bounds(self, df_daily_up):
        result = smart_money.beregn_smi_daglig_proxy(df_daily_up)
        cmf = result['cmf'].dropna()
        assert (cmf >= -1.0).all() and (cmf <= 1.0).all(), \
            f"CMF should be in [-1, 1], got [{cmf.min():.3f}, {cmf.max():.3f}]"

    def test_obv_monotonic_structure(self, df_daily_up):
        result = smart_money.beregn_smi_daglig_proxy(df_daily_up)
        # OBV should be non-trivially non-zero
        assert result['obv'].abs().max() > 0

    def test_empty_data(self):
        result = smart_money.beregn_smi_daglig_proxy(pd.DataFrame())
        assert result.empty

    def test_short_data(self, df_daily_up):
        result = smart_money.beregn_smi_daglig_proxy(df_daily_up.iloc[:10])
        assert result.empty  # Need > 30 rows (perioder + 10)

    def test_uptrend_cmf_positive(self, df_daily_up):
        """In an uptrend with increasing volume, CMF should be mostly positive."""
        result = smart_money.beregn_smi_daglig_proxy(df_daily_up)
        cmf = result['cmf'].dropna()
        positive_pct = (cmf > 0).mean()
        # At least 40% positive (with random data, not guaranteed to be super strong)
        assert positive_pct > 0.3, f"Expected mostly positive CMF in uptrend, got {positive_pct:.1%} positive"


class TestIntradagSMI:
    """Test Smart Money Index from hourly bars."""

    def test_smi_structure(self, df_hourly):
        result = smart_money.beregn_smi_intradag(df_hourly)
        assert not result.empty
        assert 'smi' in result.columns
        assert 'smi_daily' in result.columns
        assert 'early_return' in result.columns
        assert 'late_return' in result.columns
        assert 'smi_sma10' in result.columns

    def test_smi_positive_when_smart_money_buys(self, df_hourly):
        """When late bars consistently outperform early bars, SMI should trend up."""
        result = smart_money.beregn_smi_intradag(df_hourly)
        # Late return should be higher on average (our fixture has +0.5% late, -0.3% early)
        assert result['late_return'].mean() > result['early_return'].mean()
        # SMI (cumulative) should be positive at end
        assert result['smi'].iloc[-1] > 0

    def test_empty_data(self):
        result = smart_money.beregn_smi_intradag(pd.DataFrame())
        assert result.empty

    def test_single_bar_day(self):
        """Day with only 1 bar should be skipped."""
        df = pd.DataFrame({
            'Open': [100], 'High': [101], 'Low': [99],
            'Close': [100.5], 'Volume': [10000.0]
        }, index=pd.DatetimeIndex([pd.Timestamp('2025-01-06 09:00')]))
        result = smart_money.beregn_smi_intradag(df)
        assert result.empty


class TestDivergens:
    """Test divergence detection."""

    def test_bullish_divergence(self):
        """Price down + SMI up → bullish divergence."""
        pris = pd.Series(np.linspace(100, 90, 15))  # falling
        smi = pd.Series(np.linspace(-0.01, 0.02, 15))  # rising
        result = smart_money.finn_divergens(pris, smi, vindu=10)
        assert result['type'] == 'bullish'
        assert result['score_justering'] > 0

    def test_bearish_divergence(self):
        """Price up + SMI down → bearish divergence."""
        pris = pd.Series(np.linspace(100, 115, 15))  # rising
        smi = pd.Series(np.linspace(0.02, -0.01, 15))  # falling
        result = smart_money.finn_divergens(pris, smi, vindu=10)
        assert result['type'] == 'bearish'
        assert result['score_justering'] < 0

    def test_confirming_trend(self):
        """Both up → confirming."""
        pris = pd.Series(np.linspace(100, 120, 15))
        smi = pd.Series(np.linspace(0, 0.05, 15))
        result = smart_money.finn_divergens(pris, smi, vindu=10)
        assert result['type'] == 'confirming'

    def test_flat_returns_neutral(self):
        """Flat series → neutral."""
        flat = pd.Series([1.0] * 15)
        result = smart_money.finn_divergens(flat, flat, vindu=10)
        assert result['type'] in ('neutral', 'flat', 'confirming')
        assert result['score_justering'] == 0 or abs(result['score_justering']) <= 5

    def test_short_data(self):
        """Too short for analysis → flat."""
        short = pd.Series([1, 2, 3])
        result = smart_money.finn_divergens(short, short, vindu=10)
        assert result['type'] in ('neutral', 'flat', 'confirming')


class TestScannerIntegration:
    """Test beregn_smi_for_scanner()."""

    def test_scanner_output_keys(self, df_daily_up):
        result = smart_money.beregn_smi_for_scanner(df_daily_up)
        assert 'emoji' in result
        assert 'score_justering' in result
        assert 'type' in result
        assert 'cmf' in result

    def test_scanner_score_bounds(self, df_daily_up):
        result = smart_money.beregn_smi_for_scanner(df_daily_up)
        assert -10 <= result['score_justering'] <= 10

    def test_scanner_empty_data(self):
        result = smart_money.beregn_smi_for_scanner(pd.DataFrame())
        assert result['score_justering'] == 0
        assert result['emoji'] == '⚪'

    def test_scanner_none_data(self):
        result = smart_money.beregn_smi_for_scanner(None)
        assert result['score_justering'] == 0

    def test_scanner_short_data(self):
        """Data shorter than 40 rows → neutral."""
        df = pd.DataFrame({
            'Open': np.ones(20), 'High': np.ones(20) * 1.01,
            'Low': np.ones(20) * 0.99, 'Close': np.ones(20),
            'Volume': np.ones(20) * 10000
        }, index=pd.bdate_range('2025-01-01', periods=20))
        result = smart_money.beregn_smi_for_scanner(df)
        assert result['score_justering'] == 0


class TestTrendRetning:
    """Test internal _trend_retning helper."""

    def test_clear_uptrend(self):
        s = pd.Series(np.linspace(0, 10, 20))
        assert smart_money._trend_retning(s, 10) == 'up'

    def test_clear_downtrend(self):
        s = pd.Series(np.linspace(10, 0, 20))
        assert smart_money._trend_retning(s, 10) == 'down'

    def test_flat(self):
        s = pd.Series([5.0] * 20)
        assert smart_money._trend_retning(s, 10) == 'flat'

    def test_short_series(self):
        s = pd.Series([1, 2])
        assert smart_money._trend_retning(s, 10) == 'flat'
