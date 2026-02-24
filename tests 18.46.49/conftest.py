"""
Fixtures for InveStock DEV test suite.

Provides synthetic DataFrames with known prices for deterministic testing.
All fixtures use business-day frequency to match real market data.
"""
import pytest
import pandas as pd
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────

def _make_ohlcv(close: np.ndarray, dates: pd.DatetimeIndex,
                noise_pct: float = 0.005) -> pd.DataFrame:
    """Build OHLCV DataFrame from a Close series with small H/L noise."""
    close = np.asarray(close, dtype=float)
    high = close * (1 + noise_pct)
    low = close * (1 - noise_pct)
    opn = (close + low) / 2  # open between low and close
    volume = np.full(len(close), 1_000_000.0)
    return pd.DataFrame({
        'Open': opn, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


# ── Core Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def df_uptrend():
    """
    300 business days of steady uptrend: ~100 → ~170.
    Daily returns average +0.2% but with noise so RSI has both gain and loss days.
    Enough data for SMA_200 to be valid.
    """
    n = 300
    dates = pd.bdate_range('2022-01-03', periods=n, freq='B')
    rng = np.random.RandomState(42)
    daily_returns = 1.002 + rng.normal(0, 0.005, n)  # avg +0.2% with noise
    daily_returns = np.clip(daily_returns, 0.97, 1.03)
    close = 100 * np.cumprod(daily_returns)
    return _make_ohlcv(close, dates)


@pytest.fixture
def df_downtrend():
    """
    300 business days of steady downtrend: 200 → ~110.
    Price drops 0.2% per day.  RSI should be low (<40).
    """
    n = 300
    dates = pd.bdate_range('2022-01-03', periods=n, freq='B')
    close = 200 * np.cumprod(np.ones(n) * 0.998)
    return _make_ohlcv(close, dates)


@pytest.fixture
def df_golden_cross():
    """
    DataFrame where SMA_50 crosses above SMA_200 at a known date.
    
    Strategy:
    - First 250 days: declining price (SMA50 < SMA200)
    - Day 251+: sharp rally so SMA50 catches up and crosses SMA200.
    
    The cross should occur around day ~280-290.
    """
    n = 350
    dates = pd.bdate_range('2021-01-04', periods=n, freq='B')
    
    # Phase 1 (0-249): slow decline from 100 to ~90
    p1 = np.linspace(100, 90, 250)
    # Phase 2 (250-349): strong rally from 90 to ~130
    p2 = np.linspace(90, 130, 100)
    close = np.concatenate([p1, p2])
    
    return _make_ohlcv(close, dates)


@pytest.fixture
def df_death_cross():
    """
    DataFrame where SMA_50 crosses BELOW SMA_200.
    
    - First 250 days: rising price (SMA50 > SMA200)
    - Day 251+: sharp decline so SMA50 drops below SMA200.
    """
    n = 350
    dates = pd.bdate_range('2021-01-04', periods=n, freq='B')
    
    # Phase 1 (0-249): slow rise from 100 to 120
    p1 = np.linspace(100, 120, 250)
    # Phase 2 (250-349): decline from 120 to 80
    p2 = np.linspace(120, 80, 100)
    close = np.concatenate([p1, p2])
    
    return _make_ohlcv(close, dates)


@pytest.fixture
def df_rsi_known():
    """
    14 gains of +1 followed by 14 losses of -1.
    After 14 straight gains: RSI should be ~100.
    After mixed period: RSI should decay toward 50.
    
    We provide 60 days total for the EWM to stabilize.
    """
    n = 60
    dates = pd.bdate_range('2023-01-02', periods=n, freq='B')
    
    close = np.zeros(n)
    close[0] = 100.0
    # 14 consecutive gains of +1
    for i in range(1, 15):
        close[i] = close[i-1] + 1.0
    # 14 consecutive losses of -1
    for i in range(15, 29):
        close[i] = close[i-1] - 1.0
    # Then oscillate ±0.5
    for i in range(29, n):
        close[i] = close[i-1] + (0.5 if i % 2 == 0 else -0.5)
    
    return _make_ohlcv(close, dates, noise_pct=0.001)


@pytest.fixture
def df_vcp():
    """
    DataFrame that satisfies VCP (Volatility Contraction Pattern) conditions:
    - MA stacking: Close > SMA50 > SMA150 > SMA200 (all rising)
    - Near 52w high (>75%)
    - Above 52w low (>130%)
    - Progressive contraction in ranges
    - Volume contraction
    - Daily tightness <3.5%
    
    Build ~300 days: gentle uptrend with decreasing volatility at end.
    """
    n = 300
    dates = pd.bdate_range('2022-01-03', periods=n, freq='B')
    
    # Base: steady uptrend
    base = np.linspace(80, 160, n)
    
    # Add noise that decreases over time (contraction)
    noise = np.zeros(n)
    for i in range(n):
        if i < 100:
            noise[i] = np.sin(i * 0.5) * 8   # Wide swings early
        elif i < 200:
            noise[i] = np.sin(i * 0.5) * 4   # Medium
        else:
            noise[i] = np.sin(i * 0.5) * 1   # Tight at end
    
    close = base + noise
    
    # Build OHLCV with matching noise profile for High/Low
    high = np.zeros(n)
    low = np.zeros(n)
    for i in range(n):
        if i < 100:
            spread = 0.04
        elif i < 200:
            spread = 0.025
        else:
            spread = 0.01   # Very tight daily spread
        high[i] = close[i] * (1 + spread)
        low[i] = close[i] * (1 - spread)
    
    opn = (close + low) / 2
    
    # Volume: decreasing over the last 30 days
    volume = np.full(n, 1_000_000.0)
    volume[270:] = 500_000  # Volume contraction
    
    return pd.DataFrame({
        'Open': opn, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


@pytest.fixture
def df_exit_signals():
    """
    DataFrame designed to trigger multiple exit signals:
    - Large drawdown (>7% from 20-day high)
    - Price below SMA50 with high volume at end
    - MACD bearish crossover
    
    250 days of uptrend, then 50 days of sharp crash.
    """
    n = 300
    dates = pd.bdate_range('2022-01-03', periods=n, freq='B')
    
    # Phase 1: uptrend
    p1 = np.linspace(100, 150, 250)
    # Phase 2: crash from 150 to 100 (>7% drawdown)
    p2 = np.linspace(150, 100, 50)
    close = np.concatenate([p1, p2])
    
    # High volume during crash phase
    volume = np.full(n, 1_000_000.0)
    volume[250:] = 3_000_000.0  # 3× normal volume
    
    high = close * 1.005
    low = close * 0.995
    opn = (close + low) / 2
    
    return pd.DataFrame({
        'Open': opn, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


@pytest.fixture
def df_backtest_known():
    """
    DataFrame with a predictable price path for backtest verification.
    
    300 days. Price goes from 100 → 100 (flat) with known bumps:
    - Day 50: price = 100
    - Day 70 (50+20): price = 110  → 10% gross gain
    - Day 150: price = 100
    - Day 170 (150+20): price = 95  → -5% gross loss
    
    We'll inject signals at days 50 and 150 to test known outcomes.
    """
    n = 300
    dates = pd.bdate_range('2022-01-03', periods=n, freq='B')
    
    close = np.full(n, 100.0)
    # Create controlled price movements
    # Bump 1: days 50-70, rise to 110
    for i in range(50, 71):
        close[i] = 100 + (10 * (i - 50) / 20)
    # Revert back
    for i in range(71, 100):
        close[i] = 110 - (10 * (i - 70) / 30)
    
    # Bump 2: days 150-170, drop to 95
    for i in range(150, 171):
        close[i] = 100 - (5 * (i - 150) / 20)
    # Stay around 95
    for i in range(171, 200):
        close[i] = 95 + (5 * (i - 170) / 30)
    
    return _make_ohlcv(close, dates, noise_pct=0.001)


@pytest.fixture
def df_minimal():
    """Very short DataFrame (20 rows) to test edge cases / early returns."""
    n = 20
    dates = pd.bdate_range('2024-01-02', periods=n, freq='B')
    close = np.linspace(100, 105, n)
    return _make_ohlcv(close, dates)
