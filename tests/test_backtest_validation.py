"""
Backtest validation against known historical data.

Tests real market data from EQNR.OL, DNB.OL, MOWI.OL to verify that:
1. Known signals are detected at correct dates (±1 day tolerance)
2. RSI values match expected levels at known market events
3. Backtest return calculations match manual calculations (with tx costs)
4. Signal counts are stable and reproducible
5. Backtest statistics are internally consistent

Data: Downloaded live via yfinance (requires internet).
If yfinance fails, tests are skipped gracefully.
"""
import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logic
import config

# ── Helpers ──────────────────────────────────────────────────────────

def _download(ticker: str, start='2018-01-01', end='2025-12-31') -> pd.DataFrame:
    """Download data via yfinance, return empty df on failure."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def _signal_at_date(signaler: pd.DataFrame, col: str, target_date: str,
                    tolerance_days: int = 1) -> bool:
    """Check if a signal fires at target_date ± tolerance."""
    sig_dates = signaler[signaler[col]].index
    target = pd.Timestamp(target_date)
    for d in sig_dates:
        if abs((d - target).days) <= tolerance_days:
            return True
    return False


def _manual_gross_return(df: pd.DataFrame, entry_date: str, hold: int = 20) -> float:
    """Compute gross return (no costs) for a manual trade."""
    idx = df.index.get_loc(pd.Timestamp(entry_date))
    if idx + hold >= len(df):
        return None
    entry = df.iloc[idx]['Close']
    exit_ = df.iloc[idx + hold]['Close']
    return (exit_ - entry) / entry * 100


def _manual_net_return(df: pd.DataFrame, entry_date: str, hold: int = 20,
                       kurtasje: float = 0.05, spread: float = 0.10) -> float:
    """Compute net return (with costs) matching logic.backtest_strategi formula."""
    idx = df.index.get_loc(pd.Timestamp(entry_date))
    if idx + hold >= len(df):
        return None
    raw_entry = df.iloc[idx]['Close']
    raw_exit = df.iloc[idx + hold]['Close']
    friksjons_faktor = kurtasje / 100 + spread / 200
    entry_pris = raw_entry * (1 + friksjons_faktor)
    exit_pris = raw_exit * (1 - friksjons_faktor)
    return (exit_pris - entry_pris) / entry_pris * 100


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def eqnr_data():
    df_raw = _download('EQNR.OL')
    if df_raw.empty:
        pytest.skip('Could not download EQNR.OL data')
    df = logic.beregn_tekniske_indikatorer(df_raw)
    signaler = logic.sjekk_strategier(df)
    return df, signaler


@pytest.fixture(scope='module')
def dnb_data():
    df_raw = _download('DNB.OL')
    if df_raw.empty:
        pytest.skip('Could not download DNB.OL data')
    df = logic.beregn_tekniske_indikatorer(df_raw)
    signaler = logic.sjekk_strategier(df)
    return df, signaler


@pytest.fixture(scope='module')
def mowi_data():
    df_raw = _download('MOWI.OL')
    if df_raw.empty:
        pytest.skip('Could not download MOWI.OL data')
    df = logic.beregn_tekniske_indikatorer(df_raw)
    signaler = logic.sjekk_strategier(df)
    return df, signaler


# ═══════════════════════════════════════════════════════════════════════
# 1.  KNOWN SIGNAL DETECTION  (±1 day tolerance)
# ═══════════════════════════════════════════════════════════════════════

class TestKnownSignals:
    """Verify that the engine detects signals at historically known dates."""

    # -- DNB Golden Cross signals --
    @pytest.mark.parametrize('date', ['2019-03-29', '2019-10-22', '2022-12-27'])
    def test_dnb_golden_cross(self, dnb_data, date):
        _, signaler = dnb_data
        assert _signal_at_date(signaler, 'Golden_Cross', date), \
            f'DNB Golden Cross expected at {date}'

    # -- MOWI Golden Cross signals --
    @pytest.mark.parametrize('date', ['2022-03-25', '2024-10-16', '2025-08-29'])
    def test_mowi_golden_cross(self, mowi_data, date):
        _, signaler = mowi_data
        assert _signal_at_date(signaler, 'Golden_Cross', date), \
            f'MOWI Golden Cross expected at {date}'

    # -- EQNR Golden Cross signals --
    @pytest.mark.parametrize('date', ['2025-02-04', '2025-07-15'])
    def test_eqnr_golden_cross(self, eqnr_data, date):
        _, signaler = eqnr_data
        assert _signal_at_date(signaler, 'Golden_Cross', date), \
            f'EQNR Golden Cross expected at {date}'

    # -- DNB Wyckoff Spring --
    @pytest.mark.parametrize('date', ['2018-04-12', '2018-12-18'])
    def test_dnb_wyckoff_spring(self, dnb_data, date):
        _, signaler = dnb_data
        assert _signal_at_date(signaler, 'Wyckoff_Spring', date), \
            f'DNB Wyckoff Spring expected at {date}'

    # -- EQNR Momentum Burst --
    @pytest.mark.parametrize('date', ['2019-09-05', '2023-08-09'])
    def test_eqnr_momentum_burst(self, eqnr_data, date):
        _, signaler = eqnr_data
        assert _signal_at_date(signaler, 'Momentum_Burst', date), \
            f'EQNR Momentum Burst expected at {date}'

    # -- MOWI Momentum Burst --
    @pytest.mark.parametrize('date', ['2019-02-13', '2020-11-09', '2021-08-25'])
    def test_mowi_momentum_burst(self, mowi_data, date):
        _, signaler = mowi_data
        assert _signal_at_date(signaler, 'Momentum_Burst', date), \
            f'MOWI Momentum Burst expected at {date}'


# ═══════════════════════════════════════════════════════════════════════
# 2.  RSI SPOT-CHECKS AT KNOWN MARKET EVENTS
# ═══════════════════════════════════════════════════════════════════════

class TestRSISpotChecks:
    """
    Verify RSI at key market dates.

    COVID crash 2020-03-23: RSI should be very low (<35).
    Mid-2022 correction: RSI should be moderate.
    """

    def test_eqnr_covid_crash_rsi(self, eqnr_data):
        """EQNR RSI on 2020-03-23 (COVID crash bottom) ~ 30.7."""
        df, _ = eqnr_data
        rsi = df.loc['2020-03-23', 'RSI']
        assert 20 < rsi < 40, f'EQNR COVID RSI expected ~30, got {rsi:.1f}'

    def test_dnb_covid_crash_rsi(self, dnb_data):
        """DNB RSI on 2020-03-23 ~ 25.4."""
        df, _ = dnb_data
        rsi = df.loc['2020-03-23', 'RSI']
        assert 15 < rsi < 35, f'DNB COVID RSI expected ~25, got {rsi:.1f}'

    def test_mowi_covid_crash_rsi(self, mowi_data):
        """MOWI RSI on 2020-03-23 ~ 37.5."""
        df, _ = mowi_data
        rsi = df.loc['2020-03-23', 'RSI']
        assert 25 < rsi < 50, f'MOWI COVID RSI expected ~37, got {rsi:.1f}'

    def test_dnb_2024_rsi_moderate(self, dnb_data):
        """DNB RSI on 2024-01-02 ~ 58.6 (moderate, no extremes)."""
        df, _ = dnb_data
        rsi = df.loc['2024-01-02', 'RSI']
        assert 45 < rsi < 75, f'DNB 2024 RSI expected moderate (~59), got {rsi:.1f}'


# ═══════════════════════════════════════════════════════════════════════
# 3.  BACKTEST RETURN CALCULATION VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class TestBacktestReturnMath:
    """
    Manually compute expected trade returns and compare with backtest output.

    We pick specific signal dates where we know the entry/exit prices,
    compute the expected net return with transaction costs, and verify
    the backtest produces matching results.
    """

    def test_dnb_golden_cross_backtest_returns(self, dnb_data):
        """
        DNB Golden Cross at 2019-03-29:
          entry close = ~104.36, exit(+20d) close = ~109.69
          gross return = ~5.11%
          With costs (0.05% kurtasje + 0.10% spread): net should be ~4.91%
        """
        df, _ = dnb_data
        manual_net = _manual_net_return(df, '2019-03-29', hold=20)
        assert manual_net is not None

        # Build signals manually at the known date
        signaler = pd.DataFrame(index=df.index)
        signaler['Golden_Cross'] = False
        # Set True at the exact date + 2 more to reach the 3-signal minimum
        for d in ['2019-03-29', '2019-10-22', '2022-12-27']:
            if pd.Timestamp(d) in df.index:
                signaler.loc[pd.Timestamp(d), 'Golden_Cross'] = True

        result = logic.backtest_strategi(df, signaler, 'Golden_Cross',
                                         holdingperiode=20,
                                         kurtasje_pct=0.05, spread_pct=0.10,
                                         trailing_stop_atr=999.0,
                                         profit_target_atr=999.0)
        assert result is not None, 'Backtest should produce results for 3 signals'

        # Find the trade at 2019-03-29
        trade_found = False
        for trade in result['resultater']:
            if str(trade['dato'].date()) == '2019-03-29':
                trade_found = True
                diff = abs(trade['avkastning'] - manual_net)
                assert diff < 0.01, \
                    f"Backtest return {trade['avkastning']:.4f}% != manual {manual_net:.4f}%"
                break
        assert trade_found, 'Trade at 2019-03-29 not found in backtest results'

    def test_mowi_golden_cross_backtest_returns(self, mowi_data):
        """MOWI Golden Cross at 2022-03-25: gross ~6.13%, verify net with costs."""
        df, _ = mowi_data
        manual_net = _manual_net_return(df, '2022-03-25', hold=20)
        assert manual_net is not None

        signaler = pd.DataFrame(index=df.index)
        signaler['Golden_Cross'] = False
        for d in ['2022-03-25', '2024-10-16', '2025-08-29']:
            if pd.Timestamp(d) in df.index:
                signaler.loc[pd.Timestamp(d), 'Golden_Cross'] = True

        result = logic.backtest_strategi(df, signaler, 'Golden_Cross',
                                         holdingperiode=20,
                                         kurtasje_pct=0.05, spread_pct=0.10,
                                         trailing_stop_atr=999.0,
                                         profit_target_atr=999.0)
        assert result is not None

        for trade in result['resultater']:
            if str(trade['dato'].date()) == '2022-03-25':
                diff = abs(trade['avkastning'] - manual_net)
                assert diff < 0.01, \
                    f"Backtest {trade['avkastning']:.4f}% != manual {manual_net:.4f}%"
                break

    def test_transaction_cost_impact(self, dnb_data):
        """
        Verify that transaction costs reduce returns by ~0.20% per round-trip.
        
        Round-trip cost = 2 × 0.05% + 0.10% = 0.20%
        """
        df, _ = dnb_data

        # Use a real signal date
        gross = _manual_gross_return(df, '2019-03-29', hold=20)
        net = _manual_net_return(df, '2019-03-29', hold=20)
        assert gross is not None and net is not None

        cost_drag = gross - net
        # Cost drag should be approximately 0.20% (can vary slightly due
        # to compounding on entry/exit, but should be in 0.15-0.30% range)
        assert 0.10 < cost_drag < 0.40, \
            f'Transaction cost drag should be ~0.20%, got {cost_drag:.3f}%'


# ═══════════════════════════════════════════════════════════════════════
# 4.  SIGNAL STABILITY & REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════

class TestSignalStability:
    """Verify signals are deterministic — same data → same signals."""

    def test_eqnr_signals_reproducible(self, eqnr_data):
        """Run indicators + strategies twice on same data → identical."""
        df, signaler1 = eqnr_data
        signaler2 = logic.sjekk_strategier(df)
        pd.testing.assert_frame_equal(signaler1, signaler2)

    def test_dnb_signals_reproducible(self, dnb_data):
        df, signaler1 = dnb_data
        signaler2 = logic.sjekk_strategier(df)
        pd.testing.assert_frame_equal(signaler1, signaler2)

    def test_mowi_signals_reproducible(self, mowi_data):
        df, signaler1 = mowi_data
        signaler2 = logic.sjekk_strategier(df)
        pd.testing.assert_frame_equal(signaler1, signaler2)


# ═══════════════════════════════════════════════════════════════════════
# 5.  SIGNAL COUNT SANITY
# ═══════════════════════════════════════════════════════════════════════

class TestSignalCountSanity:
    """
    Verify signal counts are within expected ranges.
    
    Over ~2000 trading days (8 years), signal frequencies should be:
    - Golden Cross: rare, 1-10 signals
    - Ichimoku Breakout: moderate, 10-80
    - VCP: moderate to many, 20-150
    - Pocket Pivot: many, 20-150
    - Any strategy: should have at least some signals for active stocks
    """

    def test_eqnr_signal_counts(self, eqnr_data):
        _, signaler = eqnr_data
        total = signaler.sum().sum()
        assert total > 10, f'EQNR should have >10 total signals, got {total}'
        assert signaler['Golden_Cross'].sum() <= 15, \
            'Golden Cross should be rare (<15 over 8 years)'

    def test_dnb_signal_counts(self, dnb_data):
        _, signaler = dnb_data
        total = signaler.sum().sum()
        assert total > 20, f'DNB should have >20 total signals, got {total}'
        # DNB is an active stock — should fire on most strategies
        strategies_with_signals = (signaler.sum() > 0).sum()
        assert strategies_with_signals >= 5, \
            f'DNB should trigger at least 5 different strategies, got {strategies_with_signals}'

    def test_mowi_signal_counts(self, mowi_data):
        _, signaler = mowi_data
        total = signaler.sum().sum()
        assert total > 20, f'MOWI should have >20 total signals, got {total}'


# ═══════════════════════════════════════════════════════════════════════
# 6.  BACKTEST STATISTICS CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestBacktestConsistency:
    """Verify backtest output is internally consistent."""

    def _run_backtest(self, df, signaler, strategy):
        """Helper to run backtest with enough signals."""
        result = logic.backtest_strategi(df, signaler, strategy, holdingperiode=20)
        return result

    def test_dnb_ichimoku_backtest_consistency(self, dnb_data):
        """DNB Ichimoku has ~40 signals — backtest should be internally consistent."""
        df, signaler = dnb_data
        result = self._run_backtest(df, signaler, 'Ichimoku_Breakout')
        if result is None:
            pytest.skip('Not enough Ichimoku signals for DNB')

        # Winners + losers = total
        assert result['vinnere'] + result['tapere'] == result['antall_signaler']

        # Win rate should match
        expected_wr = result['vinnere'] / result['antall_signaler'] * 100
        assert abs(result['win_rate'] - expected_wr) < 0.2

        # Costs should be positive
        assert result['total_kostnad_pst'] > 0
        assert result['snitt_kostnad_per_trade'] > 0

        # Best trade >= average >= worst trade
        assert result['beste_trade'] >= result['snitt_avkastning']
        assert result['snitt_avkastning'] >= result['verste_trade']

    def test_eqnr_vcp_backtest_consistency(self, eqnr_data):
        """EQNR VCP has ~53 signals — verify stats."""
        df, signaler = eqnr_data
        result = self._run_backtest(df, signaler, 'VCP_Pattern')
        if result is None:
            pytest.skip('Not enough VCP signals for EQNR')

        assert result['vinnere'] + result['tapere'] == result['antall_signaler']
        assert result['total_kostnad_pst'] > 0
        assert result['beste_trade'] >= result['verste_trade']

    def test_mowi_pocket_pivot_backtest_consistency(self, mowi_data):
        """MOWI Pocket Pivot has ~62 signals — verify stats."""
        df, signaler = mowi_data
        result = self._run_backtest(df, signaler, 'Pocket_Pivot')
        if result is None:
            pytest.skip('Not enough Pocket Pivot signals for MOWI')

        assert result['vinnere'] + result['tapere'] == result['antall_signaler']

        # Profit factor check: if there are both winners and losers
        if result['tapere'] > 0 and result['vinnere'] > 0:
            assert result['profit_factor'] > 0, 'Profit factor should be positive'


# ═══════════════════════════════════════════════════════════════════════
# 7.  EXIT SIGNAL VALIDATION ON REAL DATA
# ═══════════════════════════════════════════════════════════════════════

class TestExitSignalsRealData:
    """Verify exit signals fire during known market crashes."""

    def test_eqnr_covid_exit(self, eqnr_data):
        """
        EQNR crashed from ~170 to ~75 in March 2020.
        Exit signals should fire during the crash period.
        """
        df, _ = eqnr_data
        exits = logic.beregn_exit_signaler_historisk(df, min_signaler=2)
        
        if not exits.empty:
            # Check for exits in Feb-Apr 2020
            covid_exits = exits[(exits.index >= '2020-02-01') & (exits.index <= '2020-04-30')]
            assert len(covid_exits) > 0, \
                'Exit signals should fire during EQNR COVID crash (Feb-Apr 2020)'

    def test_dnb_covid_exit(self, dnb_data):
        """DNB crashed from ~130 to ~70 in March 2020."""
        df, _ = dnb_data
        exits = logic.beregn_exit_signaler_historisk(df, min_signaler=2)
        
        if not exits.empty:
            covid_exits = exits[(exits.index >= '2020-02-01') & (exits.index <= '2020-04-30')]
            assert len(covid_exits) > 0, \
                'Exit signals should fire during DNB COVID crash (Feb-Apr 2020)'

    def test_mowi_drawdown_detection(self, mowi_data):
        """MOWI had drawdown in 2022-2023. Exit signals should appear."""
        df, _ = mowi_data
        exits = logic.beregn_exit_signaler_historisk(df, min_signaler=2)
        
        # MOWI had significant moves — should have some exit signals overall
        assert not exits.empty, 'MOWI should have some exit signals over 8 years'
        assert len(exits) >= 5, \
            f'MOWI should have >=5 exit signal dates, got {len(exits)}'
