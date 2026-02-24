# tests/test_markedsbredde.py
"""
Tester for 7.4 — Forbedret markedsbredde-analyse.
Dekker: McClellan Oscillator, A/D-linje, bredde-indikatorer.
"""

import pandas as pd
import numpy as np
import pytest
import logic


class TestMcClellanOscillator:
    """Tester McClellan Oscillator-beregning."""
    
    def test_basic_beregning(self):
        """McClellan skal returnere verdier for ≥39 dager."""
        np.random.seed(42)
        adv = pd.Series(np.random.randint(10, 40, size=60))
        dec = pd.Series(np.random.randint(10, 40, size=60))
        
        result = logic.beregn_mcclellan_oscillator(adv, dec)
        assert len(result) == 60
        assert not result.isna().all()
    
    def test_for_kort_data(self):
        """Mindre enn 39 dager skal gi tom Series."""
        adv = pd.Series([20, 25, 30])
        dec = pd.Series([15, 20, 10])
        
        result = logic.beregn_mcclellan_oscillator(adv, dec)
        assert result.empty
    
    def test_bullish_gir_positiv(self):
        """Økende bullish trend → positiv oscillator."""
        # Start nøytralt, bli stadig mer bullish → EMA19 reagerer raskere enn EMA39
        adv = pd.Series([25]*20 + [40]*30)
        dec = pd.Series([25]*20 + [10]*30)
        
        result = logic.beregn_mcclellan_oscillator(adv, dec)
        assert result.iloc[-1] > 0
    
    def test_bearish_gir_negativ(self):
        """Økende bearish trend → negativ oscillator."""
        # Start nøytralt, bli stadig mer bearish → EMA19 reagerer raskere enn EMA39
        adv = pd.Series([25]*20 + [10]*30)
        dec = pd.Series([25]*20 + [40]*30)
        
        result = logic.beregn_mcclellan_oscillator(adv, dec)
        assert result.iloc[-1] < 0
    
    def test_balansert_nær_null(self):
        """Balansert marked → oscillator nær null."""
        adv = pd.Series([25] * 50)
        dec = pd.Series([25] * 50)
        
        result = logic.beregn_mcclellan_oscillator(adv, dec)
        assert abs(result.iloc[-1]) < 0.1


class TestADLinje:
    """Tester Advance/Decline-linje."""
    
    def test_kumulativ(self):
        """A/D-linjen skal være kumulativ sum av netto advances."""
        adv = pd.Series([30, 20, 35, 25])
        dec = pd.Series([20, 30, 15, 25])
        
        result = logic.beregn_ad_linje(adv, dec)
        
        # Netto: 10, -10, 20, 0
        # Kumulativ: 10, 0, 20, 20
        assert result.iloc[0] == 10
        assert result.iloc[1] == 0
        assert result.iloc[2] == 20
        assert result.iloc[3] == 20
    
    def test_stigende_ad_linje(self):
        """Konsekvent bullish → stigende A/D-linje."""
        adv = pd.Series([35] * 10)
        dec = pd.Series([15] * 10)
        
        result = logic.beregn_ad_linje(adv, dec)
        assert result.iloc[-1] > result.iloc[0]
        assert result.is_monotonic_increasing
    
    def test_tom_data(self):
        """Tom data → tom resultat."""
        result = logic.beregn_ad_linje(pd.Series(dtype=int), pd.Series(dtype=int))
        assert result.empty


class TestBreddeIndikatorer:
    """Tester samlet bredde-indikator-beregning."""
    
    def _lag_test_df(self, n_days=300, close_start=100, trend_up=True):
        """Hjelpefunksjon for å lage test-DataFrames med SMA."""
        dates = pd.date_range('2025-01-01', periods=n_days, freq='B')
        if trend_up:
            close = np.linspace(close_start, close_start * 1.5, n_days) + np.random.randn(n_days) * 2
        else:
            close = np.linspace(close_start, close_start * 0.6, n_days) + np.random.randn(n_days) * 2
        
        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(100000, 1000000, n_days)
        }, index=dates)
        
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        return df
    
    def test_alle_bullish(self):
        """Alle tickers i opptrend → høy SMA%-andel."""
        np.random.seed(42)
        tickers = {f'T{i}.OL': self._lag_test_df(trend_up=True) for i in range(5)}
        
        result = logic.beregn_bredde_indikatorer(tickers)
        
        assert result['total_analyzed'] == 5
        assert result['pct_over_sma200'] > 50
        assert result['pct_over_sma50'] > 50
        assert result['ad_ratio'] >= 0
    
    def test_alle_bearish(self):
        """Alle tickers i nedtrend → lav SMA%-andel."""
        np.random.seed(42)
        tickers = {f'T{i}.OL': self._lag_test_df(trend_up=False) for i in range(5)}
        
        result = logic.beregn_bredde_indikatorer(tickers)
        
        assert result['pct_over_sma200'] < 50
    
    def test_for_kort_data_ignoreres(self):
        """Tickers med <200 dager skal hoppes over."""
        short_df = pd.DataFrame({
            'Close': [100] * 50,
            'High': [102] * 50,
            'Low': [98] * 50,
            'SMA_200': [99] * 50,
            'SMA_50': [99] * 50,
        }, index=pd.date_range('2025-01-01', periods=50))
        
        result = logic.beregn_bredde_indikatorer({'SHORT.OL': short_df})
        assert result['total_analyzed'] == 0
    
    def test_52ukers_hoey(self):
        """Ticker på 52-ukers høy skal telles."""
        np.random.seed(42)
        df = self._lag_test_df(n_days=300, trend_up=True)
        # Sett siste pris til all-time high
        df.iloc[-1, df.columns.get_loc('Close')] = df['High'].max() * 1.01
        df.iloc[-1, df.columns.get_loc('High')] = df['High'].max() * 1.01
        
        result = logic.beregn_bredde_indikatorer({'BULL.OL': df})
        assert result['new_52w_high'] >= 1
    
    def test_tom_dict_gir_nullverdier(self):
        """Tom input → alle verdier null."""
        result = logic.beregn_bredde_indikatorer({})
        assert result['total_analyzed'] == 0
        assert result['pct_over_sma200'] == 0
        assert result['advances'] == 0
    
    def test_ad_ratio_unngår_div_null(self):
        """Ingen declines → ratio beregnes uten feil."""
        np.random.seed(42)
        # Lag en ticker som garantert stiger siste dag
        df = self._lag_test_df(trend_up=True)
        df.iloc[-1, df.columns.get_loc('Close')] = df.iloc[-2, df.columns.get_loc('Close')] + 10
        
        result = logic.beregn_bredde_indikatorer({'UP.OL': df})
        # Ingen div-by-zero-feil
        assert result['ad_ratio'] >= 0
    
    def test_output_keys(self):
        """Sjekk at alle forventede nøkler er i output."""
        result = logic.beregn_bredde_indikatorer({})
        expected_keys = [
            'total_analyzed', 'pct_over_sma200', 'pct_over_sma50',
            'advances', 'declines', 'new_52w_high', 'new_52w_low', 'ad_ratio'
        ]
        for key in expected_keys:
            assert key in result, f"Mangler nøkkel: {key}"
