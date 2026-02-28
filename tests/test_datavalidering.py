# tests/test_datavalidering.py
"""
Tester for datavalidering & robusthet.
Dekker: retry-logikk, integritetsjekk.
"""

import json
import os
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock


# ============================================================================
# Data-integritet (data.py)
# ============================================================================

class TestDataIntegritet:
    """Tester parquet-validering og robusthet i hent_data()."""
    
    def test_negative_priser_fjernes(self):
        """Rader med negative priser skal filtreres bort."""
        import data
        
        df = pd.DataFrame({
            'Open': [100, -5, 102],
            'High': [105, -1, 108],
            'Low': [99, -10, 101],
            'Close': [103, -3, 106],
            'Volume': [1000, 500, 1200],
            'Ticker': ['A', 'A', 'A']
        }, index=pd.date_range('2026-01-01', periods=3))
        
        # Mock parquet-lesing
        with patch('pandas.read_parquet', return_value=df), \
             patch('os.path.exists', return_value=True):
            result = data.hent_data()
        
        assert len(result) == 2
        assert (result['Close'] > 0).all()
    
    def test_tom_parquet_gir_tom_df(self):
        """Tom parquet-fil skal gi tom DataFrame, ikke krasj."""
        import data
        
        with patch('pandas.read_parquet', return_value=pd.DataFrame()), \
             patch('os.path.exists', return_value=True):
            result = data.hent_data()
        
        assert result.empty


class TestYfinanceRetry:
    """Tester retry-mekanisme for yfinance."""
    
    def test_lykkes_paa_foerste_forsoek(self):
        import data
        
        mock_df = pd.DataFrame({'Close': [100, 101]})
        with patch('yfinance.download', return_value=mock_df):
            result = data._yf_download_med_retry(tickers='TEST', start='2020-01-01', max_retries=3)
        
        assert not result.empty
    
    def test_lykkes_paa_andre_forsoek(self):
        import data
        
        mock_df = pd.DataFrame({'Close': [100, 101]})
        # Først feil, deretter suksess
        with patch('yfinance.download', side_effect=[Exception("timeout"), mock_df]), \
             patch('time.sleep'):  # Ikke vent i tester
            result = data._yf_download_med_retry(tickers='TEST', start='2020-01-01', max_retries=3)
        
        assert not result.empty
    
    def test_gir_opp_etter_max_retries(self):
        import data
        
        with patch('yfinance.download', side_effect=Exception("vedvarende feil")), \
             patch('time.sleep'):
            result = data._yf_download_med_retry(tickers='TEST', start='2020-01-01', max_retries=2)
        
        assert result.empty
    
    def test_tom_df_trigger_retry(self):
        import data
        
        empty = pd.DataFrame()
        mock_df = pd.DataFrame({'Close': [100]})
        
        with patch('yfinance.download', side_effect=[empty, mock_df]), \
             patch('time.sleep'):
            result = data._yf_download_med_retry(tickers='TEST', start='2020-01-01', max_retries=2)
        
        assert not result.empty
