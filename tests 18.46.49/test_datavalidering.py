# tests/test_datavalidering.py
"""
Tester for 7.1 — Datavalidering & robusthet.
Dekker: input-validering, backup-mekanisme, retry-logikk, integritetsjekk.
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
# Portefølje input-validering
# ============================================================================

class TestAddPositionValidering:
    """Tester input-validering i add_position()."""
    
    def setup_method(self):
        """Opprett temp-mappe for portefølje-fil."""
        self.tmpdir = tempfile.mkdtemp()
        self.pf_path = os.path.join(self.tmpdir, 'portfolio.json')
        self.hist_path = os.path.join(self.tmpdir, 'portfolio_history.json')
    
    def teardown_method(self):
        """Rydd opp."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def _add_with_mock(self, **kwargs):
        """Kaller add_position med mocket filsti."""
        import portfolio
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path), \
             patch.object(portfolio, '_get_history_path', return_value=self.hist_path):
            return portfolio.add_position(**kwargs)
    
    def test_gyldig_posisjon(self):
        ok, msg = self._add_with_mock(ticker="EQNR.OL", quantity=100, buy_price=300.0)
        assert ok is True
        assert "EQNR.OL" in msg
    
    def test_ugyldig_tom_ticker(self):
        ok, msg = self._add_with_mock(ticker="", quantity=100, buy_price=300.0)
        assert ok is False
        assert "Ugyldig ticker" in msg
    
    def test_ugyldig_negativ_kvantitet(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=-5, buy_price=200.0)
        assert ok is False
        assert "Ugyldig antall" in msg
    
    def test_ugyldig_null_kvantitet(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=0, buy_price=200.0)
        assert ok is False
        assert "Ugyldig antall" in msg
    
    def test_ugyldig_negativ_pris(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=-50.0)
        assert ok is False
        assert "Ugyldig kjøpskurs" in msg
    
    def test_ugyldig_null_pris(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=0)
        assert ok is False
        assert "Ugyldig kjøpskurs" in msg
    
    def test_urimelig_hoey_pris(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=200_000)
        assert ok is False
        assert "Urimelig" in msg
    
    def test_ugyldig_datoformat(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=200.0, buy_date="15/02/2026")
        assert ok is False
        assert "datoformat" in msg.lower()
    
    def test_gyldig_dato(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=200.0, buy_date="2026-02-15")
        assert ok is True
    
    def test_stop_loss_over_kjoepskurs(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=200.0, stop_loss=250.0)
        assert ok is False
        assert "Stop loss" in msg
    
    def test_stop_loss_lik_kjoepskurs(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=200.0, stop_loss=200.0)
        assert ok is False
        assert "Stop loss" in msg
    
    def test_stop_loss_negativ(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=100, buy_price=200.0, stop_loss=-10.0)
        assert ok is False
        assert "Ugyldig stop loss" in msg
    
    def test_float_kvantitet_rundes_til_int(self):
        ok, msg = self._add_with_mock(ticker="DNB.OL", quantity=50.7, buy_price=200.0)
        assert ok is True
        # Sjekk at kvantiteten ble rundet
        import portfolio
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path):
            pf = portfolio.load_portfolio()
            assert pf['positions']['DNB.OL']['quantity'] == 50
    
    def test_ticker_konverteres_til_uppercase(self):
        ok, msg = self._add_with_mock(ticker="dnb.ol", quantity=100, buy_price=200.0)
        assert ok is True
        import portfolio
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path):
            pf = portfolio.load_portfolio()
            assert "DNB.OL" in pf['positions']


class TestSellPositionValidering:
    """Tester input-validering i sell_position()."""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pf_path = os.path.join(self.tmpdir, 'portfolio.json')
        self.hist_path = os.path.join(self.tmpdir, 'portfolio_history.json')
        # Opprett en posisjon å selge
        pf = {
            'positions': {
                'EQNR.OL': {
                    'quantity': 100, 'avg_price': 300.0, 'buy_date': '2026-01-01',
                    'last_added': '2026-01-01', 'stop_loss': 276.0, 'trailing_high': 300.0,
                    'notes': '', 'strategy': 'Test'
                }
            },
            'created': '2026-01-01', 'last_updated': '2026-01-01', 'version': '1.0'
        }
        os.makedirs(self.tmpdir, exist_ok=True)
        with open(self.pf_path, 'w') as f:
            json.dump(pf, f)
    
    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def _sell_with_mock(self, **kwargs):
        import portfolio
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path), \
             patch.object(portfolio, '_get_history_path', return_value=self.hist_path):
            return portfolio.sell_position(**kwargs)
    
    def test_gyldig_salg(self):
        ok, msg, result = self._sell_with_mock(ticker="EQNR.OL", quantity=50, sell_price=320.0)
        assert ok is True
        assert result['profit_pct'] > 0
    
    def test_ugyldig_negativ_salgskurs(self):
        ok, msg, result = self._sell_with_mock(ticker="EQNR.OL", sell_price=-100.0)
        assert ok is False
        assert "Ugyldig salgskurs" in msg
    
    def test_selg_mer_enn_eier(self):
        ok, msg, result = self._sell_with_mock(ticker="EQNR.OL", quantity=999, sell_price=320.0)
        assert ok is False
        assert "Kan ikke selge" in msg
    
    def test_ticker_ikke_i_portefolje(self):
        ok, msg, result = self._sell_with_mock(ticker="FAKE.OL", sell_price=100.0)
        assert ok is False
        assert "finnes ikke" in msg
    
    def test_mangler_salgskurs(self):
        ok, msg, result = self._sell_with_mock(ticker="EQNR.OL")
        assert ok is False
        assert "Salgskurs" in msg


# ============================================================================
# Backup-mekanisme
# ============================================================================

class TestBackupMekanisme:
    """Tester backup og restore av portefølje."""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pf_path = os.path.join(self.tmpdir, 'portfolio.json')
        self.hist_path = os.path.join(self.tmpdir, 'portfolio_history.json')
    
    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_backup_opprettes_ved_lagring(self):
        """Backup skal opprettes automatisk ved save_portfolio."""
        import portfolio
        
        # Lag original fil
        pf = {'positions': {'EQNR.OL': {'quantity': 100}}, 'version': '1.0'}
        with open(self.pf_path, 'w') as f:
            json.dump(pf, f)
        
        # Lagre ny versjon
        pf2 = {'positions': {'EQNR.OL': {'quantity': 200}}, 'version': '1.0'}
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path):
            portfolio.save_portfolio(pf2)
        
        # Sjekk at backup eksisterer
        assert os.path.exists(f"{self.pf_path}.bak")
        
        # Sjekk at backup har gammel verdi
        with open(f"{self.pf_path}.bak") as f:
            bak = json.load(f)
        assert bak['positions']['EQNR.OL']['quantity'] == 100
    
    def test_restore_fra_backup(self):
        """Restore skal fungere når hovedfilen er korrupt."""
        import portfolio
        
        # Lag gyldig backup
        pf = {'positions': {'DNB.OL': {'quantity': 50}}, 'version': '1.0',
              'created': '2026-01-01', 'last_updated': '2026-01-01'}
        with open(f"{self.pf_path}.bak", 'w') as f:
            json.dump(pf, f)
        
        # Lag korrupt hovedfil
        with open(self.pf_path, 'w') as f:
            f.write("{korrupt json!!")
        
        # Load skal automatisk gjenopprette fra backup
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path):
            result = portfolio.load_portfolio()
        
        assert 'DNB.OL' in result['positions']
        assert result['positions']['DNB.OL']['quantity'] == 50
    
    def test_restore_ingen_backup(self):
        """Uten backup skal det returneres tom portefølje."""
        import portfolio
        
        # Lag korrupt hovedfil uten backup
        with open(self.pf_path, 'w') as f:
            f.write("{broken}")
        
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path):
            result = portfolio.load_portfolio()
        
        assert result['positions'] == {}
    
    def test_tom_fil_haandteres(self):
        """Tom fil skal ikke krasje."""
        import portfolio
        
        with open(self.pf_path, 'w') as f:
            f.write("")
        
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path):
            result = portfolio.load_portfolio()
        
        assert result['positions'] == {}
    
    def test_atomisk_skriving(self):
        """Hvis skriving feiler midtveis, skal original fil beholdes."""
        import portfolio
        
        # Lag original
        pf = {'positions': {'MOWI.OL': {'quantity': 75}}, 'version': '1.0'}
        with open(self.pf_path, 'w') as f:
            json.dump(pf, f)
        
        # Simuler feil ved å gjøre tmp-filen ulasbar
        with patch.object(portfolio, '_get_portfolio_path', return_value=self.pf_path), \
             patch('shutil.move', side_effect=OSError("disk full")):
            result = portfolio.save_portfolio({'positions': {'NY': {}}, 'version': '1.0'})
        
        assert result is False
        # Original fil skal fortsatt eksistere og være gyldig
        with open(self.pf_path) as f:
            data = json.load(f)
        assert 'MOWI.OL' in data['positions']


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
