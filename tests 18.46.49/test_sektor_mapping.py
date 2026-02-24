# tests/test_sektor_mapping.py
"""
Tester for 7.2 — Dynamisk sektor-mapping.
Dekker: yfinance-henting, caching, fallback, oversettelse, oppdatering.
"""

import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Sektor-oversettelse
# ============================================================================

class TestSektorOversettelse:
    """Tester at engelske sektornavn mappes korrekt til norske."""

    def test_energy_til_energi(self):
        from data import _SEKTOR_OVERSETTELSE
        assert _SEKTOR_OVERSETTELSE["Energy"] == "Energi"

    def test_financial_services_til_finans(self):
        from data import _SEKTOR_OVERSETTELSE
        assert _SEKTOR_OVERSETTELSE["Financial Services"] == "Finans"

    def test_technology_til_teknologi(self):
        from data import _SEKTOR_OVERSETTELSE
        assert _SEKTOR_OVERSETTELSE["Technology"] == "Teknologi"

    def test_communication_services_til_telekom(self):
        from data import _SEKTOR_OVERSETTELSE
        assert _SEKTOR_OVERSETTELSE["Communication Services"] == "Telekom"

    def test_alle_sektorer_dekket(self):
        """Alle yfinance-sektorer skal ha en norsk mapping."""
        from data import _SEKTOR_OVERSETTELSE
        yfinance_sektorer = [
            "Energy", "Basic Materials", "Industrials", "Consumer Cyclical",
            "Consumer Defensive", "Healthcare", "Financial Services",
            "Technology", "Communication Services", "Utilities", "Real Estate"
        ]
        for sektor in yfinance_sektorer:
            assert sektor in _SEKTOR_OVERSETTELSE, f"Mangler mapping for '{sektor}'"


# ============================================================================
# Cache lese/skrive
# ============================================================================

class TestSektorCache:
    """Tester lesing og skriving av sektor-cache."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.tmpdir, "sektor_mapping.json")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tom_cache_gir_tom_dict(self):
        import data
        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file):
            result = data._last_sektor_cache()
        assert result == {}

    def test_lagre_og_lese_cache(self):
        import data
        cache = {"EQNR.OL": "Energi", "DNB.OL": "Finans"}
        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file):
            data._lagre_sektor_cache(cache)
            result = data._last_sektor_cache()
        assert result == cache

    def test_korrupt_cache_gir_tom_dict(self):
        import data
        with open(self.cache_file, 'w') as f:
            f.write("{korrupt json!!")
        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file):
            result = data._last_sektor_cache()
        assert result == {}

    def test_cache_inneholder_norske_tegn(self):
        import data
        cache = {"MOWI.OL": "Sjømat", "BAKKA.OL": "Sjømat"}
        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file):
            data._lagre_sektor_cache(cache)
            result = data._last_sektor_cache()
        assert result["MOWI.OL"] == "Sjømat"


# ============================================================================
# Oppdater sektor-mapping (yfinance-henting)
# ============================================================================

class TestOppdaterSektorMapping:
    """Tester oppdatering av sektorer fra yfinance."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.tmpdir, "sektor_mapping.json")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_henter_sektor_for_nye_tickers(self):
        import data

        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': 'Energy', 'industry': 'Oil & Gas'}

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', return_value=mock_ticker):
            result = data.oppdater_sektor_mapping(["EQNR.OL"])

        assert result["EQNR.OL"] == "Energi"

    def test_bruker_cache_for_eksisterende(self):
        import data

        # Pre-fyll cache
        cache = {"EQNR.OL": "Energi"}
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker') as mock_yf:
            result = data.oppdater_sektor_mapping(["EQNR.OL"])

        # yfinance.Ticker should NOT have been called (cache hit)
        mock_yf.assert_not_called()
        assert result["EQNR.OL"] == "Energi"

    def test_fallback_til_industri(self):
        import data

        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': '', 'industry': 'Oil & Gas Exploration'}

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', return_value=mock_ticker):
            result = data.oppdater_sektor_mapping(["OKEA.OL"])

        assert result["OKEA.OL"] == "Energi"

    def test_shipping_industri_fallback(self):
        import data

        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': '', 'industry': 'Marine Shipping'}

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', return_value=mock_ticker):
            result = data.oppdater_sektor_mapping(["FRO.OL"])

        assert result["FRO.OL"] == "Shipping"

    def test_sjoemat_industri_fallback(self):
        import data

        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': '', 'industry': 'Salmon Farming'}

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', return_value=mock_ticker):
            result = data.oppdater_sektor_mapping(["MOWI.OL"])

        assert result["MOWI.OL"] == "Sjømat"

    def test_ingen_info_gir_annet(self):
        import data

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', return_value=mock_ticker):
            result = data.oppdater_sektor_mapping(["UKJENT.OL"])

        assert result["UKJENT.OL"] == "Annet"

    def test_yfinance_feil_hopper_over_ticker(self):
        import data

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', side_effect=Exception("API error")):
            result = data.oppdater_sektor_mapping(["FEIL.OL"])

        # Ticker skal IKKE være i cache (kan prøves igjen senere)
        assert "FEIL.OL" not in result

    def test_blanding_cache_og_nye(self):
        import data

        # Pre-fyll cache med en ticker
        cache = {"EQNR.OL": "Energi"}
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': 'Financial Services'}

        with patch.object(data, 'SEKTOR_MAPPING_FILE', self.cache_file), \
             patch('yfinance.Ticker', return_value=mock_ticker):
            result = data.oppdater_sektor_mapping(["EQNR.OL", "DNB.OL"])

        assert result["EQNR.OL"] == "Energi"
        assert result["DNB.OL"] == "Finans"


# ============================================================================
# Dynamisk hent_sektor med fallback
# ============================================================================

class TestHentSektorDynamisk:
    """Tester at hent_sektor() bruker dynamisk cache med hardkodet fallback."""

    def test_hardkodet_fungerer(self):
        import logic
        # Reset til hardkodet for å unngå at dynamisk disk-cache påvirker
        original = dict(logic.SEKTOR_MAPPING)
        logic.SEKTOR_MAPPING = dict(logic._SEKTOR_MAPPING_HARDKODET)
        try:
            assert logic.hent_sektor("EQNR.OL") == "Energi"
            assert logic.hent_sektor("DNB.OL") == "Finans"
            assert logic.hent_sektor("MOWI.OL") == "Sjømat"
        finally:
            logic.SEKTOR_MAPPING = original

    def test_ukjent_ticker_gir_annet(self):
        import logic
        assert logic.hent_sektor("FINNESIKKE.OL") == "Annet"

    def test_oppdater_sektor_cache(self):
        """oppdater_sektor_cache() skal oppdatere SEKTOR_MAPPING."""
        import logic

        mock_cache = {"NY_TICKER.OL": "Teknologi", "EQNR.OL": "Energi"}
        with patch('data.hent_sektor_mapping', return_value=mock_cache):
            logic.oppdater_sektor_cache()

        assert logic.SEKTOR_MAPPING.get("NY_TICKER.OL") == "Teknologi"
        # Hardkodede skal fortsatt være der
        assert logic.SEKTOR_MAPPING.get("EQNR.OL") == "Energi"

        # Rydd opp
        if "NY_TICKER.OL" in logic.SEKTOR_MAPPING:
            del logic.SEKTOR_MAPPING["NY_TICKER.OL"]

    def test_dynamisk_overskriver_hardkodet(self):
        """Dynamisk cache-verdier skal ha høyere prioritet enn hardkodet."""
        import logic

        # Simuler at yfinance sier VAR.OL egentlig er 'Industri' (ikke 'Energi')
        mock_cache = {"VAR.OL": "Industri"}
        with patch('data.hent_sektor_mapping', return_value=mock_cache):
            logic.oppdater_sektor_cache()

        assert logic.SEKTOR_MAPPING["VAR.OL"] == "Industri"

        # Rydd opp — sett tilbake til hardkodet
        logic.SEKTOR_MAPPING["VAR.OL"] = "Energi"
