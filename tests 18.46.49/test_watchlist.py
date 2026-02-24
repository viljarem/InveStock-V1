# tests/test_watchlist.py
"""
Tester for 7.3 — Forbedret Watchlist med metadata.
Dekker: migrering, CRUD med metadata, bakoverkompatibilitet, P&L-beregning.
"""

import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch
from datetime import datetime


class TestMigrering:
    """Tester automatisk migrering fra gammel til ny format."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wl_file = os.path.join(self.tmpdir, "watchlist.json")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migrer_flat_liste(self):
        """Gammel liste-format skal migreres automatisk."""
        import utils
        with open(self.wl_file, 'w') as f:
            json.dump(["EQNR.OL", "DNB.OL"], f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils._load_watchlist_raw()

        assert "EQNR.OL" in result
        assert "DNB.OL" in result
        assert "added_date" in result["EQNR.OL"]
        assert result["EQNR.OL"]["notes"] == "Migrert fra gammel liste"

    def test_migrer_gammel_dict_format(self):
        """Gammel {tickers: [], added_dates: {}} format skal migreres."""
        import utils
        old = {"tickers": ["MOWI.OL"], "added_dates": {"MOWI.OL": "2026-01-15"}}
        with open(self.wl_file, 'w') as f:
            json.dump(old, f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils._load_watchlist_raw()

        assert "MOWI.OL" in result
        assert result["MOWI.OL"]["added_date"] == "2026-01-15"

    def test_ny_format_trenger_ingen_migrering(self):
        """Nytt dict-format skal brukes direkte."""
        import utils
        new_data = {
            "EQNR.OL": {
                "added_date": "2026-02-19",
                "reason": "VCP",
                "price_at_add": 300.0,
                "target_price": 350.0,
                "notes": "Test"
            }
        }
        with open(self.wl_file, 'w') as f:
            json.dump(new_data, f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils._load_watchlist_raw()

        assert result == new_data

    def test_tom_fil_gir_tom_dict(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', os.path.join(self.tmpdir, "finnes_ikke.json")):
            result = utils._load_watchlist_raw()
        assert result == {}

    def test_korrupt_fil_gir_tom_dict(self):
        import utils
        with open(self.wl_file, 'w') as f:
            f.write("{korrupt!!")
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils._load_watchlist_raw()
        assert result == {}

    def test_migrering_lagres_til_disk(self):
        """Etter migrering skal nytt format være lagret."""
        import utils
        with open(self.wl_file, 'w') as f:
            json.dump(["EQNR.OL"], f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils._load_watchlist_raw()  # Trigger migrering

        # Les på nytt — skal nå være dict
        with open(self.wl_file, 'r') as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "EQNR.OL" in data


class TestBakoverkompatibilitet:
    """Tester at load_watchlist() returnerer liste (bakoverkompatibelt)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wl_file = os.path.join(self.tmpdir, "watchlist.json")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_watchlist_returnerer_liste(self):
        import utils
        data = {"EQNR.OL": {"added_date": "2026-01-01", "reason": "", "price_at_add": None, "target_price": None, "notes": ""}}
        with open(self.wl_file, 'w') as f:
            json.dump(data, f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils.load_watchlist()

        assert isinstance(result, list)
        assert "EQNR.OL" in result

    def test_in_operator_fungerer(self):
        """'ticker in load_watchlist()' skal fungere som før."""
        import utils
        data = {"DNB.OL": {"added_date": "2026-01-01", "reason": "", "price_at_add": None, "target_price": None, "notes": ""}}
        with open(self.wl_file, 'w') as f:
            json.dump(data, f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            wl = utils.load_watchlist()

        assert "DNB.OL" in wl
        assert "FAKE.OL" not in wl

    def test_save_watchlist_med_liste_bevarer_metadata(self):
        """save_watchlist(liste) skal bevare metadata for eksisterende tickers."""
        import utils
        data = {
            "EQNR.OL": {"added_date": "2026-01-01", "reason": "VCP", "price_at_add": 300.0, "target_price": 350.0, "notes": ""},
            "DNB.OL": {"added_date": "2026-01-02", "reason": "", "price_at_add": None, "target_price": None, "notes": ""}
        }
        with open(self.wl_file, 'w') as f:
            json.dump(data, f)

        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            # Lagre med liste (fjern DNB)
            utils.save_watchlist(["EQNR.OL"])
            result = utils._load_watchlist_raw()

        assert "EQNR.OL" in result
        assert "DNB.OL" not in result
        assert result["EQNR.OL"]["reason"] == "VCP"
        assert result["EQNR.OL"]["price_at_add"] == 300.0


class TestCRUDMedMetadata:
    """Tester tillegging, fjerning og oppdatering med metadata."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wl_file = os.path.join(self.tmpdir, "watchlist.json")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_med_metadata(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils.add_to_watchlist("EQNR.OL", reason="VCP-signal", price_at_add=305.50)
            assert result is True

            meta = utils.load_watchlist_metadata()
            assert meta["EQNR.OL"]["reason"] == "VCP-signal"
            assert meta["EQNR.OL"]["price_at_add"] == 305.50
            assert meta["EQNR.OL"]["added_date"] == datetime.now().strftime("%Y-%m-%d")

    def test_add_uten_metadata(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("DNB.OL")
            meta = utils.load_watchlist_metadata()
            assert meta["DNB.OL"]["reason"] == ""
            assert meta["DNB.OL"]["price_at_add"] is None

    def test_add_duplikat_returnerer_false(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("EQNR.OL")
            result = utils.add_to_watchlist("EQNR.OL")
            assert result is False

    def test_remove(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("EQNR.OL")
            result = utils.remove_from_watchlist("EQNR.OL")
            assert result is True
            assert "EQNR.OL" not in utils.load_watchlist()

    def test_remove_ikkeeksisterende(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils.remove_from_watchlist("FAKE.OL")
            assert result is False

    def test_is_in_watchlist(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("EQNR.OL")
            assert utils.is_in_watchlist("EQNR.OL") is True
            assert utils.is_in_watchlist("FAKE.OL") is False

    def test_update_metadata(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("EQNR.OL", price_at_add=300.0)
            utils.update_watchlist_metadata("EQNR.OL", target_price=350.0, notes="Mulig oppkjøp")

            meta = utils.load_watchlist_metadata()
            assert meta["EQNR.OL"]["target_price"] == 350.0
            assert meta["EQNR.OL"]["notes"] == "Mulig oppkjøp"
            assert meta["EQNR.OL"]["price_at_add"] == 300.0  # Uendret

    def test_update_metadata_ikkeeksisterende(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            result = utils.update_watchlist_metadata("FAKE.OL", notes="test")
            assert result is False

    def test_load_watchlist_metadata(self):
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("EQNR.OL", reason="AI", price_at_add=305.0)
            utils.add_to_watchlist("DNB.OL", reason="VCP", price_at_add=200.0)

            meta = utils.load_watchlist_metadata()
            assert len(meta) == 2
            assert meta["EQNR.OL"]["reason"] == "AI"
            assert meta["DNB.OL"]["price_at_add"] == 200.0

    def test_flere_operasjoner_konsekvent(self):
        """Legg til, fjern, legg til igjen — skal fungere rent."""
        import utils
        with patch.object(utils, 'WATCHLIST_FILE', self.wl_file):
            utils.add_to_watchlist("A.OL", reason="R1", price_at_add=100.0)
            utils.add_to_watchlist("B.OL", reason="R2", price_at_add=200.0)
            utils.remove_from_watchlist("A.OL")
            utils.add_to_watchlist("C.OL", reason="R3", price_at_add=300.0)

            wl = utils.load_watchlist()
            assert sorted(wl) == ["B.OL", "C.OL"]

            meta = utils.load_watchlist_metadata()
            assert "A.OL" not in meta
            assert meta["C.OL"]["reason"] == "R3"
