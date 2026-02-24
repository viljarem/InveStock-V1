"""
Tests for insider_monitor.py — Insider trade monitor (Newsweb API).

Tests cover:
  - Trade type classification (kjøp/salg/ukjent)
  - Role weight estimation
  - Amount extraction from text
  - Insider score calculation
  - Scanner integration (emoji output)
  - Summary DataFrame generation
  - Cache read/write/TTL
  - Edge cases (empty data, missing fields, bad API)
  - API access functions (mocked)

All tests are fully mocked — no real API calls.
"""
import sys, os
import json
import time
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import insider_monitor


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_handler():
    """Sample list of insider trades (typical API output after parsing)."""
    return [
        {
            "ticker": "EQNR.OL",
            "issuer_sign": "EQNR",
            "issuer_name": "Equinor ASA",
            "tittel": "Mandatory notification of trade - purchase by CEO",
            "dato": "2025-06-10",
            "dato_tid": "2025-06-10T08:00:00+00:00",
            "message_id": 666100,
            "type": "kjøp",
        },
        {
            "ticker": "EQNR.OL",
            "issuer_sign": "EQNR",
            "issuer_name": "Equinor ASA",
            "tittel": "Mandatory notification of trade - purchase by CFO",
            "dato": "2025-06-08",
            "dato_tid": "2025-06-08T10:00:00+00:00",
            "message_id": 666099,
            "type": "kjøp",
        },
        {
            "ticker": "DNB.OL",
            "issuer_sign": "DNB",
            "issuer_name": "DNB Bank ASA",
            "tittel": "Mandatory notification of trade - sale by board member",
            "dato": "2025-06-05",
            "dato_tid": "2025-06-05T14:00:00+00:00",
            "message_id": 666080,
            "type": "salg",
        },
        {
            "ticker": "NHY.OL",
            "issuer_sign": "NHY",
            "issuer_name": "Norsk Hydro ASA",
            "tittel": "Meldepliktig handel - kjøp av styreleder",
            "dato": "2025-05-20",
            "dato_tid": "2025-05-20T09:00:00+00:00",
            "message_id": 665900,
            "type": "kjøp",
        },
        {
            "ticker": "MOWI.OL",
            "issuer_sign": "MOWI",
            "issuer_name": "Mowi ASA",
            "tittel": "Mandatory notification of trade - primary insider allocation",
            "dato": "2025-04-15",
            "dato_tid": "2025-04-15T12:00:00+00:00",
            "message_id": 665500,
            "type": "kjøp",
        },
    ]


@pytest.fixture
def sample_api_list_response():
    """Mock Newsweb API response for /v1/newsreader/list."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    date1 = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date2 = (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "data": {
            "messages": [
                {
                    "messageId": 666158,
                    "issuerSign": "EQNR",
                    "issuerName": "Equinor ASA",
                    "title": "Mandatory notification of trade - purchase by CEO",
                    "publishedTime": date1,
                    "category": {"id": 1102},
                    "markets": [{"name": "Oslo Børs"}],
                },
                {
                    "messageId": 666157,
                    "issuerSign": "STB",
                    "issuerName": "Storebrand ASA",
                    "title": "Mandatory notification of trade - sale by director",
                    "publishedTime": date2,
                    "category": {"id": 1102},
                    "markets": [{"name": "Oslo Børs"}],
                },
            ],
            "overflow": False,
        }
    }


@pytest.fixture
def sample_api_urls_response():
    """Mock urls.json response."""
    return {
        "version": "3.10.17",
        "build_time": "2025-10-16T17:19:01.818Z",
        "api_large": "https://api3.oslo.oslobors.no",
        "public_url": "",
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestKlassifiserType:
    """Test _klassifiser_type() trade direction classifier."""

    def test_purchase_english(self):
        assert insider_monitor._klassifiser_type("Mandatory notification - purchase") == "kjøp"

    def test_purchase_norwegian(self):
        assert insider_monitor._klassifiser_type("Meldepliktig handel - kjøp") == "kjøp"

    def test_sale_english(self):
        assert insider_monitor._klassifiser_type("Mandatory notification - sale") == "salg"

    def test_sale_norwegian(self):
        assert insider_monitor._klassifiser_type("Meldepliktig handel - salg av aksjer") == "salg"

    def test_disposal(self):
        assert insider_monitor._klassifiser_type("Disposal of shares") == "salg"

    def test_sold(self):
        assert insider_monitor._klassifiser_type("Board member sold 10000 shares") == "salg"

    def test_acquisition(self):
        assert insider_monitor._klassifiser_type("Acquisition of shares") == "kjøp"

    def test_allocation(self):
        assert insider_monitor._klassifiser_type("Share allocation to primary insider") == "kjøp"

    def test_award(self):
        assert insider_monitor._klassifiser_type("Share award program") == "kjøp"

    def test_unknown(self):
        assert insider_monitor._klassifiser_type("Some random title") == "ukjent"

    def test_case_insensitive(self):
        assert insider_monitor._klassifiser_type("MANDATORY NOTIFICATION - PURCHASE") == "kjøp"
        assert insider_monitor._klassifiser_type("SALE OF SHARES") == "salg"

    def test_empty_string(self):
        assert insider_monitor._klassifiser_type("") == "ukjent"

    def test_salg_overrides_kjøp_if_first(self):
        """If 'sale' appears, it should classify as salg."""
        result = insider_monitor._klassifiser_type("sale of shares")
        assert result == "salg"


# ═══════════════════════════════════════════════════════════════════════
# 2. ROLE ESTIMATION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestEstimerRolle:
    """Test _estimer_rolle() role weight estimation."""

    def test_ceo_english(self):
        assert insider_monitor._estimer_rolle("Purchase by CEO") == 1.0

    def test_ceo_norwegian(self):
        assert insider_monitor._estimer_rolle("Kjøp av Adm. Dir") == 1.0

    def test_cfo(self):
        assert insider_monitor._estimer_rolle("Trade by CFO of company") == 0.9

    def test_chair(self):
        assert insider_monitor._estimer_rolle("Styreleder har kjøpt aksjer") == 0.85

    def test_board_member(self):
        assert insider_monitor._estimer_rolle("Board member purchased shares") == 0.7

    def test_primary_insider(self):
        assert insider_monitor._estimer_rolle("Primary insider notification") == 0.5

    def test_unknown_role(self):
        assert insider_monitor._estimer_rolle("Some person did something") == 0.3

    def test_empty_string(self):
        assert insider_monitor._estimer_rolle("") == 0.3

    def test_highest_role_wins(self):
        """If multiple roles match, the highest weight should be used."""
        # Contains both "CEO" and "board member"
        result = insider_monitor._estimer_rolle("CEO who is also board member")
        assert result == 1.0

    def test_case_insensitive(self):
        assert insider_monitor._estimer_rolle("chairman of the board") == 0.85


# ═══════════════════════════════════════════════════════════════════════
# 3. AMOUNT EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestEstimerBeløp:
    """Test _estimer_beløp_fra_tekst() amount extraction."""

    def test_nok_simple(self):
        result = insider_monitor._estimer_beløp_fra_tekst("Total amount: NOK 1234567")
        assert result == 1234567.0

    def test_nok_with_commas(self):
        result = insider_monitor._estimer_beløp_fra_tekst("Total: NOK 1,234,567")
        assert result == 1234567.0

    def test_nok_with_spaces(self):
        result = insider_monitor._estimer_beløp_fra_tekst("Beløp: NOK 1 234 567")
        assert result == 1234567.0

    def test_shares_at_price(self):
        result = insider_monitor._estimer_beløp_fra_tekst(
            "10000 shares at NOK 176.56"
        )
        assert result is not None
        assert result == pytest.approx(1765600.0, rel=0.01)

    def test_aksjer_til_pris(self):
        result = insider_monitor._estimer_beløp_fra_tekst(
            "5000 aksjer til NOK 200.00"
        )
        assert result is not None
        assert result == pytest.approx(1000000.0, rel=0.01)

    def test_no_amount(self):
        result = insider_monitor._estimer_beløp_fra_tekst("Some text without amounts")
        assert result is None

    def test_empty_string(self):
        result = insider_monitor._estimer_beløp_fra_tekst("")
        assert result is None

    def test_multiple_amounts_returns_max(self):
        result = insider_monitor._estimer_beløp_fra_tekst(
            "Price: NOK 100. Total: NOK 500000"
        )
        assert result == 500000.0


# ═══════════════════════════════════════════════════════════════════════
# 4. INSIDER SCORE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestBeregInsiderScore:
    """Test beregn_insider_score() scoring logic."""

    def test_no_trades_returns_zero(self, sample_handler):
        result = insider_monitor.beregn_insider_score("NONEXISTENT.OL", sample_handler)
        assert result["score"] == 0
        assert result["antall_kjøp"] == 0
        assert result["antall_salg"] == 0
        assert result["siste_handel"] is None
        assert result["handler"] == []

    def test_buy_gives_positive_score(self, sample_handler):
        result = insider_monitor.beregn_insider_score("EQNR.OL", sample_handler)
        assert result["score"] > 0
        assert result["antall_kjøp"] == 2
        assert result["antall_salg"] == 0

    def test_sell_gives_negative_score(self, sample_handler):
        result = insider_monitor.beregn_insider_score("DNB.OL", sample_handler)
        assert result["score"] < 0
        assert result["antall_salg"] == 1

    def test_score_bounded(self, sample_handler):
        """Score should be within [-100, 100]."""
        result = insider_monitor.beregn_insider_score("EQNR.OL", sample_handler)
        assert -100 <= result["score"] <= 100

    def test_result_has_expected_keys(self, sample_handler):
        result = insider_monitor.beregn_insider_score("EQNR.OL", sample_handler)
        expected_keys = {"score", "antall_kjøp", "antall_salg", "antall_ukjent", "siste_handel", "handler"}
        assert set(result.keys()) == expected_keys

    def test_siste_handel_date(self, sample_handler):
        result = insider_monitor.beregn_insider_score("EQNR.OL", sample_handler)
        assert result["siste_handel"] == "2025-06-10"

    def test_handler_list_sorted_newest_first(self, sample_handler):
        result = insider_monitor.beregn_insider_score("EQNR.OL", sample_handler)
        dates = [h["dato"] for h in result["handler"]]
        assert dates == sorted(dates, reverse=True)

    def test_ceo_scores_higher_than_board_member(self):
        """CEO purchase should score higher than board member purchase."""
        handler_ceo = [{
            "ticker": "TEST.OL", "issuer_sign": "TEST",
            "issuer_name": "Test", "tittel": "Purchase by CEO",
            "dato": "2025-06-10", "dato_tid": "2025-06-10T08:00:00+00:00",
            "message_id": 1, "type": "kjøp",
        }]
        handler_bm = [{
            "ticker": "TEST.OL", "issuer_sign": "TEST",
            "issuer_name": "Test", "tittel": "Purchase by board member",
            "dato": "2025-06-10", "dato_tid": "2025-06-10T08:00:00+00:00",
            "message_id": 2, "type": "kjøp",
        }]
        score_ceo = insider_monitor.beregn_insider_score("TEST.OL", handler_ceo)["score"]
        score_bm = insider_monitor.beregn_insider_score("TEST.OL", handler_bm)["score"]
        assert score_ceo > score_bm

    def test_recent_trade_scores_higher(self):
        """More recent trade should contribute more to score."""
        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).isoformat()
        old_str = "2025-01-01T08:00:00+00:00"

        handler_new = [{
            "ticker": "T.OL", "issuer_sign": "T", "issuer_name": "T",
            "tittel": "Purchase by primary insider",
            "dato": "2025-06-10", "dato_tid": now_str,
            "message_id": 1, "type": "kjøp",
        }]
        handler_old = [{
            "ticker": "T.OL", "issuer_sign": "T", "issuer_name": "T",
            "tittel": "Purchase by primary insider",
            "dato": "2025-01-01", "dato_tid": old_str,
            "message_id": 2, "type": "kjøp",
        }]
        score_new = insider_monitor.beregn_insider_score("T.OL", handler_new)["score"]
        score_old = insider_monitor.beregn_insider_score("T.OL", handler_old)["score"]
        assert score_new > score_old

    def test_empty_handler_list(self):
        result = insider_monitor.beregn_insider_score("EQNR.OL", [])
        assert result["score"] == 0

    def test_max_10_handler_in_result(self):
        """Result should contain at most 10 trades."""
        handler = []
        for i in range(15):
            handler.append({
                "ticker": "X.OL", "issuer_sign": "X", "issuer_name": "X",
                "tittel": "Purchase by CEO", "dato": f"2025-06-{10-i:02d}",
                "dato_tid": f"2025-06-{10-i:02d}T08:00:00+00:00",
                "message_id": i, "type": "kjøp",
            })
        result = insider_monitor.beregn_insider_score("X.OL", handler)
        assert len(result["handler"]) <= 10


# ═══════════════════════════════════════════════════════════════════════
# 5. SCANNER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestScannerIntegration:
    """Test beregn_insider_for_scanner() emoji output."""

    def test_strong_buy_green(self, sample_handler):
        result = insider_monitor.beregn_insider_for_scanner("EQNR.OL", sample_handler)
        # EQNR has 2 buys by CEO/CFO — should be positive
        assert "🟢" in result or "🟡" in result
        assert "+" in result

    def test_sell_red(self, sample_handler):
        result = insider_monitor.beregn_insider_for_scanner("DNB.OL", sample_handler)
        # DNB has 1 sale — should be negative
        assert "🔴" in result or "🟡" in result

    def test_no_trades_returns_dash(self, sample_handler):
        result = insider_monitor.beregn_insider_for_scanner("NONEXISTENT.OL", sample_handler)
        assert result == "—"

    def test_returns_string(self, sample_handler):
        result = insider_monitor.beregn_insider_for_scanner("EQNR.OL", sample_handler)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
# 6. SUMMARY DATAFRAME TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestSammendrag:
    """Test hent_insider_sammendrag() DataFrame generation."""

    def test_returns_dataframe(self, sample_handler):
        df = insider_monitor.hent_insider_sammendrag(sample_handler)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, sample_handler):
        df = insider_monitor.hent_insider_sammendrag(sample_handler)
        assert "Dato" in df.columns
        assert "Selskap" in df.columns
        assert "Type" in df.columns

    def test_type_has_emojis(self, sample_handler):
        df = insider_monitor.hent_insider_sammendrag(sample_handler)
        types = df["Type"].tolist()
        assert any("🟢" in str(t) for t in types)  # At least one kjøp
        assert any("🔴" in str(t) for t in types)  # At least one salg

    def test_correct_row_count(self, sample_handler):
        df = insider_monitor.hent_insider_sammendrag(sample_handler)
        assert len(df) == len(sample_handler)

    def test_empty_handler(self):
        df = insider_monitor.hent_insider_sammendrag([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═══════════════════════════════════════════════════════════════════════
# 7. CACHE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCache:
    """Test cache read/write/TTL logic."""

    def test_write_and_read_cache(self, tmp_path, monkeypatch, sample_handler):
        cache_file = str(tmp_path / "test_insider_cache.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)

        insider_monitor._skriv_cache(sample_handler)
        result = insider_monitor._les_cache()

        assert result is not None
        assert len(result) == len(sample_handler)
        assert result[0]["ticker"] == sample_handler[0]["ticker"]

    def test_cache_ttl_expired(self, tmp_path, monkeypatch, sample_handler):
        cache_file = str(tmp_path / "test_insider_cache.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)
        monkeypatch.setattr(insider_monitor, "CACHE_TTL_SEKUNDER", 1)

        insider_monitor._skriv_cache(sample_handler)

        # Manually set timestamp to past
        with open(cache_file, "r") as f:
            cache = json.load(f)
        cache["timestamp"] = time.time() - 100
        with open(cache_file, "w") as f:
            json.dump(cache, f)

        result = insider_monitor._les_cache()
        assert result is None

    def test_cache_fresh(self, tmp_path, monkeypatch, sample_handler):
        cache_file = str(tmp_path / "test_insider_cache.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)
        monkeypatch.setattr(insider_monitor, "CACHE_TTL_SEKUNDER", 3600)

        insider_monitor._skriv_cache(sample_handler)
        result = insider_monitor._les_cache()
        assert result is not None

    def test_cache_missing_file(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "nonexistent.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)
        result = insider_monitor._les_cache()
        assert result is None

    def test_cache_corrupt_file(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "corrupt.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)
        with open(cache_file, "w") as f:
            f.write("not valid json {{{")
        result = insider_monitor._les_cache()
        assert result is None

    def test_tøm_cache(self, tmp_path, monkeypatch, sample_handler):
        cache_file = str(tmp_path / "to_delete.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)
        insider_monitor._skriv_cache(sample_handler)
        assert os.path.exists(cache_file)
        insider_monitor.tøm_cache()
        assert not os.path.exists(cache_file)


# ═══════════════════════════════════════════════════════════════════════
# 8. API ACCESS TESTS (MOCKED)
# ═══════════════════════════════════════════════════════════════════════


class TestAPIAccess:
    """Test API access functions with mocked requests."""

    def test_hent_api_base_success(self, monkeypatch, sample_api_urls_response):
        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return sample_api_urls_response

        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
        result = insider_monitor._hent_api_base()
        assert result == "https://api3.oslo.oslobors.no"

    def test_hent_api_base_failure(self, monkeypatch):
        import requests
        def fail(*a, **kw):
            raise requests.ConnectionError("No network")
        monkeypatch.setattr(requests, "get", fail)
        result = insider_monitor._hent_api_base()
        assert result is None

    def test_api_post_success(self, monkeypatch, sample_api_list_response):
        class MockResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return sample_api_list_response

        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **kw: MockResponse())
        result = insider_monitor._api_post("/v1/newsreader/list", "https://api3.oslo.oslobors.no")
        assert result is not None
        assert "data" in result

    def test_api_post_failure(self, monkeypatch):
        import requests
        def fail(*a, **kw):
            raise requests.ConnectionError("No network")
        monkeypatch.setattr(requests, "post", fail)
        result = insider_monitor._api_post("/v1/newsreader/list", "https://api3.oslo.oslobors.no")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 9. INTEGRATION: hent_innsidehandler WITH MOCKS
# ═══════════════════════════════════════════════════════════════════════


class TestHentInnsidehandler:
    """Test the full hent_innsidehandler flow with mocked API."""

    def test_returns_parsed_handler(self, monkeypatch, tmp_path,
                                     sample_api_urls_response, sample_api_list_response):
        # Disable cache
        cache_file = str(tmp_path / "no_cache.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)

        import requests

        class MockGetResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return sample_api_urls_response

        class MockPostResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return sample_api_list_response

        monkeypatch.setattr(requests, "get", lambda *a, **kw: MockGetResp())
        monkeypatch.setattr(requests, "post", lambda *a, **kw: MockPostResp())

        result = insider_monitor.hent_innsidehandler(dager=90)
        assert isinstance(result, list)
        assert len(result) == 2  # Two messages in sample
        assert result[0]["ticker"] == "EQNR.OL"
        assert result[0]["type"] == "kjøp"
        assert result[1]["ticker"] == "STB.OL"
        assert result[1]["type"] == "salg"

    def test_uses_cache_if_fresh(self, monkeypatch, tmp_path, sample_handler):
        cache_file = str(tmp_path / "cached.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)
        insider_monitor._skriv_cache(sample_handler)

        # API should NOT be called — if it is, it'll fail
        import requests
        def fail(*a, **kw):
            raise RuntimeError("Should not call API when cache is fresh!")
        monkeypatch.setattr(requests, "get", fail)
        monkeypatch.setattr(requests, "post", fail)

        result = insider_monitor.hent_innsidehandler(dager=90)
        assert len(result) == len(sample_handler)

    def test_empty_when_api_fails(self, monkeypatch, tmp_path):
        cache_file = str(tmp_path / "no_cache2.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)

        import requests
        def fail(*a, **kw):
            raise requests.ConnectionError("No network")
        monkeypatch.setattr(requests, "get", fail)

        result = insider_monitor.hent_innsidehandler(dager=90)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════
# 10. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_handler_with_missing_fields(self):
        """Handler entries with missing optional fields should not crash."""
        handler = [{
            "ticker": "X.OL",
            "issuer_sign": "X",
            "issuer_name": "X Corp",
            "tittel": "",
            "dato": "2025-06-01",
            "dato_tid": "2025-06-01T08:00:00+00:00",
            "message_id": 1,
            "type": "ukjent",
        }]
        result = insider_monitor.beregn_insider_score("X.OL", handler)
        assert isinstance(result["score"], (int, float))

    def test_handler_with_bad_date(self):
        """Invalid date should not crash scoring."""
        handler = [{
            "ticker": "X.OL",
            "issuer_sign": "X",
            "issuer_name": "X Corp",
            "tittel": "Purchase by CEO",
            "dato": "invalid-date",
            "dato_tid": "not-a-date",
            "message_id": 1,
            "type": "kjøp",
        }]
        result = insider_monitor.beregn_insider_score("X.OL", handler)
        assert isinstance(result["score"], (int, float))

    def test_very_old_trade_has_low_weight(self):
        """A trade from 89 days ago should have very low time weight."""
        handler = [{
            "ticker": "X.OL", "issuer_sign": "X", "issuer_name": "X",
            "tittel": "Purchase by CEO",
            "dato": "2025-01-01",
            "dato_tid": "2025-01-01T08:00:00+00:00",
            "message_id": 1, "type": "kjøp",
        }]
        result = insider_monitor.beregn_insider_score("X.OL", handler)
        # Score should be positive but small due to time decay
        assert 0 < result["score"] < 20

    def test_sammendrag_with_none_handler(self, monkeypatch, tmp_path):
        """hent_insider_sammendrag(None) should try to fetch, return empty on failure."""
        cache_file = str(tmp_path / "empty.json")
        monkeypatch.setattr(insider_monitor, "INSIDER_CACHE_FILE", cache_file)

        import requests
        def fail(*a, **kw):
            raise requests.ConnectionError("No network")
        monkeypatch.setattr(requests, "get", fail)

        df = insider_monitor.hent_insider_sammendrag(None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_ticker_matching_is_exact(self, sample_handler):
        """Score for 'EQN.OL' (partial match) should be zero."""
        result = insider_monitor.beregn_insider_score("EQN.OL", sample_handler)
        assert result["score"] == 0

    def test_multiple_buys_amplify_score(self):
        """Multiple buys by same ticker should increase score."""
        handler_1 = [{
            "ticker": "X.OL", "issuer_sign": "X", "issuer_name": "X",
            "tittel": "Purchase by CEO",
            "dato": "2025-06-10", "dato_tid": "2025-06-10T08:00:00+00:00",
            "message_id": 1, "type": "kjøp",
        }]
        handler_3 = handler_1 * 3  # 3 identical trades
        score_1 = insider_monitor.beregn_insider_score("X.OL", handler_1)["score"]
        score_3 = insider_monitor.beregn_insider_score("X.OL", handler_3)["score"]
        assert score_3 > score_1
