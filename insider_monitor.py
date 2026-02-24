"""
Insider Monitor — Overvåk meldepliktige handler fra Oslo Børs Newsweb.

🧪 BETA-MODUL: Avhenger av uoffisiell Newsweb API (Euronext).
   Kan slutte å virke uten forvarsel.

Dataflyt:
  1) Hent insider-handler fra Newsweb API (POST til api3.oslo.oslobors.no)
  2) Parse meldinger for rolle, type (kjøp/salg), beløp
  3) Beregn en insider-score per ticker
  4) Cache til disk med TTL for å unngå API-overbelastning

Kategorier i Newsweb:
  1102 = "Meldepliktig handel for primærinnsidere"
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger("InveStock.InsiderMonitor")

# ═══════════════════════════════════════════════════════════════════════
# Konfigurasjon
# ═══════════════════════════════════════════════════════════════════════

NEWSWEB_URLS_ENDPOINT = "https://newsweb.oslobors.no/urls.json"
NEWSWEB_CATEGORY_INSIDER = 1102  # Meldepliktig handel for primærinnsidere

CACHE_DIR = "data_storage"
INSIDER_CACHE_FILE = os.path.join(CACHE_DIR, "insider_cache.json")
CACHE_TTL_SEKUNDER = 3600  # 1 time mellom API-kall

# Mapping fra issuerSign til Yahoo Finance ticker
# Newsweb bruker tickers uten .OL-suffiks
_TICKER_SUFFIKS = ".OL"

# Vekter for insider-score
ROLLE_VEKT = {
    "ceo": 1.0,
    "adm. dir": 1.0,
    "administrerende direktør": 1.0,
    "chief executive": 1.0,
    "cfo": 0.9,
    "finansdirektør": 0.9,
    "chief financial": 0.9,
    "styreleder": 0.85,
    "chair": 0.85,
    "chairman": 0.85,
    "styremedlem": 0.7,
    "board member": 0.7,
    "director": 0.7,
    "evp": 0.6,
    "svp": 0.6,
    "vp": 0.5,
    "primary insider": 0.5,
    "primærinnsider": 0.5,
}

# ═══════════════════════════════════════════════════════════════════════
# 1. API-tilgang
# ═══════════════════════════════════════════════════════════════════════

def _hent_api_base() -> Optional[str]:
    """Hent API base-URL fra Newsweb urls.json."""
    try:
        import requests
        resp = requests.get(NEWSWEB_URLS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        base = data.get("api_large", "")
        if base:
            logger.info("Newsweb API base: %s", base)
            return base
        logger.warning("Ingen api_large i urls.json")
        return None
    except Exception as e:
        logger.warning("Kunne ikke hente Newsweb API base: %s", e)
        return None


def _api_post(endpoint: str, api_base: str, timeout: int = 15) -> Optional[dict]:
    """POST-forespørsel til Newsweb API."""
    try:
        import requests
        url = f"{api_base}{endpoint}"
        resp = requests.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("API-kall feilet (%s): %s", endpoint, e)
        return None


# ═══════════════════════════════════════════════════════════════════════
# 2. Hent og parse insider-handler
# ═══════════════════════════════════════════════════════════════════════

def hent_innsidehandler(dager: int = 90) -> list[dict]:
    """
    Hent meldepliktige handler fra Newsweb.
    
    Returns:
        Liste med dicts: {ticker, tittel, dato, type, issuer_sign,
                          issuer_name, message_id, body}
    """
    cached = _les_cache()
    if cached is not None:
        logger.debug("Bruker cached insider-data (%d handler)", len(cached))
        return cached

    api_base = _hent_api_base()
    if not api_base:
        logger.warning("Ingen API base — returnerer tom liste")
        return []

    # Hent liste over insider-meldinger
    data = _api_post(
        f"/v1/newsreader/list?category={NEWSWEB_CATEGORY_INSIDER}",
        api_base,
    )
    if not data or "data" not in data:
        logger.warning("Ugyldig API-respons for insider-liste")
        return []

    messages = data["data"].get("messages", [])
    if not messages:
        logger.info("Ingen insider-meldinger funnet")
        return []

    # Filtrer på tidsperiode
    cutoff = datetime.now(timezone.utc) - timedelta(days=dager)
    handler = []

    for msg in messages:
        try:
            pub_time = datetime.fromisoformat(
                msg["publishedTime"].replace("Z", "+00:00")
            )
            if pub_time < cutoff:
                continue

            issuer_sign = msg.get("issuerSign", "")
            ticker = f"{issuer_sign}{_TICKER_SUFFIKS}" if issuer_sign else ""

            handel = {
                "ticker": ticker,
                "issuer_sign": issuer_sign,
                "issuer_name": msg.get("issuerName", ""),
                "tittel": msg.get("title", ""),
                "dato": pub_time.strftime("%Y-%m-%d"),
                "dato_tid": pub_time.isoformat(),
                "message_id": msg.get("messageId", 0),
                "type": _klassifiser_type(msg.get("title", "")),
            }
            handler.append(handel)
        except Exception as e:
            logger.debug("Kunne ikke parse melding %s: %s", msg.get("id"), e)
            continue

    logger.info("Hentet %d insider-handler fra Newsweb", len(handler))
    _skriv_cache(handler)
    return handler


def hent_melding_detaljer(message_id: int, api_base: Optional[str] = None) -> Optional[dict]:
    """Hent full meldingstekst for en enkelt insider-melding."""
    if not api_base:
        api_base = _hent_api_base()
    if not api_base:
        return None

    data = _api_post(f"/v1/newsreader/message?messageId={message_id}", api_base)
    if not data or "data" not in data:
        return None

    msg = data["data"].get("message", {})
    return {
        "body": msg.get("body", ""),
        "tittel": msg.get("title", ""),
        "issuer_sign": msg.get("issuerSign", ""),
        "issuer_name": msg.get("issuerName", ""),
        "dato": msg.get("publishedTime", ""),
        "vedlegg": len(msg.get("attachments", [])),
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. Klassifisering og scoring
# ═══════════════════════════════════════════════════════════════════════

def _klassifiser_type(tittel: str) -> str:
    """Gjett om meldingen gjelder kjøp eller salg basert på tittelen."""
    tittel_lower = tittel.lower()

    # Salg-indikatorer
    salg_ord = ["sale", "salg", "sold", "solgt", "disposal", "avhend"]
    if any(ord in tittel_lower for ord in salg_ord):
        return "salg"

    # Kjøp-indikatorer
    kjøp_ord = [
        "purchase", "kjøp", "bought", "acquisition",
        "bought", "allocation", "tildel", "award",
    ]
    if any(ord in tittel_lower for ord in kjøp_ord):
        return "kjøp"

    # Default — de fleste obligatoriske meldinger er kjøp
    return "ukjent"


def _estimer_rolle(tekst: str) -> float:
    """Estimer rollvekt fra meldingstekst (0.3-1.0)."""
    tekst_lower = tekst.lower()
    best_vekt = 0.3  # Default: ukjent rolle

    for rolle_nøkkel, vekt in ROLLE_VEKT.items():
        if rolle_nøkkel in tekst_lower:
            best_vekt = max(best_vekt, vekt)

    return best_vekt


def _estimer_beløp_fra_tekst(tekst: str) -> Optional[float]:
    """
    Forsøk å trekke ut handelsbeløp fra meldingsteksten.
    
    Typiske mønstre:
      "NOK 1,234,567" eller "NOK 1 234 567" eller "10,201 shares at NOK 176.56"
    """
    # Mønster 1: "NOK X" direkte beløp
    nok_pattern = re.findall(
        r'NOK\s*([\d,.\s]+)', tekst, re.IGNORECASE
    )
    beløp_kandidater = []
    for match in nok_pattern:
        try:
            # Fjern mellomrom og konverter komma
            clean = match.strip().replace(" ", "").replace(",", "")
            val = float(clean)
            if val > 0:
                beløp_kandidater.append(val)
        except ValueError:
            continue

    # Mønster 2: "X shares at NOK Y" → antall * pris
    aksje_pris = re.findall(
        r'([\d,.\s]+)\s*(?:shares?|aksjer?)\s*(?:at|til|@)\s*NOK\s*([\d,.\s]+)',
        tekst, re.IGNORECASE
    )
    for antall_str, pris_str in aksje_pris:
        try:
            antall = float(antall_str.strip().replace(" ", "").replace(",", ""))
            pris = float(pris_str.strip().replace(" ", "").replace(",", ""))
            beløp_kandidater.append(antall * pris)
        except ValueError:
            continue

    if beløp_kandidater:
        return max(beløp_kandidater)
    return None


def beregn_insider_score(ticker: str, handler: list[dict],
                         detaljer_cache: Optional[dict] = None) -> dict:
    """
    Beregn en insider-score for en ticker basert på nylige handler.
    
    Score-skala: -100 til +100
      Positiv = netto kjøp (bullish)
      Negativ = netto salg (bearish)
    
    Faktorer:
      - Type: kjøp (+), salg (-)
      - Rolle: CEO/CFO vektes mer enn styremedlem
      - Antall: flere handler i samme retning forsterker
      - Recentness: nyere handler vektes mer
    
    Args:
        ticker: Yahoo Finance ticker (f.eks. "EQNR.OL")
        handler: Liste fra hent_innsidehandler()
        detaljer_cache: Valgfri dict med message_id -> detaljer
    
    Returns:
        Dict med score, antall_kjøp, antall_salg, siste_handel, handler_liste
    """
    ticker_handler = [h for h in handler if h["ticker"] == ticker]

    if not ticker_handler:
        return {
            "score": 0,
            "antall_kjøp": 0,
            "antall_salg": 0,
            "antall_ukjent": 0,
            "siste_handel": None,
            "handler": [],
        }

    nå = datetime.now(timezone.utc)
    total_score = 0.0
    antall_kjøp = 0
    antall_salg = 0
    antall_ukjent = 0

    for h in ticker_handler:
        # Type-retning
        if h["type"] == "kjøp":
            retning = 1.0
            antall_kjøp += 1
        elif h["type"] == "salg":
            retning = -1.0
            antall_salg += 1
        else:
            retning = 0.3  # Ukjent, men de fleste er kjøp
            antall_ukjent += 1

        # Rolle-vekt (fra tittel)
        rolle_vekt = _estimer_rolle(h.get("tittel", ""))

        # Tid-decay: nyere handler vektes mer
        try:
            handel_dato = datetime.fromisoformat(h["dato_tid"])
            dager_siden = max(1, (nå - handel_dato).days)
        except (ValueError, KeyError):
            dager_siden = 30

        tid_vekt = 1.0 / (1.0 + dager_siden / 30.0)

        # Kombiner
        handel_score = retning * rolle_vekt * tid_vekt * 30.0
        total_score += handel_score

    # Klipp til [-100, 100]
    score = max(-100, min(100, total_score))

    # Sorter etter dato (nyeste først)
    ticker_handler.sort(key=lambda x: x.get("dato_tid", ""), reverse=True)
    siste = ticker_handler[0].get("dato") if ticker_handler else None

    return {
        "score": round(score, 1),
        "antall_kjøp": antall_kjøp,
        "antall_salg": antall_salg,
        "antall_ukjent": antall_ukjent,
        "siste_handel": siste,
        "handler": ticker_handler[:10],  # Maks 10 nyeste
    }


def beregn_insider_for_scanner(ticker: str, handler: Optional[list] = None) -> str:
    """
    Rask versjon for scanner-tabellen.
    
    Returns:
        Emoji-streng: "🟢 +45" / "🔴 -30" / "—"
    """
    if handler is None:
        handler = hent_innsidehandler()

    resultat = beregn_insider_score(ticker, handler)
    score = resultat["score"]

    if score == 0 and resultat["antall_kjøp"] == 0:
        return "—"

    if score > 20:
        return f"🟢 +{score:.0f}"
    elif score > 0:
        return f"🟡 +{score:.0f}"
    elif score < -20:
        return f"🔴 {score:.0f}"
    elif score < 0:
        return f"🟡 {score:.0f}"
    else:
        return f"⚪ {score:.0f}"


def hent_insider_sammendrag(handler: Optional[list] = None) -> pd.DataFrame:
    """
    Lag en oppsummerings-DataFrame over nylige insider-handler.
    Brukes i views for å vise en tabell.
    """
    if handler is None:
        handler = hent_innsidehandler()

    if not handler:
        return pd.DataFrame()

    df = pd.DataFrame(handler)
    if df.empty:
        return df

    # Velg og omdøp kolonner
    kolonner = {
        "dato": "Dato",
        "issuer_name": "Selskap",
        "ticker": "Ticker",
        "tittel": "Melding",
        "type": "Type",
    }
    tilgjengelige = {k: v for k, v in kolonner.items() if k in df.columns}
    df = df[list(tilgjengelige.keys())].rename(columns=tilgjengelige)

    # Formater type
    type_emoji = {"kjøp": "🟢 Kjøp", "salg": "🔴 Salg", "ukjent": "⚪ Ukjent"}
    if "Type" in df.columns:
        df["Type"] = df["Type"].map(lambda x: type_emoji.get(x, x))

    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. Cache
# ═══════════════════════════════════════════════════════════════════════

def _les_cache() -> Optional[list]:
    """Les cached insider-data hvis den er fersk nok."""
    try:
        if not os.path.exists(INSIDER_CACHE_FILE):
            return None

        with open(INSIDER_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        timestamp = cache.get("timestamp", 0)
        if time.time() - timestamp > CACHE_TTL_SEKUNDER:
            logger.debug("Insider-cache utløpt")
            return None

        return cache.get("handler", [])
    except Exception as e:
        logger.debug("Kunne ikke lese insider-cache: %s", e)
        return None


def _skriv_cache(handler: list) -> None:
    """Skriv insider-data til cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache = {
            "timestamp": time.time(),
            "handler": handler,
        }
        with open(INSIDER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.debug("Insider-cache oppdatert (%d handler)", len(handler))
    except Exception as e:
        logger.warning("Kunne ikke skrive insider-cache: %s", e)


def tøm_cache() -> None:
    """Slett insider-cache manuelt."""
    try:
        if os.path.exists(INSIDER_CACHE_FILE):
            os.remove(INSIDER_CACHE_FILE)
            logger.info("Insider-cache slettet")
    except Exception as e:
        logger.warning("Kunne ikke slette insider-cache: %s", e)
