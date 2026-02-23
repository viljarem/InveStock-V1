# data.py
import yfinance as yf
import pandas as pd
import os
import json
import time
import datetime
from config import DATA_FILE, MARKET_DATA_FILE, HISTORICAL_YEARS, TICKER_LIST_FILE
from config import BRENT_DATA_FILE, USDNOK_DATA_FILE, SEKTOR_MAPPING_FILE
from log_config import get_logger

logger = get_logger(__name__)


def _yf_download_med_retry(max_retries=3, **kwargs):
    """
    Wrapper rundt yf.download med exponential backoff.
    Returnerer DataFrame (kan være tom ved gjentatte feil).
    """
    label = kwargs.get('tickers', kwargs.get('ticker', '?'))
    if isinstance(label, list):
        label = f"{len(label)} tickers"
    
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(**kwargs)
            if not df.empty:
                return df
            logger.warning(f"yfinance returnerte tom DataFrame for {label} (forsøk {attempt}/{max_retries})")
        except Exception as e:
            logger.warning(f"yfinance feil for {label} (forsøk {attempt}/{max_retries}): {e}")
        
        if attempt < max_retries:
            delay = 2 ** attempt  # 2s, 4s
            logger.info(f"Venter {delay}s før neste forsøk...")
            time.sleep(delay)
    
    logger.error(f"Alle {max_retries} forsøk feilet for {label}")
    return pd.DataFrame()

def hent_oppdaterte_tickers():
    """
    Henter selskaper fra Oslo Børs. 
    Prioriterer 'mine_tickere.txt' i rotmappen hvis den eksisterer.
    Returnerer en alfabetisk sortert liste.
    """
    filsti_bruker = "mine_tickere.txt"
    tickers = []
    
    if os.path.exists(filsti_bruker):
        try:
            with open(filsti_bruker, "r") as f:
                # Les rader, fjern uønsket mellomrom og ignorer tomme linjer
                tickers = [line.strip().upper() for line in f if line.strip()]
            
            # Sikrer .OL suffiks for Oslo Børs hvis det mangler
            tickers = [t if t.endswith(".OL") else f"{t}.OL" for t in tickers]
        except Exception:
            tickers = []
    
    # Hvis filen ikke finnes eller er tom, benyttes standardlisten
    if not tickers:
        tickers = [
            "EQNR.OL", "DNB.OL", "NHY.OL", "ORK.OL", "MOWI.OL", "YAR.OL", 
            "TEL.OL", "GJF.OL", "STB.OL", "AKRBP.OL", "TOM.OL", "SALM.OL",
            "KOG.OL", "SUBC.OL", "NAS.OL", "FRO.OL", "MPCC.OL", "VAR.OL",
            "HAFNI.OL", "LSG.OL", "BAKKA.OL", "NOD.OL", "PGS.OL", "RECSI.OL",
            "SCATC.OL", "ADE.OL", "BOUV.OL", "KIT.OL", "BORR.OL", "ELK.OL",
            "ENTRA.OL", "EPR.OL", "FLNG.OL", "GOGL.OL", "HSHP.OL", "IDEX.OL",
            "KID.OL", "KOA.OL", "LER.OL", "LINK.OL", "MPC.OL", "MULTI.OL",
            "NEL.OL", "NORBIT.OL", "NANOV.OL", "OKEA.OL", "OTOVO.OL", "PEXIP.OL",
            "PHOTO.OL", "PROTECTOR.OL", "SRBANK.OL", "SSG.OL", "TGS.OL", "VEI.OL",
            "VOLUE.OL", "WWI.OL", "XXL.OL", "BELCO.OL", "BWLPG.OL", "CADELER.OL"
        ]
    
    # Sorterer listen alfabetisk for konsistent visning i menyer
    tickers = sorted(list(set(tickers)))
    
    # Lagrer den aktive listen som brukes til en lokal referansefil
    try:
        with open(TICKER_LIST_FILE, "w") as f:
            f.write("\n".join(tickers))
    except Exception:
        pass
        
    return tickers

def last_ned_data():
    """Laster ned 10 aars historikk for markedet og indekser."""
    tickers = hent_oppdaterte_tickers()
    start_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORICAL_YEARS*365)).strftime('%Y-%m-%d')
    
    # 1. Last ned enkeltaksjer
    try:
        # Data hentes i bulk for optimal hastighet
        data = _yf_download_med_retry(
            tickers=tickers,
            start=start_date,
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False
        )
        df_list = []

        if isinstance(data.columns, pd.MultiIndex):
            tilgjengelige_tickers = set(data.columns.get_level_values(0))
        else:
            tilgjengelige_tickers = set()

        for ticker in tickers:
            try:
                if ticker not in tilgjengelige_tickers:
                    continue

                temp_df = data[ticker].dropna(how='all').copy()
                if temp_df.empty:
                    continue

                temp_df['Ticker'] = ticker
                df_list.append(temp_df)
            except (KeyError, Exception):
                continue

        if df_list:
            full_df = pd.concat(df_list)
            full_df.reset_index(inplace=True)
            full_df.to_parquet(DATA_FILE, engine='pyarrow', compression='snappy')
    except Exception:
        pass

    # 2. Last ned Benchmark (OSEBX)
    try:
        benchmark = _yf_download_med_retry(tickers="^OSEBX", start=start_date, auto_adjust=True)
        if not benchmark.empty:
            benchmark.to_parquet(MARKET_DATA_FILE, engine='pyarrow')
    except Exception:
        pass
    
    # 3. Oppdater sektor-mapping for alle tickers
    try:
        oppdater_sektor_mapping(tickers)
    except Exception as e:
        logger.warning(f"Kunne ikke oppdatere sektor-mapping: {e}")
        
    return True

def hent_data():
    """Henter aksjedata fra disk. Validerer integritet."""
    if not os.path.exists(DATA_FILE):
        logger.info("Datafil mangler, laster ned...")
        last_ned_data()
    
    try:
        df = pd.read_parquet(DATA_FILE)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)

        # Sikre numeriske typer for indikator-/signalberegning
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        required_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols)

        df.sort_index(inplace=True)
        
        # === INTEGRITETSJEKK ===
        if df.empty:
            logger.error("Datafilen er tom etter filtrering")
            return pd.DataFrame()
        
        # Sjekk for negative/null-priser
        if 'Close' in df.columns:
            neg_count = (df['Close'] <= 0).sum()
            if neg_count > 0:
                logger.warning(f"Fjerner {neg_count} rader med ugyldig pris (≤ 0)")
                df = df[df['Close'] > 0]
        
        # Sjekk dato-range
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            dato_range = (df.index.max() - df.index.min()).days
            if dato_range < 30:
                logger.warning(f"Bare {dato_range} dager med data — kan være ufullstendig")
        
        n_tickers = df['Ticker'].nunique() if 'Ticker' in df.columns else 0
        logger.debug(f"Lastet {len(df)} rader, {n_tickers} tickers, {df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else '?'} til {df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else '?'}")
        
        return df
    except Exception as e:
        logger.error(f"Feil ved lesing av data: {e}")
        return pd.DataFrame()

def hent_markedsdata_df(force_refresh=False):
    """
    Henter benchmark-data (Oslo Børs).
    Prøver flere alternative tickers siden Yahoo Finance kan være ustabil.
    """
    should_download = force_refresh or not os.path.exists(MARKET_DATA_FILE)
    
    # Sjekk om eksisterende data er utdatert
    if not should_download and os.path.exists(MARKET_DATA_FILE):
        try:
            df_check = pd.read_parquet(MARKET_DATA_FILE)
            if isinstance(df_check.columns, pd.MultiIndex):
                df_check.columns = df_check.columns.get_level_values(0)
            if not df_check.empty:
                last_date = pd.to_datetime(df_check.index.max())
                days_old = (datetime.datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
                if days_old > 3:  # Data er mer enn 3 dager gammel
                    should_download = True
                    logger.info(f"Markedsdata er {days_old} dager gammel, laster ned på nytt")
        except Exception as e:
            logger.warning(f"Feil ved sjekk av fil: {e}")
            should_download = True
    
    if should_download:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORICAL_YEARS*365)).strftime('%Y-%m-%d')
        
        # Liste over alternative tickers for Oslo Børs
        # OSEBX.OL fungerer! (testet 2024)
        alternative_tickers = [
            "OSEBX.OL",    # Oslo Børs Benchmark Index ✓
            "OBX.OL",      # OBX Index ✓
            "^OSEAX",      # Oslo Børs All-share Index ✓
            "EQNR.OL",     # Equinor som proxy ✓
        ]
        
        benchmark = pd.DataFrame()
        used_ticker = None
        
        for ticker in alternative_tickers:
            try:
                logger.info(f"Prøver å laste ned {ticker}...")
                temp = _yf_download_med_retry(tickers=ticker, start=start_date, auto_adjust=True, progress=False, max_retries=2)
                
                if not temp.empty:
                    # Sjekk at data er oppdatert (ikke eldre enn 7 dager)
                    if isinstance(temp.columns, pd.MultiIndex):
                        temp.columns = temp.columns.get_level_values(0)
                    
                    last_date = pd.to_datetime(temp.index.max())
                    days_old = (datetime.datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
                    
                    if days_old <= 7:
                        benchmark = temp
                        used_ticker = ticker
                        logger.info(f"Bruker {ticker}: {temp.index.min().strftime('%Y-%m-%d')} til {temp.index.max().strftime('%Y-%m-%d')}")
                        break
                    else:
                        logger.info(f"{ticker} er {days_old} dager gammel, prøver neste...")
            except Exception as e:
                logger.error(f"Feil med {ticker}: {e}")
                continue
        
        # Hvis ingen ticker fungerte, lag syntetisk indeks fra våre aksjer
        if benchmark.empty:
            logger.warning("Ingen indeks-ticker fungerte, lager syntetisk indeks fra aksjedata...")
            benchmark = _lag_syntetisk_indeks(start_date)
            used_ticker = "SYNTETISK"
        
        if not benchmark.empty:
            # Håndter MultiIndex kolonner fra yfinance
            if isinstance(benchmark.columns, pd.MultiIndex):
                benchmark.columns = benchmark.columns.get_level_values(0)
            
            # Sørg for at indeks er ren datetime
            benchmark.index = pd.to_datetime(benchmark.index)
            if benchmark.index.tz is not None:
                benchmark.index = benchmark.index.tz_localize(None)
            
            benchmark.to_parquet(MARKET_DATA_FILE, engine='pyarrow')
            logger.info(f"Lagret markedsdata ({used_ticker}): {benchmark.index.min().strftime('%Y-%m-%d')} til {benchmark.index.max().strftime('%Y-%m-%d')}")
        else:
            logger.error("Kunne ikke hente markedsdata fra noen kilde")
    
    # Les fra fil
    try:
        df = pd.read_parquet(MARKET_DATA_FILE)
        
        # Håndter MultiIndex kolonner
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Sørg for at indeks er datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
        
        # Fjern timezone hvis den finnes
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Feil ved lesing av markedsdata: {e}", exc_info=True)
        return pd.DataFrame()


def _lag_syntetisk_indeks(start_date: str) -> pd.DataFrame:
    """
    Lager en syntetisk markedsindeks basert på de største aksjene.
    Brukes som fallback hvis ingen indeks-ticker fungerer.
    """
    # Bruk de største og mest likvide aksjene som proxy
    proxy_tickers = ["EQNR.OL", "DNB.OL", "NHY.OL", "ORK.OL", "MOWI.OL", "TEL.OL"]
    
    try:
        data = _yf_download_med_retry(tickers=proxy_tickers, start=start_date, auto_adjust=True, progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Hent Close-priser
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close']
        else:
            closes = data[['Close']]
        
        # Normaliser hver aksje til 100 ved start og beregn gjennomsnitt
        normalized = closes / closes.iloc[0] * 100
        synthetic_index = normalized.mean(axis=1)
        
        # Lag DataFrame med samme struktur som yfinance output
        result = pd.DataFrame({
            'Open': synthetic_index,
            'High': synthetic_index,
            'Low': synthetic_index,
            'Close': synthetic_index,
            'Volume': 0
        }, index=synthetic_index.index)
        
        return result
    except Exception as e:
        logger.error(f"Feil ved laging av syntetisk indeks: {e}")
        return pd.DataFrame()

def filtrer_likvide_aksjer(df, min_dagsomsetning=500000):
    """Ekskluderer illikvide aksjer basert på 20-dagers snittomsetning."""
    if df.empty:
        return df

    df_c = df.copy()

    for col in ['Close', 'Volume']:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')

    df_c = df_c.dropna(subset=['Close', 'Volume'])
    if df_c.empty:
        return df

    df_c['Turnover'] = df_c['Close'] * df_c['Volume']

    # Raskere enn groupby-apply-lambda: ta siste 20 per ticker, så mean
    df_last20 = df_c.sort_index().groupby('Ticker', group_keys=False).tail(20)
    avg_turnover = df_last20.groupby('Ticker')['Turnover'].mean()

    likvide = avg_turnover[avg_turnover >= min_dagsomsetning].index.tolist()
    if not likvide:
        return df

    return df[df['Ticker'].isin(likvide)]


def ticker_til_navn(ticker):
    """Konverterer ticker til selskapsnavn."""
    # Enkel mapping for de vanligste aksjene
    navn_mapping = {
        "EQNR.OL": "Equinor",
        "DNB.OL": "DNB Bank",
        "NHY.OL": "Norsk Hydro",
        "ORK.OL": "Orkla",
        "MOWI.OL": "Mowi",
        "YAR.OL": "Yara International",
        "TEL.OL": "Telenor",
        "GJF.OL": "Gjensidige Forsikring",
        "STB.OL": "Storebrand",
        "AKRBP.OL": "Aker BP",
        "TOM.OL": "Tomra Systems",
        "SALM.OL": "SalMar",
        "KOG.OL": "Kongsberg Gruppen",
        "SUBC.OL": "Subsea 7",
        "NAS.OL": "Norwegian Air Shuttle",
        "FRO.OL": "Frontline",
        "MPCC.OL": "MPC Container Ships",
        "VAR.OL": "Vår Energi",
        "HAFNI.OL": "Hafnia",
        "LSG.OL": "Lerøy Seafood",
        "BAKKA.OL": "Bakkafrost",
        "NOD.OL": "Nordic Semiconductor",
        "PGS.OL": "PGS",
        "RECSI.OL": "REC Silicon",
        "SCATC.OL": "Scatec",
        "ADE.OL": "Adevinta",
        "BOUV.OL": "Bouvet",
        "KIT.OL": "Kitron",
        "BORR.OL": "Borr Drilling",
        "ELK.OL": "Elkem",
        "NEL.OL": "Nel",
        "OTOVO.OL": "Otovo",
        "PEXIP.OL": "Pexip",
        "VOLUE.OL": "Volue",
        "XXL.OL": "XXL",
        "CADELER.OL": "Cadeler",
        "BWLPG.OL": "BW LPG",
    }
    
    # Returner navnet hvis det finnes, ellers ticker uten .OL
    return navn_mapping.get(ticker, ticker.replace(".OL", ""))

def hent_siste_oppdateringstid():
    """Returnerer siste oppdateringstid basert på lokale datafiler."""
    filer = [DATA_FILE, MARKET_DATA_FILE]
    eksisterende = [f for f in filer if os.path.exists(f)]
    if not eksisterende:
        return None
    siste_mtime = max(os.path.getmtime(f) for f in eksisterende)
    return datetime.datetime.fromtimestamp(siste_mtime)


# =============================================================================
# MAKRO-DATA: Oljepris (Brent) og USDNOK
# =============================================================================

def _hent_makro_serie(ticker_yf, parquet_fil, label, max_dager_gammel=3):
    """
    Generisk funksjon for å hente og cache makro-tidsserie.
    Laster ned på nytt hvis data er eldre enn max_dager_gammel.
    """
    should_download = not os.path.exists(parquet_fil)
    
    if not should_download:
        try:
            df_check = pd.read_parquet(parquet_fil)
            if isinstance(df_check.columns, pd.MultiIndex):
                df_check.columns = df_check.columns.get_level_values(0)
            if not df_check.empty:
                last_date = pd.to_datetime(df_check.index.max())
                days_old = (datetime.datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
                if days_old > max_dager_gammel:
                    logger.info(f"{label}-data er {days_old} dager gammel, oppdaterer")
                    should_download = True
        except Exception:
            should_download = True
    
    if should_download:
        try:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORICAL_YEARS * 365)).strftime('%Y-%m-%d')
            logger.info(f"Laster ned {label} ({ticker_yf})...")
            df = _yf_download_med_retry(tickers=ticker_yf, start=start_date, auto_adjust=True, progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_parquet(parquet_fil, engine='pyarrow')
                logger.info(f"Lagret {label}: {df.index.min().strftime('%Y-%m-%d')} til {df.index.max().strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.warning(f"Kunne ikke hente {label}: {e}")
    
    # Les fra fil
    try:
        df = pd.read_parquet(parquet_fil)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def hent_brent_data(force_refresh=False):
    """Henter Brent Crude oljepris (BZ=F)."""
    if force_refresh and os.path.exists(BRENT_DATA_FILE):
        os.remove(BRENT_DATA_FILE)
    return _hent_makro_serie("BZ=F", BRENT_DATA_FILE, "Brent Crude")


def hent_usdnok_data(force_refresh=False):
    """Henter USD/NOK-kurs (USDNOK=X)."""
    if force_refresh and os.path.exists(USDNOK_DATA_FILE):
        os.remove(USDNOK_DATA_FILE)
    return _hent_makro_serie("USDNOK=X", USDNOK_DATA_FILE, "USD/NOK")


# =============================================================================
# DYNAMISK SEKTOR-MAPPING
# =============================================================================

# Mapper yfinance engelske sektor-navn til norske
_SEKTOR_OVERSETTELSE = {
    "Energy": "Energi",
    "Basic Materials": "Materialer",
    "Industrials": "Industri",
    "Consumer Cyclical": "Konsum",
    "Consumer Defensive": "Konsum",
    "Healthcare": "Helse",
    "Financial Services": "Finans",
    "Technology": "Teknologi",
    "Communication Services": "Telekom",
    "Utilities": "Fornybar",
    "Real Estate": "Eiendom",
}


def _last_sektor_cache() -> dict:
    """Laster cached sektor-mapping fra disk."""
    if os.path.exists(SEKTOR_MAPPING_FILE):
        try:
            with open(SEKTOR_MAPPING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Kunne ikke lese sektor-cache: {e}")
    return {}


def _lagre_sektor_cache(cache: dict) -> None:
    """Lagrer sektor-cache til disk."""
    try:
        os.makedirs(os.path.dirname(SEKTOR_MAPPING_FILE), exist_ok=True)
        with open(SEKTOR_MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.debug(f"Sektor-cache lagret: {len(cache)} tickers")
    except Exception as e:
        logger.warning(f"Kunne ikke lagre sektor-cache: {e}")


def oppdater_sektor_mapping(tickers: list) -> dict:
    """
    Henter sektorer fra yfinance for tickers som mangler i cache.
    
    Args:
        tickers: Liste med ticker-symboler (f.eks. ['EQNR.OL', 'DNB.OL'])
        
    Returns:
        Oppdatert sektor-mapping {ticker: norsk_sektor}
    """
    cache = _last_sektor_cache()
    
    # Finn tickers som mangler i cache
    mangler = [t for t in tickers if t not in cache]
    
    if not mangler:
        logger.debug(f"Alle {len(tickers)} tickers har sektor i cache")
        return cache
    
    logger.info(f"Henter sektor-info fra yfinance for {len(mangler)} nye tickers...")
    
    for ticker in mangler:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            sektor_en = info.get('sector', '')
            
            if sektor_en:
                sektor_no = _SEKTOR_OVERSETTELSE.get(sektor_en, sektor_en)
                cache[ticker] = sektor_no
                logger.debug(f"  {ticker}: {sektor_en} → {sektor_no}")
            else:
                # Prøv industri som fallback
                industri = info.get('industry', '')
                if industri:
                    # Map vanlige industrier til sektorer
                    if any(w in industri.lower() for w in ['oil', 'gas', 'energy', 'drill']):
                        cache[ticker] = "Energi"
                    elif any(w in industri.lower() for w in ['ship', 'marine', 'tanker', 'lng']):
                        cache[ticker] = "Shipping"
                    elif any(w in industri.lower() for w in ['salmon', 'fish', 'seafood', 'aqua']):
                        cache[ticker] = "Sjømat"
                    elif any(w in industri.lower() for w in ['bank', 'insur', 'financ']):
                        cache[ticker] = "Finans"
                    elif any(w in industri.lower() for w in ['software', 'tech', 'semi', 'it ']):
                        cache[ticker] = "Teknologi"
                    elif any(w in industri.lower() for w in ['solar', 'wind', 'renew', 'hydrogen']):
                        cache[ticker] = "Fornybar"
                    else:
                        cache[ticker] = "Annet"
                    logger.debug(f"  {ticker}: industri '{industri}' → {cache[ticker]}")
                else:
                    cache[ticker] = "Annet"
                    logger.debug(f"  {ticker}: ingen sektor/industri funnet → Annet")
        except Exception as e:
            logger.warning(f"  Kunne ikke hente sektor for {ticker}: {e}")
            # Ikke legg til i cache — prøv igjen neste gang
    
    _lagre_sektor_cache(cache)
    return cache


def hent_sektor_mapping() -> dict:
    """
    Returnerer komplett sektor-mapping.
    Laster fra cache, ingen yfinance-kall.
    """
    return _last_sektor_cache()