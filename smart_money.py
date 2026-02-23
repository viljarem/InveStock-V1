"""
Smart Money Flow — Detekter institusjonell vs retail aktivitet.

Konsept (Wall Street-klassiker):
  - Retail handler tidlig på dagen (emosjonelt, reaksjon på nyheter)
  - Institusjonelle handler sent (informert, strategisk)
  - SMI = kumulativ sum av (siste-time-endring − første-time-endring)
  - Divergens mellom SMI og pris gir sterke signaler

To modi:
  1) Intradag-basert (1h bar fra yfinance, ~60 dagers historikk)
  2) Daglig proxy (Chaikin A/D + OBV, ubegrenset historikk)
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

logger = logging.getLogger("InveStock.SmartMoney")


# ═══════════════════════════════════════════════════════════════════════
# 1.  DATA-HENTING (intradag)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=300)
def hent_intraday_data(ticker: str, interval: str = "1h",
                       period: str = "60d") -> pd.DataFrame:
    """
    Henter intradag-data fra yfinance for Oslo Børs.
    
    Args:
        ticker: Yahoo Finance ticker (f.eks. 'EQNR.OL')
        interval: '1h' for best SMI-beregning
        period: '60d' maks for 1h data fra yfinance
    
    Returns:
        DataFrame med OHLCV og datetime-index i Europe/Oslo tz
    """
    try:
        ticker_yf = ticker if ticker.endswith('.OL') else f"{ticker}.OL"
        df = yf.download(
            ticker_yf, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=False
        )
        if df.empty:
            return pd.DataFrame()
        
        # Flatt MultiIndex-kolonner hvis nødvendig
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Konverter til Oslo-tid
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Europe/Oslo')
        
        df = df.dropna(subset=['Close'])
        return df
    
    except Exception as e:
        logger.warning(f"Kunne ikke hente intradag-data for {ticker}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# 2.  SMI FRA INTRADAG-DATA
# ═══════════════════════════════════════════════════════════════════════

def beregn_smi_intradag(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Beregner Smart Money Index fra intradag (1h) data.
    
    Formel per dag:
      early_return = (close_første_bar - open_første_bar) / open_første_bar
      late_return  = (close_siste_bar - open_siste_bar) / open_siste_bar
      smi_daily    = late_return - early_return
      SMI          = cumsum(smi_daily)
    
    Oslo Børs: 09:00-16:25 → Første bar ≈ 09:00, siste ≈ 15:00 eller 16:00
    
    Returns:
        DataFrame index=dato, cols=[smi_daily, smi, smi_sma10, smi_sma20,
                                    early_return, late_return]
    """
    if df_intraday.empty or len(df_intraday) < 10:
        return pd.DataFrame()
    
    df = df_intraday.copy()
    df['date'] = df.index.date
    
    daglige = []
    for dato, dag_df in df.groupby('date'):
        if len(dag_df) < 2:
            continue
        
        # Første bar (tidligst) og siste bar (senest) på dagen
        første = dag_df.iloc[0]
        siste = dag_df.iloc[-1]
        
        open_first = float(første['Open'])
        close_first = float(første['Close'])
        open_last = float(siste['Open'])
        close_last = float(siste['Close'])
        
        if open_first <= 0 or open_last <= 0:
            continue
        
        early_return = (close_first - open_first) / open_first
        late_return = (close_last - open_last) / open_last
        
        # SMI: sen handel minus tidlig handel
        smi_daily = late_return - early_return
        
        # Daglig close og volum for kontekst
        daglige.append({
            'date': pd.Timestamp(dato),
            'smi_daily': smi_daily,
            'early_return': early_return,
            'late_return': late_return,
            'close': close_last,
            'volume': float(dag_df['Volume'].sum()),
        })
    
    if not daglige:
        return pd.DataFrame()
    
    result = pd.DataFrame(daglige).set_index('date')
    result.index = pd.DatetimeIndex(result.index)
    
    # Kumulativ SMI
    result['smi'] = result['smi_daily'].cumsum()
    
    # Glidende snitt for trendanalyse
    result['smi_sma10'] = result['smi'].rolling(10, min_periods=5).mean()
    result['smi_sma20'] = result['smi'].rolling(20, min_periods=10).mean()
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3.  DAGLIG PROXY (Chaikin A/D + OBV)
# ═══════════════════════════════════════════════════════════════════════

def beregn_smi_daglig_proxy(df: pd.DataFrame, perioder: int = 20) -> pd.DataFrame:
    """
    Proxy for Smart Money Flow basert på daglige data.
    
    Bruker to etablerte institusjonelle indikatorer:
    1) Chaikin Money Flow (CMF): Viser om volum akkumuleres nær høy/lav
       - CLV = ((Close - Low) - (High - Close)) / (High - Low)
       - CMF = sum(CLV * Volume, 20) / sum(Volume, 20)
    2) OBV-divergens: On-Balance Volume vs pris-trend
    
    Kombinert gir disse en daglig «smart money»-score.
    
    Returns:
        DataFrame med cols=[cmf, obv, obv_sma20, sm_proxy, sm_proxy_sma10]
    """
    if df is None or len(df) < perioder + 10:
        return pd.DataFrame()
    
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    
    result = pd.DataFrame(index=df.index)
    
    # 1) Chaikin Money Flow
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    clv = clv.fillna(0)
    
    money_flow_volume = clv * volume
    result['cmf'] = (
        money_flow_volume.rolling(perioder, min_periods=perioder).sum() /
        volume.rolling(perioder, min_periods=perioder).sum()
    )
    
    # 2) On-Balance Volume
    obv_direction = np.sign(close.diff())
    result['obv'] = (obv_direction * volume).cumsum()
    result['obv_sma20'] = result['obv'].rolling(20, min_periods=10).mean()
    
    # 3) Kombinert proxy-score (-1 til +1 skala)
    # CMF er allerede -1 til +1
    # Normaliser OBV-trend som avvik fra SMA
    obv_norm = (result['obv'] - result['obv_sma20']) / result['obv_sma20'].abs().replace(0, 1)
    obv_norm = obv_norm.clip(-2, 2) / 2  # Skaler til ~[-1, 1]
    
    # Vektet kombinasjon: CMF er mer pålitelig
    result['sm_proxy'] = 0.6 * result['cmf'].fillna(0) + 0.4 * obv_norm.fillna(0)
    result['sm_proxy_sma10'] = result['sm_proxy'].rolling(10, min_periods=5).mean()
    
    return result


# ═══════════════════════════════════════════════════════════════════════
# 4.  DIVERGENS-DETEKSJON
# ═══════════════════════════════════════════════════════════════════════

def _trend_retning(serie: pd.Series, vindu: int = 10) -> str:
    """Returnerer 'up', 'down', eller 'flat' basert på lineær regresjon over vindu."""
    if len(serie) < vindu:
        return 'flat'
    
    s = serie.iloc[-vindu:].dropna()
    if len(s) < max(5, vindu // 2):
        return 'flat'
    
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    
    # Lineær regresjon: helling
    slope = np.polyfit(x, y, 1)[0]
    
    # Normaliser helling relativt til spredningen
    std = np.std(y)
    if std == 0:
        return 'flat'
    
    norm_slope = slope / std * len(s)
    
    if norm_slope > 0.5:
        return 'up'
    elif norm_slope < -0.5:
        return 'down'
    return 'flat'


def finn_divergens(pris_serie: pd.Series, smi_serie: pd.Series,
                   vindu: int = 10) -> dict:
    """
    Detekterer divergens mellom pris og Smart Money-indikator.
    
    Bullish divergens: pris ↓ + SMI ↑ = institusjonell akkumulering
    Bearish divergens: pris ↑ + SMI ↓ = retail-drevet (svakt rally)
    
    Returns:
        {
            'type': 'bullish' | 'bearish' | 'confirming' | 'neutral',
            'emoji': str,
            'beskrivelse': str,
            'score_justering': int (-10 til +10),
            'pris_trend': str,
            'smi_trend': str,
        }
    """
    pris_trend = _trend_retning(pris_serie, vindu)
    smi_trend = _trend_retning(smi_serie, vindu)
    
    if pris_trend == 'down' and smi_trend == 'up':
        return {
            'type': 'bullish',
            'emoji': '🟢💰',
            'beskrivelse': 'Bullish divergens — Smart Money akkumulerer',
            'score_justering': 10,
            'pris_trend': pris_trend,
            'smi_trend': smi_trend,
        }
    elif pris_trend == 'up' and smi_trend == 'down':
        return {
            'type': 'bearish',
            'emoji': '🔴💰',
            'beskrivelse': 'Bearish divergens — Smart Money distribuerer',
            'score_justering': -10,
            'pris_trend': pris_trend,
            'smi_trend': smi_trend,
        }
    elif pris_trend == smi_trend and pris_trend != 'flat':
        # Pris og SMI bekrefter hverandre
        juster = 5 if pris_trend == 'up' else -5
        return {
            'type': 'confirming',
            'emoji': '✅💰' if pris_trend == 'up' else '⚠️💰',
            'beskrivelse': f'Bekreftet trend — pris og Smart Money begge {pris_trend}',
            'score_justering': juster,
            'pris_trend': pris_trend,
            'smi_trend': smi_trend,
        }
    else:
        return {
            'type': 'neutral',
            'emoji': '⚪💰',
            'beskrivelse': 'Ingen klar Smart Money-divergens',
            'score_justering': 0,
            'pris_trend': pris_trend,
            'smi_trend': smi_trend,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.  KOMPLETT ANALYSE
# ═══════════════════════════════════════════════════════════════════════

def analyser_smart_money(ticker: str, df_daglig: pd.DataFrame = None,
                         bruk_intradag: bool = True) -> dict:
    """
    Full Smart Money-analyse for en ticker.
    
    Prøver intradag-data først (mest presis). Faller tilbake til daglig proxy.
    
    Args:
        ticker: Yahoo Finance ticker
        df_daglig: Daglig OHLCV DataFrame (brukes for proxy)
        bruk_intradag: Om intradag-data skal forsøkes
    
    Returns:
        {
            'kilde': 'intradag' | 'daglig_proxy',
            'divergens': dict fra finn_divergens(),
            'smi_verdi': float (siste SMI eller proxy-verdi),
            'smi_trend': 'up' | 'down' | 'flat',
            'emoji': str,
            'score_justering': int,
            'detaljer': dict med ekstra info,
        }
    """
    # Forsøk 1: Intradag-basert SMI
    if bruk_intradag:
        df_intra = hent_intraday_data(ticker, interval="1h", period="60d")
        if not df_intra.empty and len(df_intra) >= 20:
            smi_df = beregn_smi_intradag(df_intra)
            if not smi_df.empty and len(smi_df) >= 10:
                divergens = finn_divergens(
                    smi_df['close'], smi_df['smi'], vindu=10
                )
                smi_trend = _trend_retning(smi_df['smi'], 10)
                
                return {
                    'kilde': 'intradag',
                    'divergens': divergens,
                    'smi_verdi': float(smi_df['smi'].iloc[-1]),
                    'smi_trend': smi_trend,
                    'emoji': divergens['emoji'],
                    'score_justering': divergens['score_justering'],
                    'detaljer': {
                        'dager_analysert': len(smi_df),
                        'smi_sma10': float(smi_df['smi_sma10'].iloc[-1]) 
                            if pd.notna(smi_df['smi_sma10'].iloc[-1]) else None,
                        'smi_sma20': float(smi_df['smi_sma20'].iloc[-1])
                            if pd.notna(smi_df['smi_sma20'].iloc[-1]) else None,
                        'early_return_snitt': float(smi_df['early_return'].mean()),
                        'late_return_snitt': float(smi_df['late_return'].mean()),
                    },
                    'smi_df': smi_df,
                }
    
    # Forsøk 2: Daglig proxy
    if df_daglig is not None and len(df_daglig) >= 40:
        proxy_df = beregn_smi_daglig_proxy(df_daglig)
        if not proxy_df.empty and len(proxy_df.dropna(subset=['sm_proxy'])) >= 10:
            pris = df_daglig['Close'].astype(float)
            sm = proxy_df['sm_proxy'].dropna()
            
            # Match indekser
            felles = pris.index.intersection(sm.index)
            if len(felles) >= 10:
                divergens = finn_divergens(
                    pris.loc[felles], sm.loc[felles], vindu=10
                )
                smi_trend = _trend_retning(sm, 10)
                
                return {
                    'kilde': 'daglig_proxy',
                    'divergens': divergens,
                    'smi_verdi': float(sm.iloc[-1]),
                    'smi_trend': smi_trend,
                    'emoji': divergens['emoji'],
                    'score_justering': divergens['score_justering'],
                    'detaljer': {
                        'cmf': float(proxy_df['cmf'].iloc[-1])
                            if pd.notna(proxy_df['cmf'].iloc[-1]) else None,
                        'obv_trend': _trend_retning(proxy_df['obv'], 20),
                    },
                    'proxy_df': proxy_df,
                }
    
    # Ingen data tilgjengelig
    return {
        'kilde': 'ingen',
        'divergens': {
            'type': 'neutral', 'emoji': '⚪💰',
            'beskrivelse': 'Ingen Smart Money-data tilgjengelig',
            'score_justering': 0, 'pris_trend': 'flat', 'smi_trend': 'flat',
        },
        'smi_verdi': 0.0,
        'smi_trend': 'flat',
        'emoji': '⚪💰',
        'score_justering': 0,
        'detaljer': {},
    }


# ═══════════════════════════════════════════════════════════════════════
# 6.  SCANNER-OPTIMALISERT (rask, daglig-proxy only)
# ═══════════════════════════════════════════════════════════════════════

def beregn_smi_for_scanner(df_daglig: pd.DataFrame) -> dict:
    """
    Rask Smart Money-beregning for Scanner (bruker kun daglig proxy).
    
    Unngår intradag-API-kall (60 tickers × yfinance = for tregt).
    Bruker Chaikin Money Flow + OBV fra eksisterende daglige data.
    
    Returns:
        {
            'emoji': str,
            'score_justering': int (-10 til +10),
            'type': str,
            'cmf': float,
        }
    """
    if df_daglig is None or len(df_daglig) < 40:
        return {
            'emoji': '⚪',
            'score_justering': 0,
            'type': 'neutral',
            'cmf': 0.0,
        }
    
    proxy_df = beregn_smi_daglig_proxy(df_daglig)
    if proxy_df.empty:
        return {
            'emoji': '⚪',
            'score_justering': 0,
            'type': 'neutral',
            'cmf': 0.0,
        }
    
    pris = df_daglig['Close'].astype(float)
    sm = proxy_df['sm_proxy'].dropna()
    
    felles = pris.index.intersection(sm.index)
    if len(felles) < 10:
        return {
            'emoji': '⚪',
            'score_justering': 0,
            'type': 'neutral',
            'cmf': 0.0,
        }
    
    divergens = finn_divergens(pris.loc[felles], sm.loc[felles], vindu=10)
    
    cmf_val = float(proxy_df['cmf'].iloc[-1]) if pd.notna(proxy_df['cmf'].iloc[-1]) else 0.0
    
    # Forenklet emoji for scanner-kolonne
    emoji_map = {
        'bullish': '🟢',
        'bearish': '🔴',
        'confirming': '✅' if divergens.get('pris_trend') == 'up' else '⚠️',
        'neutral': '⚪',
    }
    
    return {
        'emoji': emoji_map.get(divergens['type'], '⚪'),
        'score_justering': divergens['score_justering'],
        'type': divergens['type'],
        'cmf': round(cmf_val, 3),
    }
