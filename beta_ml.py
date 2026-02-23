import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import fbeta_score
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from log_config import get_logger

logger = get_logger(__name__)
import pickle
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import logic

# Import portfolio for portefølje-integrasjon
try:
    import portfolio
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

# Import regime model for markedstemperatur-integrasjon
try:
    import regime_model
    import data
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

# Regime feature cache (for feature engineering)
_REGIME_FEATURE_CACHE = {
    'ts': None,
    'df_regimes': None,
    'n_regimes': 3
}
_REGIME_CACHE_TTL_HOURS = 6

# Import Lightweight Charts
try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LWC_INSTALLED = True
except ImportError:
    LWC_INSTALLED = False

# Import Fundamental Data for fundamental analyse
try:
    import fundamental_data
    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_AVAILABLE = False

# Import Insider Monitor for meldepliktige handler
try:
    import insider_monitor
    INSIDER_AVAILABLE = True
except ImportError:
    INSIDER_AVAILABLE = False

# =============================================================================
# MODELL CACHING SYSTEM
# =============================================================================

CACHE_DIR = "data_storage/ml_models"
CACHE_MAX_AGE_HOURS = 24  # Cache utløper etter 24 timer

def _get_cache_key(ticker: str, horisont: int, target_pct: float, stop_pct: float, last_date: str) -> str:
    """Genererer unik cache-nøkkel basert på parametere."""
    key_str = f"{ticker}_{horisont}_{target_pct:.4f}_{stop_pct:.4f}_{last_date}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]

def _get_cache_path(cache_key: str) -> str:
    """Returnerer filsti for cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"model_{cache_key}.pkl")

def _is_cache_valid(cache_path: str) -> bool:
    """Sjekker om cache-fil er gyldig (eksisterer og ikke utløpt)."""
    if not os.path.exists(cache_path):
        return False
    
    # Sjekk alder
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time
    
    return age < timedelta(hours=CACHE_MAX_AGE_HOURS)

def _load_cached_model(cache_path: str):
    """Laster modell fra cache."""
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def _save_model_to_cache(model, cache_path: str):
    """Lagrer modell til cache."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception:
        return False

def clear_model_cache():
    """Sletter all modell-cache."""
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.startswith("model_") and f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(CACHE_DIR, f))
                except:
                    pass

def get_cache_stats() -> dict:
    """Returnerer statistikk om cache."""
    if not os.path.exists(CACHE_DIR):
        return {'count': 0, 'size_mb': 0, 'oldest': None}
    
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("model_") and f.endswith(".pkl")]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    
    oldest = None
    if files:
        oldest_time = min(os.path.getmtime(os.path.join(CACHE_DIR, f)) for f in files)
        oldest = datetime.fromtimestamp(oldest_time)
    
    return {
        'count': len(files),
        'size_mb': round(total_size / (1024 * 1024), 2),
        'oldest': oldest
    }


# =============================================================================
# PATTERN VISION INTEGRASJON
# =============================================================================

def _apply_insider_confidence_boost(ml_score: float, insider_score: float) -> float:
    """
    Justerer ML score basert på insider-aktivitet.
    
    Logikk:
    - Sterke insider-kjøp (score > 30) → boost opptil +5 poeng
    - Sterke insider-salg (score < -30) → straff opptil -3 poeng
    - Moderat aktivitet → liten justering
    - Ingen aktivitet (score = 0) → ingen endring
    
    Returns:
        boost_amount (positiv eller negativ)
    """
    if insider_score == 0:
        return 0
    
    if insider_score > 0:
        # Kjøp-signal: skalér lineært opp til maks +5
        # score 30 → +1.5, score 60 → +3, score 100 → +5
        boost = min(5.0, insider_score / 100 * 5.0)
        # Kun boost hvis ML allerede er noe positiv
        if ml_score < 45:
            boost *= 0.3  # Minimal boost hvis ML er bearish
        return round(boost, 1)
    else:
        # Salg-signal: skalér lineært ned til maks -3
        # score -30 → -0.9, score -60 → -1.8, score -100 → -3
        penalty = max(-3.0, insider_score / 100 * 3.0)
        return round(penalty, 1)


# === CHART HELPERS ===
def _to_timestamp(idx):
    """Konverter pandas index til unix timestamp."""
    try:
        if hasattr(idx, 'timestamp'):
            return int(idx.timestamp())
        return int(pd.Timestamp(idx).timestamp())
    except:
        return 0

def _df_to_ohlc(df: pd.DataFrame) -> list:
    """Konverter DataFrame til OHLC-format."""
    data = []
    for idx, row in df.iterrows():
        try:
            ts = _to_timestamp(idx)
            if ts > 0:
                data.append({
                    'time': ts,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                })
        except:
            continue
    return data

def _df_to_line(df: pd.DataFrame, column: str) -> list:
    """Konverter kolonne til line-format."""
    if column not in df.columns:
        return []
    data = []
    for idx, val in df[column].dropna().items():
        try:
            ts = _to_timestamp(idx)
            if ts > 0 and pd.notna(val):
                data.append({'time': ts, 'value': float(val)})
        except:
            continue
    return data

def _series_to_line(series: pd.Series) -> list:
    """Konverter pandas Series til line-format."""
    data = []
    for idx, val in series.dropna().items():
        try:
            ts = _to_timestamp(idx)
            if ts > 0 and pd.notna(val):
                data.append({'time': ts, 'value': float(val)})
        except:
            continue
    return data

def _df_to_volume(df: pd.DataFrame) -> list:
    """Konverter volum med farger."""
    data = []
    for idx, row in df.iterrows():
        try:
            ts = _to_timestamp(idx)
            if ts > 0:
                is_up = row['Close'] >= row['Open']
                data.append({
                    'time': ts,
                    'value': float(row['Volume']),
                    'color': 'rgba(38, 166, 154, 0.5)' if is_up else 'rgba(239, 83, 80, 0.5)'
                })
        except:
            continue
    return data

def _ai_score_to_area(series: pd.Series, buy_level: float = 60, sell_level: float = 40) -> list:
    """Konverter AI Score series til area format med farge basert på verdi."""
    data = []
    for idx, val in series.dropna().items():
        try:
            ts = _to_timestamp(idx)
            if ts > 0 and pd.notna(val):
                # Farge basert på score - grønn over kjøp, rød under salg
                if val >= buy_level:
                    color = 'rgba(0, 200, 5, 0.7)'  # Grønn - kjøpssone
                elif val <= sell_level:
                    color = 'rgba(239, 83, 80, 0.7)'  # Rød - salgssone
                else:
                    color = 'rgba(158, 158, 158, 0.5)'  # Grå - nøytral
                data.append({'time': ts, 'value': float(val), 'color': color})
        except:
            continue
    return data


def _render_ai_chart(df: pd.DataFrame, ml_scores: pd.Series, ticker: str, 
                     indicators: dict = None, show_rsi: bool = True):
    """
    Render Lightweight Charts med AI Score subchart.
    """
    if not LWC_INSTALLED:
        st.error("streamlit-lightweight-charts er ikke installert")
        return
    
    if df.empty:
        st.warning("Ingen data å vise")
        return
    
    indicators = indicators or {}
    
    # Tema-konfigurasjon (mørkt tema)
    theme_cfg = {
        'bg': '#0e1117',
        'text': '#d1d4dc',
        'grid': 'rgba(255, 255, 255, 0.04)',
        'border': 'rgba(255, 255, 255, 0.1)',
    }
    
    # Farger
    COLORS = {
        'up': '#26a69a',
        'down': '#ef5350',
        'sma_50': '#45B7D1',
        'sma_200': '#DDA0DD',
        'bb': 'rgba(33, 150, 243, 0.6)',
        'ai_score': '#4e8cff',
        'rsi': '#AB47BC',
    }
    
    # === HOVEDCHART SERIES ===
    main_series = []
    
    # Candlesticks
    candle_data = _df_to_ohlc(df)
    if not candle_data:
        st.warning("Kunne ikke konvertere prisdata")
        return
    
    main_series.append({
        'type': 'Candlestick',
        'data': candle_data,
        'options': {
            'upColor': COLORS['up'],
            'downColor': COLORS['down'],
            'borderUpColor': COLORS['up'],
            'borderDownColor': COLORS['down'],
            'wickUpColor': COLORS['up'],
            'wickDownColor': COLORS['down'],
        }
    })
    
    # SMA 50
    if indicators.get('sma_50') and 'SMA_50' in df.columns:
        sma50_data = _df_to_line(df, 'SMA_50')
        if sma50_data:
            main_series.append({
                'type': 'Line',
                'data': sma50_data,
                'options': {
                    'color': COLORS['sma_50'],
                    'lineWidth': 2,
                    'crosshairMarkerVisible': False,
                    'lastValueVisible': True,
                    'priceLineVisible': False,
                }
            })
    
    # SMA 200
    if indicators.get('sma_200') and 'SMA_200' in df.columns:
        sma200_data = _df_to_line(df, 'SMA_200')
        if sma200_data:
            main_series.append({
                'type': 'Line',
                'data': sma200_data,
                'options': {
                    'color': COLORS['sma_200'],
                    'lineWidth': 2,
                    'crosshairMarkerVisible': False,
                    'lastValueVisible': True,
                    'priceLineVisible': False,
                }
            })
    
    # Bollinger Bands
    if indicators.get('bb'):
        for col in ['BB_Upper', 'BB_Lower']:
            if col in df.columns:
                bb_data = _df_to_line(df, col)
                if bb_data:
                    main_series.append({
                        'type': 'Line',
                        'data': bb_data,
                        'options': {
                            'color': COLORS['bb'],
                            'lineWidth': 1,
                            'crosshairMarkerVisible': False,
                            'lastValueVisible': False,
                            'priceLineVisible': False,
                        }
                    })
    
    # === BYGG CHARTS ===
    charts = []
    
    # 1. Hovedchart (Pris)
    charts.append({
        'chart': {
            'height': 350,
            'layout': {
                'background': {'type': 'solid', 'color': theme_cfg['bg']},
                'textColor': theme_cfg['text'],
            },
            'grid': {
                'vertLines': {'color': theme_cfg['grid']},
                'horzLines': {'color': theme_cfg['grid']},
            },
            'crosshair': {'mode': 0},
            'rightPriceScale': {
                'borderColor': theme_cfg['border'],
                'scaleMargins': {'top': 0.1, 'bottom': 0.2},
            },
            'timeScale': {
                'borderColor': theme_cfg['border'],
                'timeVisible': True,
                'secondsVisible': False,
            },
        },
        'series': main_series,
    })
    
    # 2. AI Score subchart
    if not ml_scores.empty:
        ai_data = _series_to_line(ml_scores)
        ai_data_colored = _ai_score_to_area(ml_scores, buy_level=60, sell_level=40)
        if ai_data:
            # Lag referanselinjer for kjøp (60) og salg (40)
            ref_sell = [{'time': d['time'], 'value': 40} for d in ai_data]
            ref_buy = [{'time': d['time'], 'value': 60} for d in ai_data]
            
            charts.append({
                'chart': {
                    'height': 120,
                    'layout': {
                        'background': {'type': 'solid', 'color': theme_cfg['bg']},
                        'textColor': theme_cfg['text'],
                    },
                    'grid': {
                        'vertLines': {'color': theme_cfg['grid']},
                        'horzLines': {'color': theme_cfg['grid']},
                    },
                    'rightPriceScale': {
                        'borderColor': theme_cfg['border'],
                        'scaleMargins': {'top': 0.1, 'bottom': 0.1},
                    },
                    'timeScale': {'visible': False},
                },
                'series': [
                    {
                        'type': 'Histogram',
                        'data': ai_data_colored,
                        'options': {
                            'priceFormat': {'type': 'price', 'precision': 0},
                            'lastValueVisible': True,
                            'priceLineVisible': False,
                        }
                    },
                    {
                        'type': 'Line',
                        'data': ref_sell,
                        'options': {
                            'color': 'rgba(239, 83, 80, 0.6)',
                            'lineWidth': 1,
                            'lineStyle': 2,
                            'crosshairMarkerVisible': False,
                            'lastValueVisible': False,
                            'priceLineVisible': False,
                            'title': 'Salg',
                        }
                    },
                    {
                        'type': 'Line',
                        'data': ref_buy,
                        'options': {
                            'color': 'rgba(0, 200, 5, 0.6)',
                            'lineWidth': 1,
                            'lineStyle': 2,
                            'crosshairMarkerVisible': False,
                            'lastValueVisible': False,
                            'priceLineVisible': False,
                            'title': 'Kjøp',
                        }
                    },
                ],
            })
    
    # 3. RSI subchart
    if show_rsi and 'RSI' in df.columns:
        rsi_data = _df_to_line(df, 'RSI')
        if rsi_data:
            rsi_70 = [{'time': d['time'], 'value': 70} for d in rsi_data]
            rsi_30 = [{'time': d['time'], 'value': 30} for d in rsi_data]
            
            charts.append({
                'chart': {
                    'height': 100,
                    'layout': {
                        'background': {'type': 'solid', 'color': theme_cfg['bg']},
                        'textColor': theme_cfg['text'],
                    },
                    'grid': {
                        'vertLines': {'color': theme_cfg['grid']},
                        'horzLines': {'color': theme_cfg['grid']},
                    },
                    'rightPriceScale': {'borderColor': theme_cfg['border']},
                    'timeScale': {'visible': False},
                },
                'series': [
                    {'type': 'Line', 'data': rsi_data, 'options': {'color': COLORS['rsi'], 'lineWidth': 2}},
                    {'type': 'Line', 'data': rsi_70, 'options': {'color': 'rgba(239, 83, 80, 0.5)', 'lineWidth': 1, 'lineStyle': 2, 'crosshairMarkerVisible': False, 'lastValueVisible': False, 'priceLineVisible': False}},
                    {'type': 'Line', 'data': rsi_30, 'options': {'color': 'rgba(38, 166, 154, 0.5)', 'lineWidth': 1, 'lineStyle': 2, 'crosshairMarkerVisible': False, 'lastValueVisible': False, 'priceLineVisible': False}},
                ],
            })
    
    # 4. Volum subchart
    vol_data = _df_to_volume(df)
    if vol_data:
        charts.append({
            'chart': {
                'height': 80,
                'layout': {
                    'background': {'type': 'solid', 'color': theme_cfg['bg']},
                    'textColor': theme_cfg['text'],
                },
                'grid': {
                    'vertLines': {'color': theme_cfg['grid']},
                    'horzLines': {'color': theme_cfg['grid']},
                },
                'rightPriceScale': {'borderColor': theme_cfg['border']},
                'timeScale': {'visible': False},
            },
            'series': [{
                'type': 'Histogram',
                'data': vol_data,
                'options': {'priceFormat': {'type': 'volume'}},
            }],
        })
    
    # === RENDER ===
    try:
        # Inkluder alle toggle-verdier i key for å tvinge re-render ved endringer
        ind_str = "_".join([f"{k}{v}" for k, v in sorted(indicators.items())])
        chart_key = f"ai_chart_{ticker}_{len(df)}_{show_rsi}_{ind_str}_{hash(str(df.index[0]))}"
        renderLightweightCharts(charts, key=chart_key)
    except Exception as e:
        st.error(f"Feil ved rendering av chart: {e}")


# =============================================================================
# DATA VALIDERING - Må være først siden andre funksjoner bruker dem
# =============================================================================

def valider_ticker_data(df, ticker):
    """
    Validerer om ticker-data er aktiv og av god kvalitet.
    Returnerer (is_valid, reason) tuple.
    """
    if df.empty:
        return False, "Tom dataframe"
    
    if len(df) < 252:  # Mindre enn 1 års data
        return False, f"Kun {len(df)} datapunkter (krever minimum 252)"
    
    # Sjekk om ticker er inaktiv (ingen handel siste 30 dager)
    try:
        siste_dato = df.index.max()
        dager_siden_handel = (pd.Timestamp.now() - siste_dato).days
        if dager_siden_handel > 30:
            return False, f"Ingen handel siste {dager_siden_handel} dager"
    except Exception:
        pass  # Ignorer hvis datoberegning feiler
    
    # Sjekk volum (må ha gjennomsnittlig volum > 1000 siste 30 dager)
    try:
        recent_volume = df['Volume'].tail(30).mean()
        if recent_volume < 1000:
            return False, f"Lavt volum: {recent_volume:.0f}/dag"
    except Exception:
        pass
    
    # Sjekk for unormal prisdata
    try:
        close = df['Close']
        
        # Sjekk for null/negative priser
        if (close <= 0).any():
            return False, "Inneholder null eller negative priser"
        
        # Sjekk for ekstreme prisbevegelser (>50% på en dag) som kan indikere feil data
        daily_changes = close.pct_change().abs()
        extreme_moves = (daily_changes > 0.5).sum()
        if extreme_moves > 5:  # Maks 5 ekstreme bevegelser totalt
            return False, f"For mange ekstreme prisbevegelser: {extreme_moves}"
        
        # Sjekk for konstante priser (døde ticker)
        recent_prices = close.tail(10)
        if recent_prices.nunique() == 1:
            return False, "Konstant pris siste 10 dager"
        
        # Sjekk for unormalt lav volatilitet (kan indikere suspendert handel)
        recent_volatility = close.tail(30).pct_change().std() * np.sqrt(252)
        if pd.notna(recent_volatility) and recent_volatility < 0.05:  # Under 5% årlig volatilitet
            return False, f"Unormalt lav volatilitet: {recent_volatility:.1%}"
    except Exception as e:
        return False, f"Valideringsfeil: {str(e)[:30]}"
    
    return True, "OK"

def sikre_numeriske_data(df):
    """
    Sikrer at alle numeriske kolonner er rene og håndterer problematiske verdier.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Liste over kolonner som må være numeriske
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in numeric_cols:
        if col in df.columns:
            # Konverter til numerisk, erstatt feil med NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Erstatt 0-verdier i pris-kolonner med NaN
            if col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].replace(0, np.nan)
            
            # Erstatt 0-volum med 1 (for å unngå divisjon med 0)
            if col == 'Volume':
                df[col] = df[col].fillna(1).replace(0, 1)
    
    # Fjern rader der alle pris-kolonner er NaN
    price_cols = ['Open', 'High', 'Low', 'Close']
    df = df.dropna(subset=price_cols, how='all')
    
    return df

# =============================================================================
# FEATURE ENGINEERING - Avansert indikatorberegning
# =============================================================================

def beregn_avanserte_features(df):
    """
    Beregner et bredt spekter av tekniske features for ML-modellen.
    Inkluderer robust error-håndtering for problematiske data.
    """
    if df.empty or len(df) < 252:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        
        # Sikre numeriske data først
        df = sikre_numeriske_data(df)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sørg for basisindikatorer
        if 'RSI' not in df.columns:
            df = logic.beregn_tekniske_indikatorer(df)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Sikkerhetsjekk for grunnleggende data
        if close.isna().all() or (close <= 0).all():
            return pd.DataFrame()
        
        # --- MOMENTUM FEATURES (med robuste beregninger) ---
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = close.pct_change(period) * 100
            df[f'ROC_{period}'] = df[f'ROC_{period}'].clip(-50, 50)
        
        # Momentum oscillator
        close_shift_10 = close.shift(10)
        close_shift_20 = close.shift(20)
        df['MOM_10'] = np.where(close_shift_10 > 0, (close - close_shift_10) / close_shift_10 * 100, 0)
        df['MOM_20'] = np.where(close_shift_20 > 0, (close - close_shift_20) / close_shift_20 * 100, 0)
        df['MOM_10'] = df['MOM_10'].clip(-50, 50)
        df['MOM_20'] = df['MOM_20'].clip(-50, 50)
        
        # Williams %R (med sikker divisjon)
        highest_14 = high.rolling(14).max()
        lowest_14 = low.rolling(14).min()
        range_14 = highest_14 - lowest_14
        df['Williams_R'] = np.where(range_14 > 0, -100 * (highest_14 - close) / range_14, -50)
        
        # RSI divergens
        if 'RSI' in df.columns:
            rsi_pct = df['RSI'].pct_change(5)
            df['RSI_Divergence'] = (rsi_pct - df['ROC_5']).clip(-20, 20)
        
        # --- VOLATILITET FEATURES ---
        if 'ATR' in df.columns:
            df['ATR_Norm'] = np.where(close > 0, df['ATR'] / close * 100, 0)
            df['ATR_Norm'] = df['ATR_Norm'].clip(0, 50)
        
        # Bollinger Band features
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            bb_range = df['BB_Upper'] - df['BB_Lower']
            df['BB_Width'] = np.where(df['BB_Middle'] > 0, bb_range / df['BB_Middle'] * 100, 0)
            df['BB_Width'] = df['BB_Width'].clip(0, 50)
            df['BB_PctB'] = np.where(bb_range > 0, (close - df['BB_Lower']) / bb_range, 0.5)
            df['BB_PctB'] = df['BB_PctB'].clip(0, 1)
        
        # Historisk volatilitet
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        for period in [10, 20]:
            hv = returns.rolling(period).std() * np.sqrt(252) * 100
            df[f'HV_{period}'] = hv.clip(0, 200)
        
        # Keltner Channel posisjon
        if 'ATR' in df.columns:
            keltner_mid = close.ewm(span=20, adjust=False).mean()
            keltner_range = 2 * df['ATR']
            keltner_lower = keltner_mid - keltner_range
            df['Keltner_Pos'] = np.where(keltner_range > 0, (close - keltner_lower) / (2 * keltner_range), 0.5)
            df['Keltner_Pos'] = df['Keltner_Pos'].clip(0, 1)

        # --- REGIME FEATURES ---
        if REGIME_AVAILABLE:
            try:
                df_regimes = _get_market_regime_df(n_regimes=3)
                if df_regimes is not None and not df_regimes.empty:
                    regime_cols = ['rolling_return', 'volatility'] + [
                        c for c in df_regimes.columns if c.startswith('prob_regime_')
                    ]

                    regime_feat = df_regimes[regime_cols].copy()
                    if not isinstance(regime_feat.index, pd.DatetimeIndex):
                        regime_feat.index = pd.to_datetime(regime_feat.index)
                    if regime_feat.index.tz is not None:
                        regime_feat.index = regime_feat.index.tz_localize(None)

                    regime_feat = regime_feat.sort_index()
                    aligned = regime_feat.reindex(df.index, method='ffill')

                    df['Regime_Rolling_Return'] = aligned.get('rolling_return', pd.Series(index=df.index))
                    df['Regime_Volatility'] = aligned.get('volatility', pd.Series(index=df.index))

                    for i in range(3):
                        col = f'prob_regime_{i}'
                        df[f'Regime_Prob_{i}'] = aligned.get(col, pd.Series(index=df.index))
                else:
                    df['Regime_Rolling_Return'] = 0.0
                    df['Regime_Volatility'] = 0.0
                    df['Regime_Prob_0'] = 0.0
                    df['Regime_Prob_1'] = 0.0
                    df['Regime_Prob_2'] = 0.0
            except Exception:
                df['Regime_Rolling_Return'] = 0.0
                df['Regime_Volatility'] = 0.0
                df['Regime_Prob_0'] = 0.0
                df['Regime_Prob_1'] = 0.0
                df['Regime_Prob_2'] = 0.0
        
        # --- VOLUM FEATURES ---
        for period in [20, 50]:
            vol_ma = volume.rolling(period).mean()
            df[f'Vol_Ratio_{period}'] = np.where(vol_ma > 0, volume / vol_ma, 1.0)
            df[f'Vol_Ratio_{period}'] = df[f'Vol_Ratio_{period}'].clip(0, 10)
        
        vol_ma_20 = volume.rolling(20).mean()
        vol_ma_50 = volume.rolling(50).mean()
        df['Vol_Trend'] = np.where(vol_ma_50 > 0, vol_ma_20 / vol_ma_50, 1.0)
        df['Vol_Trend'] = df['Vol_Trend'].clip(0, 5)
        
        # OBV trend
        price_change = np.sign(close.diff().fillna(0))
        obv = (price_change * volume).cumsum()
        obv_pct = obv.pct_change(10) * 100
        df['OBV_Trend'] = obv_pct.fillna(0).clip(-100, 100)
        
        # Money Flow Index (MFI)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df['MFI'] = np.where(negative_flow > 0, 100 - (100 / (1 + positive_flow / negative_flow)), 50)
        df['MFI'] = df['MFI'].clip(0, 100)
        
        # --- TREND FEATURES ---
        for sma_col in ['SMA_50', 'SMA_200']:
            if sma_col in df.columns:
                ratio_col = f'Price_{sma_col}_Ratio'
                df[ratio_col] = np.where(df[sma_col] > 0, close / df[sma_col], 1.0)
                df[ratio_col] = df[ratio_col].clip(0.5, 2.0)
        
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            df['SMA50_SMA200_Ratio'] = np.where(df['SMA_200'] > 0, df['SMA_50'] / df['SMA_200'], 1.0)
            df['SMA50_SMA200_Ratio'] = df['SMA50_SMA200_Ratio'].clip(0.8, 1.2)
        
        # ADX og DI beregning - FIKSET: Konverter numpy arrays til pandas Series
        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        
        # Bruk pandas .abs() i stedet for np.abs() på Series
        low_diff_abs = low.diff().abs()
        plus_dm = high.diff().where((high.diff() > low_diff_abs) & (high.diff() > 0), 0).clip(lower=0)
        minus_dm = (-low.diff()).where((low_diff_abs > high.diff()) & (low.diff() < 0), 0).clip(lower=0)
        
        plus_dm_smooth = plus_dm.rolling(14).mean()
        minus_dm_smooth = minus_dm.rolling(14).mean()
        
        # Beregn DI som pandas Series
        plus_di = pd.Series(np.where(atr_14 > 0, 100 * plus_dm_smooth / atr_14, 0), index=df.index)
        minus_di = pd.Series(np.where(atr_14 > 0, 100 * minus_dm_smooth / atr_14, 0), index=df.index)
        
        di_sum = plus_di + minus_di
        dx = pd.Series(np.where(di_sum > 0, 100 * (plus_di - minus_di).abs() / di_sum, 0), index=df.index)
        
        df['ADX'] = dx.rolling(14).mean().clip(0, 100)
        df['DI_Diff'] = (plus_di - minus_di).clip(-100, 100)
        
        # MACD features
        if 'MACD_Hist' in df.columns:
            df['MACD_Hist_Change'] = df['MACD_Hist'].diff().clip(-5, 5)
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Signal_Dist'] = (df['MACD'] - df['MACD_Signal']).clip(-10, 10)
        
        # --- PRIS MØNSTRE ---
        body = close - df['Open']
        range_hl = high - low
        
        df['Body_Ratio'] = np.where(range_hl > 0, body / range_hl, 0)
        df['Body_Ratio'] = df['Body_Ratio'].clip(-1, 1)
        
        close_max = np.maximum(close, df['Open'])
        close_min = np.minimum(close, df['Open'])
        df['Upper_Shadow'] = np.where(range_hl > 0, (high - close_max) / range_hl, 0)
        df['Upper_Shadow'] = df['Upper_Shadow'].clip(0, 1)
        df['Lower_Shadow'] = np.where(range_hl > 0, (close_min - low) / range_hl, 0)
        df['Lower_Shadow'] = df['Lower_Shadow'].clip(0, 1)
        
        # Consecutive days
        up_days = (close > close.shift(1)).astype(int)
        down_days = (close < close.shift(1)).astype(int)
        df['Consec_Up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum().clip(0, 10)
        df['Consec_Down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum().clip(0, 10)
        
        # --- RELATIVE STYRKE ---
        if 'High_52w' in df.columns and 'Low_52w' in df.columns:
            df['Dist_52w_High'] = np.where(df['High_52w'] > 0, (close - df['High_52w']) / df['High_52w'] * 100, 0)
            df['Dist_52w_High'] = df['Dist_52w_High'].clip(-50, 5)
            df['Dist_52w_Low'] = np.where(df['Low_52w'] > 0, (close - df['Low_52w']) / df['Low_52w'] * 100, 0)
            df['Dist_52w_Low'] = df['Dist_52w_Low'].clip(-5, 500)
        
        # Ichimoku signal
        if 'ISA_9' in df.columns and 'ISB_26' in df.columns:
            df['Ichimoku_Signal'] = np.where(close > df['ISA_9'], 1, np.where(close < df['ISB_26'], -1, 0))
        else:
            df['Ichimoku_Signal'] = 0
        
        # --- RENGJØR ALLE FEATURES ---
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                if col.startswith(('RSI', 'MFI', 'BB_PctB', 'Keltner_Pos')):
                    df[col] = df[col].fillna(50)
                elif col.startswith(('Vol_Ratio', 'Price_', 'SMA', 'Vol_Trend')):
                    df[col] = df[col].fillna(1.0)
                else:
                    df[col] = df[col].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Feil i feature engineering: {e}")
        return pd.DataFrame()

def velg_features():
    """Returnerer listen over features brukt i modellen."""
    base_features = [
        # Momentum (mer konservativ liste)
        'RSI', 'ROC_5', 'ROC_10', 'ROC_20', 'MOM_10', 'Williams_R',
        # Volatilitet
        'ATR_Norm', 'BB_Width', 'BB_PctB', 'HV_10', 'Keltner_Pos',
        # Volum
        'Vol_Ratio_20', 'Vol_Trend', 'OBV_Trend', 'MFI',
        # Trend
        'Price_SMA_50_Ratio', 'Price_SMA_200_Ratio', 'SMA50_SMA200_Ratio',
        'ADX', 'DI_Diff',
        # Pris mønstre
        'Body_Ratio', 'Upper_Shadow', 'Lower_Shadow',
        # Relative
        'Dist_52w_High', 'Dist_52w_Low',
        # Teknisk
        'Ichimoku_Signal'
    ]

    if REGIME_AVAILABLE:
        base_features.extend([
            'Regime_Rolling_Return',
            'Regime_Volatility',
            'Regime_Prob_0',
            'Regime_Prob_1',
            'Regime_Prob_2'
        ])

    return base_features

# =============================================================================
# MARKEDSREGIME INTEGRASJON
# =============================================================================

def _get_market_regime():
    """
    Henter nåværende markedsregime fra regime_model.
    Returnerer regime-info dict eller None hvis ikke tilgjengelig.
    """
    if not REGIME_AVAILABLE:
        return None

    try:
        # Hent markedsdata (SPY/indeks)
        df_market = data.hent_markedsdata_df(force_refresh=False)
        
        if df_market is None or len(df_market) < 50:
            return None
        
        # Kjør regimeanalyse med 3 regimer (standard)
        regime_data = regime_model.full_regime_analyse(df_market, n_regimes=3)
        
        if regime_data and regime_data.get('current_info'):
            return regime_data['current_info']
        
        return None
        
    except Exception as e:
        logger.warning(f"Kunne ikke hente regime: {e}")
        return None


def _get_market_regime_df(n_regimes: int = 3) -> pd.DataFrame:
    """Henter regime-DataFrame (cached) for feature engineering."""
    if not REGIME_AVAILABLE:
        return pd.DataFrame()

    try:
        now = datetime.now()
        cache_ts = _REGIME_FEATURE_CACHE.get('ts')
        cache_df = _REGIME_FEATURE_CACHE.get('df_regimes')
        cache_n = _REGIME_FEATURE_CACHE.get('n_regimes', n_regimes)

        if cache_ts and cache_df is not None and cache_n == n_regimes:
            age = now - cache_ts
            if age < timedelta(hours=_REGIME_CACHE_TTL_HOURS):
                return cache_df

        df_market = data.hent_markedsdata_df(force_refresh=False)
        if df_market is None or len(df_market) < 50:
            return pd.DataFrame()

        regime_data = regime_model.full_regime_analyse(df_market, n_regimes=n_regimes)
        if not regime_data or regime_data.get('df_regimes') is None:
            return pd.DataFrame()

        df_regimes = regime_data['df_regimes'].copy()
        if not isinstance(df_regimes.index, pd.DatetimeIndex):
            df_regimes.index = pd.to_datetime(df_regimes.index)
        if df_regimes.index.tz is not None:
            df_regimes.index = df_regimes.index.tz_localize(None)

        _REGIME_FEATURE_CACHE['ts'] = now
        _REGIME_FEATURE_CACHE['df_regimes'] = df_regimes
        _REGIME_FEATURE_CACHE['n_regimes'] = n_regimes

        return df_regimes

    except Exception:
        return pd.DataFrame()


def _get_regime_score_adjustment(regime_info: dict) -> float:
    """
    Beregner justeringsfaktor for AI Score basert på markedsregime.
    
    Returns:
        float: Multiplikator for AI Score (0.7 - 1.0)
    """
    if not regime_info:
        return 1.0
    
    regime_name = regime_info.get('name', '')
    confidence = regime_info.get('probability', 0.5)
    
    # Base-justering basert på regime-type
    adjustments = {
        'Bull Market': 1.0,      # Ingen justering i bull market
        'Mild Bull': 1.0,        # Ingen justering
        'Nøytral': 0.95,         # 5% reduksjon
        'Mild Bear': 0.85,       # 15% reduksjon
        'Bear Market': 0.70,     # 30% reduksjon
    }
    
    base_adj = adjustments.get(regime_name, 0.95)
    
    # Jo høyere konfidens på bear regime, jo sterkere justering
    if 'Bear' in regime_name and confidence > 0.6:
        # Forsterker justering ved høy konfidens
        extra_reduction = (confidence - 0.6) * 0.2  # Max 8% ekstra reduksjon
        base_adj -= extra_reduction
    
    return max(0.5, base_adj)  # Aldri lavere enn 50%


def _render_regime_badge(regime_info: dict):
    """
    Renderer et kompakt regime-badge i sidebaren eller toppen av AI Scanner.
    """
    if not regime_info:
        return
    
    name = regime_info.get('name', 'Ukjent')
    emoji = regime_info.get('emoji', '❓')
    color = regime_info.get('color', '#808080')
    prob = regime_info.get('probability', 0) * 100
    action = regime_info.get('action', '')
    days = regime_info.get('streak_days', 0)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}20 0%, {color}08 100%);
                border-left: 4px solid {color}; border-radius: 8px; padding: 12px 15px;
                margin-bottom: 15px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 1.3rem;">{emoji}</span>
                <span style="color: {color}; font-weight: 600; margin-left: 8px; font-size: 1rem;">
                    {name}
                </span>
            </div>
            <div style="text-align: right;">
                <span style="color: #888; font-size: 0.8rem;">{prob:.0f}%</span>
                <span style="color: #666; font-size: 0.7rem; margin-left: 6px;">({days}d)</span>
            </div>
        </div>
        <div style="color: #aaa; font-size: 0.75rem; margin-top: 6px;">
            {action}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENSEMBLE ML-MODELL (Oppdatert med bedre datavalidering)
# =============================================================================

class EnsembleStockPredictor:
    """
    Ensemble-modell med robust datahåndtering for døde/problematiske tickers.
    """
    
    def __init__(self, horisont=10, target_pct=0.04, stop_pct=None, min_adx=15, min_hv=8.0, min_atr_norm=0.5):
        self.horisont = horisont
        self.target_pct = target_pct
        self.stop_pct = stop_pct if stop_pct is not None else target_pct * 0.6
        self.min_adx = min_adx
        self.min_hv = min_hv
        self.min_atr_norm = min_atr_norm
        self.scaler = StandardScaler()
        self.features = velg_features()
        self.models = {}
        self.calibrators = {}
        self.weights = {'xgb': 0.4, 'rf': 0.3, 'gb': 0.3}
        self.is_fitted = False
        self.metrics = {}
        self.signal_threshold = 60.0
        
    def _validate_data_quality(self, df):
        """Validerer datakvalitet før ML-prosessering."""
        if df.empty:
            return False, "Tom dataframe"
        if len(df) < 100:
            return False, f"For lite data: {len(df)} rader"

        available_features = [f for f in self.features if f in df.columns]
        if len(available_features) < len(self.features) * 0.7:
            return False, f"Mangler for mange features: {len(available_features)}/{len(self.features)}"

        feature_data = df[available_features]
        nan_pct = feature_data.isna().mean().mean()
        if nan_pct > 0.3:
            return False, f"For mange NaN verdier: {nan_pct:.1%}"

        return True, "OK"

    def _create_target(self, df):
        """
        Strengere target:
        - bruker CLOSE på horisont
        - krever at stop-barriere ikke er truffet i perioden
        """
        close = df['Close']
        low = df['Low']
        
        future_close = close.shift(-self.horisont)
        
        # Forward-looking min-low (ekskluderer nåværende bar for å unngå lekkasje)
        low_rev = low.shift(-1).iloc[::-1]
        future_low = low_rev.rolling(self.horisont, min_periods=self.horisont).min().iloc[::-1]
        
        hit_target = future_close >= close * (1 + self.target_pct)
        hit_stop = future_low <= close * (1 - self.stop_pct)
        
        target = (hit_target & ~hit_stop).astype(int)
        return target

    def _apply_noise_filter(self, df):
        """Avviser lav trendstyrke eller ekstrem lav volatilitet."""
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        if 'ADX' in df.columns:
            mask &= df['ADX'] >= self.min_adx
        if 'HV_10' in df.columns:
            mask &= df['HV_10'] >= self.min_hv
        elif 'ATR_Norm' in df.columns:
            mask &= df['ATR_Norm'] >= self.min_atr_norm
        return df[mask]

    def _compute_sample_weights(self, y):
        pos = int(y.sum())
        neg = int(len(y) - pos)
        weight_pos = (neg / pos) if pos > 0 else 1.0
        return np.where(y == 1, weight_pos, 1.0)

    def _purged_time_series_split(self, n_samples, n_splits=3):
        idx = np.arange(n_samples)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(idx):
            purge_before = test_idx.min() - self.horisont
            if purge_before > 0:
                train_idx = train_idx[train_idx < purge_before]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def _ensemble_predict_proba(self, X_scaled):
        if not self.models:
            return np.full(len(X_scaled), 0.5)
        total_w = sum(self.weights.values()) if self.weights else 1.0
        agg = np.zeros(len(X_scaled))
        for name, model in self.models.items():
            mdl = self.calibrators.get(name, model)
            try:
                probs = mdl.predict_proba(X_scaled)[:, 1]
            except Exception:
                probs = np.full(len(X_scaled), 0.5)
            agg += probs * self.weights.get(name, 1 / len(self.models))
        return agg / total_w

    def _find_optimal_threshold(self, probs, y, min_thr=0.30, max_thr=0.80, steps=26):
        """Finner terskel som maksimerer F0.5 på kalibreringsdata."""
        if probs is None or len(probs) == 0:
            return 0.60
        if y is None or len(y) == 0:
            return 0.60

        thresholds = np.linspace(min_thr, max_thr, steps)
        best_thr = 0.60
        best_score = -1.0

        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            score = fbeta_score(y, preds, beta=0.5, zero_division=0)
            if score > best_score:
                best_score = score
                best_thr = thr

        return float(best_thr)

    def _prepare_data(self, df):
        """Forbereder data med omfattende validering."""
        df_work = df.copy()
        
        # Validér grunnleggende datakvalitet
        is_valid, reason = self._validate_data_quality(df_work)
        if not is_valid:
            return pd.DataFrame(), []
        
        # Opprett target
        df_work['Target'] = self._create_target(df_work)
        
        # Støyfilter
        df_work = self._apply_noise_filter(df_work)
        if df_work.empty:
            return pd.DataFrame(), []
        
        # Velg tilgjengelige features
        available_features = [f for f in self.features if f in df_work.columns]
        
        # Fjern rader med for mange NaN
        required_cols = available_features + ['Target']
        df_clean = df_work.dropna(subset=required_cols, thresh=len(required_cols)*0.8)
        
        # Final validation
        if len(df_clean) < 50:
            return pd.DataFrame(), []
        
        return df_clean, available_features
    
    def fit(self, df, validate=True):
        """Trener modellen med omfattende datavalidering."""
        try:
            df_clean, feature_cols = self._prepare_data(df)
            
            if df_clean.empty or len(feature_cols) == 0:
                return False
            
            # Separer data
            X = df_clean[feature_cols].iloc[:-self.horisont]
            y = df_clean['Target'].iloc[:-self.horisont]
            
            if len(X) < 50:
                return False
            
            # CV med purging + F0.5
            if validate:
                cv_scores = {'xgb': [], 'rf': [], 'gb': []}
                cv_precision = {'xgb': [], 'rf': [], 'gb': []}
                for train_idx, test_idx in self._purged_time_series_split(len(X), n_splits=3):
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                    
                    scaler_cv = StandardScaler().fit(X_train.replace([np.inf, -np.inf], np.nan).fillna(0))
                    X_train_scaled = scaler_cv.transform(X_train.replace([np.inf, -np.inf], np.nan).fillna(0))
                    X_test_scaled = scaler_cv.transform(X_test.replace([np.inf, -np.inf], np.nan).fillna(0))
                    
                    sw = self._compute_sample_weights(y_train)
                    
                    models_cv = {
                        'xgb': xgb.XGBClassifier(
                            n_estimators=50, max_depth=3, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            verbosity=0, use_label_encoder=False, random_state=42,
                            n_jobs=1, scale_pos_weight=max(1.0, sw.max())
                        ),
                        'rf': RandomForestClassifier(
                            n_estimators=50, max_depth=5, min_samples_split=10,
                            min_samples_leaf=5, random_state=42, n_jobs=1
                        ),
                        'gb': GradientBoostingClassifier(
                            n_estimators=50, max_depth=3, learning_rate=0.1,
                            subsample=0.8, random_state=42
                        )
                    }
                    
                    for name, mdl in models_cv.items():
                        try:
                            mdl.fit(X_train_scaled, y_train, sample_weight=sw)
                            probs = mdl.predict_proba(X_test_scaled)[:, 1]
                            preds = (probs >= 0.5).astype(int)
                            prec = precision_score(y_test, preds, zero_division=0)
                            f05 = fbeta_score(y_test, preds, beta=0.5, zero_division=0)
                            cv_precision[name].append(prec)
                            cv_scores[name].append(f05)
                        except Exception:
                            cv_precision[name].append(0.0)
                            cv_scores[name].append(0.0)
                
                # Oppdater vekter basert på F0.5
                new_weights = {}
                for name in cv_scores:
                    score = float(np.mean(cv_scores[name])) if cv_scores[name] else 0.01
                    new_weights[name] = max(0.01, score)
                total = sum(new_weights.values())
                self.weights = {k: v / total for k, v in new_weights.items()}
                self.metrics = {
                    'precision': {k: float(np.mean(cv_precision[k])) if cv_precision[k] else 0.0 for k in cv_precision},
                    'f0.5': {k: float(np.mean(cv_scores[k])) if cv_scores[k] else 0.0 for k in cv_scores}
                }
            
            # Kalibreringsholdout (purged)
            n_total = len(X)
            train_end = int(n_total * 0.8)
            calib_start = train_end + self.horisont
            
            use_calibration = calib_start < n_total - 10
            if use_calibration:
                X_train = X.iloc[:train_end]
                y_train = y.iloc[:train_end]
                X_calib = X.iloc[calib_start:]
                y_calib = y.iloc[calib_start:]
                
                self.scaler.fit(X_train.replace([np.inf, -np.inf], np.nan).fillna(0))
                X_train_scaled = self.scaler.transform(X_train.replace([np.inf, -np.inf], np.nan).fillna(0))
                X_calib_scaled = self.scaler.transform(X_calib.replace([np.inf, -np.inf], np.nan).fillna(0))
                
                sw = self._compute_sample_weights(y_train)
                
                # Tren base-modeller
                self.models['xgb'] = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    verbosity=0, use_label_encoder=False, random_state=42,
                    n_jobs=1, scale_pos_weight=max(1.0, sw.max())
                )
                self.models['xgb'].fit(X_train_scaled, y_train, sample_weight=sw)
                
                self.models['rf'] = RandomForestClassifier(
                    n_estimators=50, max_depth=5, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=1
                )
                self.models['rf'].fit(X_train_scaled, y_train, sample_weight=sw)
                
                self.models['gb'] = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                self.models['gb'].fit(X_train_scaled, y_train, sample_weight=sw)
                
                # Kalibrer
                self.calibrators = {}
                for name, model in self.models.items():
                    try:
                        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
                        calibrator.fit(X_calib_scaled, y_calib)
                        self.calibrators[name] = calibrator
                    except Exception:
                        pass

                # Optimal terskel basert på kalibreringssett
                try:
                    calib_probs = self._ensemble_predict_proba(X_calib_scaled)
                    self.signal_threshold = round(self._find_optimal_threshold(calib_probs, y_calib) * 100, 1)
                except Exception:
                    self.signal_threshold = 60.0
            else:
                # Fallback: tren på full data (uten kalibrering)
                X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                self.scaler.fit(X_clean)
                X_scaled = self.scaler.transform(X_clean)
                sw_full = self._compute_sample_weights(y)
                
                self.models['xgb'] = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    verbosity=0, use_label_encoder=False, random_state=42,
                    n_jobs=1, scale_pos_weight=max(1.0, sw_full.max())
                )
                self.models['xgb'].fit(X_scaled, y, sample_weight=sw_full)
                
                self.models['rf'] = RandomForestClassifier(
                    n_estimators=50, max_depth=5, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=1
                )
                self.models['rf'].fit(X_scaled, y, sample_weight=sw_full)
                
                self.models['gb'] = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                self.models['gb'].fit(X_scaled, y, sample_weight=sw_full)
            
            self.is_fitted = True
            self.feature_cols = feature_cols
            
            return True
            
        except Exception as e:
            logger.error(f"Feil under modelltrening: {e}")
            return False
    
    def predict_proba(self, df):
        """Robust prediksjon med omfattende error handling."""
        if not self.is_fitted:
            return 50.0
        
        try:
            df_work = beregn_avanserte_features(df)
            if df_work.empty:
                return 50.0
            
            # Støyfilter: avvis siste rad hvis lav trend/vol
            latest_row = df_work.tail(1)
            latest_row = self._apply_noise_filter(latest_row)
            if latest_row.empty:
                return 50.0
            
            # Valider data quality
            is_valid, _ = self._validate_data_quality(df_work)
            if not is_valid:
                return 50.0
            
            # Sjekk at vi har de nødvendige features
            missing_features = [f for f in self.feature_cols if f not in df_work.columns]
            if len(missing_features) > len(self.feature_cols) * 0.3:
                return 50.0
            
            # Hent siste rad
            latest = df_work[self.feature_cols].tail(1)
            
            # Fylle manglende features med nøytrale verdier
            for col in self.feature_cols:
                if col not in latest.columns:
                    if col.startswith(('RSI', 'MFI')):
                        latest[col] = 50
                    elif col.startswith(('Vol_Ratio', 'Price_')):
                        latest[col] = 1.0
                    else:
                        latest[col] = 0
            
            # Sikre riktig rekkefølge
            latest = latest[self.feature_cols]
            
            # Rens data
            latest_clean = latest.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if latest_clean.isna().any().any():
                return 50.0
            
            # Skaler
            latest_scaled = self.scaler.transform(latest_clean)
            
            if np.isnan(latest_scaled).any() or np.isinf(latest_scaled).any():
                return 50.0
            
            weighted_prob = self._ensemble_predict_proba(latest_scaled)[0]
            return round(float(weighted_prob * 100), 1)
            
        except Exception as e:
            logger.error(f"Feil under prediksjon: {e}")
            return 50.0

    def predict_historical(self, df, dager=60):
        """
        Beregner historiske sannsynligheter for grafvisning.
        """
        if not self.is_fitted:
            return pd.Series()
        
        try:
            df_work = beregn_avanserte_features(df)
            if df_work.empty or len(df_work) < dager:
                return pd.Series()
            
            test_data = df_work[self.feature_cols].tail(dager)
            valid_indices = test_data.dropna().index
            
            if len(valid_indices) == 0:
                return pd.Series()
            
            test_scaled = self.scaler.transform(test_data.loc[valid_indices].replace([np.inf, -np.inf], np.nan).fillna(0))
            
            probs = self._ensemble_predict_proba(test_scaled)
            return pd.Series(probs * 100, index=valid_indices)
        except Exception as e:
            logger.error(f"Feil i predict_historical: {e}")
            return pd.Series()
    
    def get_feature_importance(self):
        """Returnerer feature importance fra XGBoost-modellen."""
        if 'xgb' not in self.models or not hasattr(self, 'feature_cols'):
            return {}
        
        try:
            importance = self.models['xgb'].feature_importances_
            return dict(zip(self.feature_cols, importance))
        except:
            return {}
    
    def backtest(self, df, lookback_signals=20):
        """
        Kjører backtest på historiske signaler for denne tickeren.
        Returnerer dict med treffsikkerhet-statistikk.
        
        Args:
            df: DataFrame med features
            lookback_signals: Antall historiske signaler å evaluere
        
        Returns:
            dict med 'total_signals', 'hits', 'misses', 'hit_rate', 'avg_return', 'details'
        """
        if not self.is_fitted:
            return None
        
        try:
            df_work = beregn_avanserte_features(df)
            if df_work.empty or len(df_work) < 100:
                return None
            
            # Beregn historiske prediksjoner (unngå siste horisont dager)
            eval_end = len(df_work) - self.horisont - 1
            if eval_end < 50:
                return None
            
            # Hent features for evaluering
            test_data = df_work[self.feature_cols].iloc[50:eval_end]
            valid_indices = test_data.dropna().index
            
            if len(valid_indices) < 10:
                return None
            
            # Skaler og prediker
            test_scaled = self.scaler.transform(
                test_data.loc[valid_indices].replace([np.inf, -np.inf], np.nan).fillna(0)
            )
            probs = self._ensemble_predict_proba(test_scaled) * 100
            
            # Finn signaler (score >= optimal terskel)
            signal_mask = probs >= self.signal_threshold
            signal_indices = valid_indices[signal_mask]
            signal_probs = probs[signal_mask]
            
            if len(signal_indices) == 0:
                return {'total_signals': 0, 'hits': 0, 'misses': 0, 'hit_rate': 0, 'avg_return': 0, 'details': []}
            
            # Begrens til siste N signaler
            if len(signal_indices) > lookback_signals:
                signal_indices = signal_indices[-lookback_signals:]
                signal_probs = signal_probs[-lookback_signals:]
            
            # Evaluer hvert signal
            hits = 0
            misses = 0
            returns = []
            details = []
            
            close = df_work['Close']
            low = df_work['Low']
            
            for idx, prob in zip(signal_indices, signal_probs):
                try:
                    # Finn posisjon i dataframe
                    pos = df_work.index.get_loc(idx)
                    if pos + self.horisont >= len(df_work):
                        continue
                    
                    entry_price = close.iloc[pos]
                    future_close = close.iloc[pos + self.horisont]
                    
                    # Sjekk om stop ble truffet
                    future_lows = low.iloc[pos + 1:pos + self.horisont + 1]
                    if len(future_lows) == 0:
                        continue
                    min_low = future_lows.min()
                    hit_stop = min_low <= entry_price * (1 - self.stop_pct)
                    
                    # Beregn faktisk avkastning
                    actual_return = (future_close - entry_price) / entry_price * 100
                    
                    # Hit = nådde mål OG traff ikke stop
                    hit_target = future_close >= entry_price * (1 + self.target_pct)
                    is_hit = hit_target and not hit_stop
                    
                    if is_hit:
                        hits += 1
                    else:
                        misses += 1
                    
                    returns.append(actual_return)
                    details.append({
                        'dato': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        'score': round(prob, 1),
                        'avkastning': round(actual_return, 2),
                        'treff': is_hit
                    })
                except Exception:
                    continue
            
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            avg_return = np.mean(returns) if returns else 0
            
            return {
                'total_signals': total,
                'hits': hits,
                'misses': misses,
                'hit_rate': round(hit_rate, 1),
                'avg_return': round(avg_return, 2),
                'details': details[-10:]  # Siste 10 for visning
            }
            
        except Exception as e:
            logger.error(f"Backtest feil: {e}")
            return None

# =============================================================================
# KONFIDENSANALYSE
# =============================================================================

def beregn_konfidensintervall(predictor, df, n_bootstrap=25):
    """
    Beregner konfidensintervall ved bootstrap-sampling (redusert antall for ytelse).
    """
    try:
        df_work = beregn_avanserte_features(df)
        if df_work.empty:
            return 50.0, 40, 60
        
        # Sjekk at vi har nødvendige features
        if not hasattr(predictor, 'feature_cols') or len(predictor.feature_cols) == 0:
            return 50.0, 40, 60
        
        predictions = []
        for _ in range(n_bootstrap):
            try:
                # Legg til liten støy i features for å simulere usikkerhet
                latest = df_work[predictor.feature_cols].tail(1).copy()
                
                # Fyll manglende features med nøytrale verdier
                for col in predictor.feature_cols:
                    if col not in latest.columns:
                        if col.startswith(('RSI', 'MFI')):
                            latest[col] = 50
                        elif col.startswith(('Vol_Ratio', 'Price_')):
                            latest[col] = 1.0
                        else:
                            latest[col] = 0
                
                latest_vals = latest[predictor.feature_cols].values
                noise = np.random.normal(0, 0.01, len(predictor.feature_cols))
                latest_with_noise = latest_vals + noise
                
                latest_scaled = predictor.scaler.transform(latest_with_noise.reshape(1, -1))
                prob = predictor._ensemble_predict_proba(latest_scaled)[0]
                predictions.append(prob * 100)
            except Exception:
                predictions.append(50.0)
        
        if predictions:
            mean_pred = np.mean(predictions)
            lower = np.percentile(predictions, 10)
            upper = np.percentile(predictions, 90)
            return round(mean_pred, 1), round(lower, 1), round(upper, 1)
        else:
            return 50.0, 40, 60
            
    except Exception as e:
        logger.warning(f"Feil i konfidensintervall: {e}")
        return 50.0, 40, 60

# =============================================================================
# PARALLELL ANALYSE MED CACHING
# =============================================================================

def _analyser_ticker(ticker, df_full, valgt_horisont, valgt_maal, valgt_stop, terskel, use_cache=True, insider_handler=None):
    """
    Analyserer én ticker med modell-caching.
    Returnerer (resultat_dict, feil_tuple, cache_hit) - én av de to første er None.
    """
    try:
        df_ticker = df_full[df_full['Ticker'] == ticker].copy()
        
        # Validér ticker data først
        is_valid, reason = valider_ticker_data(df_ticker, ticker)
        if not is_valid:
            return None, (ticker, reason), False
        
        if len(df_ticker) <= 300:
            return None, (ticker, f"For lite data: {len(df_ticker)} rader"), False
        
        df_ticker_features = beregn_avanserte_features(df_ticker)
        
        if df_ticker_features.empty:
            return None, (ticker, "Feature engineering feilet"), False
        
        # === CACHE LOGIKK ===
        last_date = df_ticker_features.index.max().strftime('%Y-%m-%d')
        cache_key = _get_cache_key(ticker, valgt_horisont, valgt_maal/100, valgt_stop/100, last_date)
        cache_path = _get_cache_path(cache_key)
        
        predictor = None
        cache_hit = False
        
        # Prøv å laste fra cache
        if use_cache and _is_cache_valid(cache_path):
            cached = _load_cached_model(cache_path)
            if cached is not None:
                predictor = cached
                cache_hit = True
        
        # Tren ny modell hvis ikke cached
        if predictor is None:
            predictor = EnsembleStockPredictor(
                horisont=valgt_horisont, 
                target_pct=valgt_maal/100,
                stop_pct=valgt_stop/100
            )
            
            if not predictor.fit(df_ticker_features, validate=True):
                return None, (ticker, "Modelltrening feilet"), False
            
            # Lagre til cache
            if use_cache:
                _save_model_to_cache(predictor, cache_path)
        
        score = predictor.predict_proba(df_ticker_features)
        
        if score < terskel:
            return None, None, cache_hit  # Ikke feil, bare under terskel
        
        # Beregn konfidensintervall
        _, lower, upper = beregn_konfidensintervall(predictor, df_ticker_features)
        
        # Kjør backtest for historisk treffsikkerhet
        backtest_result = predictor.backtest(df_ticker_features, lookback_signals=20)
        
        vurdering, _ = gi_vurdering(score)
        
        # === FUNDAMENTAL DATA ===
        fund_data = None
        fund_score = None
        if FUNDAMENTAL_AVAILABLE:
            try:
                fund_data = fundamental_data.hent_fundamental_data(ticker, use_cache=True)
                if fund_data:
                    fund_score, _ = fundamental_data.beregn_fundamental_score(fund_data)
            except:
                pass
        
        # === INSIDER DATA ===
        insider_result = None
        insider_score = 0
        insider_boost = 0
        if INSIDER_AVAILABLE and insider_handler is not None:
            try:
                insider_result = insider_monitor.beregn_insider_score(ticker, insider_handler)
                insider_score = insider_result.get("score", 0)
                # Insider confidence boost: sterke kjøp fra innsidere → boost AI score
                insider_boost = _apply_insider_confidence_boost(score, insider_score)
                if insider_boost != 0:
                    score = min(100, max(0, score + insider_boost))
            except Exception:
                pass
        
        resultat = {
            "Ticker": ticker,
            "AI Score": score,
            "Konfidens": f"{lower:.0f}-{upper:.0f}%",
            "Vurdering": vurdering,
            "Pris": df_ticker_features['Close'].iloc[-1],
            "RSI": df_ticker_features['RSI'].iloc[-1],
            "ADX": df_ticker_features['ADX'].iloc[-1] if 'ADX' in df_ticker_features.columns else 0,
            "Vol_Trend": df_ticker_features['Vol_Trend'].iloc[-1] if 'Vol_Trend' in df_ticker_features.columns else 1.0,
            "Regime Ret": round(float(df_ticker_features['Regime_Rolling_Return'].iloc[-1]) * 100, 2) if 'Regime_Rolling_Return' in df_ticker_features.columns else 0.0,
            "Regime Vol": round(float(df_ticker_features['Regime_Volatility'].iloc[-1]) * 100, 2) if 'Regime_Volatility' in df_ticker_features.columns else 0.0,
            "Regime P0": round(float(df_ticker_features['Regime_Prob_0'].iloc[-1]) * 100, 1) if 'Regime_Prob_0' in df_ticker_features.columns else 0.0,
            "Regime P1": round(float(df_ticker_features['Regime_Prob_1'].iloc[-1]) * 100, 1) if 'Regime_Prob_1' in df_ticker_features.columns else 0.0,
            "Regime P2": round(float(df_ticker_features['Regime_Prob_2'].iloc[-1]) * 100, 1) if 'Regime_Prob_2' in df_ticker_features.columns else 0.0,
            "predictor": predictor,
            "backtest": backtest_result,
            "cached": cache_hit,
            # Fundamental data
            "fundamental": fund_data,
            "Fund Score": fund_score,
            # Insider data
            "insider": insider_result,
            "Insider Score": insider_score,
            "insider_boost": insider_boost
        }
        return resultat, None, cache_hit
        
    except Exception as e:
        return None, (ticker, f"Feil: {str(e)[:50]}"), False


def kjor_parallell_analyse(tickers, df_full, valgt_horisont, valgt_maal, valgt_stop, terskel, max_workers=4, use_cache=True, progress_callback=None, insider_handler=None):
    """
    Kjører analyse på flere tickers parallelt med caching.
    Returnerer (resultater, feilede_tickers, cache_stats).
    
    Args:
        progress_callback: Optional callable(completed, total, ticker) for progress updates
        insider_handler: Pre-hentet liste med insider-handler fra Newsweb
    """
    resultater = []
    feilede_tickers = []
    cache_hits = 0
    cache_misses = 0
    completed = 0
    total = len(tickers)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit alle jobber
        futures = {
            executor.submit(
                _analyser_ticker, ticker, df_full, 
                valgt_horisont, valgt_maal, valgt_stop, terskel, use_cache,
                insider_handler
            ): ticker for ticker in tickers
        }
        
        # Samle resultater etter hvert som de blir ferdige
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                resultat, feil, was_cached = future.result()
                if resultat is not None:
                    resultater.append(resultat)
                    if was_cached:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                elif feil is not None:
                    feilede_tickers.append(feil)
            except Exception as e:
                feilede_tickers.append((ticker, f"Executor feil: {str(e)[:40]}"))
            
            # Oppdater progress
            completed += 1
            if progress_callback:
                progress_callback(completed, total, ticker)
    
    cache_info = {'hits': cache_hits, 'misses': cache_misses}
    return resultater, feilede_tickers, cache_info


# =============================================================================
# STREAMLIT UI
# =============================================================================

def gi_vurdering(score, lower=None, upper=None):
    """Returnerer tekstlig vurdering basert på AI Score."""
    if score >= 75:
        return "Sterkt Kjøp", "#00C805"
    if score >= 60:
        return "Kjøp", "#FFC107"
    if score >= 50:
        return "Hold/Vurder", "#FF9800"
    return "Nøytral", "#9E9E9E"


@st.dialog("AI Scanner — Deep Dive", width="large")
def _vis_ai_popup():
    """Popup med chart (inkl. historisk AI Score) og nøkkeldata for valgt ticker."""
    info = st.session_state.get('_ai_popup')
    if not info:
        return

    valgt_ticker = info['ticker']
    df_full = info['df_full']
    df_res = info['df_res']
    ml_results = info.get('ml_results', [])

    # Finn predictor
    predictor = None
    ticker_result = None
    for r in ml_results:
        if r['Ticker'] == valgt_ticker:
            predictor = r['predictor']
            ticker_result = r
            break

    if not predictor:
        st.warning(f"Ingen AI-data for {valgt_ticker}")
        return

    df_view = df_full[df_full['Ticker'] == valgt_ticker].copy()
    df_view_features = beregn_avanserte_features(df_view)

    if df_view_features.empty:
        st.warning("Ikke nok data for analyse.")
        return

    score = df_res[df_res['Ticker'] == valgt_ticker]['AI Score'].values[0]
    vurdering, color = gi_vurdering(score)
    _, lower, upper = beregn_konfidensintervall(predictor, df_view_features)
    ml_trend = predictor.predict_historical(df_view_features, dager=365)
    current_price = df_view_features['Close'].iloc[-1]

    # === HEADER ===
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
                padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
        <div>
            <span style="font-size: 22px; font-weight: 700; color: #fff;">{valgt_ticker}</span>
            <span style="color: #8892b0; margin-left: 8px;">Pris: {current_price:.2f}</span>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 2rem; font-weight: 700; color: {color};">{score:.0f}%</span>
            <span style="color: {color}; margin-left: 6px;">{vurdering}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === CHART MED AI SCORE SUBCHART ===
    end_date = df_view_features.index.max()
    start_date = end_date - pd.DateOffset(days=365)
    df_plot = df_view_features[df_view_features.index >= start_date].copy()
    ml_plot = ml_trend[ml_trend.index >= start_date] if not ml_trend.empty else pd.Series()

    if LWC_INSTALLED and not df_plot.empty:
        _render_ai_chart(
            df=df_plot, ml_scores=ml_plot, ticker=valgt_ticker,
            indicators={'sma_50': True, 'sma_200': True, 'bb': False},
            show_rsi=False,
        )

    # === NØKKELTALL ===
    metrics_df = df_view_features.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RSI", f"{metrics_df['RSI']:.0f}")
    adx_val = metrics_df.get('ADX', 0) if pd.notna(metrics_df.get('ADX')) else 0
    m2.metric("ADX", f"{adx_val:.0f}")
    mfi_val = metrics_df.get('MFI', 50) if pd.notna(metrics_df.get('MFI')) else 50
    m3.metric("MFI", f"{mfi_val:.0f}")
    vol_ratio = metrics_df.get('Vol_Ratio_20', 1.0) if pd.notna(metrics_df.get('Vol_Ratio_20')) else 1.0
    m4.metric("Volum", f"{vol_ratio:.1f}x")
    m5.metric("Konfidens", f"{lower:.0f}-{upper:.0f}%")

    # === COMPACT EXTRAS ===
    # Fundamental + Insider i en rad
    fund_data = ticker_result.get('fundamental') if ticker_result else None
    fund_score = ticker_result.get('Fund Score') if ticker_result else None
    insider_data = ticker_result.get('insider') if ticker_result else None

    extras = []
    if fund_data and fund_score is not None:
        extras.append("Fundamentals")
    if insider_data and (insider_data.get('antall_kjøp', 0) > 0 or insider_data.get('antall_salg', 0) > 0):
        extras.append("Insider")

    if extras:
        popup_tabs = st.tabs(extras)
        tab_i = 0

        if fund_data and fund_score is not None:
            with popup_tabs[tab_i]:
                score_color = "#64ffda" if fund_score >= 70 else "#ffaa00" if fund_score >= 50 else "#ff4444"
                st.markdown(f"**Fundamental Score:** <span style='color:{score_color}'>{fund_score:.0f}/100</span>", unsafe_allow_html=True)
                fc1, fc2, fc3, fc4 = st.columns(4)
                pe = fund_data.get('pe_trailing') or fund_data.get('pe_forward')
                if pe: fc1.metric("P/E", f"{pe:.1f}")
                pb = fund_data.get('pb_ratio')
                if pb: fc2.metric("P/B", f"{pb:.2f}")
                roe = fund_data.get('roe')
                if roe: fc3.metric("ROE", f"{roe*100:.1f}%")
                div_yield = fund_data.get('dividend_yield')
                if div_yield: fc4.metric("Utbytte", f"{div_yield:.1f}%")
            tab_i += 1

        if insider_data and (insider_data.get('antall_kjøp', 0) > 0 or insider_data.get('antall_salg', 0) > 0):
            with popup_tabs[tab_i]:
                ins_s = insider_data.get('score', 0)
                ins_color = "#64ffda" if ins_s > 20 else "#ffaa00" if ins_s >= 0 else "#ff4444"
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Insider-score", f"{ins_s:+.0f}")
                ic2.metric("Kjøp", insider_data.get('antall_kjøp', 0))
                ic3.metric("Salg", insider_data.get('antall_salg', 0))
                handler_liste = insider_data.get('handler', [])
                if handler_liste:
                    for h in handler_liste[:5]:
                        emoji = "🟢" if h.get('type') == 'kjøp' else "🔴" if h.get('type') == 'salg' else "❓"
                        st.markdown(f"- {emoji} **{h.get('dato', '')}** — {h.get('tittel', '')[:50]}")

    # === TA KNAPP ===
    if st.button("Åpne i Teknisk Analyse", type="primary", key="ai_popup_ta", use_container_width=True):
        st.session_state['valgt_ticker'] = valgt_ticker
        st.session_state['navigate_to'] = "Teknisk Analyse"
        st.rerun()


def vis_beta_side(df_full, tickers):
    """
    Hovedfunksjonen for Beta AI Scanner-siden.
    """
    required_cols = {"Ticker", "Open", "High", "Low", "Close", "Volume"}
    if df_full is None or df_full.empty:
        st.warning("Ingen data tilgjengelig for Beta-modulen.")
        return
    missing = required_cols - set(df_full.columns)
    if missing:
        st.error(f"Mangler nødvendige kolonner for Beta-modulen: {', '.join(sorted(missing))}")
        return
    if not tickers:
        st.warning("Ingen tickers etter filtrering.")
        return

    st.title("AI-Drevet Aksjescanner")
    
    st.caption(f"Analyserer {len(tickers)} likvide aksjer basert på valgt minimumsomsetning.")
    
    # === MARKEDSREGIME VISNING ===
    regime_info = None
    regime_adjustment = 1.0
    use_regime_filter = False
    
    if REGIME_AVAILABLE:
        with st.expander("Markedsregime", expanded=True):
            regime_col1, regime_col2 = st.columns([3, 1])
            
            with regime_col1:
                with st.spinner("Henter markedsregime..."):
                    regime_info = _get_market_regime()
                
                if regime_info:
                    _render_regime_badge(regime_info)
                else:
                    st.info("Regime-data ikke tilgjengelig")
            
            with regime_col2:
                if regime_info:
                    use_regime_filter = st.checkbox(
                        "Juster score", 
                        value=True,
                        help="Reduser AI Score i bear-markeder for mer konservative signaler"
                    )
                    if use_regime_filter:
                        regime_adjustment = _get_regime_score_adjustment(regime_info)
                        if regime_adjustment < 1.0:
                            st.caption(f"Score × {regime_adjustment:.0%}")
                        else:
                            st.caption("Ingen justering")
    
    # Metodikkbeskrivelse
    with st.expander("Metodikk og Modellarkitektur", expanded=False):
        st.markdown("""
        ### Ensemble Machine Learning-modell
        
        Denne modulen benytter en **ensemble-tilnærming** som kombinerer tre kraftige algoritmer:
        
        | Algoritme | Vekt | Styrke |
        |-----------|------|--------|
        | **XGBoost** | Dynamisk | Fanger komplekse ikke-lineære mønstre |
        | **Random Forest** | Dynamisk | Robust mot overfitting, håndterer støy |
        | **Gradient Boosting** | Dynamisk | God på sekvensiell læring |
        
        ### Feature Engineering (30+ indikatorer)
        
        **Momentum-indikatorer:**
        - RSI (14), Williams %R, Rate of Change (5, 10, 20 dager)
        - RSI-pris divergens, Momentum oscillator
        
        **Volatilitetsmål:**
        - Normalisert ATR, Bollinger Band Width, Historisk volatilitet
        - Keltner Channel posisjon, Bollinger %B
        
        **Volumindikatorer:**
        - Volumratio (20/50-dagers), OBV-trend, Money Flow Index (MFI)
        
        **Trendindikatorer:**
        - ADX (trendstyrke), DI+/DI- differanse
        - Pris/SMA-ratioer, MACD histogram-endring
        
        **Prismønstre:**
        - Candlestick body-ratio, skygger, consecutive up/down dager
        
        ### Ytelsesopptimalisering
        
        **Parallell prosessering:**
        - 4 samtidige tråder for akselerasjon
        - 80-90% raskere analyse av større porteføljer
        
        **Intelligent modell-caching:**
        - 24-timers cache for trente modeller
        - Automatisk invalidering ved nye data
        - Drastisk redusert analysetid for repeat-analyser
        
        ### Robuste prediksjoner
        
        **Strengere labels (mål/stop):**
        - **Close-basert target** (ikke future max)
        - **Barrierer**: mål og stop-loss innen horisont
        
        **Walk-Forward + Purging:**
        - Tidsserie-splitt med purged gap (fjerner overlapp)
        - Reduserer lekkasje og “flaks”-treff
        
        **Sample‑vekting + F0.5:**
        - Positiv klasse vektlegges
        - Optimaliseres for **precision/F0.5**
        
        ### Validering og konfidens
        
        **Historisk treffsikkerhet:**
        - Real-time backtest per ticker (siste 20 signaler)
        - Aggregert hit rate og gjennomsnittlig avkastning
        - Transparent ytelsesrapportering
        
        **Markedsregime-integrasjon:**
        - HMM-basert regimedeteksjon (Bull/Neutral/Bear)
        - Dynamisk score-justering basert på markedsforhold
        - Konservative signaler i bear-markeder (ned til 70% av score)
        
        **Insider-integrasjon:**
        - Meldepliktige handler fra Oslo Børs Newsweb (siste 90 dager)
        - Netto kjøp/salg-score per ticker (-100 til +100)
        - Confidence boost/straff: opptil +5p for sterke kjøp, -3p for sterke salg
        
        **Støyfilter:**
        - Avviser lav trendstyrke (ADX) og ekstrem lav volatilitet
        
        **Kalibrering:**
        - Platt‑kalibrering (sigmoid) for mer pålitelige terskler
        """)
    
    st.markdown("---")
    
    # Parametervalg
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        valgt_horisont = st.slider("Horisont (dager)", 5, 30, 10, 
                                   help="Antall dager frem i tid modellen predikerer")
    with col2:
        valgt_maal = st.slider("Mål oppgang (%)", 2.0, 15.0, 5.0, 0.5,
                               help="Minimum prisøkning for å definere 'suksess'")
    with col3:
        valgt_stop = st.slider("Stop-loss (%)", 1.0, 12.0, 3.0, 0.5,
                               help="Stop-barriere innen horisont (strammer labels)")
    with col4:
        terskel = st.slider("Min. AI Score", 1, 100, 55,
                           help="Filtrer resultater med score under denne verdien")
    
    # Cache-innstillinger
    with st.expander("Cache-innstillinger", expanded=False):
        cache_stats = get_cache_stats()
        cache_col1, cache_col2, cache_col3 = st.columns(3)
        cache_col1.metric("Cachede modeller", cache_stats['count'])
        cache_col2.metric("Cache-størrelse", f"{cache_stats['size_mb']} MB")
        if cache_stats['oldest']:
            cache_col3.metric("Eldste", cache_stats['oldest'].strftime('%H:%M'))
        
        use_cache = st.checkbox("Bruk modell-cache", value=True, 
                               help="Gjenbruk tidligere trente modeller for raskere analyse")
        if st.button("Tøm cache"):
            clear_model_cache()
            st.success("Cache tømt!")
            st.rerun()
    
    # === HURTIGKNAPPER FOR PORTEFØLJE ===
    portfolio_tickers = []
    if PORTFOLIO_AVAILABLE:
        try:
            pf = portfolio.load_portfolio()
            portfolio_tickers = list(pf.get('positions', {}).keys())
        except:
            pass
    
    # Vis hurtigknapper
    if portfolio_tickers:
        st.markdown("#### Hurtiganalyse")
        qc1, qc2 = st.columns(2)
        
        with qc1:
            if st.button(f"Alle ({len(tickers)})", type="primary", use_container_width=True):
                st.session_state['ml_quick_tickers'] = None  # None = bruk alle
                st.session_state['ml_run_analysis'] = True
        
        with qc2:
            if portfolio_tickers:
                # Filtrer portefølje til kun tickers som finnes i data
                valid_portfolio = [t for t in portfolio_tickers if t in tickers]
                btn_disabled = len(valid_portfolio) == 0
                if st.button(f"Portefølje ({len(valid_portfolio)})", 
                           use_container_width=True, disabled=btn_disabled):
                    st.session_state['ml_quick_tickers'] = valid_portfolio
                    st.session_state['ml_run_analysis'] = True
                if btn_disabled and portfolio_tickers:
                    st.caption("Ingen portefølje-tickers i datasettet")
            else:
                st.button("Portefølje (0)", use_container_width=True, disabled=True)
                st.caption("Tom portefølje")
        
        st.markdown("---")
    else:
        # Vis standard knapp hvis ingen lister
        if st.button("Start AI-Analyse", type="primary"):
            st.session_state['ml_quick_tickers'] = None
            st.session_state['ml_run_analysis'] = True
    
    # Start analyse hvis knapp trykket
    if st.session_state.get('ml_run_analysis', False):
        st.session_state['ml_run_analysis'] = False  # Reset
        
        # Bruk quick_tickers hvis satt, ellers alle
        analysis_tickers = st.session_state.get('ml_quick_tickers') or tickers
        st.session_state['ml_quick_tickers'] = None  # Reset
        
        import time
        start_tid = time.time()
        
        # Pre-hent insider-handler (én gang, deles av alle tickers)
        insider_handler = None
        if INSIDER_AVAILABLE:
            try:
                insider_handler = insider_monitor.hent_innsidehandler(dager=90)
            except Exception:
                insider_handler = None
        
        # Progress UI
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info(f"Starter analyse av {len(analysis_tickers)} aksjer...")
        
        # Progress callback
        def update_progress(completed, total, ticker):
            pct = completed / total
            progress_bar.progress(pct)
            status_text.info(f"Analysert {completed}/{total} aksjer ({pct*100:.0f}%) — sist: {ticker}")
        
        # Kjør parallell analyse
        resultater, feilede_tickers, cache_info = kjor_parallell_analyse(
            tickers=analysis_tickers,
            df_full=df_full,
            valgt_horisont=valgt_horisont,
            valgt_maal=valgt_maal,
            valgt_stop=valgt_stop,
            terskel=terskel,
            max_workers=4,
            use_cache=use_cache,
            progress_callback=update_progress,
            insider_handler=insider_handler
        )
        
        slutt_tid = time.time()
        tid_brukt = slutt_tid - start_tid
        
        # Rydd opp progress UI
        progress_bar.empty()
        status_text.empty()
        
        # Lagre cache-info
        st.session_state['ml_cache_info'] = cache_info
        
        # Lagre regime-info for justering i visning
        st.session_state['ml_regime_info'] = regime_info
        st.session_state['ml_regime_adjustment'] = regime_adjustment if use_regime_filter else 1.0
        
        # Lagre resultater
        if resultater:
            st.session_state['ml_results'] = resultater
            # Inkluder hit_rate og regime-justert score i DataFrame for visning
            df_data = []
            adj = regime_adjustment if use_regime_filter else 1.0
            for r in resultater:
                # Ekskluder komplekse objekter som ikke skal i tabellen
                exclude_keys = ['predictor', 'backtest', 'cached', 'fundamental', 'insider']
                row = {k: v for k, v in r.items() if k not in exclude_keys}
                
                # Regime-justert score
                original_score = row.get('AI Score', 0)
                if adj < 1.0:
                    row['Justert Score'] = round(original_score * adj, 1)
                
                # Legg til backtest hit_rate hvis tilgjengelig
                if r.get('backtest') and r['backtest'].get('total_signals', 0) > 0:
                    row['Hit Rate'] = r['backtest']['hit_rate']
                    row['Snitt Avk.'] = r['backtest']['avg_return']
                else:
                    row['Hit Rate'] = None
                    row['Snitt Avk.'] = None
                
                # Sjekk exit-signal og strategi-signal for denne aksjen
                ticker = row.get('Ticker', '')
                df_ticker = df_full[df_full['Ticker'] == ticker]
                if not df_ticker.empty:
                    df_tek = logic.beregn_tekniske_indikatorer(df_ticker.copy())
                    
                    # Exit-signal
                    exit_info = logic.sjekk_exit_signaler(df_tek)
                    row['Exit⚠️'] = '⚠️' if exit_info.get('skal_selge', False) else ''
                    
                    # Sjekk om det finnes ferske strategisignaler (siste 5 dager)
                    signaler = logic.sjekk_strategier(df_tek)
                    aktive_strat = []
                    if not signaler.empty and len(signaler) >= 5:
                        for strat_col in signaler.columns:
                            if signaler[strat_col].iloc[-5:].any():
                                aktive_strat.append(strat_col.replace('_', ' '))
                    row['Strategi✓'] = '✓' if aktive_strat else ''
                    row['_aktive_strat'] = aktive_strat  # For visning
                else:
                    row['Exit⚠️'] = ''
                    row['Strategi✓'] = ''
                    row['_aktive_strat'] = []
                
                # Fundamental Score - lagre kun score og emoji, IKKE hele dict
                fund_score = r.get('Fund Score')
                fund_data = r.get('fundamental') or {}
                if fund_score is not None:
                    row['Fund'] = fund_score
                    # Fundamental status som enkel emoji
                    if fund_score >= 70:
                        row['F'] = '🟢'
                    elif fund_score >= 50:
                        row['F'] = '🟡'
                    else:
                        row['F'] = '🔴'
                else:
                    row['Fund'] = None
                    row['F'] = ''
                
                # Lagre fundamental data som intern kolonne (ikke vises)
                row['_fundamental'] = fund_data
                
                # Ekstra nøkkeltall for visning
                if fund_data:
                    row['_pe'] = fund_data.get('pe_trailing') or fund_data.get('pe_forward')
                    row['_roe'] = fund_data.get('roe')
                    row['_div'] = fund_data.get('dividend_yield')
                
                # Insider-info
                insider_data = r.get('insider') or {}
                ins_score = r.get('Insider Score', 0)
                ins_boost = r.get('insider_boost', 0)
                if ins_score != 0 or insider_data.get('antall_kjøp', 0) > 0 or insider_data.get('antall_salg', 0) > 0:
                    if ins_score > 20:
                        row['Ins.'] = f"🟢 +{ins_score:.0f}"
                    elif ins_score > 0:
                        row['Ins.'] = f"🟡 +{ins_score:.0f}"
                    elif ins_score < -20:
                        row['Ins.'] = f"🔴 {ins_score:.0f}"
                    elif ins_score < 0:
                        row['Ins.'] = f"🟡 {ins_score:.0f}"
                    else:
                        row['Ins.'] = f"⚪ {ins_score:.0f}"
                else:
                    row['Ins.'] = '—'
                row['_insider'] = insider_data
                row['_insider_boost'] = ins_boost
                
                df_data.append(row)
            
            # Sorter etter justert score hvis tilgjengelig, ellers original
            sort_col = 'Justert Score' if adj < 1.0 else 'AI Score'
            st.session_state['ml_df'] = pd.DataFrame(df_data).sort_values(by=sort_col, ascending=False)
            st.session_state['ml_terskel'] = terskel
            st.session_state['ml_tid'] = tid_brukt
        else:
            st.session_state['ml_results'] = []
            st.session_state['ml_df'] = pd.DataFrame()
            st.warning("Ingen aksjer møtte kravet til AI-score.")
        
        # Vis feilede tickers
        if feilede_tickers:
            with st.expander(f"{len(feilede_tickers)} tickers ble hoppet over", expanded=False):
                error_df = pd.DataFrame(feilede_tickers, columns=['Ticker', 'Årsak'])
                st.dataframe(error_df, width="stretch", hide_index=True)
    
    # Vis resultater
    if 'ml_df' in st.session_state and not st.session_state['ml_df'].empty:
        df_res = st.session_state['ml_df'].copy()
        
        # Fjern interne kolonner fra visning
        display_cols = [c for c in df_res.columns if not c.startswith('_')]
        df_res_display = df_res[display_cols]
        
        saved_terskel = st.session_state.get('ml_terskel', terskel)
        tid_brukt = st.session_state.get('ml_tid', 0)
        cache_info = st.session_state.get('ml_cache_info', {'hits': 0, 'misses': 0})
        saved_regime_info = st.session_state.get('ml_regime_info')
        saved_regime_adj = st.session_state.get('ml_regime_adjustment', 1.0)
        
        # === APPLE-INSPIRERT HEADER ===
        score_col = 'Justert Score' if 'Justert Score' in df_res.columns else 'AI Score'
        
        # Telle opp highlights
        n_total = len(df_res)
        n_strong = len(df_res[df_res[score_col] >= 75])
        n_fund = len(df_res[(df_res['Fund'].notna()) & (df_res['Fund'] >= 70)]) if 'Fund' in df_res.columns else 0
        n_strat = len(df_res[df_res['Strategi✓'] == '✓']) if 'Strategi✓' in df_res.columns else 0
        n_exit = len(df_res[df_res['Exit⚠️'] == '⚠️']) if 'Exit⚠️' in df_res.columns else 0
        n_insider = len(df_res[df_res['Ins.'].str.startswith('🟢', na=False)]) if 'Ins.' in df_res.columns else 0
        
        # Kompakt header med key metrics
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    border-radius: 16px; padding: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                    <h2 style="margin: 0; color: #ffffff; font-weight: 600;">{n_total} kandidater funnet</h2>
                    <p style="margin: 4px 0 0 0; color: #8892b0; font-size: 0.9rem;">
                        AI Score ≥ {saved_terskel}% • {tid_brukt:.1f}s
                    </p>
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div style="text-align: center;">
                        <span style="font-size: 1.8rem; font-weight: 700; color: #64ffda;">{n_strong}</span>
                        <p style="margin: 0; color: #8892b0; font-size: 0.75rem;">Sterke kjøp</p>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 1.8rem; font-weight: 700; color: #bd93f9;">{n_pv}</span>
                        <p style="margin: 0; color: #8892b0; font-size: 0.75rem;">Pattern Vision</p>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 1.8rem; font-weight: 700; color: #50fa7b;">{n_fund}</span>
                        <p style="margin: 0; color: #8892b0; font-size: 0.75rem;">Sterke fundamentals</p>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 1.8rem; font-weight: 700; color: #e6c07b;">{n_insider}</span>
                        <p style="margin: 0; color: #8892b0; font-size: 0.75rem;">Insider-kjøp</p>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 1.8rem; font-weight: 700; color: #ffb86c;">{n_strat}</span>
                        <p style="margin: 0; color: #8892b0; font-size: 0.75rem;">Strategi-signal</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # === KOMPAKT INSIGHTS BAR (kun hvis relevant) ===
        insights = []
        if n_exit > 0:
            insights.append(f"{n_exit} med exit-signal")
        if saved_regime_info and saved_regime_adj < 1.0:
            insights.append(f"{saved_regime_info['emoji']} Markedsregime justert")
        
        if insights:
            st.caption(" • ".join(insights))
        
        # === FANER: Alle / Portefølje / Watchlist ===
        watchlist = load_watchlist()
        portfolio_tickers = []
        if PORTFOLIO_AVAILABLE:
            try:
                pf = portfolio.load_portfolio()
                portfolio_tickers = list(pf.get('positions', {}).keys())
            except:
                pass
        
        # Filtrer DataFrames for hver fane
        df_portfolio = df_res[df_res['Ticker'].isin(portfolio_tickers)] if portfolio_tickers else pd.DataFrame()
        df_watchlist = df_res[df_res['Ticker'].isin(watchlist)] if watchlist else pd.DataFrame()
        
        # Lag fane-titler med antall
        tab_alle = f"Alle ({len(df_res)})"
        tab_portfolio = f"Portefølje ({len(df_portfolio)})"
        tab_watchlist = f"Watchlist ({len(df_watchlist)})"
        
        tab1, tab2, tab3 = st.tabs([tab_alle, tab_portfolio, tab_watchlist])
        
        def render_ai_table(df_view, tab_key):
            """Renderer interaktiv AI-tabell — klikk på rad for popup."""
            if df_view.empty:
                st.info("Ingen aksjer i denne kategorien.")
                return None
            
            df_view = df_view.copy()
            
            # Fjern interne kolonner
            display_cols = [c for c in df_view.columns if not c.startswith('_')]
            df_display = df_view[display_cols]
            
            # Definer kolonner som skal vises i rekkefølge
            visible_columns = ['Ticker', 'AI Score', 'Vurdering', 'F', 'Fund', 'Ins.', 'RSI', 'Strategi✓', 'Exit⚠️']
            if 'Justert Score' in df_display.columns:
                visible_columns.insert(2, 'Justert Score')
            
            # Filtrer til kun synlige kolonner som eksisterer
            visible_columns = [c for c in visible_columns if c in df_display.columns]
            df_show = df_display[visible_columns]
            
            event = st.dataframe(
                df_show,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"ai_table_{tab_key}",
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "AI Score": st.column_config.ProgressColumn(
                        "AI", format="%.0f%%", min_value=0, max_value=100, width="small"
                    ),
                    "Justert Score": st.column_config.ProgressColumn(
                        "Justert", format="%.0f%%", min_value=0, max_value=100, width="small"
                    ),
                    "Vurdering": st.column_config.TextColumn("Signal", width="small"),
                    "F": st.column_config.TextColumn("F", width="small", help="Fundamental: God/OK/Svak"),
                    "Fund": st.column_config.NumberColumn("Score", format="%.0f", width="small", help="Fundamental Score 0-100"),
                    "Ins.": st.column_config.TextColumn("Ins.", width="small", help="Insider-score: Netto kjøp/Netto salg"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.0f", width="small", help="RSI indikator"),
                    "Strategi✓": st.column_config.TextColumn("Strat", width="small", help="Strategi-signal"),
                    "Exit⚠️": st.column_config.TextColumn("Exit", width="small", help="Exit-signal"),
                }
            )
            
            selected_rows = event.selection.rows if event.selection else []
            
            # Popup ved valg
            if selected_rows:
                valgt_idx = selected_rows[0]
                valgt_ticker = df_view.iloc[valgt_idx]['Ticker']

                st.session_state['_ai_popup'] = {
                    'ticker': valgt_ticker,
                    'df_full': df_full,
                    'df_res': df_res,
                    'ml_results': st.session_state.get('ml_results', []),
                }
                return valgt_ticker
            return None
        
        # Render tabeller i hver fane
        with tab1:
            st.markdown("##### Alle AI-kandidater")
            valgt_alle = render_ai_table(df_res, "alle")
        
        with tab2:
            if portfolio_tickers:
                st.markdown(f"##### Dine {len(portfolio_tickers)} posisjoner med AI-score")
                if not df_portfolio.empty:
                    valgt_portfolio = render_ai_table(df_portfolio, "portfolio")
                else:
                    st.info("Ingen av dine posisjoner har høy nok AI-score for denne skanningen.")
            else:
                st.info("Du har ingen posisjoner i porteføljen ennå.")
        
        with tab3:
            st.markdown("##### Din Watchlist")
            if watchlist:
                if not df_watchlist.empty:
                    valgt_watchlist = render_ai_table(df_watchlist, "watchlist")
                else:
                    st.info("Ingen watchlist-aksjer har høy nok AI-score for denne skanningen.")
                    st.caption(f"Watchlist: {', '.join(watchlist)}")
            else:
                st.info("Din watchlist er tom. Velg en aksje i 'Alle'-fanen og klikk 'Legg til Watchlist'.")
        
        # Finn valgt ticker fra hvilken som helst fane (popup håndterer resten)
        valgt_ticker = valgt_alle or (valgt_portfolio if 'valgt_portfolio' in dir() else None) or (valgt_watchlist if 'valgt_watchlist' in dir() else None)
        
        st.caption("Klikk på en rad for chart og detaljer. AI-prediksjoner gir ingen garanti for fremtidig avkastning.")

        # Popup utenfor tabs — @st.dialog må kalles fra hoved-script-kroppen
        if st.session_state.get('_ai_popup'):
            _vis_ai_popup()