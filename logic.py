# logic.py
import pandas as pd
import numpy as np
import config
from log_config import get_logger

logger = get_logger(__name__)


def beregn_tekniske_indikatorer(df):
    """Beregner alle tekniske indikatorer for analyse."""
    if df.empty:
        return df
        
    df = df.copy()
    
    # RSI (Wilder's Smoothing / EMA — matcher TradingView og Bloomberg)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # SMA
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)  # behold indeks (pandas-safe)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 52-ukers High/Low
    df['High_52w'] = df['High'].rolling(window=252).max()
    df['Low_52w'] = df['Low'].rolling(window=252).min()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)
    
    # Ichimoku Cloud (korrekt implementasjon)
    # Tenkan-sen (Conversion Line): 9-perioder
    df['Tenkan'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    # Kijun-sen (Base Line): 26-perioder
    df['Kijun'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    # Senkou Span A (Leading Span A): snitt av Tenkan+Kijun, forskjøvet 26 perioder FREMOVER
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    # Senkou Span B (Leading Span B): 52-perioder, forskjøvet 26 perioder FREMOVER
    df['Senkou_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    # Chikou Span (Lagging Span): close forskjøvet 26 perioder BAKOVER
    df['Chikou'] = df['Close'].shift(-26)
    
    # Behold gamle kolonnenavn for bakoverkompatibilitet (chart_utils, etc.)
    df['ISA_9'] = df['Tenkan']
    df['ISB_26'] = df['Kijun']
    df['ICS_26'] = df['Chikou']
    
    return df


def sjekk_strategier(df):
    """Implementerer tekniske strategier med strenge krav."""
    signaler = pd.DataFrame(index=df.index)
    
    col_list = ['Kort_Sikt_RSI', 'Golden_Cross', 'Momentum_Burst', 
                'Ichimoku_Breakout', 'Wyckoff_Spring', 'Bull_Race_Prep', 'VCP_Pattern', 'Pocket_Pivot',
                'Strength_Pullback']
    
    for c in col_list:
        signaler[c] = False

    if df.empty:
        return signaler

    vol_avg_20 = df['Volume'].rolling(20).mean()

    # === KORT SIKT RSI ===
    if 'SMA_200' in df.columns and 'RSI' in df.columns:
        over_sma200 = df['Close'] > df['SMA_200']
        rsi_oversold = df['RSI'] < 30
        rsi_was_healthy = df['RSI'].rolling(10).max() > 50
        
        if 'SMA_50' in df.columns:
            not_falling = df['Close'] > df['SMA_50'] * 0.95
        else:
            not_falling = True
        
        mask = over_sma200 & rsi_oversold & rsi_was_healthy & not_falling
        signaler['Kort_Sikt_RSI'] = mask.fillna(False)
    
    # === GOLDEN CROSS ===
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        cross = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        sma50_rising = df['SMA_50'] > df['SMA_50'].shift(5)
        sma200_rising = df['SMA_200'] > df['SMA_200'].shift(20)
        price_above = df['Close'] > df['SMA_50']
        
        mask = cross & sma50_rising & sma200_rising & price_above
        signaler['Golden_Cross'] = mask.fillna(False)
    
    # === MOMENTUM BURST ===
    high_20 = df['High'].shift(1).rolling(20).max()
    breakout = df['Close'] > high_20
    high_volume = df['Volume'] > vol_avg_20 * 2
    
    if 'SMA_50' in df.columns:
        over_sma50 = df['Close'] > df['SMA_50']
    else:
        over_sma50 = True
    
    if 'RSI' in df.columns:
        not_overbought = df['RSI'] < 70
    else:
        not_overbought = True
    
    mask = breakout & high_volume & over_sma50 & not_overbought
    signaler['Momentum_Burst'] = mask.fillna(False)
    
    # === BULL RACE PREP ===
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        squeeze = bb_width < bb_width.rolling(100).quantile(0.2)
        breakout = df['Close'] > df['BB_Upper']
        high_vol = df['Volume'] > vol_avg_20 * 1.5
        
        if 'SMA_50' in df.columns:
            uptrend = df['Close'] > df['SMA_50']
        else:
            uptrend = True
        
        mask = squeeze.shift(1) & breakout & high_vol & uptrend
        signaler['Bull_Race_Prep'] = mask.fillna(False)
    
    # === WYCKOFF SPRING ===
    support = df['Low'].rolling(window=40).min().shift(1)
    spring = (df['Low'] < support) & (df['Close'] > support)
    range_today = df['High'] - df['Low']
    strong_close = (df['Close'] - df['Low']) > (range_today * 0.6)
    vol_surge = df['Volume'] > df['Volume'].shift(1)
    
    mask = spring & strong_close & vol_surge
    signaler['Wyckoff_Spring'] = mask.fillna(False)

    # === ICHIMOKU BREAKOUT (korrigert med korrekt Kumo-sky) ===
    if all(col in df.columns for col in ['Senkou_A', 'Senkou_B', 'Tenkan', 'Kijun']):
        # Kumo-topp: høyeste av Senkou A og B (den faktiske skyen)
        kumo_top = pd.concat([df['Senkou_A'], df['Senkou_B']], axis=1).max(axis=1)
        # Pris bryter opp gjennom skyen
        was_below = df['Close'].shift(1) < kumo_top.shift(1)
        now_above = df['Close'] > kumo_top
        # Tenkan over Kijun (bullish momentum)
        tk_bullish = df['Tenkan'] > df['Kijun']
        # Chikou Span over pris for 26 dager siden (bekreftelse)
        chikou_ok = df['Close'] > df['Close'].shift(26)
        
        mask = was_below & now_above & tk_bullish & chikou_ok
        signaler['Ichimoku_Breakout'] = mask.fillna(False)
    elif all(col in df.columns for col in ['ISA_9', 'ISB_26']):
        # Fallback til gammel metode hvis nye kolonner mangler
        kumo_top = pd.concat([df['ISA_9'], df['ISB_26']], axis=1).max(axis=1)
        was_below = df['Close'].shift(1) < kumo_top.shift(1)
        now_above = df['Close'] > kumo_top
        tk_bullish = df['ISA_9'] > df['ISB_26']
        mask = was_below & now_above & tk_bullish
        signaler['Ichimoku_Breakout'] = mask.fillna(False)
        
    # === VCP (MINERVINI) — med kontraksjonsdeteksjon ===
    if all(col in df.columns for col in ['SMA_50', 'SMA_150', 'SMA_200', 'High_52w', 'Low_52w']):
        # Stage 2 krav: Moving average-stacking
        trend_ok = (df['Close'] > df['SMA_50']) & \
                   (df['SMA_50'] > df['SMA_150']) & \
                   (df['SMA_150'] > df['SMA_200']) & \
                   (df['SMA_200'] > df['SMA_200'].shift(20))
        
        near_high = df['Close'] > df['High_52w'] * 0.75
        above_low = df['Close'] > df['Low_52w'] * 1.30
        
        # Kontraksjonsdeteksjon: progressivt trangere range over 3 vinduer
        # Vindu 1: dag -60 til -40, Vindu 2: dag -40 til -20, Vindu 3: dag -20 til nå
        range_1 = (df['High'].rolling(20).max().shift(40) - df['Low'].rolling(20).min().shift(40)) / df['Close'].shift(40)
        range_2 = (df['High'].rolling(20).max().shift(20) - df['Low'].rolling(20).min().shift(20)) / df['Close'].shift(20)
        range_3 = (df['High'].rolling(20).max() - df['Low'].rolling(20).min()) / df['Close']
        
        # Minst 2 av 3 vinduer viser kontraksjon
        kontraksjon_1_2 = range_2 < range_1
        kontraksjon_2_3 = range_3 < range_2
        har_kontraksjoner = kontraksjon_1_2 | kontraksjon_2_3  # minst én kontraksjon
        
        # Volum-kontraksjon: siste 10 dagars volum < forrige 20 dagars volum × 0.8
        vol_siste_10 = df['Volume'].rolling(10).mean()
        vol_forrige_20 = df['Volume'].rolling(20).mean().shift(10)
        vol_kontraksjon = vol_siste_10 < (vol_forrige_20 * 0.8)
        
        # Tightness: daglig spread
        daily_spread = (df['High'] - df['Low']) / df['Close']
        tightness = daily_spread.rolling(10).mean() < 0.035
        
        mask = trend_ok & near_high & above_low & har_kontraksjoner & tightness & vol_kontraksjon
        signaler['VCP_Pattern'] = mask.fillna(False)

    # === POCKET PIVOT (Gil Morales / Chris Kacher) ===
    # Identifiserer institusjonell akkumulering med høy treffprosent
    if 'SMA_50' in df.columns and 'RSI' in df.columns:
        # 1. I opptrend (pris over SMA 50, SMA 50 stigende)
        uptrend = (df['Close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_50'].shift(10))
        
        # 2. Ikke for ekstendert (maks 10% over SMA 50)
        not_extended = df['Close'] < df['SMA_50'] * 1.10
        
        # 3. Volum > høyeste volum på ned-dager siste 10 dager
        # Finn ned-dager (close < open)
        down_days = df['Close'] < df['Open']
        down_volume = df['Volume'].where(down_days, 0)
        max_down_vol_10 = down_volume.rolling(10).max()
        pocket_volume = df['Volume'] > max_down_vol_10
        
        # 4. Lukker i øvre 50% av dagens range (styrketegn)
        daily_range = df['High'] - df['Low']
        close_position = (df['Close'] - df['Low']) / daily_range.replace(0, np.nan)
        strong_close = close_position > 0.5
        
        # 5. RSI ikke overkjøpt (under 70) og ikke oversolgt (over 40)
        rsi_healthy = (df['RSI'] > 40) & (df['RSI'] < 70)
        
        # 6. Pris ikke under forrige dags lav (unngå svakhet)
        above_prev_low = df['Close'] > df['Low'].shift(1)
        
        # 7. Volum over snittet (bekrefter interesse)
        vol_above_avg = df['Volume'] > vol_avg_20
        
        mask = uptrend & not_extended & pocket_volume & strong_close & rsi_healthy & above_prev_low & vol_above_avg
        signaler['Pocket_Pivot'] = mask.fillna(False)
    else:
        signaler['Pocket_Pivot'] = False

    # === STRENGTH PULLBACK (Pullback til støtte i sterk trend) ===
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        close = df['Close']
        sma50 = df['SMA_50']
        sma200 = df['SMA_200']
        vol = df['Volume']
        
        # SMA20 (beregn hvis ikke finnes)
        if 'SMA_20' in df.columns:
            sma20 = df['SMA_20']
        else:
            sma20 = close.rolling(20).mean()
        
        # 1. Stage 2 trend (sterk bull) - pris > SMA50 > SMA200, SMA50 stigende
        trend_strong = (
            (close > sma50) & 
            (sma50 > sma200) & 
            (sma50 > sma50.shift(20))  # SMA50 stigende over 20 dager
        )
        
        # 2. Nær støtte (SMA20) - innenfor ±2%
        near_support = (close > sma20 * 0.98) & (close < sma20 * 1.02)
        
        # 3. Volum dry-up på pullback (ingen distribusjon)
        vol_dry = vol < vol_avg_20 * 0.85
        
        # 4. Konstruktiv pullback (higher lows) - ikke panikksalg
        low = df['Low']
        higher_lows = (low > low.shift(1)) | (low > low.shift(2))
        
        # 5. Entry trigger: Bounce tilbake (close over yesterday's high)
        bounce = close > df['High'].shift(1)
        
        # 6. Trend-styrke filter (ADX eller SMA50 slope)
        if 'ADX' in df.columns:
            strong_trend = df['ADX'] > 25
        else:
            # Fallback: bruk slope av SMA50 (>2% over 10 dager)
            sma50_slope = (sma50 / sma50.shift(10) - 1) * 100
            strong_trend = sma50_slope > 2
        
        # 7. Ikke for dyp pullback (max 8% fra 20-dagers topp)
        recent_high = close.rolling(20).max()
        pullback_depth = ((recent_high - close) / recent_high * 100)
        shallow_pullback = pullback_depth < 8
        
        mask = (
            trend_strong & 
            near_support & 
            vol_dry & 
            higher_lows &
            bounce &
            strong_trend &
            shallow_pullback
        )
        
        # Cooldown: 10 dager mellom signaler (unngå clustering)
        # Sjekk om det var et signal i de FOREGÅENDE 10 dagene (shift(1) for å ikke blokkere dagens)
        recent_signal = mask.shift(1).rolling(10, min_periods=1).max().fillna(0).astype(bool)
        signaler['Strength_Pullback'] = (mask & ~recent_signal).fillna(False)
    else:
        signaler['Strength_Pullback'] = False

    return signaler


def beregn_signal_kvalitet(df, signal_dato, strategi_key, regime=None):
    """Beregner kvalitetsscore (0-100) for et signal med walk-forward.
    
    Args:
        regime: Valgfri regime-navn for adaptiv vekting.
                Bull → momentum/volum vektes tyngre
                Bear → trend/volatilitet vektes tyngre
    """
    if signal_dato not in df.index:
        return 0, {}
    
    idx = df.index.get_loc(signal_dato)
    if idx < 50:
        return 0, {}
    
    df_hist = df.iloc[:idx+1]
    row = df_hist.iloc[-1]
    close = row['Close']
    
    score = 0
    faktorer = {}
    
    # 1. Volum (0-20) — kombinerer ratio + akselerasjon
    vol_hist = df_hist['Volume'].iloc[-21:-1] if len(df_hist) > 20 else df_hist['Volume'].iloc[:-1]
    vol_avg = vol_hist.mean() if len(vol_hist) > 0 else 1
    vol_ratio = row['Volume'] / vol_avg if vol_avg > 0 else 1
    
    # Volum-akselerasjon: sjekk om volum øker over siste 3-5 dager
    vol_accel_bonus = 0
    if len(df_hist) >= 6:
        siste_5_vol = df_hist['Volume'].iloc[-5:]
        forrige_5_vol = df_hist['Volume'].iloc[-10:-5] if len(df_hist) >= 10 else df_hist['Volume'].iloc[:-5]
        if len(forrige_5_vol) > 0 and forrige_5_vol.mean() > 0:
            accel_ratio = siste_5_vol.mean() / forrige_5_vol.mean()
            # Sjekk stigende trend i volum (dag-for-dag)
            vol_stigende = all(siste_5_vol.iloc[i] >= siste_5_vol.iloc[i-1] * 0.9 for i in range(1, min(4, len(siste_5_vol))))
            if accel_ratio >= 1.5 and vol_stigende:
                vol_accel_bonus = 5  # Sterk akselerasjon
            elif accel_ratio >= 1.2:
                vol_accel_bonus = 3  # Moderat akselerasjon
    
    if strategi_key == 'VCP_Pattern':
        vol_score = 20 if vol_ratio < 0.7 else 15 if vol_ratio < 1.0 else 8 if vol_ratio < 1.5 else 3
    else:
        vol_base = 15 if vol_ratio >= 2.0 else 12 if vol_ratio >= 1.5 else 8 if vol_ratio >= 1.0 else 4
        vol_score = min(20, vol_base + vol_accel_bonus)
    
    score += vol_score
    faktorer['volum'] = {'score': vol_score, 'ratio': round(vol_ratio, 2), 'akselerasjon': vol_accel_bonus}
    
    # 2. Trend (0-25)
    sma_50 = df_hist['Close'].iloc[-50:].mean() if len(df_hist) >= 50 else None
    sma_200 = df_hist['Close'].iloc[-200:].mean() if len(df_hist) >= 200 else None
    
    if sma_50 is not None and sma_200 is not None:
        if close > sma_50 > sma_200:
            trend_score = 25
        elif close > sma_50 and close > sma_200:
            trend_score = 20
        elif close > sma_50:
            trend_score = 15
        elif close > sma_200:
            trend_score = 10
        else:
            trend_score = 0
    elif sma_50 is not None:
        trend_score = 15 if close > sma_50 else 5
    else:
        trend_score = 10
    
    score += trend_score
    faktorer['trend'] = {
        'score': trend_score,
        'over_sma50': (close > sma_50) if sma_50 is not None else None
    }
    
    # 3. RSI (0-20) — Wilder's EMA
    if len(df_hist) >= 15:
        delta = df_hist['Close'].diff()
        _gain_s = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        _loss_s = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        _rs = _gain_s / _loss_s.replace(0, np.nan)
        _rsi_series = 100 - (100 / (1 + _rs))
        rsi = float(_rsi_series.iloc[-1]) if not pd.isna(_rsi_series.iloc[-1]) else 50
        if pd.isna(rsi):
            rsi = 50
    else:
        rsi = 50
    
    if 30 <= rsi <= 50:
        rsi_score = 20
    elif 50 < rsi <= 60:
        rsi_score = 15
    elif 60 < rsi <= 70:
        rsi_score = 10
    elif rsi > 70:
        rsi_score = 0
    else:
        rsi_score = 15
    
    score += rsi_score
    faktorer['rsi'] = {'score': rsi_score, 'verdi': round(rsi, 1)}
    
    # 4. Volatilitet (0-15)
    if len(df_hist) >= 15:
        high_low = df_hist['High'] - df_hist['Low']
        high_close = abs(df_hist['High'] - df_hist['Close'].shift())
        low_close = abs(df_hist['Low'] - df_hist['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.iloc[-14:].mean()
        atr_pct = atr / close * 100 if close > 0 else 0
    else:
        atr_pct = 3
    
    atr_score = 15 if atr_pct < 2 else 12 if atr_pct < 4 else 8 if atr_pct < 6 else 3
    score += atr_score
    faktorer['volatilitet'] = {'score': atr_score, 'atr_pct': round(atr_pct, 2)}
    
    # 5. Posisjon (0-20)
    if len(df_hist) >= 50:
        high_52w = df_hist['High'].max()
        low_52w = df_hist['Low'].min()
        if high_52w > low_52w:
            range_pos = (close - low_52w) / (high_52w - low_52w)
            if 0.3 <= range_pos <= 0.7:
                pos_score = 20
            elif 0.2 <= range_pos < 0.3 or 0.7 < range_pos <= 0.85:
                pos_score = 15
            elif range_pos > 0.85:
                pos_score = 5
            else:
                pos_score = 12
        else:
            pos_score = 10
            range_pos = 0.5
    else:
        pos_score = 10
        range_pos = 0.5
    
    score += pos_score
    faktorer['posisjon'] = {'score': pos_score, 'range_pct': round(range_pos * 100, 1)}
    
    # Adaptiv regime-vekting: juster score basert på markedsregime
    if regime:
        # Standard vekter: volum=20, trend=25, rsi=20, volatilitet=15, posisjon=20 (=100)
        # Regime-spesifikke multiplikatorer (normalisert til sum=100)
        REGIME_VEKTER = {
            'Bull Market': {
                'volum': 1.3, 'trend': 0.8, 'rsi': 1.2, 'volatilitet': 0.7, 'posisjon': 1.0
            },
            'Mild Bull': {
                'volum': 1.15, 'trend': 0.9, 'rsi': 1.1, 'volatilitet': 0.85, 'posisjon': 1.0
            },
            'Nøytral': {
                'volum': 1.0, 'trend': 1.0, 'rsi': 1.0, 'volatilitet': 1.0, 'posisjon': 1.0
            },
            'Mild Bear': {
                'volum': 0.8, 'trend': 1.3, 'rsi': 0.9, 'volatilitet': 1.2, 'posisjon': 0.8
            },
            'Bear Market': {
                'volum': 0.7, 'trend': 1.4, 'rsi': 0.8, 'volatilitet': 1.3, 'posisjon': 0.8
            },
        }
        vekter = REGIME_VEKTER.get(regime)
        if vekter:
            # Re-beregn score med regime-vekter
            score = (
                vol_score * vekter['volum'] +
                trend_score * vekter['trend'] +
                rsi_score * vekter['rsi'] +
                atr_score * vekter['volatilitet'] +
                pos_score * vekter['posisjon']
            )
            score = min(100, max(0, round(score)))
            faktorer['regime_vekting'] = regime
    
    return min(score, 100), faktorer


def klassifiser_signal_kvalitet(score, faktorer=None):
    """Returnerer kvalitetsklasse basert på score.
    
    Med faktorer: A-kvalitet krever også trend ≥ 15 og volum ≥ 10
    (ikke bare totalsum ≥ 75). Forhindrer at aksjer med én dominerende
    faktor får A-stempel.
    """
    if score >= 75:
        # Strengere A-krav: sjekk minstekrav per nøkkelfaktor
        if faktorer:
            trend_score = faktorer.get('trend', {}).get('score', 0)
            volum_score = faktorer.get('volum', {}).get('score', 0)
            if trend_score < 15 or volum_score < 10:
                return 'B', '🟡', 'Moderat kvalitet (mangler faktor-bredde)'
        return 'A', '🟢', 'Høy kvalitet'
    elif score >= 55:
        return 'B', '🟡', 'Moderat kvalitet'
    elif score >= 35:
        return 'C', '🟠', 'Lav kvalitet'
    else:
        return 'D', '🔴', 'Unngå'


def hent_strategi_detaljer(key):
    """Returnerer beskrivelse og horisont for en strategi."""
    data = {
        "Kort Sikt (RSI Dip)": {
            "horisont": "Kortsiktig (1-5 dager)",
            "beskrivelse": "**Kriterier:** Pris over SMA 200. RSI under 30. RSI var over 50 siste 10 dager. Pris maks 5% under SMA 50."
        },
        "Momentum Burst": {
            "horisont": "Kortsiktig (1-10 dager)",
            "beskrivelse": "**Kriterier:** Breakout over 20-dagers høy. Volum minst 2x snitt. Pris over SMA 50. RSI under 70."
        },
        "Golden Cross": {
            "horisont": "Langsiktig (3-12 mnd)",
            "beskrivelse": "**Kriterier:** SMA 50 krysser over SMA 200. Begge MA-er stigende. Pris over SMA 50."
        },
        "Ichimoku Breakout": {
            "horisont": "Mellomlang (2-8 uker)",
            "beskrivelse": "**Kriterier:** Pris bryter opp gjennom Kumo-skyen fra undersiden. Tenkan-sen over Kijun-sen."
        },
        "Wyckoff Spring": {
            "horisont": "Kort-mellomlang (3-20 dager)",
            "beskrivelse": "**Kriterier:** Intradag-brudd under 40-dagers støtte, men lukker over. Lukker i øvre 60% av range. Volum økende."
        },
        "Bull Race Prep": {
            "horisont": "Kortsiktig (1-10 dager)",
            "beskrivelse": "**Kriterier:** Bollinger Bands squeeze (nedre 20-persentil). Breakout over øvre band. Volum over 1.5x snitt. Over SMA 50."
        },
        "VCP (Minervini)": {
            "horisont": "Mellomlang (2-8 uker)",
            "beskrivelse": "**Stage 2 VCP:** Pris > SMA 50 > SMA 150 > SMA 200 (stigende). Innen 25% av 52-ukers høy. Volatilitet krymper. Volum under 70% av snitt."
        },
        "Pocket Pivot": {
            "horisont": "Kort-mellomlang (5-20 dager)",
            "beskrivelse": "**Kriterier:** Pris i opptrend over SMA 50. Volum > høyeste volum på ned-dager siste 10 dager. Lukker i øvre 50% av range. RSI 40-70. Maks 10% over SMA 50."
        },
        "Strength Pullback": {
            "horisont": "Kort-mellomlang (5-20 dager)",
            "beskrivelse": "**Kriterier:** Stage 2 trend (pris > SMA50 > SMA200). Pullback til SMA20 (±2%). Volum dry-up (<85% av snitt). Bounce over forrige dags høy. Maks 8% drawdown."
        }
    }
    return data.get(key, {"horisont": "Ukjent", "beskrivelse": "Ingen beskrivelse."})


def finn_stotte_motstand(df, perioder=100):
    """
    Finner støtte- og motstandsnivåer basert på pivot points og volumklynger.
    Returnerer (stotte_nivåer, motstand_nivåer) som lister med dict.
    """
    if len(df) < perioder:
        return [], []
    
    df_period = df.iloc[-perioder:].copy()
    high = df_period['High']
    low = df_period['Low']
    close = df_period['Close']
    volume = df_period['Volume']
    current_price = close.iloc[-1]
    
    nivåer = []
    
    # 1. Pivot Points (swing highs/lows)
    for i in range(5, len(df_period) - 5):
        # Swing High (lokal topp)
        if high.iloc[i] == high.iloc[i-5:i+6].max():
            nivåer.append({
                'pris': float(high.iloc[i]),
                'type': 'motstand' if high.iloc[i] > current_price else 'stotte',
                'styrke': 1,
                'kilde': 'pivot'
            })
        # Swing Low (lokal bunn)
        if low.iloc[i] == low.iloc[i-5:i+6].min():
            nivåer.append({
                'pris': float(low.iloc[i]),
                'type': 'stotte' if low.iloc[i] < current_price else 'motstand',
                'styrke': 1,
                'kilde': 'pivot'
            })
    
    # 2. 52-ukers høy/lav (viktige psykologiske nivåer)
    if len(df) >= 252:
        high_52w = df['High'].iloc[-252:].max()
        low_52w = df['Low'].iloc[-252:].min()
        nivåer.append({'pris': float(high_52w), 'type': 'motstand', 'styrke': 3, 'kilde': '52w_high'})
        nivåer.append({'pris': float(low_52w), 'type': 'stotte', 'styrke': 3, 'kilde': '52w_low'})
    
    # 3. Runde tall (psykologiske nivåer)
    price_range = high.max() - low.min()
    round_interval = 10 if price_range > 100 else 5 if price_range > 20 else 1
    
    lower_round = int(low.min() / round_interval) * round_interval
    upper_round = int(high.max() / round_interval + 1) * round_interval
    
    for rund in range(lower_round, upper_round + 1, round_interval):
        if low.min() * 0.95 < rund < high.max() * 1.05:
            nivåer.append({
                'pris': float(rund),
                'type': 'motstand' if rund > current_price else 'stotte',
                'styrke': 1,
                'kilde': 'round'
            })
    
    # 4. Volumvektet pris-klynger (hvor mest handel skjer)
    try:
        price_bins = pd.cut(close, bins=20)
        vol_profile = volume.groupby(price_bins).sum()
        top_zones = vol_profile.nlargest(3)
        
        for interval in top_zones.index:
            if pd.notna(interval):
                mid_price = (interval.left + interval.right) / 2
                nivåer.append({
                    'pris': float(mid_price),
                    'type': 'motstand' if mid_price > current_price else 'stotte',
                    'styrke': 2,
                    'kilde': 'volume'
                })
    except Exception:
        pass
    
    # Konsolider nivåer som er nære hverandre (innenfor 2.5% - økt fra 1.5%)
    def konsolider_nivåer(nivå_liste, terskel_pct=0.025):  # Økt terskel
        if not nivå_liste:
            return []
        
        sortert = sorted(nivå_liste, key=lambda x: x['pris'])
        konsolidert = [sortert[0]]
        
        for nivå in sortert[1:]:
            siste = konsolidert[-1]
            if abs(nivå['pris'] - siste['pris']) / siste['pris'] < terskel_pct:
                # Slå sammen: behold høyeste styrke og oppdater pris til gjennomsnitt
                total_styrke = siste['styrke'] + nivå['styrke']
                vektet_pris = (siste['pris'] * siste['styrke'] + nivå['pris'] * nivå['styrke']) / total_styrke
                siste['styrke'] = min(total_styrke, 5)  # Maks styrke 5
                siste['pris'] = vektet_pris
            else:
                konsolidert.append(nivå)
        
        return konsolidert
    
    alle_nivåer = konsolider_nivåer(nivåer)
    
    # Filtrer ut nivåer som er for nærme nåværende pris (< 1.5% - økt fra 0.5%)
    filtrert = [n for n in alle_nivåer if abs(n['pris'] - current_price) / current_price > 0.015]
    
    # Sorter etter styrke og returner færre, men bedre nivåer
    stotte = sorted([n for n in filtrert if n['type'] == 'stotte'], key=lambda x: -x['styrke'])[:3]  # Redusert fra 5 til 3
    motstand = sorted([n for n in filtrert if n['type'] == 'motstand'], key=lambda x: -x['styrke'])[:3]  # Redusert fra 5 til 3
    
    # Ytterligere filtrering: kun nivåer med styrke >= 2 eller de 2 beste
    stotte = [n for n in stotte if n['styrke'] >= 2][:2] if any(n['styrke'] >= 2 for n in stotte) else stotte[:2]
    motstand = [n for n in motstand if n['styrke'] >= 2][:2] if any(n['styrke'] >= 2 for n in motstand) else motstand[:2]
    
    return stotte, motstand


def finn_siste_signal_info(df, signaler, strategi_key):
    """Finner informasjon om siste signal for en strategi."""
    if strategi_key not in signaler.columns:
        return {'dato': 'Ingen data', 'dager_siden': float('inf'), 'utvikling_pst': 0.0}
    
    signal_dates = signaler[signaler[strategi_key] == True].index
    
    if len(signal_dates) == 0:
        return {'dato': 'Ingen signal', 'dager_siden': float('inf'), 'utvikling_pst': 0.0}
    
    siste = signal_dates[-1]
    dager = (df.index[-1] - siste).days
    
    try:
        pris_signal = df.loc[siste, 'Close']
        pris_nå = df['Close'].iloc[-1]
        utvikling = ((pris_nå - pris_signal) / pris_signal) * 100
    except:
        utvikling = 0.0
    
    return {'dato': siste.strftime('%Y-%m-%d'), 'dager_siden': dager, 'utvikling_pst': round(utvikling, 2)}


def filtrer_signaler_med_kvalitet(df, signaler, strategi_key, min_kvalitet='C', max_dager_siden=30, min_volum_ratio=0.8, regime=None):
    """Filtrerer signaler basert på kvalitet."""
    if df is None or signaler is None:
        return None
    if len(df) == 0 or signaler.empty or len(df.index) == 0:
        return None
    if strategi_key not in signaler.columns:
        return None
    
    signal_mask = signaler[strategi_key].fillna(False).astype(bool)
    signal_dates = signaler.index[signal_mask]
    if len(signal_dates) == 0:
        return None
    
    kvalitet_grenser = {'A': 75, 'B': 55, 'C': 35, 'D': 0}
    min_score = kvalitet_grenser.get(min_kvalitet, 35)
    siste_dato = df.index[-1]
    
    for signal_dato in reversed(signal_dates):
        dager = (siste_dato - signal_dato).days
        if dager > max_dager_siden:
            continue
        
        score, faktorer = beregn_signal_kvalitet(df, signal_dato, strategi_key, regime=regime)
        if score < min_score:
            continue
        
        vol_ratio = faktorer.get('volum', {}).get('ratio', 0)
        # VCP og Strength_Pullback krever LAV volum (volume dry-up), så de unntas fra min_volum filter
        if strategi_key not in ['VCP_Pattern', 'Strength_Pullback'] and vol_ratio < min_volum_ratio:
            continue
        
        klasse, emoji, beskrivelse = klassifiser_signal_kvalitet(score, faktorer)
        
        try:
            pris_signal = df.loc[signal_dato, 'Close']
            pris_nå = df['Close'].iloc[-1]
            utvikling = ((pris_nå - pris_signal) / pris_signal) * 100 if dager > 0 else 0.0
            
            # Peak-avkastning: høyeste Close etter signal
            idx_signal = df.index.get_loc(signal_dato)
            if idx_signal < len(df) - 1:
                peak_pris = df['Close'].iloc[idx_signal + 1:].max()
                peak_utvikling = ((peak_pris - pris_signal) / pris_signal) * 100
            else:
                peak_utvikling = utvikling
        except Exception:
            utvikling = 0.0
            peak_utvikling = 0.0
        
        return {
            'dato': signal_dato.strftime('%Y-%m-%d'),
            'dager_siden': dager,
            'utvikling_pst': round(utvikling, 2),
            'peak_utvikling_pst': round(peak_utvikling, 2),
            'kvalitet_score': score,
            'kvalitet_klasse': klasse,
            'kvalitet_emoji': emoji,
            'kvalitet_beskrivelse': beskrivelse,
            'faktorer': faktorer
        }
    
    return None


def hent_signaler_for_chart(df, signaler, strategi_keys, start_date=None):
    """Henter signaler med kvalitet for chart-visning."""
    resultat = {}

    if df is None or signaler is None or len(df) == 0 or signaler.empty:
        return {k: pd.DataFrame() for k in (strategi_keys or [])}
    
    for strat_key in (strategi_keys or []):
        if strat_key not in signaler.columns:
            resultat[strat_key] = pd.DataFrame()
            continue
        
        signal_mask = signaler[strat_key].fillna(False).astype(bool)
        if start_date is not None:
            signal_mask = signal_mask[signal_mask.index >= start_date]
        
        signal_dates = signal_mask[signal_mask == True].index
        if len(signal_dates) == 0:
            resultat[strat_key] = pd.DataFrame()
            continue
        
        siste_dato = df.index[-1]
        signal_data = []
        
        for signal_dato in signal_dates:
            if signal_dato not in df.index:
                continue
            
            score, faktorer = beregn_signal_kvalitet(df, signal_dato, strat_key)
            klasse, emoji, beskrivelse = klassifiser_signal_kvalitet(score, faktorer)
            
            row = df.loc[signal_dato]
            dager = (siste_dato - signal_dato).days
            
            try:
                utvikling = ((df['Close'].iloc[-1] - row['Close']) / row['Close']) * 100 if dager > 0 else 0.0
            except Exception:
                utvikling = 0.0
            
            signal_data.append({
                'dato': signal_dato,
                'pris': row['Close'],
                'lav': row['Low'],
                'kvalitet_score': score,
                'kvalitet_klasse': klasse,
                'kvalitet_emoji': emoji,
                'kvalitet_beskrivelse': beskrivelse,
                'utvikling_pst': round(utvikling, 2),
                'dager_siden': dager,
                'volum_ratio': faktorer.get('volum', {}).get('ratio', 0),
                'rsi': faktorer.get('rsi', {}).get('verdi', 50),
                'trend_score': faktorer.get('trend', {}).get('score', 0),
                'faktorer': faktorer
            })
        
        resultat[strat_key] = pd.DataFrame(signal_data).set_index('dato') if signal_data else pd.DataFrame()
    
    return resultat


def beregn_risk_reward(entry, stop, kapital, risiko_pst):
    """Beregner posisjonsstørrelse basert på risiko."""
    if entry <= stop:
        return None
    risiko_kr = kapital * (risiko_pst / 100)
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    antall = int(risiko_kr / risk_per_share)
    return {
        "antall": antall,
        "total_investering": antall * entry,
        "risiko_kr": risiko_kr,
        "target_2r": entry + (risk_per_share * 2)
    }


def beregn_kelly_criterion(vinnrate: float, avg_gevinst: float, avg_tap: float) -> dict:
    """
    Beregner optimal posisjonsstørrelse med Kelly Criterion.
    
    Kelly% = W - (1-W)/R
    Der:
        W = Vinnrate (0-1)
        R = Reward/Risk ratio (avg_gevinst / avg_tap)
    
    Args:
        vinnrate: Vinnrate som desimal (f.eks. 0.55 for 55%)
        avg_gevinst: Gjennomsnittlig gevinst per vinnende trade
        avg_tap: Gjennomsnittlig tap per tapende trade (positiv verdi)
        
    Returns:
        Dict med Kelly-beregninger
    """
    if avg_tap <= 0 or vinnrate <= 0 or vinnrate >= 1:
        return None
    
    # Reward/Risk ratio
    R = avg_gevinst / avg_tap
    
    # Full Kelly
    W = vinnrate
    kelly_pct = W - ((1 - W) / R)
    
    # Kelly kan være negativ = ingen edge, ikke trade
    if kelly_pct <= 0:
        return {
            'kelly_full': 0,
            'kelly_half': 0,
            'kelly_quarter': 0,
            'har_edge': False,
            'reward_risk': round(R, 2),
            'anbefaling': 'Ingen edge - ikke trade dette systemet',
            'forventet_verdi': round((W * avg_gevinst) - ((1-W) * avg_tap), 2)
        }
    
    # Begrens til maks 50% (selv full Kelly bør ikke overstige dette)
    kelly_full = min(kelly_pct, 0.50)
    
    return {
        'kelly_full': round(kelly_full * 100, 1),      # Prosent
        'kelly_half': round(kelly_full * 50, 1),       # Half Kelly
        'kelly_quarter': round(kelly_full * 25, 1),   # Quarter Kelly
        'har_edge': True,
        'reward_risk': round(R, 2),
        'anbefaling': _kelly_anbefaling(kelly_full),
        'forventet_verdi': round((W * avg_gevinst) - ((1-W) * avg_tap), 2)
    }


def _kelly_anbefaling(kelly: float) -> str:
    """Gir anbefaling basert på Kelly-verdi."""
    if kelly >= 0.25:
        return 'Sterk edge! Bruk Half Kelly for sikkerhet.'
    elif kelly >= 0.15:
        return 'God edge. Half Kelly anbefales.'
    elif kelly >= 0.05:
        return 'Moderat edge. Quarter Kelly anbefales.'
    else:
        return 'Svak edge. Vær forsiktig, bruk liten posisjon.'


def beregn_kelly_fra_historikk(transaksjoner: list) -> dict:
    """
    Beregner Kelly Criterion fra historiske transaksjoner.
    
    Args:
        transaksjoner: Liste med dicts som har 'profit' og 'type' keys
        
    Returns:
        Kelly-beregning basert på faktisk historikk
    """
    # Filtrer ut salg med profit-data
    salg = [t for t in transaksjoner if t.get('type') == 'SELL' and 'profit' in t]
    
    if len(salg) < 5:  # Trenger minimum 5 handler for statistikk
        return {
            'kelly_full': 0,
            'kelly_half': 0,
            'kelly_quarter': 0,
            'har_edge': None,
            'anbefaling': f'Trenger minimum 5 handler (har {len(salg)})',
            'antall_handler': len(salg)
        }
    
    # Beregn statistikk
    gevinster = [t['profit'] for t in salg if t['profit'] > 0]
    tap = [abs(t['profit']) for t in salg if t['profit'] < 0]
    
    if not gevinster or not tap:
        return {
            'kelly_full': 0,
            'kelly_half': 0, 
            'kelly_quarter': 0,
            'har_edge': None,
            'anbefaling': 'Trenger både gevinster og tap for beregning',
            'antall_handler': len(salg)
        }
    
    vinnrate = len(gevinster) / len(salg)
    avg_gevinst = sum(gevinster) / len(gevinster)
    avg_tap = sum(tap) / len(tap)
    
    result = beregn_kelly_criterion(vinnrate, avg_gevinst, avg_tap)
    if result:
        result['antall_handler'] = len(salg)
        result['vinnrate'] = round(vinnrate * 100, 1)
        result['avg_gevinst'] = round(avg_gevinst, 0)
        result['avg_tap'] = round(avg_tap, 0)
    
    return result


# === NYE FUNKSJONER: FORBEDRINGER ===

def beregn_relativ_styrke(df, benchmark_df=None, alle_tickers_avkastning=None):
    """
    Beregner IBD-vektet Relative Strength (RS) Rating.
    
    IBD-metode (Investor's Business Daily — industristandard):
        RS = 40% × 3mnd + 20% × 6mnd + 20% × 9mnd + 20% × 12mnd
        
    Nyere prisutvikling vektes tyngre for å fange momentum-akselerasjon.
    
    Returnerer 1-99 persentilrangering der 99 = sterkeste aksjen.
    Hvis alle_tickers_avkastning er gitt, beregnes rangering relativt til universet.
    Ellers brukes benchmark som referanse.
    """
    perioder = {'3mnd': 63, '6mnd': 126, '9mnd': 189, '12mnd': 252}
    vekter = {'3mnd': 0.40, '6mnd': 0.20, '9mnd': 0.20, '12mnd': 0.20}
    
    min_perioder = perioder['3mnd']  # Trenger minst 3 mnd data
    if len(df) < min_perioder:
        return None
    
    # Beregn vektet avkastning for aksjen
    aksje_score = 0.0
    perioder_brukt = {}
    for label, dager in perioder.items():
        if len(df) >= dager:
            avk = (df['Close'].iloc[-1] / df['Close'].iloc[-dager] - 1) * 100
        else:
            # Bruk tilgjengelig data, skalert til perioden
            avk = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            avk = avk * (dager / len(df))  # Annualiserings-justering
        perioder_brukt[label] = round(avk, 2)
        aksje_score += vekter[label] * avk
    
    # Beregn benchmark vektet avkastning
    marked_score = 0.0
    if benchmark_df is not None and len(benchmark_df) >= min_perioder:
        close_col = 'Close' if 'Close' in benchmark_df.columns else benchmark_df.columns[0]
        for label, dager in perioder.items():
            if len(benchmark_df) >= dager:
                avk = (benchmark_df[close_col].iloc[-1] / benchmark_df[close_col].iloc[-dager] - 1) * 100
            else:
                avk = (benchmark_df[close_col].iloc[-1] / benchmark_df[close_col].iloc[0] - 1) * 100
            marked_score += vekter[label] * avk
    else:
        marked_score = 8.0  # Antar ~8% årlig gjennomsnitt for OSEBX
    
    # RS-ratio (aksje vs marked)
    rs_ratio = aksje_score / marked_score if marked_score != 0 else 1.0
    
    # Persentilrangering
    if alle_tickers_avkastning is not None and len(alle_tickers_avkastning) > 5:
        # Ekte persentilrangering vs hele universet
        antall_lavere = sum(1 for v in alle_tickers_avkastning if v < aksje_score)
        rs_rating = int(round(antall_lavere / len(alle_tickers_avkastning) * 98 + 1))
    else:
        # Estimert rangering basert på RS-ratio
        # RS-ratio 1.0 = 50, lineær skalering med cap
        rs_rating = int(50 + (rs_ratio - 1) * 30)
    
    rs_rating = min(99, max(1, rs_rating))
    
    return {
        'aksje_avkastning': round(aksje_score, 2),
        'marked_avkastning': round(marked_score, 2),
        'rs_ratio': round(rs_ratio, 2),
        'rs_rating': rs_rating,
        'perioder': perioder_brukt  # Detaljert info for UI
    }


def sjekk_multi_timeframe(df):
    """
    Multi-timeframe konvergens-analyse.
    
    Resampler daglige data til ukentlig og sjekker om den overordnede trenden
    bekrefter daglige kjøpssignaler.
    
    Returnerer:
        dict med 'status' ('bullish'/'neutral'/'bearish'),
        'emoji' (✅/⚠️/❌), og 'score_justering' (+15/0/-15)
    """
    if len(df) < 100:
        return {'status': 'neutral', 'emoji': '⚠️', 'score_justering': 0, 'detaljer': 'For lite data'}
    
    try:
        # Resample til ukentlig
        weekly = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(weekly) < 40:
            return {'status': 'neutral', 'emoji': '⚠️', 'score_justering': 0, 'detaljer': 'For lite ukentlig data'}
        
        # Ukentlige indikatorer
        w_sma10 = weekly['Close'].rolling(10).mean()   # ≈ SMA 50 daglig
        w_sma40 = weekly['Close'].rolling(40).mean()   # ≈ SMA 200 daglig
        
        # Ukentlig RSI (Wilder's)
        w_delta = weekly['Close'].diff()
        w_gain = (w_delta.where(w_delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        w_loss = (-w_delta.where(w_delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        w_rs = w_gain / w_loss.replace(0, np.nan)
        w_rsi = 100 - (100 / (1 + w_rs))
        
        siste_close = weekly['Close'].iloc[-1]
        siste_sma10 = w_sma10.iloc[-1]
        siste_sma40 = w_sma40.iloc[-1]
        siste_rsi = w_rsi.iloc[-1]
        
        if pd.isna(siste_sma10) or pd.isna(siste_sma40) or pd.isna(siste_rsi):
            return {'status': 'neutral', 'emoji': '⚠️', 'score_justering': 0, 'detaljer': 'Mangler data'}
        
        # Klassifiser ukentlig trend
        bullish_count = 0
        bearish_count = 0
        
        if siste_close > siste_sma10:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if siste_rsi > 50:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if siste_sma10 > siste_sma40:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if bullish_count >= 3:
            return {
                'status': 'bullish',
                'emoji': '✅',
                'score_justering': 15,
                'detaljer': f'Ukentlig: RSI {siste_rsi:.0f}, over SMA10w & SMA40w'
            }
        elif bearish_count >= 3:
            return {
                'status': 'bearish',
                'emoji': '❌',
                'score_justering': -15,
                'detaljer': f'Ukentlig: RSI {siste_rsi:.0f}, under SMA10w & SMA40w'
            }
        else:
            return {
                'status': 'neutral',
                'emoji': '⚠️',
                'score_justering': 0,
                'detaljer': f'Ukentlig: RSI {siste_rsi:.0f}, blandet trend'
            }
    except Exception:
        return {'status': 'neutral', 'emoji': '⚠️', 'score_justering': 0, 'detaljer': 'Feil i MTF'}


def sjekk_exit_signaler(df):
    """
    Identifiserer exit/salgssignaler.
    Returnerer dict med ulike exit-grunner.
    """
    if len(df) < 50:
        return {
            'skal_selge': False,
            'grunner': [],
            'antall_signaler': 0,
            'drawdown_pct': 0.0
        }
    
    grunner = []
    close = df['Close'].iloc[-1]
    
    # 1. Death Cross (SMA 50 krysser under SMA 200)
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        if df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1] and df['SMA_50'].iloc[-2] >= df['SMA_200'].iloc[-2]:
            grunner.append("⚠️ Death Cross: SMA 50 krysset under SMA 200")
    
    # 2. RSI overkjøpt og snur ned
    if 'RSI' in df.columns:
        if df['RSI'].iloc[-2] > 70 and df['RSI'].iloc[-1] < df['RSI'].iloc[-2]:
            grunner.append("📉 RSI faller fra overkjøpt nivå (>70)")
    
    # 3. Pris bryter under SMA 50 med volum
    if 'SMA_50' in df.columns:
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        if close < df['SMA_50'].iloc[-1] and df['Volume'].iloc[-1] > vol_avg * 1.5:
            grunner.append("🔻 Pris brøt under SMA 50 med høyt volum")
    
    # 4. MACD bearish crossover
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        if df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
            grunner.append("📊 MACD krysset under signallinjen")
    
    # 5. ATR-basert adaptiv trailing stop (erstatter fast 7% grense)
    high_20d = df['High'].iloc[-20:].max()
    drawdown = (close - high_20d) / high_20d * 100
    
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
        atr_pct = (atr / close) * 100 if close > 0 else 3.0
        
        # Adaptiv multiplier basert på volatilitetsregime
        if atr_pct < 2.0:
            multiplier = 2.0   # Lav vol → stramt stop
        elif atr_pct < 4.0:
            multiplier = 2.5   # Normal vol
        else:
            multiplier = 3.0   # Høy vol → bredt stop
        
        trailing_stop_pris = high_20d - (atr * multiplier)
        trailing_stop_pct = (trailing_stop_pris - high_20d) / high_20d * 100
        
        if close < trailing_stop_pris:
            grunner.append(f"📉 Under ATR trailing stop ({abs(trailing_stop_pct):.1f}%, multiplier {multiplier}×)")
    else:
        # Fallback til fast grense hvis ATR mangler
        trailing_stop_pct = -7.0
        if drawdown < -7:
            grunner.append(f"📉 Ned {abs(drawdown):.1f}% fra 20-dagers topp")
    
    # 6. Brudd under Bollinger Lower Band
    if 'BB_Lower' in df.columns:
        if close < df['BB_Lower'].iloc[-1]:
            grunner.append("⬇️ Pris under nedre Bollinger Band")
    
    return {
        'skal_selge': len(grunner) >= 2,  # Selg hvis 2+ exit-signaler
        'grunner': grunner,
        'antall_signaler': len(grunner),
        'drawdown_pct': round(drawdown, 2),
        'trailing_stop_pct': round(trailing_stop_pct, 2)
    }


def beregn_exit_signaler_historisk(df, min_signaler=2):
    """
    Beregner exit-signaler for hele tidsserien (vektorisert O(n)).
    Returnerer DataFrame med datoer og type exit-signal.
    """
    if len(df) < 50:
        return pd.DataFrame()

    close = df['Close']
    volume = df['Volume']

    # --- Pre-beregn alle signal-serier som boolske vektorer ---

    # 1. Death Cross: SMA50 krysser under SMA200
    sig_death_cross = pd.Series(False, index=df.index)
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        sma50 = df['SMA_50']
        sma200 = df['SMA_200']
        valid = sma50.notna() & sma200.notna()
        below_now = sma50 < sma200
        above_prev = sma50.shift(1) >= sma200.shift(1)
        sig_death_cross = valid & below_now & above_prev

    # 2. RSI faller fra overkjøpt (>70 forrige bar, fallende)
    sig_rsi_overkjopt = pd.Series(False, index=df.index)
    if 'RSI' in df.columns:
        rsi = df['RSI']
        rsi_prev = rsi.shift(1)
        valid = rsi.notna() & rsi_prev.notna()
        sig_rsi_overkjopt = valid & (rsi_prev > 70) & (rsi < rsi_prev)

    # 3. Brudd under SMA50 med høyt volum (>1.5× 20d snitt)
    sig_under_sma50_vol = pd.Series(False, index=df.index)
    if 'SMA_50' in df.columns:
        sma50 = df['SMA_50']
        vol_avg_20 = volume.rolling(20).mean()
        valid = sma50.notna() & vol_avg_20.notna()
        sig_under_sma50_vol = valid & (close < sma50) & (volume > vol_avg_20 * 1.5)

    # 4. MACD bearish crossover
    sig_macd_bearish = pd.Series(False, index=df.index)
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = df['MACD']
        macd_sig = df['MACD_Signal']
        valid = macd.notna() & macd_sig.notna()
        below_now = macd < macd_sig
        above_prev = macd.shift(1) >= macd_sig.shift(1)
        sig_macd_bearish = valid & below_now & above_prev

    # 5. Drawdown > 7% fra 20-dagers høy
    high_20d = df['High'].rolling(20).max()
    drawdown_pct = (close - high_20d) / high_20d * 100
    sig_drawdown = drawdown_pct < -7

    # 6. Under Bollinger Lower Band
    sig_under_bb = pd.Series(False, index=df.index)
    if 'BB_Lower' in df.columns:
        bb_lower = df['BB_Lower']
        valid = bb_lower.notna()
        sig_under_bb = valid & (close < bb_lower)

    # --- Kombiner til signal-matrise ---
    signal_names = [
        'Death Cross', 'RSI fra overkjøpt', 'Under SMA50+volum',
        'MACD bearish', 'Drawdown', 'Under BB'
    ]
    signal_series = [
        sig_death_cross, sig_rsi_overkjopt, sig_under_sma50_vol,
        sig_macd_bearish, sig_drawdown, sig_under_bb
    ]

    # Tell totalt antall signaler per rad
    signal_matrix = pd.DataFrame(
        {name: series.astype(int) for name, series in zip(signal_names, signal_series)},
        index=df.index
    )
    total_signals = signal_matrix.sum(axis=1)

    # Filtrer: kun rader fra indeks 50+ med nok signaler
    valid_mask = (total_signals >= min_signaler)
    valid_mask.iloc[:50] = False  # Ignorer de første 50 radene

    if not valid_mask.any():
        return pd.DataFrame()

    # Bygg resultat
    exit_indices = df.index[valid_mask]
    exit_counts_arr = total_signals[valid_mask].values
    exit_types_arr = []

    # Drawdown-verdier for type-teksten
    dd_vals = drawdown_pct[valid_mask].values
    matrix_vals = signal_matrix.loc[valid_mask]

    for idx_pos, (idx, row) in enumerate(matrix_vals.iterrows()):
        aktive = []
        for name in signal_names:
            if row[name]:
                if name == 'Drawdown':
                    aktive.append(f"Drawdown {dd_vals[idx_pos]:.0f}%")
                else:
                    aktive.append(name)
        exit_types_arr.append(" + ".join(aktive[:2]))

    result = pd.DataFrame({
        'dato': exit_indices,
        'type': exit_types_arr,
        'antall': exit_counts_arr
    })
    result.set_index('dato', inplace=True)

    return result


def finn_kombinerte_signaler(df, signaler):
    """
    Finner aksjer der flere strategier gir signal samtidig.
    Høyere score = sterkere konvergens.
    
    Forbedret: Tettere timing (1-2 dager) gir ekstra bonus.
    Strategipar klassifisert som sterke eller svake.
    """
    strategi_kolonner = ['Kort_Sikt_RSI', 'Golden_Cross', 'Momentum_Burst', 
                         'Ichimoku_Breakout', 'Wyckoff_Spring', 'Bull_Race_Prep', 'VCP_Pattern',
                         'Pocket_Pivot']
    
    # Tell signaler siste 5 dager + separat for siste 2 dager (tight cluster)
    lookback = 5
    tight_lookback = 2
    aktive_strategier = []
    tight_strategier = []  # Signaler innen siste 2 dager
    
    for strat in strategi_kolonner:
        if strat in signaler.columns:
            if signaler[strat].iloc[-lookback:].any():
                aktive_strategier.append(strat)
            if signaler[strat].iloc[-tight_lookback:].any():
                tight_strategier.append(strat)
    
    konvergens_score = len(aktive_strategier)
    
    # Timing-bonus: tett klynging (≥2 signaler innen 2 dager) er sterkere
    timing_bonus = 0
    if len(tight_strategier) >= 3:
        timing_bonus = 8  # Svært tett klynge
    elif len(tight_strategier) >= 2:
        timing_bonus = 4  # Tett klynge
    
    # Strategipar: sterke vs svake kombinasjoner
    STERKE_PAR = [
        ('VCP_Pattern', 'Momentum_Burst'),        # VCP breakout + momentum = kraftig
        ('VCP_Pattern', 'Pocket_Pivot'),           # VCP + institusjonell akkumulering
        ('Golden_Cross', 'Ichimoku_Breakout'),     # Dobbel trendbrekk
        ('Momentum_Burst', 'Bull_Race_Prep'),      # Breakout + squeeze = eksplosivt
        ('Wyckoff_Spring', 'Pocket_Pivot'),        # Akkumulering + institusjonelt
    ]
    SVAKE_PAR = [
        ('Kort_Sikt_RSI', 'Golden_Cross'),         # Motstridende tidshorisonter
        ('Kort_Sikt_RSI', 'VCP_Pattern'),           # Dip-kjøp + setup = inkompatibelt
    ]
    
    par_bonus = 0
    aktive_set = set(aktive_strategier)
    for s1, s2 in STERKE_PAR:
        if s1 in aktive_set and s2 in aktive_set:
            par_bonus += 10
    for s1, s2 in SVAKE_PAR:
        if s1 in aktive_set and s2 in aktive_set:
            par_bonus -= 5  # Svakt par reduserer bonus
    
    total_bonus = timing_bonus + max(0, par_bonus)
    
    return {
        'aktive_strategier': aktive_strategier,
        'tight_strategier': tight_strategier,
        'antall': konvergens_score,
        'konvergens_score': konvergens_score * 15 + total_bonus,  # 0-100+ skala
        'er_sterk_konvergens': konvergens_score >= 2
    }


def backtest_strategi(df, signaler, strategi_key, holdingperiode=20,
                      kurtasje_pct=None, spread_pct=None,
                      trailing_stop_atr=1.0, profit_target_atr=2.0,
                      min_dagsomsetning=5000000):
    """
    Backtester en strategi med trailing stop og profit target.
    Walk-forward: Evaluerer hvert signal basert på fremtidig avkastning.
    
    Exit-regler (prioritert rekkefølge):
        1. Profit target: Close >= entry + profit_target_atr * ATR14
        2. Trailing stop:  Close <= peak_close - trailing_stop_atr * ATR14
        3. Maks holdingperiode: fallback etter N dager
    
    Transaksjonskostnader:
        kurtasje_pct: Kurtasje per vei i % (default fra config.KURTASJE_PCT)
        spread_pct:   Spread + slippage i % (default fra config.SPREAD_SLIPPAGE_PCT)
    
    Likviditetsfilter:
        min_dagsomsetning: Minimum dagsomsetning i NOK (default 5 mill)
    """
    if kurtasje_pct is None:
        kurtasje_pct = getattr(config, 'KURTASJE_PCT', 0.05)
    if spread_pct is None:
        spread_pct = getattr(config, 'SPREAD_SLIPPAGE_PCT', 0.10)
    
    # Enveis friksjons-faktor
    friksjons_faktor = kurtasje_pct / 100 + spread_pct / 200
    
    if strategi_key not in signaler.columns:
        return None
    
    signal_dates = signaler[signaler[strategi_key] == True].index
    
    if len(signal_dates) < 3:
        return None
    
    # Pre-beregn dagsomsetning (Close * Volume) for likviditetsfilter
    df['dagsomsetning'] = df['Close'] * df['Volume']
    
    # Pre-beregn ATR14 for hele serien
    if len(df) >= 15:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
    else:
        atr_series = pd.Series(0, index=df.index)
    
    resultater = []
    total_kostnad_pst = 0.0
    filtrert_bort_lav_likviditet = 0
    
    for signal_dato in signal_dates:
        idx = df.index.get_loc(signal_dato)
        
        # Trenger minst 2 dager fremover (dag+1 for entry, dag+2 for første exit-sjekk)
        if idx + 3 >= len(df):
            continue
        
        # LIKVIDITETSFILTER: Sjekk at entry-dagen har nok omsetning
        entry_idx = idx + 1
        entry_omsetning = df.iloc[entry_idx]['dagsomsetning']
        
        # Håndter både Series og enkeltverdi
        if hasattr(entry_omsetning, 'iloc'):
            entry_omsetning = entry_omsetning.iloc[0]
        
        if entry_omsetning < min_dagsomsetning:
            filtrert_bort_lav_likviditet += 1
            continue
        
        # REALISTISK: Handle dagen ETTER signalet (dag+1)
        # Signal på dag X → du handler på dag X+1 åpning/tidlig handel
        rå_entry = df.iloc[idx + 1]['Open'] if 'Open' in df.columns else df.iloc[idx + 1]['Close']
        entry_pris = rå_entry * (1 + friksjons_faktor)
        
        # ATR ved signal-dato (brukes til stop-beregning)
        atr_val = atr_series.iloc[idx] if not pd.isna(atr_series.iloc[idx]) else rå_entry * 0.03
        
        # Profit target basert på entry-pris
        pt_nivå = rå_entry + profit_target_atr * atr_val
        
        # Simuler dag-for-dag med trailing stop (start fra entry-dagen)
        peak_close = rå_entry
        exit_dag = min(idx + 1 + holdingperiode, len(df) - 1)
        exit_grunn = 'maks_tid'
        rå_exit = df.iloc[exit_dag]['Close']
        
        # Start fra dagen ETTER entry (dag+2 fra signal)
        for d in range(2, holdingperiode + 2):
            if idx + d >= len(df):
                exit_dag = len(df) - 1
                rå_exit = df.iloc[exit_dag]['Close']
                exit_grunn = 'data_slutt'
                break
            
            dag_close = df.iloc[idx + d]['Close']
            
            # Oppdater peak
            if dag_close > peak_close:
                peak_close = dag_close
            
            # Sjekk profit target
            if dag_close >= pt_nivå:
                exit_dag = idx + d
                rå_exit = dag_close
                exit_grunn = 'profit_target'
                break
            
            # Sjekk trailing stop
            trailing_stop_nivå = peak_close - trailing_stop_atr * atr_val
            if dag_close <= trailing_stop_nivå:
                exit_dag = idx + d
                rå_exit = dag_close
                exit_grunn = 'trailing_stop'
                break
            
            # Siste dag i holdingperiode (justert for +1 dag forsinkelse)  
            if d == holdingperiode + 1:
                exit_dag = idx + d
                rå_exit = dag_close
                exit_grunn = 'maks_tid'
        
        exit_pris = rå_exit * (1 - friksjons_faktor)
        avkastning = (exit_pris - entry_pris) / entry_pris * 100
        dager_holdt = exit_dag - (idx + 1)  # Fra entry-dag, ikke signal-dag
        
        # Round-trip kostnad
        kostnad_denne = (2 * kurtasje_pct + spread_pct)
        total_kostnad_pst += kostnad_denne
        
        # Maks drawdown i perioden (fra entry-dag)
        periode_data = df.iloc[idx + 1:exit_dag + 1]
        periode_low = periode_data['Low'].min()
        max_drawdown = (periode_low - entry_pris) / entry_pris * 100
        
        # Maks gevinst i perioden (fra entry-dag)
        periode_high = periode_data['High'].max()
        max_gain = (periode_high * (1 - friksjons_faktor) - entry_pris) / entry_pris * 100
        
        resultater.append({
            'dato': signal_dato,
            'entry': round(entry_pris, 2),
            'exit': round(exit_pris, 2),
            'avkastning': avkastning,
            'gevinst': avkastning > 0,
            'dager_holdt': dager_holdt,
            'exit_grunn': exit_grunn,
            'max_drawdown': max_drawdown,
            'max_gain': max_gain
        })
    
    if not resultater:
        return None
    
    # Beregn statistikk
    df_res = pd.DataFrame(resultater)
    
    # Exit-fordeling
    exit_fordeling = df_res['exit_grunn'].value_counts().to_dict()
    
    return {
        'antall_signaler': len(df_res),
        'vinnere': int(df_res['gevinst'].sum()),
        'tapere': int((~df_res['gevinst']).sum()),
        'win_rate': round(df_res['gevinst'].mean() * 100, 1),
        'snitt_avkastning': round(df_res['avkastning'].mean(), 2),
        'median_avkastning': round(df_res['avkastning'].median(), 2),
        'beste_trade': round(df_res['avkastning'].max(), 2),
        'verste_trade': round(df_res['avkastning'].min(), 2),
        'snitt_max_drawdown': round(df_res['max_drawdown'].mean(), 2),
        'snitt_dager_holdt': round(df_res['dager_holdt'].mean(), 1),
        'exit_fordeling': exit_fordeling,
        'profit_factor': round(
            df_res[df_res['avkastning'] > 0]['avkastning'].sum() / 
            abs(df_res[df_res['avkastning'] < 0]['avkastning'].sum()) 
            if df_res[df_res['avkastning'] < 0]['avkastning'].sum() != 0 else 999, 2
        ),
        'total_kostnad_pst': round(total_kostnad_pst, 2),
        'snitt_kostnad_per_trade': round(total_kostnad_pst / len(df_res), 3),
        'filtrert_lav_likviditet': filtrert_bort_lav_likviditet,
        'resultater': resultater[-10:]  # Siste 10 trades
    }


# Hardkodet sektor-mapping for Oslo Børs (fallback dersom yfinance-cache mangler)
_SEKTOR_MAPPING_HARDKODET = {
    # Energi
    "EQNR.OL": "Energi", "AKRBP.OL": "Energi", "VAR.OL": "Energi", "OKEA.OL": "Energi",
    "PGS.OL": "Energi", "TGS.OL": "Energi", "BORR.OL": "Energi", "SUBC.OL": "Energi",
    # Shipping
    "FRO.OL": "Shipping", "HAFNI.OL": "Shipping", "MPCC.OL": "Shipping", "GOGL.OL": "Shipping",
    "BWLPG.OL": "Shipping", "FLNG.OL": "Shipping", "CADELER.OL": "Shipping",
    # Sjømat
    "MOWI.OL": "Sjømat", "SALM.OL": "Sjømat", "LSG.OL": "Sjømat", "BAKKA.OL": "Sjømat",
    # Finans
    "DNB.OL": "Finans", "STB.OL": "Finans", "GJF.OL": "Finans", "SRBANK.OL": "Finans",
    # Industri
    "NHY.OL": "Industri", "YAR.OL": "Industri", "ELK.OL": "Industri", "TOM.OL": "Industri",
    "KOG.OL": "Industri", "NORBIT.OL": "Industri",
    # Teknologi
    "NOD.OL": "Teknologi", "KIT.OL": "Teknologi", "BOUV.OL": "Teknologi", 
    "PEXIP.OL": "Teknologi", "VOLUE.OL": "Teknologi",
    # Konsum
    "ORK.OL": "Konsum", "XXL.OL": "Konsum", "KID.OL": "Konsum",
    # Telekom
    "TEL.OL": "Telekom",
    # Fornybar
    "SCATC.OL": "Fornybar", "NEL.OL": "Fornybar", "OTOVO.OL": "Fornybar", "RECSI.OL": "Fornybar",
}

# SEKTOR_MAPPING — brukes eksternt. Bygges dynamisk med cache + hardkodet fallback.
SEKTOR_MAPPING = dict(_SEKTOR_MAPPING_HARDKODET)

# Last inn dynamisk cache ved import
try:
    from data import hent_sektor_mapping as _hent_sektor_cache
    _dyn_cache = _hent_sektor_cache()
    if _dyn_cache:
        # Dynamisk cache overskriver hardkodet for tickers som finnes i begge
        SEKTOR_MAPPING.update(_dyn_cache)
except Exception:
    pass  # Dynamisk cache ikke tilgjengelig, bruk kun hardkodet


def hent_sektor(ticker):
    """
    Returnerer sektor for en ticker.
    Prioriterer dynamisk yfinance-cache, faller tilbake på hardkodet mapping.
    """
    return SEKTOR_MAPPING.get(ticker, "Annet")


def oppdater_sektor_cache():
    """
    Oppdaterer den in-memory sektor-mappingen fra disk-cache.
    Kalles etter at data.oppdater_sektor_mapping() har kjørt.
    """
    global SEKTOR_MAPPING
    try:
        from data import hent_sektor_mapping as _hent_sektor_cache
        dyn = _hent_sektor_cache()
        if dyn:
            # Start med hardkodet, overskriv med dynamisk
            SEKTOR_MAPPING = dict(_SEKTOR_MAPPING_HARDKODET)
            SEKTOR_MAPPING.update(dyn)
            logger.debug(f"Sektor-mapping oppdatert: {len(SEKTOR_MAPPING)} tickers")
    except Exception as e:
        logger.warning(f"Kunne ikke oppdatere sektor-cache: {e}")


def filtrer_sektor_konsentrasjon(resultater, maks_per_sektor=3):
    """
    Filtrerer resultater for å unngå for mange aksjer i samme sektor.
    Beholder kun de beste per sektor.
    """
    if not resultater:
        return resultater
    
    # Grupper per sektor
    sektor_grupper = {}
    for r in resultater:
        ticker = r.get('Ticker', '')
        sektor = hent_sektor(ticker)
        if sektor not in sektor_grupper:
            sektor_grupper[sektor] = []
        sektor_grupper[sektor].append(r)
    
    # Behold kun topp N per sektor (sortert etter score)
    filtrert = []
    for sektor, aksjer in sektor_grupper.items():
        sortert = sorted(aksjer, key=lambda x: x.get('Score', 0), reverse=True)
        filtrert.extend(sortert[:maks_per_sektor])
    
    return filtrert


def beregn_sektor_momentum(df_clean, perioder=20):
    """
    Beregner sektor-momentum og rangering for alle sektorer.
    
    Grupperer tickers etter SEKTOR_MAPPING, beregner gjennomsnittlig
    avkastning per sektor siste N dager, og returnerer rangering 1-99.
    
    Args:
        df_clean: Full DataFrame med alle tickers (kolonne 'Ticker' påkrevd)
        perioder: Antall dager for momentum-beregning (default 20)
        
    Returns:
        dict med:
          'sektor_data': {sektor: {avkastning, rangering, antall_aksjer}}
          'ticker_sektor_rs': {ticker: sektor_rangering}  (for enkel oppslag)
          'topp_sektorer': liste med topp-3
          'bunn_sektorer': liste med bunn-3
    """
    if df_clean is None or df_clean.empty:
        return None
    
    # Beregn avkastning per ticker siste N dager
    sektor_avkastninger = {}
    
    tickers = df_clean['Ticker'].unique() if 'Ticker' in df_clean.columns else []
    
    for ticker in tickers:
        df_t = df_clean[df_clean['Ticker'] == ticker]
        if len(df_t) < perioder:
            continue
        
        avk = (df_t['Close'].iloc[-1] / df_t['Close'].iloc[-perioder] - 1) * 100
        sektor = hent_sektor(ticker)
        
        if sektor not in sektor_avkastninger:
            sektor_avkastninger[sektor] = []
        sektor_avkastninger[sektor].append(avk)
    
    if not sektor_avkastninger:
        return None
    
    # Beregn gjennomsnitt per sektor
    sektor_snitt = {}
    for sektor, avk_liste in sektor_avkastninger.items():
        sektor_snitt[sektor] = {
            'avkastning': round(np.mean(avk_liste), 2),
            'antall_aksjer': len(avk_liste)
        }
    
    # Ranger sektorer (høyest avkastning = høyest rangering)
    sortert = sorted(sektor_snitt.keys(), key=lambda s: sektor_snitt[s]['avkastning'])
    antall = len(sortert)
    
    for i, sektor in enumerate(sortert):
        # Persentil-rangering 1-99
        sektor_snitt[sektor]['rangering'] = int(round((i / max(antall - 1, 1)) * 98 + 1))
    
    # Bygg ticker → sektor_rangering lookup
    ticker_sektor_rs = {}
    for ticker in tickers:
        sektor = hent_sektor(ticker)
        if sektor in sektor_snitt:
            ticker_sektor_rs[ticker] = sektor_snitt[sektor]['rangering']
        else:
            ticker_sektor_rs[ticker] = 50
    
    topp_sektorer = sortert[-3:][::-1] if antall >= 3 else sortert[::-1]
    bunn_sektorer = sortert[:3] if antall >= 3 else sortert
    
    return {
        'sektor_data': sektor_snitt,
        'ticker_sektor_rs': ticker_sektor_rs,
        'topp_sektorer': topp_sektorer,
        'bunn_sektorer': bunn_sektorer
    }


# =============================================================================
# MAKRO-SIGNALER: Oljepris og Valuta
# =============================================================================

# Sektorer som påvirkes av oljepris
OLJE_SEKTORER = {"Energi", "Shipping"}

# Sektorer som påvirkes av svak/sterk NOK
SVAK_NOK_BOOST = {"Sjømat", "Industri", "Shipping"}    # Eksportører — tjener på svak NOK
STERK_NOK_BOOST = {"Finans", "Konsum", "Telekom"}      # Innenlandsvendte — tjener på sterk NOK


def analyser_oljepris(df_brent):
    """
    Analyserer Brent-oljepris for å gi signal til energiaksjer.
    
    Returns:
        dict med 'signal' ('bullish'/'bearish'/'neutral'),
        'emoji', 'score_justering', 'pris', 'sma50', 'endring_20d'
    """
    if df_brent is None or df_brent.empty or len(df_brent) < 60:
        return {'signal': 'neutral', 'emoji': '⚠️', 'score_justering': 0,
                'pris': None, 'sma50': None, 'endring_20d': None}
    
    close = df_brent['Close']
    siste_pris = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    endring_20d = float((close.iloc[-1] / close.iloc[-20] - 1) * 100)
    sma50_stigende = float(close.rolling(50).mean().iloc[-1]) > float(close.rolling(50).mean().iloc[-5])
    
    if siste_pris > sma50 and sma50_stigende:
        return {'signal': 'bullish', 'emoji': '🛢️🟢', 'score_justering': 10,
                'pris': round(siste_pris, 2), 'sma50': round(sma50, 2),
                'endring_20d': round(endring_20d, 1)}
    elif siste_pris < sma50 and not sma50_stigende:
        return {'signal': 'bearish', 'emoji': '🛢️🔴', 'score_justering': -10,
                'pris': round(siste_pris, 2), 'sma50': round(sma50, 2),
                'endring_20d': round(endring_20d, 1)}
    else:
        return {'signal': 'neutral', 'emoji': '🛢️⚪', 'score_justering': 0,
                'pris': round(siste_pris, 2), 'sma50': round(sma50, 2),
                'endring_20d': round(endring_20d, 1)}


def analyser_usdnok(df_usdnok):
    """
    Analyserer USD/NOK for å gi signal til valutasensitive sektorer.
    
    Stigende USDNOK = svak NOK → bra for eksportører
    Fallende USDNOK = sterk NOK → bra for innenlandsvendte
    
    Returns:
        dict med 'trend' ('svak_nok'/'sterk_nok'/'neutral'),
        'emoji', 'kurs', 'sma20', 'sma50', 'endring_20d'
    """
    if df_usdnok is None or df_usdnok.empty or len(df_usdnok) < 60:
        return {'trend': 'neutral', 'emoji': '💱⚪',
                'kurs': None, 'sma20': None, 'sma50': None, 'endring_20d': None}
    
    close = df_usdnok['Close']
    siste_kurs = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    endring_20d = float((close.iloc[-1] / close.iloc[-20] - 1) * 100)
    
    if sma20 > sma50 and siste_kurs > sma20:
        # Stigende USDNOK = svak NOK
        return {'trend': 'svak_nok', 'emoji': '💱📈',
                'kurs': round(siste_kurs, 2), 'sma20': round(sma20, 2),
                'sma50': round(sma50, 2), 'endring_20d': round(endring_20d, 1)}
    elif sma20 < sma50 and siste_kurs < sma20:
        # Fallende USDNOK = sterk NOK
        return {'trend': 'sterk_nok', 'emoji': '💱📉',
                'kurs': round(siste_kurs, 2), 'sma20': round(sma20, 2),
                'sma50': round(sma50, 2), 'endring_20d': round(endring_20d, 1)}
    else:
        return {'trend': 'neutral', 'emoji': '💱⚪',
                'kurs': round(siste_kurs, 2), 'sma20': round(sma20, 2),
                'sma50': round(sma50, 2), 'endring_20d': round(endring_20d, 1)}


def makro_score_justering(ticker, olje_signal, valuta_signal):
    """
    Beregner samlet makro score-justering for en ticker basert på
    oljepris og valutasignaler.
    
    Returns:
        int: score-justering (kan være negativ)
    """
    sektor = hent_sektor(ticker)
    justering = 0
    
    # Oljepris-effekt
    if olje_signal and sektor in OLJE_SEKTORER:
        justering += olje_signal.get('score_justering', 0)
    
    # Valuta-effekt
    if valuta_signal:
        trend = valuta_signal.get('trend', 'neutral')
        if trend == 'svak_nok' and sektor in SVAK_NOK_BOOST:
            justering += 5
        elif trend == 'svak_nok' and sektor in STERK_NOK_BOOST:
            justering -= 5
        elif trend == 'sterk_nok' and sektor in STERK_NOK_BOOST:
            justering += 5
        elif trend == 'sterk_nok' and sektor in SVAK_NOK_BOOST:
            justering -= 5
    
    return justering


# ────────────────── REGIME-TILPASSET SIGNALFILTERING ──────────────────

REGIME_SIGNAL_KRAV = {
    'Bull Market': {
        'min_kvalitet': 'D',       # alle tillatt
        'mtf_krav': False,          # ingen MTF-krav
        'min_rs': 0,                # ingen RS-krav
        'kun_strategier': None,     # alle strategier tillatt
        'beskrivelse': '🟢 Bull — Alle signaler tillatt',
        'kort': 'Alle tillatt',
    },
    'Mild Bull': {
        'min_kvalitet': 'B',
        'mtf_krav': False,
        'min_rs': 0,
        'kun_strategier': None,
        'beskrivelse': '📈 Mild Bull — Minimum B-kvalitet',
        'kort': 'Min B-kvalitet',
    },
    'Nøytral': {
        'min_kvalitet': 'B',
        'mtf_krav': True,           # MTF må vise bullish
        'min_rs': 0,
        'kun_strategier': None,
        'beskrivelse': '➡️ Nøytral — Min B-kvalitet + MTF-bullish',
        'kort': 'B + MTF✅',
    },
    'Mild Bear': {
        'min_kvalitet': 'A',
        'mtf_krav': True,
        'min_rs': 70,
        'kun_strategier': None,
        'beskrivelse': '📉 Mild Bear — Min A-kvalitet + MTF-bullish + RS > 70',
        'kort': 'A + MTF✅ + RS>70',
    },
    'Bear Market': {
        'min_kvalitet': 'A',
        'mtf_krav': True,
        'min_rs': 70,
        'kun_strategier': ['VCP_Pattern', 'Pocket_Pivot'],
        'beskrivelse': '🔥 Bear — KUN VCP/Pocket Pivot med A-kvalitet',
        'kort': 'VCP/PP, A-kvalitet',
    },
}


def regime_signal_krav(regime_name: str) -> dict:
    """
    Returnerer filtreringskrav for signaler basert på markedsregime.
    
    Logikk (Minervini-inspirert):
      Bull       → Alle signaler tillatt
      Mild Bull  → Min B-kvalitet
      Nøytral    → Min B-kvalitet + MTF-bullish
      Mild Bear  → Min A-kvalitet + MTF-bullish + RS > 70
      Bear       → KUN VCP/Pocket Pivot med A-kvalitet + MTF + RS > 70
    """
    return REGIME_SIGNAL_KRAV.get(regime_name, REGIME_SIGNAL_KRAV['Nøytral'])


def sjekk_regime_filter(signal_info: dict, strat_key: str,
                        mtf_data: dict, rs_rating: float,
                        regime_name: str) -> tuple:
    """
    Sjekker om et signal passerer regime-tilpasset filtrering.
    
    Args:
        signal_info: dict med 'kvalitet_klasse', 'kvalitet_score'
        strat_key: strateginøkkel (f.eks. 'VCP_Pattern')
        mtf_data: dict fra sjekk_multi_timeframe() med 'status'
        rs_rating: IBD-vektet RS (1-99)
        regime_name: nåværende regime (f.eks. 'Bear Market')
    
    Returns:
        (passerer: bool, grunn: str | None)
    """
    krav = regime_signal_krav(regime_name)
    
    kvalitet_grenser = {'A': 75, 'B': 55, 'C': 35, 'D': 0}
    min_score = kvalitet_grenser.get(krav['min_kvalitet'], 35)
    
    # 1. Strategi-restriksjon (Bear: kun VCP/Pocket Pivot)
    if krav['kun_strategier'] and strat_key not in krav['kun_strategier']:
        tillatte = ', '.join(krav['kun_strategier'])
        return False, f"Regime {regime_name}: kun {tillatte} tillatt"
    
    # 2. Kvalitetskrav
    if signal_info['kvalitet_score'] < min_score:
        return False, f"Regime {regime_name}: krev min {krav['min_kvalitet']}-kvalitet (score {min_score}+)"
    
    # 3. MTF-krav
    if krav['mtf_krav']:
        mtf_status = mtf_data.get('status', 'neutral') if mtf_data else 'neutral'
        if mtf_status != 'bullish':
            return False, f"Regime {regime_name}: krev MTF-bullish (er {mtf_status})"
    
    # 4. RS-krav
    if krav['min_rs'] > 0 and rs_rating < krav['min_rs']:
        return False, f"Regime {regime_name}: krev RS ≥ {krav['min_rs']} (er {rs_rating})"
    
    return True, None


def beregn_smart_risk_reward(df, entry_price=None):
    """
    Beregner risk/reward basert på støtte/motstand nivåer.
    Returnerer dict med entry, stop, target, og R/R ratio.
    """
    if df.empty or len(df) < 50:
        return None
    
    current_price = float(df['Close'].iloc[-1]) if entry_price is None else entry_price
    stotte, motstand = finn_stotte_motstand(df)
    
    # Finn nærmeste støtte (for stop loss)
    stop_loss = None
    if stotte:
        # Bruk nærmeste støtte under current price
        under_price = [s for s in stotte if s['pris'] < current_price * 0.99]
        if under_price:
            stop_loss = max(s['pris'] for s in under_price)  # Nærmeste under
    
    # Fallback: bruk ATR-basert stop
    if stop_loss is None:
        if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]):
            atr = float(df['ATR'].iloc[-1])
            stop_loss = current_price - (2 * atr)
        else:
            stop_loss = current_price * 0.92  # 8% stop som fallback
    
    # Finn nærmeste motstand (for target)
    target = None
    if motstand:
        # Bruk nærmeste motstand over current price
        over_price = [m for m in motstand if m['pris'] > current_price * 1.01]
        if over_price:
            target = min(m['pris'] for m in over_price)  # Nærmeste over
    
    # Fallback: bruk ATR-basert target
    if target is None:
        if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]):
            atr = float(df['ATR'].iloc[-1])
            target = current_price + (3 * atr)
        else:
            target = current_price * 1.15  # 15% target som fallback
    
    # Beregn R/R
    risk = current_price - stop_loss
    reward = target - current_price
    
    if risk <= 0:
        risk = current_price * 0.05  # Minimum 5% risk
        stop_loss = current_price - risk
    
    rr_ratio = reward / risk if risk > 0 else 0
    
    return {
        'entry': current_price,
        'stop_loss': stop_loss,
        'target': target,
        'risk': risk,
        'risk_pct': (risk / current_price) * 100,
        'reward': reward,
        'reward_pct': (reward / current_price) * 100,
        'rr_ratio': rr_ratio,
        'stotte_nivåer': stotte[:3],
        'motstand_nivåer': motstand[:3]
    }


def beregn_kjops_score(df, signaler=None):
    """
    Beregner en total kjøpsscore (0-100) basert på:
    - Signalkvalitet (A/B/C/D)
    - Risk/Reward ratio
    - Avstand fra støtte
    - Trend-styrke
    - Volum
    - NEGATIVE faktorer som trekker ned scoren
    """
    if df.empty or len(df) < 50:
        return None
    
    score = 0
    detaljer = {}
    advarsler = []
    
    current_price = float(df['Close'].iloc[-1])
    
    # === NEGATIVE FAKTORER (diskvalifiserer eller trekker fra) ===
    
    # Sjekk for Death Cross (SMA50 < SMA200) - STOR ADVARSEL
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        sma50 = df['SMA_50'].iloc[-1]
        sma200 = df['SMA_200'].iloc[-1]
        if not pd.isna(sma50) and not pd.isna(sma200):
            if sma50 < sma200:
                advarsler.append("⚠️ Death Cross aktiv (SMA50 < SMA200)")
                score -= 15  # Trekk fra poeng
    
    # Sjekk RSI overkjøpt (>70) - dårlig entry
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if not pd.isna(rsi):
            if rsi > 75:
                advarsler.append("⚠️ RSI overkjøpt (>75)")
                score -= 10
            elif rsi > 70:
                advarsler.append("⚠️ RSI høy (>70)")
                score -= 5
    
    # Sjekk for nylig stort fall (>10% siste 5 dager) - kan være knife catch
    if len(df) >= 5:
        pct_change_5d = ((current_price / df['Close'].iloc[-5]) - 1) * 100
        if pct_change_5d < -10:
            advarsler.append(f"⚠️ Stort fall siste 5 dager ({pct_change_5d:.1f}%)")
            score -= 10
        elif pct_change_5d < -7:
            advarsler.append(f"⚠️ Betydelig fall siste 5 dager ({pct_change_5d:.1f}%)")
            score -= 5
    
    # Sjekk om pris er under SMA200 (langsiktig bearish)
    if 'SMA_200' in df.columns:
        sma200 = df['SMA_200'].iloc[-1]
        if not pd.isna(sma200) and current_price < sma200:
            pct_under = ((sma200 - current_price) / sma200) * 100
            if pct_under > 15:
                advarsler.append(f"⚠️ Langt under SMA200 ({pct_under:.1f}%)")
                score -= 10
    
    # Sjekk for lav likviditet (gjennomsnittlig volum)
    if 'Volume' in df.columns:
        avg_vol = df['Volume'].iloc[-20:].mean()
        avg_close = df['Close'].iloc[-20:].mean()
        avg_turnover = avg_vol * avg_close
        if avg_turnover < 100000:  # Under 100k NOK daglig omsetning
            advarsler.append("⚠️ Lav likviditet")
            score -= 10
    
    detaljer['advarsler'] = advarsler
    
    # === POSITIVE FAKTORER ===
    
    # 1. Risk/Reward Score (0-30 poeng)
    rr = beregn_smart_risk_reward(df)
    if rr:
        rr_ratio = rr['rr_ratio']
        # Sjekk at R/R er realistisk (ikke bare fallback-verdier)
        if rr_ratio >= 3:
            rr_score = 30
        elif rr_ratio >= 2.5:
            rr_score = 25
        elif rr_ratio >= 2:
            rr_score = 20
        elif rr_ratio >= 1.5:
            rr_score = 15
        elif rr_ratio >= 1:
            rr_score = 10
        else:
            rr_score = 0  # Dårlig R/R = ingen poeng
        score += rr_score
        detaljer['risk_reward'] = {'score': rr_score, 'ratio': rr_ratio, 'data': rr}
    
    # 2. Signalkvalitet (0-25 poeng) - KUN hvis nylig signal
    if signaler is not None:
        beste_signal = None
        beste_score = 0
        
        strat_names = ['Kort_Sikt_RSI', 'Momentum_Burst', 'Golden_Cross', 
                       'Ichimoku_Breakout', 'Wyckoff_Spring', 'Bull_Race_Prep', 
                       'VCP_Pattern', 'Pocket_Pivot']
        
        for strat in strat_names:
            if strat in signaler.columns:
                signal_mask = signaler[strat].fillna(False).astype(bool)
                # Sjekk siste 5 dager
                if signal_mask.iloc[-5:].any():
                    signal_dates = signaler.index[signal_mask]
                    if len(signal_dates) > 0:
                        siste_dato = signal_dates[-1]
                        kvalitet = beregn_signal_kvalitet(df, siste_dato, strat)
                        if kvalitet and kvalitet.get('score', 0) > beste_score:
                            beste_score = kvalitet['score']
                            beste_signal = {'strategi': strat, 'kvalitet': kvalitet}
        
        if beste_signal:
            signal_score = min(25, beste_score / 4)  # Max 25 poeng
            score += signal_score
            detaljer['signal'] = {
                'score': signal_score, 
                'strategi': beste_signal['strategi'],
                'kvalitet_score': beste_score,
                'klasse': klassifiser_signal_kvalitet(beste_score)
            }
    
    # 3. Avstand fra støtte (0-15 poeng) - nærmere støtte = bedre entry
    if rr and rr.get('stotte_nivåer'):
        try:
            under_price = [s['pris'] for s in rr['stotte_nivåer'] if s['pris'] < current_price]
            if under_price:
                nærmeste_stotte = max(under_price)
                avstand_pct = ((current_price - nærmeste_stotte) / current_price) * 100
                
                if avstand_pct <= 2:
                    stotte_score = 15  # Veldig nær støtte
                elif avstand_pct <= 4:
                    stotte_score = 12
                elif avstand_pct <= 6:
                    stotte_score = 9
                elif avstand_pct <= 10:
                    stotte_score = 6
                else:
                    stotte_score = 3
                
                score += stotte_score
                detaljer['stotte_avstand'] = {'score': stotte_score, 'avstand_pct': avstand_pct}
        except:
            pass
    
    # 4. Trend-styrke (0-15 poeng)
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        sma50 = float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else current_price
        sma200 = float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else current_price
        
        trend_score = 0
        # Pris over SMA50
        if current_price > sma50:
            trend_score += 5
        # SMA50 over SMA200 (Golden Cross territory)
        if sma50 > sma200:
            trend_score += 5
        # Pris over SMA200
        if current_price > sma200:
            trend_score += 5
        
        score += trend_score
        detaljer['trend'] = {
            'score': trend_score, 
            'over_sma50': current_price > sma50,
            'over_sma200': current_price > sma200,
            'sma50_over_200': sma50 > sma200
        }
    
    # 5. Volum (0-15 poeng) - men IKKE for daglig lav volum
    if 'Volume' in df.columns:
        vol_avg = df['Volume'].iloc[-20:].mean()
        vol_today = df['Volume'].iloc[-1]
        vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1
        
        if vol_ratio >= 2:
            vol_score = 15  # Høyt volum
        elif vol_ratio >= 1.5:
            vol_score = 12
        elif vol_ratio >= 1:
            vol_score = 8
        elif vol_ratio >= 0.7:
            vol_score = 5
        else:
            vol_score = 2  # Lavt volum
        
        score += vol_score
        detaljer['volum'] = {'score': vol_score, 'ratio': vol_ratio}
    
    # Sett score til minimum 0
    score = max(0, score)
    
    # Bestem anbefaling med strengere krav
    if len(advarsler) >= 2:
        anbefaling = 'UNNGÅ'  # For mange advarsler
    elif score >= 75 and len(advarsler) == 0:
        anbefaling = 'STERK KJØP'
    elif score >= 60 and len(advarsler) <= 1:
        anbefaling = 'KJØP'
    elif score >= 45:
        anbefaling = 'HOLD/VENT'
    else:
        anbefaling = 'UNNGÅ'
    
    return {
        'total_score': min(100, score),
        'detaljer': detaljer,
        'anbefaling': anbefaling,
        'advarsler': advarsler
    }


def ranger_kjopsmuligheter(df_dict, min_score=40):
    """
    Rangerer alle aksjer etter kjøpspotensial.
    df_dict: {ticker: df} dictionary
    Returnerer sortert liste med beste kjøpsmuligheter.
    """
    resultater = []
    
    for ticker, df in df_dict.items():
        try:
            if df.empty or len(df) < 50:
                continue
            
            # Beregn signaler
            signaler = sjekk_strategier(df)
            
            # Beregn kjøpsscore
            score_data = beregn_kjops_score(df, signaler)
            if score_data is None:
                continue
            
            if score_data['total_score'] >= min_score:
                resultater.append({
                    'ticker': ticker,
                    'score': score_data['total_score'],
                    'anbefaling': score_data['anbefaling'],
                    'detaljer': score_data['detaljer'],
                    'advarsler': score_data.get('advarsler', []),
                    'pris': float(df['Close'].iloc[-1])
                })
        except Exception:
            continue
    
    # Sorter etter score (høyest først)
    resultater.sort(key=lambda x: x['score'], reverse=True)
    
    return resultater


# =============================================================================
# MARKEDSBREDDE-BEREGNINGER
# =============================================================================

def beregn_mcclellan_oscillator(advances_series: pd.Series, declines_series: pd.Series) -> pd.Series:
    """
    Beregner McClellan Oscillator.
    
    McClellan = EMA(19, A-D) - EMA(39, A-D)
    
    Args:
        advances_series: Daglige antall aksjer som stiger
        declines_series: Daglige antall aksjer som faller
        
    Returns:
        Series med McClellan Oscillator-verdier
    """
    if len(advances_series) < 39:
        return pd.Series(dtype=float)
    
    ad_net = advances_series - declines_series
    ema19 = ad_net.ewm(span=19, adjust=False).mean()
    ema39 = ad_net.ewm(span=39, adjust=False).mean()
    
    return ema19 - ema39


def beregn_ad_linje(advances_series: pd.Series, declines_series: pd.Series) -> pd.Series:
    """
    Beregner kumulativ Advance/Decline-linje.
    
    Args:
        advances_series: Daglige antall aksjer som stiger
        declines_series: Daglige antall aksjer som faller
        
    Returns:
        Series med kumulativ A/D-linje
    """
    return (advances_series - declines_series).cumsum()


def beregn_bredde_indikatorer(df_ticker_dict: dict) -> dict:
    """
    Beregner markedsbredde-indikatorer for et sett med tickers.
    
    Args:
        df_ticker_dict: {ticker: DataFrame} med tekniske indikatorer beregnet
        
    Returns:
        dict med bredde-metrics:
          pct_over_sma200, pct_over_sma50, advances, declines,
          new_52w_high, new_52w_low, ad_ratio, mcclellan (siste verdi)
    """
    total = 0
    over_sma200 = 0
    over_sma50 = 0
    advances = 0
    declines = 0
    new_high = 0
    new_low = 0
    
    for ticker, df_t in df_ticker_dict.items():
        if len(df_t) < 200:
            continue
        
        total += 1
        last_close = df_t['Close'].iloc[-1]
        prev_close = df_t['Close'].iloc[-2]
        
        # SMA 200
        if 'SMA_200' in df_t.columns:
            sma200 = df_t['SMA_200'].iloc[-1]
            if pd.notna(sma200) and last_close > sma200:
                over_sma200 += 1
        
        # SMA 50
        if 'SMA_50' in df_t.columns:
            sma50 = df_t['SMA_50'].iloc[-1]
            if pd.notna(sma50) and last_close > sma50:
                over_sma50 += 1
        
        # Advances / Declines
        if last_close > prev_close:
            advances += 1
        elif last_close < prev_close:
            declines += 1
        
        # 52-ukers høy/lav
        if len(df_t) >= 252:
            high_52w = df_t['High'].iloc[-252:].max()
            low_52w = df_t['Low'].iloc[-252:].min()
            if last_close >= high_52w * 0.99:
                new_high += 1
            if last_close <= low_52w * 1.01:
                new_low += 1
    
    pct_sma200 = (over_sma200 / total * 100) if total else 0
    pct_sma50 = (over_sma50 / total * 100) if total else 0
    ad_ratio = advances / max(declines, 1)
    
    return {
        'total_analyzed': total,
        'pct_over_sma200': round(pct_sma200, 1),
        'pct_over_sma50': round(pct_sma50, 1),
        'advances': advances,
        'declines': declines,
        'new_52w_high': new_high,
        'new_52w_low': new_low,
        'ad_ratio': round(ad_ratio, 2),
    }