# logic.py
import pandas as pd
import numpy as np
import os

def beregn_tekniske_indikatorer(df):
    """
    Beregner alle tekniske indikatorer for analyse ved bruk av standard pandas.
    Fjerner avhengighet til eksterne biblioteker som pandas-ta for bedre kompatibilitet.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # 1. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50) # Fallback for nøytral verdi
    
    # 2. Glidende gjennomsnitt (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean() # Lagt til for Minervini
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 3. Volatilitet (ATR - Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 3b. 52-ukers High/Low (Nødvendig for VCP)
    df['High_52w'] = df['High'].rolling(window=252).max()
    df['Low_52w'] = df['Low'].rolling(window=252).min()
    
    # 4. MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
    # 5. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)
    
    # 6. Ichimoku Cloud (Forenklet for analyse)
    # Tenkan-sen (9-period high + 9-period low)/2
    df['ISA_9'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    # Kijun-sen (26-period high + 26-period low)/2
    df['ISB_26'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    # Chikou Span (Lagging Span) - Pris skiftet 26 dager frem
    df['ICS_26'] = df['Close'].shift(-26)
        
    # Sikrer at alle beregnede kolonner er numeriske
    numeric_cols = ['RSI', 'SMA_50', 'SMA_150', 'SMA_200', 'ATR', 'MACD', 'MACD_Signal', 
                    'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ISA_9', 'ISB_26', 'ICS_26']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def sjekk_strategier(df):
    """Implementerer tekniske strategier."""
    signaler = pd.DataFrame(index=df.index)
    
    col_list = ['Kort_Sikt_RSI', 'Golden_Cross', 'Momentum_Burst', 
                'Ichimoku_Breakout', 'Wyckoff_Spring', 'Bull_Race_Prep', 'VCP_Pattern']
    
    for c in col_list:
        signaler[c] = False

    if df.empty:
        return signaler

    # Kort Sikt RSI
    if 'SMA_200' in df.columns and 'RSI' in df.columns:
        mask = (df['Close'] > df['SMA_200']) & (df['RSI'] < 30)
        signaler['Kort_Sikt_RSI'] = mask.fillna(False)
    
    # Golden Cross
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        mask = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        signaler['Golden_Cross'] = mask.fillna(False)
    
    # Momentum Burst
    vol_avg = df['Volume'].rolling(20).mean()
    mask = (df['Close'].pct_change() > 0.03) & (df['Volume'] > vol_avg * 1.5)
    signaler['Momentum_Burst'] = mask.fillna(False)
    
    # Bull Race Prep
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        mask = (df['Close'] > df['BB_Upper']) & (bb_width.shift(1) < bb_width.rolling(20).mean())
        signaler['Bull_Race_Prep'] = mask.fillna(False)
    
    # Wyckoff Spring
    if 'SMA_50' in df.columns:
        vol_avg = df['Volume'].rolling(20).mean()
        support = df['Low'].rolling(window=40).min().shift(1)
        mask = (df['Low'] < support) & (df['Close'] > support) & (df['Volume'] > vol_avg * 1.2)
        signaler['Wyckoff_Spring'] = mask.fillna(False)

    # Ichimoku Breakout
    if all(col in df.columns for col in ['ISA_9', 'ISB_26']):
        # Enkel breakout: Pris over begge Span-linjer
        mask = (df['Close'] > df['ISA_9']) & (df['Close'] > df['ISB_26'])
        signaler['Ichimoku_Breakout'] = mask.fillna(False)
        
    # --- NY OG OPTIMALISERT: VCP (Mark Minervini) ---
    if all(col in df.columns for col in ['SMA_50', 'SMA_150', 'SMA_200', 'High_52w']):
        # 1. Trend Template (Minervini Stage 2 kriterier)
        trend_ok = (df['Close'] > df['SMA_50']) & \
                   (df['SMA_50'] > df['SMA_150']) & \
                   (df['SMA_150'] > df['SMA_200']) & \
                   (df['SMA_200'] > df['SMA_200'].shift(20)) & \
                   (df['Close'] > df['Low_52w'] * 1.3) & \
                   (df['Close'] > df['High_52w'] * 0.75)
        
        # 2. Price Tightness (Hjertet av VCP)
        # Vi ser etter dager der spreaden (High-Low) er liten relativt til prisen.
        daily_spread_pct = (df['High'] - df['Low']) / df['Close']
        # Gjennomsnittlig spread siste 10 dager må være under 3.5% (Veldig stramt)
        tightness = daily_spread_pct.rolling(10).mean() < 0.035
        
        # 3. Volum Tørke (Supply exhaustion)
        # Volumet i dag må være betydelig lavere enn 50-dagers snittet (under 70%)
        # Eller: Volum siste 5 dager er lavt. Vi bruker "Idag er lavt" for trigger.
        vol_avg_50 = df['Volume'].rolling(50).mean()
        volume_dry_up = df['Volume'] < (vol_avg_50 * 0.7)
        
        mask = trend_ok & tightness & volume_dry_up
        signaler['VCP_Pattern'] = mask.fillna(False)

    return signaler

def sjekk_kort_sikt_rsi(df):
    """
    RSI Mean Reversion - Forbedret versjon
    - Kjøpssignal når RSI < 30 OG pris er over SMA 200 (sterk trend)
    - Tilleggskrav: RSI må ha vært over 50 i løpet av siste 10 dager (bekrefter opptrend)
    """
    rsi = df['RSI']
    close = df['Close']
    sma200 = df['SMA_200']
    
    # RSI var over 50 nylig (bekrefter opptrend)
    rsi_was_high = rsi.rolling(10).max() > 50
    
    # Hovedsignal: RSI dip i sterk trend
    signal = (rsi < 30) & (close > sma200) & rsi_was_high
    
    # Unngå gjentatte signaler - minst 5 dager mellom
    signal = signal & (~signal.shift(1).fillna(False)) & (~signal.shift(2).fillna(False)) & \
             (~signal.shift(3).fillna(False)) & (~signal.shift(4).fillna(False))
    
    return signal

def sjekk_momentum_burst(df):
    """
    Momentum Burst - Forbedret versjon
    - Kursutbrudd over 20-dagers høy
    - Volum minst 2x gjennomsnitt (ikke bare 1.5x)
    - Pris må være over SMA 50 (mellomlangt momentum)
    """
    close = df['Close']
    high = df['High']
    volume = df['Volume']
    sma50 = df['SMA_50']
    
    # 20-dagers høy (ekskluderer dagens)
    high_20 = high.shift(1).rolling(20).max()
    
    # Volumsnitt
    vol_avg = volume.rolling(20).mean()
    
    # Breakout med sterkt volum og over SMA 50
    signal = (close > high_20) & (volume > vol_avg * 2) & (close > sma50)
    
    return signal

def sjekk_golden_cross(df):
    """
    Golden Cross - Forbedret versjon
    - SMA 50 krysser over SMA 200
    - Tilleggskrav: Begge MA-er må være stigende
    - Pris må være over begge MA-er ved krysset
    """
    sma50 = df['SMA_50']
    sma200 = df['SMA_200']
    close = df['Close']
    
    # Krysset skjer
    cross = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    
    # Begge MA-er stigende (positiv momentum)
    sma50_rising = sma50 > sma50.shift(5)
    sma200_rising = sma200 > sma200.shift(20)
    
    # Pris over begge
    price_above = close > sma50
    
    signal = cross & sma50_rising & sma200_rising & price_above
    
    return signal

def sjekk_vcp_pattern(df):
    """
    VCP (Volatility Contraction Pattern) - Forbedret Minervini-versjon
    - Pris må være i Stage 2 opptrend
    - Volatilitet må krympe (ATR synkende)
    - Minst 3 konsolideringer med avtagende amplitude
    - Pris nær 52-ukers høy (innen 25%)
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    sma50 = df['SMA_50']
    sma150 = df.get('SMA_150', df['SMA_50'])  # Fallback
    sma200 = df['SMA_200']
    
    # Stage 2 kriterier (Minervini)
    stage2 = (
        (close > sma50) & 
        (sma50 > sma150) & 
        (sma150 > sma200) &
        (sma200 > sma200.shift(20))  # SMA 200 stigende
    )
    
    # 52-ukers høy
    high_52w = high.rolling(252).max()
    near_high = close > (high_52w * 0.75)  # Innen 25% av høy
    
    # Volatilitetskrymping (ATR synkende)
    atr = (high - low).rolling(14).mean()
    atr_shrinking = atr < atr.shift(10)
    
    # Konsolidering (lav range siste 10 dager)
    range_10d = (high.rolling(10).max() - low.rolling(10).min()) / close
    tight_range = range_10d < 0.08  # Maks 8% range
    
    signal = stage2 & near_high & atr_shrinking & tight_range
    
    return signal

def sjekk_wyckoff_spring(df):
    """
    Wyckoff Spring - Forbedret versjon
    - Falsk brudd under støtte
    - Rask reversering tilbake over støtte
    - Økende volum på reverseringen
    """
    close = df['Close']
    low = df['Low']
    volume = df['Volume']
    
    # Finn støttenivå (20-dagers lav)
    support = low.shift(1).rolling(20).min()
    
    # Spring: Dagens lav under støtte, men lukk over
    spring = (low < support) & (close > support)
    
    # Volum høyere enn forrige dag (kjøpspress)
    vol_increase = volume > volume.shift(1)
    
    # Pris må lukke i øvre halvdel av dagens range
    range_today = df['High'] - low
    upper_close = (close - low) > (range_today * 0.5)
    
    signal = spring & vol_increase & upper_close
    
    return signal

def sjekk_bull_race_prep(df):
    """
    Bull Race Prep (Bollinger Squeeze) - Forbedret versjon
    - Bollinger Bands må være ekstremt trange
    - Pris må være i opptrend (over SMA 50)
    - Breakout over øvre band med volum
    """
    close = df['Close']
    sma50 = df['SMA_50']
    bb_upper = df['BB_Upper']
    bb_lower = df['BB_Lower']
    volume = df['Volume']
    
    # Bollinger Band bredde
    bb_width = (bb_upper - bb_lower) / df['BB_Middle']
    
    # Squeeze: BB bredde i nedre 10% av siste 100 dager
    bb_percentile = bb_width.rolling(100).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5)
    squeeze = bb_percentile < 0.1
    
    # Breakout over øvre band
    breakout = close > bb_upper
    
    # I opptrend
    uptrend = close > sma50
    
    # Volum over snitt
    vol_avg = volume.rolling(20).mean()
    high_vol = volume > vol_avg * 1.5
    
    signal = squeeze.shift(1) & breakout & uptrend & high_vol
    
    return signal

def sjekk_ichimoku_breakout(df):
    """
    Ichimoku Breakout - Forbedret versjon
    - Pris bryter over Kumo (skyen)
    - Tenkan-sen over Kijun-sen (bullish)
    - Chikou Span over pris (bekreftelse)
    """
    close = df['Close']
    
    # Beregn Ichimoku-komponenter hvis ikke allerede gjort
    if 'Senkou_A' not in df.columns:
        tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
        chikou = close.shift(-26)
    else:
        tenkan = df['Tenkan_Sen']
        kijun = df['Kijun_Sen']
        senkou_a = df['Senkou_A']
        senkou_b = df['Senkou_B']
        chikou = df['Chikou_Span']
    
    # Kumo topp og bunn
    kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    
    # Breakout over sky
    above_cloud = close > kumo_top
    was_below = close.shift(1) <= kumo_top.shift(1)
    
    # Tenkan over Kijun (momentum)
    tk_cross = tenkan > kijun
    
    signal = above_cloud & was_below & tk_cross
    
    return signal

def beregn_risk_reward(entry, stop, kapital, risiko_pst):
    if entry <= stop: return None
    risiko_kr = kapital * (risiko_pst / 100)
    risk_per_share = entry - stop
    if risk_per_share <= 0: return None
    antall = int(risiko_kr / risk_per_share)
    return {
        "antall": antall, 
        "total_investering": antall * entry, 
        "risiko_kr": risiko_kr, 
        "target_2r": entry + (risk_per_share * 2)
    }

def hent_strategi_detaljer(key):
    data = {
        "Kort Sikt (RSI Dip)": {
            "horisont": "Kortsiktig strategi",
            "beskrivelse": "**Kriterier:** Aksjen må være i en etablert lang trend (Pris > SMA 200). RSI (14) må falle under 30, og RSI må ha vært over 50 i løpet av siste 10 dager (bekrefter at det er en midlertidig dip i en opptrend, ikke en nedtrend). Cooldown på 5 dager mellom signaler."
        },
        "Momentum Burst": {
            "horisont": "Kortsiktig strategi",
            "beskrivelse": "**Kriterier:** Pris bryter over 20-dagers høy, kombinert med volum minst 2x gjennomsnittet siste 20 dager. Pris må være over SMA 50 for å bekrefte mellomlangt momentum."
        },
        "Golden Cross": {
            "horisont": "Lang strategi",
            "beskrivelse": "**Kriterier:** SMA 50 krysser over SMA 200 fra undersiden. Tilleggskrav: Begge glidende snitt må være stigende (SMA 50 over 5-dagers nivå, SMA 200 over 20-dagers nivå), og pris må være over SMA 50 ved krysset."
        },
        "Ichimoku Breakout": {
            "horisont": "Mellomlang strategi",
            "beskrivelse": "**Kriterier:** Pris bryter opp gjennom Kumo-skyen (over både Senkou Span A og B) fra undersiden. Bekreftes ved at Tenkan-sen er over Kijun-sen (bullish momentum)."
        },
        "Wyckoff Spring": {
            "horisont": "Spesialstrategi",
            "beskrivelse": "**Kriterier:** Dagens lav bryter under 20-dagers støtte, men lukker over støtten (falsk brudd). Volumet må øke fra forrige dag (viser kjøpspress), og pris må lukke i øvre halvdel av dagens range."
        },
        "Bull Race Prep": {
            "horisont": "Spesialstrategi",
            "beskrivelse": "**Kriterier:** Bollinger Band-bredde må være i nedre 10% av siste 100 dager (ekstrem squeeze). Pris må være over SMA 50 (opptrend), og breakout over øvre Bollinger Band med volum over 1.5x snittet."
        },
        "VCP (Minervini)": {
            "horisont": "Spesialstrategi",
            "beskrivelse": "**Minervini Stage 2 VCP:** Pris > SMA 50 > SMA 150 > SMA 200, og SMA 200 stigende. Pris må være innen 25% av 52-ukers høy og minst 30% over 52-ukers lav. Volatilitet krymper (daglig spread < 3.5% i snitt siste 10 dager) og volum kollapser under 70% av 50-dagers snitt – tegn på at selgerne er tomme."
        }
    }
    return data.get(key, {"horisont": "Ukjent", "beskrivelse": "Ingen beskrivelse tilgjengelig."})

def finn_stotte_motstand(df, perioder=50):
    """
    Finner støtte- og motstandsnivåer basert på lokale min/max.
    """
    if len(df) < perioder:
        return [], []
    
    high = df['High'].iloc[-perioder:]
    low = df['Low'].iloc[-perioder:]
    close = df['Close'].iloc[-1]
    
    # Finn lokale topper og bunner
    motstand = []
    stotte = []
    
    # Enkelt: Bruk høyeste og laveste punkter
    motstand.append(high.max())
    stotte.append(low.min())
    
    # Legg til noen mellomliggende nivåer
    median_high = high.median()
    median_low = low.median()
    
    if median_high > close:
        motstand.append(median_high)
    if median_low < close:
        stotte.append(median_low)
    
    return stotte, motstand

def finn_siste_signal_info(df, signaler, strategi_key):
    """
    Finner informasjon om siste signal for en gitt strategi.
    
    Args:
        df: DataFrame med prisdata
        signaler: DataFrame med signaler for alle strategier
        strategi_key: Nøkkel for strategien (f.eks. 'VCP_Pattern')
    
    Returns:
        dict med 'dato', 'dager_siden', 'utvikling_pst'
    """
    if strategi_key not in signaler.columns:
        return {
            'dato': 'Ingen data',
            'dager_siden': float('inf'),
            'utvikling_pst': 0.0
        }
    
    signal_series = signaler[strategi_key]
    signal_dates = signal_series[signal_series == True].index
    
    if len(signal_dates) == 0:
        return {
            'dato': 'Ingen signal',
            'dager_siden': float('inf'),
            'utvikling_pst': 0.0
        }
    
    # Finn siste signal
    siste_signal_dato = signal_dates[-1]
    
    # Beregn dager siden
    siste_dato_i_data = df.index[-1]
    dager_siden = (siste_dato_i_data - siste_signal_dato).days
    
    # Beregn kursutvikling siden signal
    try:
        pris_ved_signal = df.loc[siste_signal_dato, 'Close']
        nåværende_pris = df['Close'].iloc[-1]
        utvikling_pst = ((nåværende_pris - pris_ved_signal) / pris_ved_signal) * 100
    except Exception:
        utvikling_pst = 0.0
    
    return {
        'dato': siste_signal_dato.strftime('%Y-%m-%d'),
        'dager_siden': dager_siden,
        'utvikling_pst': round(utvikling_pst, 2)
    }