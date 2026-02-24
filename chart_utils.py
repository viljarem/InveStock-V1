"""
Moderne TradingView-stil charts med Lightweight Charts.
"""
import pandas as pd
import streamlit as st

# Sjekk om lightweight charts er tilgjengelig
try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LWC_INSTALLED = True
except ImportError:
    LWC_INSTALLED = False

# === FARGEPALETT ===
COLORS = {
    'up': '#26a69a',
    'down': '#ef5350',
    'sma_10': '#FF6B6B',
    'sma_20': '#4ECDC4', 
    'sma_50': '#45B7D1',
    'sma_100': '#96CEB4',
    'sma_150': '#FFEAA7',
    'sma_200': '#DDA0DD',
    'ema_9': '#FF9F43',
    'ema_21': '#EE5A24',
    'vwap': '#9B59B6',
    'bb': 'rgba(33, 150, 243, 0.6)',
    'rsi': '#AB47BC',
    'rsi_ob': 'rgba(239, 83, 80, 0.5)',
    'rsi_os': 'rgba(38, 166, 154, 0.5)',
    'macd': '#42A5F5',
    'macd_signal': '#FFA726',
}

# Signalfarge basert på kvalitet (A=grønn, B=gul, C=oransje, D=grå)
QUALITY_COLORS = {
    'A': '#00E676',  # Sterk grønn
    'B': '#FFEB3B',  # Gul
    'C': '#FF9800',  # Oransje
    'D': '#9E9E9E',  # Grå
}

SIGNAL_COLORS = {
    "Kort_Sikt_RSI": "#00E676",
    "Momentum_Burst": "#AA00FF",
    "Golden_Cross": "#FFD600",
    "Ichimoku_Breakout": "#00BCD4",
    "Wyckoff_Spring": "#2196F3",
    "Bull_Race_Prep": "#E91E63",
    "VCP_Pattern": "#FF9800",
    "Pocket_Pivot": "#00FFFF",
    "Strength_Pullback": "#FF5252"
}

# === PRESETS ===
PRESETS = {
    "clean": {
        "name": "Ren",
        "indicators": {},
        "oscillators": {"volume": True},
        "description": "Kun pris og volum"
    },
    "swing": {
        "name": "Swing",
        "indicators": {"sma_20": True, "sma_50": True, "bb": True},
        "oscillators": {"rsi": True, "volume": True},
        "description": "20/50 SMA + BB + RSI"
    },
    "trend": {
        "name": "Trend",
        "indicators": {"sma_50": True, "sma_200": True, "ema_21": True},
        "oscillators": {"macd": True, "volume": True},
        "description": "50/200 SMA + MACD"
    },
    "scalp": {
        "name": "Scalp",
        "indicators": {"ema_9": True, "ema_21": True, "vwap": True},
        "oscillators": {"rsi": True, "volume": True},
        "description": "EMA 9/21 + VWAP"
    },
    "full": {
        "name": "Full",
        "indicators": {"sma_50": True, "sma_200": True, "bb": True},
        "oscillators": {"rsi": True, "macd": True, "volume": True},
        "description": "Alle indikatorer"
    }
}


def _to_timestamp(idx):
    """Konverter pandas index til unix timestamp."""
    try:
        if hasattr(idx, 'timestamp'):
            return int(idx.timestamp())
        return int(pd.Timestamp(idx).timestamp())
    except:
        return 0


def _to_local_timestamp(idx):
    """Konverter pandas index til unix timestamp med tidssone-offset for lokal visning."""
    try:
        if hasattr(idx, 'timestamp'):
            ts = int(idx.timestamp())
            # Legg til UTC offset for korrekt lokal visning i Lightweight Charts
            if hasattr(idx, 'utcoffset') and idx.utcoffset() is not None:
                ts += int(idx.utcoffset().total_seconds())
            return ts
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


def _df_to_histogram(df: pd.DataFrame, column: str) -> list:
    """Konverter kolonne til histogram-format."""
    if column not in df.columns:
        return []
    
    data = []
    for idx, row in df.iterrows():
        try:
            val = row[column]
            if pd.isna(val):
                continue
            ts = _to_timestamp(idx)
            if ts > 0:
                color = 'rgba(38, 166, 154, 0.7)' if val >= 0 else 'rgba(239, 83, 80, 0.7)'
                data.append({'time': ts, 'value': float(val), 'color': color})
        except:
            continue
    return data


def _create_signal_markers(signal_data: dict, signal_keys: list) -> list:
    """Lag markører for kjøpssignaler med kvalitetsindikasjon."""
    markers = []
    
    for key in signal_keys:
        if key not in signal_data or signal_data[key].empty:
            continue
        
        for idx, row in signal_data[key].iterrows():
            try:
                ts = _to_timestamp(idx)
                if ts <= 0:
                    continue
                
                # Hent kvalitetsklasse og velg farge
                qual = row.get('kvalitet_klasse', 'C')
                color = QUALITY_COLORS.get(qual, QUALITY_COLORS['C'])
                
                # Størrelse basert på kvalitet
                size = 3 if qual == 'A' else 2.5 if qual == 'B' else 2 if qual == 'C' else 1.5
                
                # Tekst viser kvalitet (A/B/C/D) og strategi-forkortelse
                strat_short = key[:3].upper()
                text = f"{strat_short}:{qual}"
                
                markers.append({
                    'time': ts,
                    'position': 'belowBar',
                    'color': color,
                    'shape': 'arrowUp',
                    'size': size,
                    'text': text,
                })
            except:
                continue
    
    return markers


def _get_theme_config(theme: str) -> dict:
    """Hent tema-konfigurasjon."""
    if theme == 'dark':
        return {
            'bg': '#0e1117',
            'text': '#d1d4dc',
            'grid': 'rgba(255, 255, 255, 0.04)',
            'border': 'rgba(255, 255, 255, 0.1)',
        }
    else:
        return {
            'bg': '#ffffff',
            'text': '#131722',
            'grid': 'rgba(0, 0, 0, 0.04)',
            'border': 'rgba(0, 0, 0, 0.1)',
        }


def render_modern_chart(
    df: pd.DataFrame,
    indicators: dict = None,
    oscillators: dict = None,
    signals: dict = None,
    signal_keys: list = None,
    theme: str = 'dark',
    chart_height: int = 450,
    support_resistance: dict = None,
    exit_signals: pd.DataFrame = None,
    insider_trades: list = None,
    chart_settings: dict = None,  # NYE AVANSERTE INNSTILLINGER
    ml_score_data: pd.Series = None,  # ML Score subchart data
    ml_threshold: float = None,  # ML signal-terskel
):
    """
    Render et moderne chart med Lightweight Charts og avanserte customization-muligheter.
    """
    if not LWC_INSTALLED:
        st.error("streamlit-lightweight-charts er ikke installert")
        return
    
    if df.empty:
        st.warning("Ingen data å vise")
        return
    
    indicators = indicators or {}
    oscillators = oscillators or {}
    signal_keys = signal_keys or []
    chart_settings = chart_settings or {}
    
    # Bruk avanserte innstillinger eller fallback til tema
    theme_cfg = _get_theme_config(theme)
    
    # Bygg avanserte chart-opsjoner
    chart_options = build_advanced_chart_options(chart_settings) if chart_settings else _get_default_chart_options(theme_cfg)
    
    # Override chart height hvis spesifisert
    if chart_settings and 'chart_height' in chart_settings:
        chart_height = chart_settings['chart_height']
    
    # === HOVEDCHART SERIES ===
    main_series = []
    
    # Candlesticks med tilpassede innstillinger
    candle_data = _df_to_ohlc(df)
    if not candle_data:
        st.warning("Kunne ikke konvertere prisdata")
        return
    
    # Bygg candlestick-serie med brukerinnstillinger
    if chart_settings:
        candle_options = build_candlestick_series_options(chart_settings)
    else:
        candle_options = {
            'upColor': COLORS['up'],
            'downColor': COLORS['down'],
            'borderUpColor': COLORS['up'],
            'borderDownColor': COLORS['down'],
            'wickUpColor': COLORS['up'],
            'wickDownColor': COLORS['down'],
        }
    
    # Konverter data basert på chart type
    chart_type = chart_settings.get('chart_type', 'candlestick') if chart_settings else 'candlestick'
    
    if chart_type == 'line':
        # Konverter til line data (kun close prices)
        line_data = []
        for item in candle_data:
            line_data.append({'time': item['time'], 'value': item['close']})
        candle_data = line_data
    elif chart_type == 'area':
        # Konverter til area data (kun close prices)  
        area_data = []
        for item in candle_data:
            area_data.append({'time': item['time'], 'value': item['close']})
        candle_data = area_data
    
    candle_series = {
        'type': candle_options.get('type', 'Candlestick'),
        'data': candle_data,
        'options': candle_options
    }
    
    # === SAMLE ALLE MARKERS ===
    # Vi samler markers SEPARAT og legger dem til candlestick-serien HELT TIL SLUTT
    # for å unngå at ulike marker-typer blandes
    
    # Signal-markører (kjøpssignaler)
    signal_markers = []
    if signals and signal_keys:
        buy_markers = _create_signal_markers(signals, signal_keys)
        if buy_markers:
            signal_markers.extend(buy_markers)
    
    # Exit-signaler (salgssignaler) - IKKE pattern vision!
    exit_markers = []
    if exit_signals is not None and not exit_signals.empty:
        for date, row in exit_signals.iterrows():
            ts = int(date.timestamp())
            exit_markers.append({
                'time': ts,
                'position': 'aboveBar',
                'color': '#FF5252',  # Rød
                'shape': 'arrowDown',
                'text': f"EXIT ({row['antall']})",
            })
    
    # Vi legger IKKE til markers til candle_series her ennå - det gjøres til slutt
    
    main_series.append(candle_series)
    
    # === INDIKATORER ===
    indicator_map = {
        'sma_10': ('SMA_10', COLORS['sma_10'], 1),
        'sma_20': ('SMA_20', COLORS['sma_20'], 1),
        'sma_50': ('SMA_50', COLORS['sma_50'], 2),
        'sma_100': ('SMA_100', COLORS['sma_100'], 1),
        'sma_150': ('SMA_150', COLORS['sma_150'], 1),
        'sma_200': ('SMA_200', COLORS['sma_200'], 2),
        'ema_9': ('EMA_9', COLORS['ema_9'], 1),
        'ema_21': ('EMA_21', COLORS['ema_21'], 1),
        'vwap': ('VWAP', COLORS['vwap'], 1),
    }
    
    for key, (col, color, width) in indicator_map.items():
        if indicators.get(key) and col in df.columns:
            line_data = _df_to_line(df, col)
            if line_data:
                main_series.append({
                    'type': 'Line',
                    'data': line_data,
                    'options': {
                        'color': color,
                        'lineWidth': width,
                        'crosshairMarkerVisible': False,
                        'lastValueVisible': True,
                        'priceLineVisible': False,
                    }
                })
    
    # Bollinger Bands
    if indicators.get('bb'):
        for col in ['BB_Upper', 'BB_Lower']:
            if col in df.columns:
                line_data = _df_to_line(df, col)
                if line_data:
                    main_series.append({
                        'type': 'Line',
                        'data': line_data,
                        'options': {
                            'color': COLORS['bb'],
                            'lineWidth': 1,
                            'crosshairMarkerVisible': False,
                            'lastValueVisible': False,
                            'priceLineVisible': False,
                        }
                    })
    
    # === STØTTE OG MOTSTAND ===
    if support_resistance:
        stotte = support_resistance.get('stotte', [])
        motstand = support_resistance.get('motstand', [])
        
        # Få hele dataområdet for linjer som ikke forsvinner ved zoom
        first_time = _to_timestamp(df.index[0])
        last_time = _to_timestamp(df.index[-1])
        
        # Lag horisontale linjer for støtte (grønn)
        for s in stotte:
            pris = s['pris'] if isinstance(s, dict) else s
            styrke = s.get('styrke', 1) if isinstance(s, dict) else 1
            line_width = max(1, min(3, 1 + styrke * 0.5))  # Begrenset linjetype
            
            # Lag data for hele perioden med buffer
            line_data = [
                {'time': first_time, 'value': pris},
                {'time': last_time, 'value': pris}
            ]
            
            main_series.append({
                'type': 'Line',
                'data': line_data,
                'options': {
                    'color': 'rgba(38, 166, 154, 0.8)',  # Litt mer synlig
                    'lineWidth': line_width,
                    'lineStyle': 2,  # Stiplet
                    'crosshairMarkerVisible': True,
                    'lastValueVisible': True,
                    'priceLineVisible': False,
                    'title': f'Støtte: {pris:.2f}',
                }
            })
        
        # Lag horisontale linjer for motstand (rød)
        for m in motstand:
            pris = m['pris'] if isinstance(m, dict) else m
            styrke = m.get('styrke', 1) if isinstance(m, dict) else 1
            line_width = max(1, min(3, 1 + styrke * 0.5))  # Begrenset linjetykkelse
            
            line_data = [
                {'time': first_time, 'value': pris},
                {'time': last_time, 'value': pris}
            ]
            
            main_series.append({
                'type': 'Line',
                'data': line_data,
                'options': {
                    'color': 'rgba(239, 83, 80, 0.8)',  # Litt mer synlig
                    'lineWidth': line_width,
                    'lineStyle': 2,  # Stiplet
                    'crosshairMarkerVisible': True,
                    'lastValueVisible': True,
                    'priceLineVisible': False,
                    'title': f'Motstand: {pris:.2f}',
                }
            })
    
    # === INSIDER-HANDLER MARKERS ===
    insider_markers = []
    if insider_trades:
        for trade in insider_trades:
            try:
                trade_date = pd.Timestamp(trade.get('dato', ''))
                # Finn nærmeste dato i df
                idx = df.index.get_indexer([trade_date], method='nearest')[0]
                if idx < 0 or idx >= len(df):
                    continue
                actual_date = df.index[idx]
                ts = _to_timestamp(actual_date)
                
                trade_type = trade.get('type', 'ukjent')
                if trade_type == 'kjøp':
                    color = '#00C853'
                    shape = 'arrowUp'
                    position = 'belowBar'
                    label = 'Kjøp'
                elif trade_type == 'salg':
                    color = '#FF5252'
                    shape = 'arrowDown'
                    position = 'aboveBar'
                    label = 'Salg'
                else:
                    color = '#FFA726'
                    shape = 'circle'
                    position = 'belowBar'
                    label = '?'
                
                insider_markers.append({
                    'time': ts,
                    'position': position,
                    'color': color,
                    'shape': shape,
                    'text': f"🏛️{label}",
                })
            except Exception:
                continue
    
    # === MØNSTER MARKERS ===
    pattern_markers = []
    if chart_settings and chart_settings.get('patterns'):
        for pattern in chart_settings['patterns']:
            try:
                pattern_date = pd.Timestamp(pattern.get('dato', ''))
                idx = df.index.get_indexer([pattern_date], method='nearest')[0]
                if idx < 0 or idx >= len(df):
                    continue
                actual_date = df.index[idx]
                ts = _to_timestamp(actual_date)
                
                retning = pattern.get('retning', 'bullish')
                if retning == 'bullish':
                    color = '#00BCD4'  # Cyan
                    shape = 'arrowUp'
                    position = 'belowBar'
                else:
                    color = '#FF9800'  # Orange
                    shape = 'arrowDown'
                    position = 'aboveBar'
                
                pattern_name = pattern.get('mønster', 'Mønster')
                
                pattern_markers.append({
                    'time': ts,
                    'position': position,
                    'color': color,
                    'shape': shape,
                    'text': f"📐 {pattern_name}",
                })
            except Exception:
                continue
    
    # === KOMBINER MARKERS OG LEGG TIL CANDLESTICK-SERIEN ===
    all_markers = signal_markers + exit_markers + insider_markers + pattern_markers
    if all_markers:
        candle_series['markers'] = all_markers
    
    # === BYGG CHARTS ===
    charts = []
    
    # Hovedchart
    main_chart = {
        'chart': {
            'height': chart_height,
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
    }
    charts.append(main_chart)
    
    # === VOLUM ===
    if oscillators.get('volume', True):
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
    
    # === RSI ===
    if oscillators.get('rsi') and 'RSI' in df.columns:
        rsi_data = _df_to_line(df, 'RSI')
        if rsi_data:
            # Lag 70/30 linjer
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
                    {'type': 'Line', 'data': rsi_70, 'options': {'color': COLORS['rsi_ob'], 'lineWidth': 1, 'lineStyle': 2}},
                    {'type': 'Line', 'data': rsi_30, 'options': {'color': COLORS['rsi_os'], 'lineWidth': 1, 'lineStyle': 2}},
                ],
            })
    
    # === MACD ===
    if oscillators.get('macd') and 'MACD' in df.columns:
        macd_line = _df_to_line(df, 'MACD')
        signal_line = _df_to_line(df, 'MACD_Signal')
        hist_data = _df_to_histogram(df, 'MACD_Hist')
        
        if macd_line:
            macd_series = []
            if hist_data:
                macd_series.append({'type': 'Histogram', 'data': hist_data, 'options': {}})
            macd_series.append({'type': 'Line', 'data': macd_line, 'options': {'color': COLORS['macd'], 'lineWidth': 2}})
            if signal_line:
                macd_series.append({'type': 'Line', 'data': signal_line, 'options': {'color': COLORS['macd_signal'], 'lineWidth': 2}})
            
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
                'series': macd_series,
            })
    
    # === ML SCORE ===
    if ml_score_data is not None and not ml_score_data.empty:
        # Filtrer til visningsperiode
        try:
            ml_filtered = ml_score_data[
                (ml_score_data.index >= df.index.min()) & 
                (ml_score_data.index <= df.index.max())
            ]
        except Exception:
            ml_filtered = ml_score_data
        
        if not ml_filtered.empty:
            # Bygg hovedchartets fulle tidsserie (for synkronisering)
            all_timestamps = []
            for i in df.index:
                try:
                    ts = int(i.timestamp())
                    if ts > 0:
                        all_timestamps.append(ts)
                except Exception:
                    continue
            
            if all_timestamps:
                thr_val = ml_threshold if ml_threshold is not None else 50
                
                # ML Score som fargelagt histogram (grønn over terskel, rød under)
                ml_hist_data = []
                for idx, val in ml_filtered.items():
                    try:
                        ts = int(idx.timestamp())
                        v = float(val)
                        color = 'rgba(0,200,83,0.7)' if v >= thr_val else 'rgba(255,82,82,0.35)'
                        ml_hist_data.append({'time': ts, 'value': v, 'color': color})
                    except Exception:
                        continue
                
                # ML Score som linje (for tydeligere avlesning)
                ml_line_data = []
                for idx, val in ml_filtered.items():
                    try:
                        ts = int(idx.timestamp())
                        ml_line_data.append({'time': ts, 'value': float(val)})
                    except Exception:
                        continue
                
                if ml_line_data:
                    ml_series = []
                    
                    # Terskel-linje over hele perioden (synkroniserer tidsakse)
                    thr_data = [{'time': ts, 'value': thr_val} for ts in all_timestamps]
                    ml_series.append({
                        'type': 'Line',
                        'data': thr_data,
                        'options': {
                            'color': '#ffa726',
                            'lineWidth': 1,
                            'lineStyle': 2,
                            'lastValueVisible': False,
                            'priceLineVisible': False,
                        }
                    })
                    
                    # Fargelagt histogram
                    ml_series.append({
                        'type': 'Histogram',
                        'data': ml_hist_data,
                        'options': {
                            'lastValueVisible': False,
                            'priceLineVisible': False,
                        }
                    })
                    
                    # Linje oppå histogrammet
                    ml_series.append({
                        'type': 'Line',
                        'data': ml_line_data,
                        'options': {
                            'color': '#64b5f6',
                            'lineWidth': 2,
                            'lastValueVisible': True,
                            'priceLineVisible': False,
                            'title': 'ML',
                        }
                    })
                    
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
                                'scaleMargins': {'top': 0.05, 'bottom': 0.05},
                            },
                            'timeScale': {'visible': True},
                        },
                        'series': ml_series,
                    })
    
    # === RENDER ===
    try:
        # Lag en stabil key basert på ticker (fra df) og preset-info
        import hashlib
        key_data = f"{len(df)}_{len(main_series)}_{len(charts)}"
        chart_key = f"lwc_{hashlib.md5(key_data.encode()).hexdigest()[:12]}"
        
        # Sjekk at vi har data
        if not charts:
            st.error("Ingen charts å vise!")
            return
        main_series_in_chart = charts[0].get('series', [])
        if not main_series_in_chart:
            st.error("Hovedchart har ingen serier!")
            return
        
        # Sjekk candlestick-data
        candle_data_check = main_series_in_chart[0].get('data', []) if main_series_in_chart else []
        if not candle_data_check:
            st.error("Candlestick-serien har ingen data!")
            return
            
        renderLightweightCharts(charts, key=chart_key)
    except Exception as e:
        st.error(f"Feil ved rendering av chart: {e}")
        import traceback
        st.code(traceback.format_exc())


def _get_default_chart_options(theme_cfg):
    """Fallback chart-opsjoner når avanserte innstillinger ikke er tilgjengelige."""
    return {
        'layout': {
            'background': {'color': theme_cfg['bg']},
            'textColor': theme_cfg['text']
        },
        'grid': {
            'vertLines': {'color': theme_cfg['grid']},
            'horzLines': {'color': theme_cfg['grid']}
        },
        'crosshair': {
            'mode': 0
        },
        'rightPriceScale': {
            'borderVisible': False
        },
        'timeScale': {
            'borderVisible': False
        }
    }


def render_chart_controls(ticker: str):
    """
    Render chart-kontroller med avanserte lightweight-charts innstillinger.
    """
    # === HOVEDKONTROLLER ===
    col_pills, col_preset, col_theme, col_settings = st.columns([2, 1, 1, 0.5])
    
    with col_pills:
        periode = st.segmented_control(
            "Periode",
            ["1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"],
            default="1Y",
            selection_mode="single",
            key="chart_periode",
            label_visibility="collapsed"
        )
        if not periode:
            periode = "1Y"
    
    with col_preset:
        preset_key = st.selectbox(
            "Oppsett",
            list(PRESETS.keys()),
            format_func=lambda x: PRESETS[x]['name'],
            index=1,
            key="chart_preset",
            label_visibility="collapsed"
        )
    
    with col_theme:
        theme = st.selectbox(
            "Tema",
            ["dark", "light"],
            format_func=lambda x: "Mørk" if x == "dark" else "Lys",
            index=0,
            key="chart_theme",
            label_visibility="collapsed"
        )
    
    with col_settings:
        # === AVANSERTE LIGHTWEIGHT-CHARTS INNSTILLINGER ===
        with st.popover("⚙️", use_container_width=True):
            st.markdown("### 🎨 **Utseende**")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["candlestick", "ohlc", "line", "area"],
                    format_func=lambda x: {
                        "candlestick": "🕯️ Candlestick",
                        "ohlc": "📊 OHLC",
                        "line": "📈 Linje",
                        "area": "📉 Område"
                    }[x],
                    key="chart_type"
                )
                
                grid_visible = st.checkbox("Vis rutenett", value=True, key="grid_visible")
                wick_visible = st.checkbox("Vis veker", value=True, key="wick_visible")
                border_visible = st.checkbox("Vis kantlinjer", value=True, key="border_visible")
                
            with col_b:
                up_color = st.color_picker(
                    "Opp-farge", 
                    value="#26A69A", 
                    key="up_color"
                )
                
                down_color = st.color_picker(
                    "Ned-farge", 
                    value="#EF5350", 
                    key="down_color"
                )
            
            st.divider()
            st.markdown("### 📌 **Akser & Skala**")
            
            col_c, col_d = st.columns(2)
            
            with col_c:
                price_scale_mode = st.selectbox(
                    "Prisskala",
                    [0, 1, 2, 3],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "Logaritmisk", 
                        2: "Prosent",
                        3: "Indeksert"
                    }[x],
                    key="price_scale_mode"
                )
                invert_scale = st.checkbox("Inverter skala", value=False, key="invert_scale")
                
            with col_d:
                price_format_precision = st.number_input(
                    "Desimaler",
                    min_value=0,
                    max_value=8,
                    value=2,
                    key="price_precision"
                )
                time_visible = st.checkbox("Vis tid", value=True, key="time_visible")
                seconds_visible = st.checkbox("Vis sekunder", value=False, key="seconds_visible")
                shift_visible_range = st.checkbox("Auto-scroll", value=True, key="auto_scroll")
            
            st.divider()
            st.markdown("### 🗒 **Crosshair & Layout**")
            
            col_e, col_f = st.columns(2)
            
            with col_e:
                crosshair_mode = st.selectbox(
                    "Crosshair modus",
                    [0, 1, 2],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "Magnet",
                        2: "Skjult"
                    }[x],
                    key="crosshair_mode"
                )
                crosshair_color = st.color_picker(
                    "Crosshair farge",
                    value="#758696",
                    key="crosshair_color"
                )
                
            with col_f:
                chart_height = st.slider(
                    "Chart høyde (px)",
                    min_value=300,
                    max_value=800,
                    value=500,
                    step=50,
                    key="chart_height"
                )
                watermark_text = st.text_input(
                    "Watermark tekst",
                    value=ticker,
                    key="watermark_text"
                )
    
    periode_map = {
        "1M": 30, "3M": 90, "6M": 180, "1Y": 365, 
        "2Y": 730, "5Y": 1825, "MAX": 9999
    }
    
    # === SAMLE ALLE INNSTILLINGER ===
    chart_settings = {
        'periode_dager': periode_map[periode],
        'preset': PRESETS[preset_key],
        'preset_key': preset_key,
        'theme': theme,
        
        # Avanserte innstillinger
        'chart_type': chart_type,
        'up_color': up_color,
        'down_color': down_color,
        'grid_visible': grid_visible,
        'wick_visible': wick_visible,
        'border_visible': border_visible,
        'price_scale_mode': price_scale_mode,
        'price_format_precision': price_format_precision,
        'invert_scale': invert_scale,
        'time_visible': time_visible,
        'seconds_visible': seconds_visible,
        'auto_scroll': shift_visible_range,
        'crosshair_mode': crosshair_mode,
        'crosshair_color': crosshair_color,
        'chart_height': chart_height,
        'watermark_text': watermark_text
    }
    
    return chart_settings


def build_advanced_chart_options(settings):
    """
    Bygger avanserte chart-opsjoner basert på brukerinnstillinger.
    """
    theme_colors = {
        'dark': {
            'background': '#1E1E1E',
            'text': '#D9D9D9',
            'grid': '#2A2A2A',
            'crosshair': '#758696'
        },
        'light': {
            'background': '#FFFFFF', 
            'text': '#333333',
            'grid': '#F0F0F0',
            'crosshair': '#9598A1'
        }
    }
    
    theme = settings.get('theme', 'dark')
    colors = theme_colors[theme]
    
    chart_options = {
        'layout': {
            'background': {'color': colors['background']},
            'textColor': colors['text'],
            'fontSize': 12,
            'fontFamily': 'Trebuchet MS, sans-serif'
        },
        'grid': {
            'vertLines': {
                'color': colors['grid'],
                'style': 1,  # Solid
                'visible': settings.get('grid_visible', True)
            },
            'horzLines': {
                'color': colors['grid'], 
                'style': 1,
                'visible': settings.get('grid_visible', True)
            }
        },
        'crosshair': {
            'mode': settings.get('crosshair_mode', 0),  # 0=Normal, 1=Magnet, 2=Hidden
            'vertLine': {
                'color': settings.get('crosshair_color', colors['crosshair']),
                'width': 1,
                'style': 3,  # Dashed
                'visible': True,
                'labelVisible': True
            },
            'horzLine': {
                'color': settings.get('crosshair_color', colors['crosshair']),
                'width': 1,
                'style': 3,
                'visible': True,
                'labelVisible': True
            }
        },
        'rightPriceScale': {
            'scaleMargins': {
                'top': 0.1,
                'bottom': 0.1
            },
            'mode': settings.get('price_scale_mode', 0),  # 0=Normal, 1=Log, 2=Percentage, 3=Indexed
            'invertScale': settings.get('invert_scale', False),
            'alignLabels': True,
            'borderVisible': False,
            'visible': True
        },
        'timeScale': {
            'visible': settings.get('time_visible', True),
            'timeVisible': True,
            'secondsVisible': settings.get('seconds_visible', False),
            'shiftVisibleRangeOnNewBar': settings.get('auto_scroll', True),
            'borderVisible': False
        },
        'watermark': {
            'visible': True,
            'fontSize': 48,
            'horzAlign': 'center',
            'vertAlign': 'center',
            'color': f"{colors['text']}20",  # 20 = low opacity
            'text': settings.get('watermark_text', ''),
            'fontFamily': 'Trebuchet MS, sans-serif',
            'fontStyle': 'bold'
        },
        'handleScroll': {
            'mouseWheel': True,
            'pressedMouseMove': True,
            'horzTouchDrag': True,
            'vertTouchDrag': True
        },
        'handleScale': {
            'axisPressedMouseMove': {
                'time': True,
                'price': True
            },
            'axisDoubleClickReset': True,
            'mouseWheel': True,
            'pinch': True
        },
        'kineticScroll': {
            'touch': True,
            'mouse': False
        }
    }
    
    return chart_options


def build_candlestick_series_options(settings):
    """
    Bygger candlestick-serie innstillinger.
    """
    chart_type = settings.get('chart_type', 'candlestick')
    
    base_options = {
        'priceFormat': {
            'type': 'price',
            'precision': settings.get('price_format_precision', 2),
            'minMove': 1 / (10 ** settings.get('price_format_precision', 2))
        }
    }
    
    if chart_type == 'candlestick':
        return {
            **base_options,
            'type': 'Candlestick',
            'upColor': settings.get('up_color', '#26A69A'),
            'downColor': settings.get('down_color', '#EF5350'),
            'borderVisible': settings.get('border_visible', True),
            'wickVisible': settings.get('wick_visible', True),
            'borderUpColor': settings.get('up_color', '#26A69A'),
            'borderDownColor': settings.get('down_color', '#EF5350'),
            'wickUpColor': settings.get('up_color', '#26A69A'),
            'wickDownColor': settings.get('down_color', '#EF5350')
        }
    elif chart_type == 'ohlc':
        return {
            **base_options,
            'type': 'Bar',
            'upColor': settings.get('up_color', '#26A69A'),
            'downColor': settings.get('down_color', '#EF5350'),
            'openVisible': True,
            'thinBars': False
        }
    elif chart_type == 'line':
        return {
            **base_options,
            'type': 'Line',
            'color': settings.get('up_color', '#26A69A'),
            'lineStyle': 0,  # Solid
            'lineWidth': 2,
            'crosshairMarkerVisible': True,
            'crosshairMarkerRadius': 4
        }
    elif chart_type == 'area':
        return {
            **base_options,
            'type': 'Area',
            'lineColor': settings.get('up_color', '#26A69A'),
            'topColor': settings.get('up_color', '#26A69A') + '80',  # 50% opacity
            'bottomColor': settings.get('up_color', '#26A69A') + '00',  # Transparent
            'lineWidth': 2
        }
    
    return base_options


def render_indicator_selector():
    """
    Render indikator-velger. Returnerer valgte indikatorer og oscillatorer.
    """
    indicators = {}
    oscillators = {'volume': True}
    
    with st.popover("📊 Indikatorer", use_container_width=True):
        st.markdown("**Moving Averages**")
        c1, c2 = st.columns(2)
        indicators['sma_20'] = c1.checkbox("SMA 20", key="cust_sma20")
        indicators['sma_50'] = c2.checkbox("SMA 50", key="cust_sma50")
        indicators['sma_200'] = c1.checkbox("SMA 200", key="cust_sma200")
        indicators['ema_21'] = c2.checkbox("EMA 21", key="cust_ema21")
        
        indicators['sma_10'] = c1.checkbox("SMA 10", key="cust_sma10")
        indicators['sma_100'] = c2.checkbox("SMA 100", key="cust_sma100")
        indicators['ema_9'] = c1.checkbox("EMA 9", key="cust_ema9")
        indicators['vwap'] = c2.checkbox("VWAP", key="cust_vwap")
        
        st.divider()
        st.markdown("**Bånd og Oscillatorer**")
        c3, c4 = st.columns(2)
        indicators['bb'] = c3.checkbox("Bollinger", key="cust_bb")
        oscillators['rsi'] = c4.checkbox("RSI", key="cust_rsi")
        oscillators['macd'] = c3.checkbox("MACD", key="cust_macd")
    
    return indicators, oscillators


def render_signal_selector(signal_keys: list, signal_names: dict):
    """
    Render signal-velger.
    """
    with st.popover("🎯 Signaler", use_container_width=True):
        st.caption("Farge indikerer kvalitet: Grønn=A, Gul=B, Oransje=C, Grå=D")
        selected = st.multiselect(
            "Velg strategier",
            signal_keys,
            default=[],
            format_func=lambda x: signal_names.get(x, x),
            key="signal_select",
            label_visibility="collapsed"
        )
    return selected

