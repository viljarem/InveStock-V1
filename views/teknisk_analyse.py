import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import logic
import data
import config
from shared_cache import cached_beregn_tekniske_indikatorer, hent_signaler_cached
from log_config import get_logger
import user_settings

logger = get_logger(__name__)

# Import AI analyzer
AI_AVAILABLE = False
try:
    import gemini_analyzer
    AI_AVAILABLE = True
except ImportError:
    logger.warning("[teknisk_analyse] Gemini analyzer ikke tilgjengelig")

# Import ML model
ML_AVAILABLE = False
try:
    import beta_ml
    ML_AVAILABLE = True
except ImportError:
    logger.warning("[teknisk_analyse] ML model ikke tilgjengelig")

LWC_AVAILABLE = False
try:
    from chart_utils import (
        render_modern_chart,
        render_chart_controls,
        render_indicator_selector,
        render_signal_selector,
        PRESETS,
        LWC_INSTALLED,
    )
    LWC_AVAILABLE = LWC_INSTALLED
except ImportError as e:
    logger.warning(f"[teknisk_analyse] ImportError: {e}")
except Exception as e:
    logger.warning(f"[teknisk_analyse] Exception: {e}")


def render():
    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    unike_tickers = st.session_state.get('unike_tickers', [])

    st.title("Teknisk Analyse")
    
    # === TICKER VALG ===
    # Sjekk om vi kommer fra portefølje med pre-valgt ticker
    default_ticker_idx = 0
    if 'valgt_ticker' in st.session_state and st.session_state['valgt_ticker'] in unike_tickers:
        default_ticker_idx = unike_tickers.index(st.session_state['valgt_ticker'])
        # Rydd opp etter bruk
        del st.session_state['valgt_ticker']
    
    col_t, col_n = st.columns([1, 2])
    with col_t:
        ticker = st.selectbox("Aksje", unike_tickers, index=default_ticker_idx, label_visibility="collapsed")
    
    # Hent og cache data
    df_full = df_clean[df_clean['Ticker'] == ticker]
    if ticker not in st.session_state['teknisk_cache']:
        st.session_state['teknisk_cache'][ticker] = cached_beregn_tekniske_indikatorer(df_full)
    df_full = st.session_state['teknisk_cache'][ticker]
    
    selskap_navn = data.ticker_til_navn(ticker) if hasattr(data, "ticker_til_navn") else ticker.replace(".OL", "")
    
    with col_n:
        st.markdown(f"### {selskap_navn}")
    
    # === PRIS-HEADER ===
    try:
        ticker_yf = ticker if ticker.endswith(".OL") else ticker + ".OL"
        hist = yf.Ticker(ticker_yf).history(period="1d", interval="1m")
        if not hist.empty:
            latest_price = float(hist['Close'].iloc[-1])
            latest_time = hist.index[-1].strftime("%H:%M")
        else:
            raise ValueError()
    except:
        latest_price = float(df_full['Close'].iloc[-1])
        latest_time = df_full.index[-1].strftime("%Y-%m-%d")
    
    prev_close = float(df_full['Close'].iloc[-2]) if len(df_full) > 1 else latest_price
    change = latest_price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0
    
    price_color = "#26a69a" if change >= 0 else "#ef5350"
    sign = "+" if change >= 0 else ""
    
    st.markdown(f"""
    <div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 16px;">
        <span style="font-size: 2rem; font-weight: 700; color: #fff;">{latest_price:.2f}</span>
        <span style="font-size: 1rem; color: {price_color}; font-weight: 600;">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</span>
        <span style="font-size: 0.8rem; color: #888;">{latest_time}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # === AI ANALYSE ===
    if AI_AVAILABLE:
        _vis_ai_teknisk_analyse(ticker, df_full)
    
    # === CHART KONTROLLER ===
    if LWC_AVAILABLE:
        # Kontroller
        ctrl = render_chart_controls(ticker)
        preset = ctrl['preset']
        
        # Signal-velger data
        signal_keys = ["Kort_Sikt_RSI", "Momentum_Burst", "Golden_Cross", 
                       "Ichimoku_Breakout", "Wyckoff_Spring", "Bull_Race_Prep", "VCP_Pattern", "Pocket_Pivot", "Strength_Pullback"]
        signal_names = {
            "Kort_Sikt_RSI": "RSI Dip",
            "Momentum_Burst": "Momentum",
            "Golden_Cross": "Golden Cross",
            "Ichimoku_Breakout": "Ichimoku",
            "Wyckoff_Spring": "Wyckoff",
            "Bull_Race_Prep": "BB Squeeze",
            "VCP_Pattern": "VCP",
            "Pocket_Pivot": "Pocket Pivot",
            "Strength_Pullback": "Strength Pullback"
        }
        
        # Popovers for indikatorer og signaler i samme rad
        col_ind, col_sig, col_space = st.columns([1, 1, 4])
        with col_ind:
            custom_ind, custom_osc = render_indicator_selector()
        with col_sig:
            selected_signals = render_signal_selector(signal_keys, signal_names)
        
        # Kombiner preset med tilpasninger
        final_ind = {**preset['indicators']}
        final_osc = {**preset['oscillators']}
        
        # Legg til tilpasninger (kun de som er aktivt valgt)
        for k, v in custom_ind.items():
            if v:
                final_ind[k] = True
        for k, v in custom_osc.items():
            if v:
                final_osc[k] = True
        
        # Kompakt layout for tilleggsfunksjoner
        st.markdown("**Tillegg på chart:**")
        
        # Les ML-historikk fra brukerinnstillinger
        settings = user_settings.load_settings()
        ml_hist_days = settings.get('teknisk_analyse', {}).get('ml_hist_days', 120)
        
        feature_col1, feature_col2, feature_col3, feature_col4, feature_col5 = st.columns(5)
        
        with feature_col1:
            show_sr = st.checkbox("📈 Støtte/Motstand", value=False, help="Automatisk beregnede nivåer")
        
        with feature_col2:
            show_exit = st.checkbox("🚨 Exit-signaler", value=False, help="Historiske exit-signaler (2+)")
        
        with feature_col3:
            show_insider = st.checkbox("💼 Insider-handler", value=False, help="Meldepliktige handler")
            
        with feature_col4:
            show_patterns = st.checkbox("📐 Mønstre", value=False, help="Tekniske prisformasjoner")
        
        with feature_col5:
            run_ml = st.button("🧠 Kjør ML", help=f"Kjør ML-modellen ({ml_hist_days}d historikk — endre i Innstillinger)", use_container_width=True) if ML_AVAILABLE else False
        
        # Filtrer data til periode
        end_date = df_full.index.max()
        days = min(ctrl['periode_dager'], len(df_full))
        start_date = end_date - pd.DateOffset(days=days)
        df_view = df_full[df_full.index >= start_date].copy()
        
        # Beregn ekstra indikatorer
        if final_ind.get('ema_9') and 'EMA_9' not in df_view.columns:
            df_view['EMA_9'] = df_view['Close'].ewm(span=9, adjust=False).mean()
        if final_ind.get('ema_21') and 'EMA_21' not in df_view.columns:
            df_view['EMA_21'] = df_view['Close'].ewm(span=21, adjust=False).mean()
        if final_ind.get('sma_10') and 'SMA_10' not in df_view.columns:
            df_view['SMA_10'] = df_view['Close'].rolling(10).mean()
        if final_ind.get('sma_20') and 'SMA_20' not in df_view.columns:
            df_view['SMA_20'] = df_view['Close'].rolling(20).mean()
        if final_ind.get('sma_100') and 'SMA_100' not in df_view.columns:
            df_view['SMA_100'] = df_view['Close'].rolling(100).mean()
        if final_ind.get('vwap') and 'VWAP' not in df_view.columns:
            cumvol = df_view['Volume'].cumsum().replace(0, pd.NA)
            df_view['VWAP'] = (df_view['Volume'] * (df_view['High'] + df_view['Low'] + df_view['Close']) / 3).cumsum() / cumvol
        
        # Hent signaldata og filtrer til visningsperioden
        signal_data = {}
        if selected_signals:
            signaler = logic.sjekk_strategier(df_full)
            all_signals = logic.hent_signaler_for_chart(df_full, signaler, selected_signals, start_date=None)
            
            # Filtrer signaler til kun de som er innenfor visningsperioden
            for key, sig_df in all_signals.items():
                if not sig_df.empty:
                    # Behold signaler som er innenfor df_view's tidsperiode
                    mask = (sig_df.index >= start_date) & (sig_df.index <= end_date)
                    signal_data[key] = sig_df[mask]
                else:
                    signal_data[key] = sig_df
        
        # Beregn støtte/motstand hvis aktivert
        sr_data = None
        if show_sr:
            stotte, motstand = logic.finn_stotte_motstand(df_full, perioder=min(100, len(df_full)))
            sr_data = {'stotte': stotte, 'motstand': motstand}
        
        # Beregn exit-signaler hvis aktivert
        exit_data = None
        if show_exit:
            exit_df = logic.beregn_exit_signaler_historisk(df_full, min_signaler=2)
            if not exit_df.empty:
                # Filtrer til visningsperioden
                mask = (exit_df.index >= start_date) & (exit_df.index <= end_date)
                exit_data = exit_df[mask]
        
        # Hent insider-handler for chart-markører
        insider_data = None
        if show_insider:
            try:
                import insider_monitor
                ins_handler = insider_monitor.hent_innsidehandler(dager=90)
                if ins_handler:
                    ticker_handler = [h for h in ins_handler if h['ticker'] == ticker]
                    if ticker_handler:
                        insider_data = ticker_handler
            except Exception:
                pass
                
        # Hent mønstre for chart-markører
        if show_patterns:
            try:
                import pattern_logic
                mønstre = pattern_logic.skann_alle_mønstre(df_full, kun_siste_n_dager=120)
                if mønstre:
                    # Filtrer til visningsperioden
                    ctrl['patterns'] = [m for m in mønstre if m['dato'] >= start_date and m['dato'] <= end_date]
            except Exception:
                pass
        
        # === ML BEREGNING (må skje FØR chart for å fylle cache) ===
        if ML_AVAILABLE and run_ml:
            _kjør_ml_beregning(ticker, df_full, hist_dager=ml_hist_days)
        
        # Hent ML score-data for subchart hvis det finnes i cache
        ml_chart_data = None
        ml_chart_threshold = None
        if ML_AVAILABLE:
            cache_key_ml = f"ml_ta_{ticker}_{ml_hist_days}"
            cached_ml = st.session_state.get(cache_key_ml)
            if cached_ml and not cached_ml.get('hist_probs', pd.Series()).empty:
                ml_chart_data = cached_ml['hist_probs']
                ml_chart_threshold = cached_ml['threshold']
        
        # Render chart med avanserte innstillinger
        render_modern_chart(
            df=df_view,
            indicators=final_ind,
            oscillators=final_osc,
            signals=signal_data,
            signal_keys=selected_signals,
            theme=ctrl['theme'],
            chart_height=ctrl.get('chart_height', 450),
            support_resistance=sr_data,
            exit_signals=exit_data,
            insider_trades=insider_data,
            chart_settings=ctrl,  # Send alle avanserte innstillinger
            ml_score_data=ml_chart_data,
            ml_threshold=ml_chart_threshold,
        )
        
        st.caption(f"Oppsett: {preset['name']} - {preset['description']}")
        
        # === ML RESULTAT (visning under chart) ===
        if ML_AVAILABLE:
            cached_ml_vis = st.session_state.get(f"ml_ta_{ticker}")
            if cached_ml_vis:
                _vis_ml_resultat(
                    ticker, cached_ml_vis['score'], cached_ml_vis['confidence'],
                    cached_ml_vis['threshold'], cached_ml_vis['backtest'],
                    cached_ml_vis.get('hist_probs', pd.Series()), df_full
                )
    
    else:
        # === PLOTLY FALLBACK ===
        st.info("Installer streamlit-lightweight-charts for bedre charts")
        
        horisont = st.selectbox("Periode", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3)
        offset_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}
        
        # Tilleggsfunksjoner for fallback chart
        show_insider = st.checkbox("💼 Insider-handler", value=False, help="Meldepliktige handler")
        
        end_date = df_full.index.max()
        start_date = end_date - pd.DateOffset(days=offset_map[horisont])
        df_view = df_full[df_full.index >= start_date]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        
        fig.add_trace(go.Candlestick(
            x=df_view.index, open=df_view['Open'], high=df_view['High'], 
            low=df_view['Low'], close=df_view['Close'], name="Pris"
        ), row=1, col=1)
        
        if 'SMA_50' in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_50'], name="SMA 50", line=dict(color='#45B7D1')), row=1, col=1)
        if 'SMA_200' in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_200'], name="SMA 200", line=dict(color='#DDA0DD')), row=1, col=1)
        
        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_view['Close'], df_view['Open'])]
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], marker_color=colors, opacity=0.5, name="Volum"), row=2, col=1)
        
        # Insider-handler markører på Plotly chart
        if show_insider:
            try:
                import insider_monitor as _im
                _ins_handler = _im.hent_innsidehandler(dager=90)
                _ticker_handler = [h for h in _ins_handler if h['ticker'] == ticker]
                for h in _ticker_handler:
                    try:
                        h_date = pd.Timestamp(h['dato'])
                        if h_date < start_date:
                            continue
                        idx = df_view.index.get_indexer([h_date], method='nearest')[0]
                        if idx < 0 or idx >= len(df_view):
                            continue
                        h_date = df_view.index[idx]
                        h_price = df_view.iloc[idx]['Low'] * 0.97 if h['type'] == 'kjøp' else df_view.iloc[idx]['High'] * 1.03
                        marker_sym = 'triangle-up' if h['type'] == 'kjøp' else 'triangle-down' if h['type'] == 'salg' else 'diamond'
                        marker_col = '#00c853' if h['type'] == 'kjøp' else '#ff5252' if h['type'] == 'salg' else '#ffa726'
                        type_label = 'Kjøp' if h['type'] == 'kjøp' else 'Salg' if h['type'] == 'salg' else '?'
                        fig.add_trace(go.Scatter(
                            x=[h_date], y=[h_price], mode='markers+text',
                            marker=dict(symbol=marker_sym, size=14, color=marker_col, line=dict(width=1, color='white')),
                            text=[f"{type_label}"], textposition='bottom center' if h['type'] == 'kjøp' else 'top center',
                            textfont=dict(size=9, color=marker_col),
                            name=f"Insider {type_label}", showlegend=False,
                            hovertext=h.get('tittel', '')[:80], hoverinfo='text',
                        ), row=1, col=1)
                    except Exception:
                        continue
            except Exception:
                pass
        
        fig.update_layout(height=550, template="plotly_dark", showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")
    
    # === KONSOLIDERT SIGNALOVERSIKT ===
    st.markdown("---")
    st.markdown("### 📊 Konsolidert Signaloversikt")
    
    all_keys = ["Kort_Sikt_RSI", "Momentum_Burst", "Golden_Cross", 
                "Ichimoku_Breakout", "Wyckoff_Spring", "Bull_Race_Prep", "VCP_Pattern", "Pocket_Pivot"]
    all_names = {
        "Kort_Sikt_RSI": "RSI Dip",
        "Momentum_Burst": "Momentum",
        "Golden_Cross": "Golden Cross",
        "Ichimoku_Breakout": "Ichimoku",
        "Wyckoff_Spring": "Wyckoff",
        "Bull_Race_Prep": "BB Squeeze",
        "VCP_Pattern": "VCP",
        "Pocket_Pivot": "Pocket Pivot"
    }
    
    signaler = logic.sjekk_strategier(df_full)
    signal_hist = logic.hent_signaler_for_chart(df_full, signaler, all_keys)
    
    # Hent mønstre hvis tilgjengelig
    mønstre_info = {}
    try:
        import pattern_logic
        mønstre = pattern_logic.skann_alle_mønstre(df_full, kun_siste_n_dager=120)
        if mønstre:
            siste_mønster = mønstre[0]  # Nyeste mønster
            mønstre_info = {
                "mønster": siste_mønster['mønster'],
                "retning": siste_mønster['retning'],
                "styrke": siste_mønster['styrke'],
                "dato": siste_mønster['dato']
            }
    except ImportError:
        pass
    
    rows = []
    for key in all_keys:
        if key not in signal_hist or signal_hist[key].empty:
            rows.append({
                "Strategi": all_names.get(key, key), 
                "Siste Signal": "Ingen", 
                "Kvalitet": "-", 
                "Utv siden": "-",
                "Peak 20d": "-"
            })
        else:
            sig_df = signal_hist[key]
            latest = sig_df.iloc[-1]
            
            # Beregn peak utvikling 20 dager etter signal
            signal_dato = sig_df.index[-1]
            end_date = min(signal_dato + pd.Timedelta(days=20), df_full.index[-1])
            future_df = df_full.loc[signal_dato:end_date]
            
            if len(future_df) > 1:
                signal_pris = df_full.loc[signal_dato, 'Close']
                peak_pris = future_df['High'].max()
                peak_utv = ((peak_pris / signal_pris) - 1) * 100
                peak_str = f"+{peak_utv:.1f}%" if peak_utv > 0 else f"{peak_utv:.1f}%"
            else:
                peak_str = "N/A"
            
            rows.append({
                "Strategi": all_names.get(key, key),
                "Siste Signal": sig_df.index[-1].strftime('%d.%m.%y'),
                "Kvalitet": latest.get('kvalitet_klasse', '-'),
                "Utv siden": f"{latest.get('utvikling_pst', 0):+.1f}%",
                "Peak 20d": peak_str
            })
    
    # Legg til mønster-rad hvis tilgjengelig
    if mønstre_info:
        retning_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(mønstre_info['retning'], "⚪")
        rows.append({
            "Strategi": f"{retning_emoji} {mønstre_info['mønster']}",
            "Siste Signal": mønstre_info['dato'].strftime('%d.%m.%y'),
            "Kvalitet": f"{mønstre_info['styrke']}/100",
            "Utv siden": "-",
            "Peak 20d": "-"
        })

    # Sorter etter dager siden signal (nyeste først)
    df_rows = pd.DataFrame(rows)
    def parse_signal_date(val):
        try:
            return pd.to_datetime(val, format='%d.%m.%y')
        except Exception:
            return pd.Timestamp.min
    
    if not df_rows.empty and 'Siste Signal' in df_rows.columns:
        df_rows['Signal_dt'] = df_rows['Siste Signal'].apply(parse_signal_date)
        df_rows = df_rows.sort_values('Signal_dt', ascending=False).drop(columns=['Signal_dt'])
    
    # Vis tabell med column config
    st.dataframe(
        df_rows, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Strategi": st.column_config.TextColumn("Strategi/Mønster", width="medium"),
            "Siste Signal": st.column_config.TextColumn("Siste Signal", width="small"),
            "Kvalitet": st.column_config.TextColumn("Kvalitet", width="small"),
            "Utv siden": st.column_config.TextColumn("Utv siden", width="small", help="Utvikling siden signal"),
            "Peak 20d": st.column_config.TextColumn("Peak 20d", width="small", help="Høyeste utvikling 20 dager etter signal")
        }
    )
    
    # === SMART MONEY FLOW ===
    with st.expander("Smart Money Flow — Institusjonell aktivitet", expanded=False):
        try:
            import smart_money
            
            sm_col1, sm_col2 = st.columns([1, 1])
            with sm_col1:
                bruk_intradag_smi = st.toggle("Bruk intradag-data (presist, men tregt)", value=False,
                                              help="Henter 1h-data fra yfinance. Mer presist, men tar ~5 sek.")
            
            with st.spinner("Beregner Smart Money Flow..."):
                sm_result = smart_money.analyser_smart_money(
                    ticker, df_daglig=df_full, bruk_intradag=bruk_intradag_smi
                )
            
            div = sm_result['divergens']
            
            # Status-visning
            sm_color = {"bullish": "#00c853", "bearish": "#ff5252",
                        "confirming": "#667eea", "neutral": "#888"}[div['type']]
            st.markdown(f"""
            <div style="background: rgba({','.join(str(int(sm_color[i:i+2],16)) for i in (1,3,5))}, 0.15);
                        border-left: 4px solid {sm_color}; border-radius: 8px;
                        padding: 14px; margin-bottom: 12px;">
                <span style="font-size: 22px;">{div['emoji']}</span>
                <b style="font-size: 16px; color: {sm_color};">{div['type'].title()}</b>
                <span style="opacity: 0.8; margin-left: 8px;">{div['beskrivelse']}</span>
                <br><span style="font-size: 12px; opacity: 0.6;">
                    Kilde: {'Intradag (1h)' if sm_result['kilde'] == 'intradag' else 'Daglig proxy (Chaikin/OBV)'}
                    &nbsp;|&nbsp; Pris-trend: {div['pris_trend']}
                    &nbsp;|&nbsp; SMI-trend: {div['smi_trend']}
                    &nbsp;|&nbsp; Score-effekt: {div['score_justering']:+d}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Detalj-metrikker
            det = sm_result.get('detaljer', {})
            if sm_result['kilde'] == 'intradag':
                d_cols = st.columns(4)
                d_cols[0].metric("SMI-verdi", f"{sm_result['smi_verdi']:.4f}")
                d_cols[1].metric("SMA 10", f"{det.get('smi_sma10', 0):.4f}" if det.get('smi_sma10') else "—")
                d_cols[2].metric("Tidlig (snitt)", f"{det.get('early_return_snitt', 0):.4f}")
                d_cols[3].metric("Sent (snitt)", f"{det.get('late_return_snitt', 0):.4f}")
                
                # Mini-chart av SMI
                smi_df = sm_result.get('smi_df')
                if smi_df is not None and len(smi_df) > 5:
                    fig_smi = go.Figure()
                    fig_smi.add_trace(go.Scatter(
                        x=smi_df.index, y=smi_df['smi'],
                        name='SMI', line=dict(color='#667eea', width=2)
                    ))
                    if 'smi_sma10' in smi_df.columns:
                        fig_smi.add_trace(go.Scatter(
                            x=smi_df.index, y=smi_df['smi_sma10'],
                            name='SMA 10', line=dict(color='#ffa726', width=1, dash='dash')
                        ))
                    fig_smi.update_layout(
                        height=200, template="plotly_dark",
                        margin=dict(l=0, r=0, t=20, b=20),
                        legend=dict(orientation="h", yanchor="top", y=1.12),
                        yaxis_title="SMI"
                    )
                    st.plotly_chart(fig_smi, use_container_width=True)
            else:
                d_cols = st.columns(3)
                d_cols[0].metric("CMF (20d)", f"{det.get('cmf', 0):.3f}" if det.get('cmf') else "—")
                d_cols[1].metric("OBV-trend", det.get('obv_trend', '—'))
                d_cols[2].metric("SM-proxy", f"{sm_result['smi_verdi']:.3f}")
            
            st.caption("Smart Money Index: Siste-time-avkastning minus første-time-avkastning (kumulativ). "
                       "Daglig proxy: Chaikin Money Flow (60%) + OBV-divergens (40%).")
            
        except ImportError:
            st.info("Smart Money-modul ikke tilgjengelig.")
        except Exception as e:
            st.warning(f"Smart Money-analyse feilet: {e}")
    
    # === INSIDER-HANDLER ===
    with st.expander("Insider-handler — Meldepliktige handler", expanded=False):
        try:
            import insider_monitor
            
            with st.spinner("Henter insider-data fra Newsweb..."):
                handler = insider_monitor.hent_innsidehandler(dager=90)
            
            if not handler:
                st.info("Ingen insider-data tilgjengelig akkurat nå.")
            else:
                # Beregn insider-score for denne tickeren
                ins_result = insider_monitor.beregn_insider_score(ticker, handler)
                score = ins_result["score"]
                
                # Fargekoding basert på score
                if score > 20:
                    ins_color = "#00c853"
                    ins_status = "KJØP"
                    ins_label = "Netto kjøp"
                elif score < -20:
                    ins_color = "#ff5252"
                    ins_status = "SALG"
                    ins_label = "Netto salg"
                else:
                    ins_color = "#888888"
                    ins_status = "NØYTRAL"
                    ins_label = "Nøytral"
                
                # Sikker hex til RGB konvertering
                def hex_til_rgb(hex_color):
                    if not hex_color or len(hex_color) != 7 or not hex_color.startswith('#'):
                        return "136, 136, 136"  # Fallback grå
                    try:
                        r = int(hex_color[1:3], 16)
                        g = int(hex_color[3:5], 16)
                        b = int(hex_color[5:7], 16)
                        return f"{r}, {g}, {b}"
                    except (ValueError, IndexError):
                        return "136, 136, 136"  # Fallback grå
                
                rgb_color = hex_til_rgb(ins_color)
                
                st.markdown(f"""
                <div style="background: rgba({rgb_color}, 0.15);
                            border-left: 4px solid {ins_color}; border-radius: 8px;
                            padding: 14px; margin-bottom: 12px;">
                    <b style="font-size: 16px; color: {ins_color};">[{ins_status}] {ins_label}</b>
                    <span style="opacity: 0.8; margin-left: 8px;">Insider-score: {score:+.0f}</span>
                    <br><span style="font-size: 12px; opacity: 0.6;">
                        Basert på meldepliktige handler siste 90 dager (kilde: Oslo Børs Newsweb)
                        — {ins_result['antall_kjøp']} kjøp, {ins_result['antall_salg']} salg
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Vis relevante handler for denne tickeren
                ticker_clean = ticker.replace(".OL", "_OL")
                sammendrag = insider_monitor.hent_insider_sammendrag(handler)
                
                if sammendrag is not None and not sammendrag.empty:
                    # Filtrer til denne tickeren
                    ticker_handler = sammendrag[
                        sammendrag['Ticker'].str.upper() == ticker_clean.split("_")[0].upper()
                    ]
                    
                    if not ticker_handler.empty:
                        st.markdown("**Handler for denne aksjen:**")
                        st.dataframe(ticker_handler, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"Ingen registrerte insider-handler for {ticker.replace('.OL', '')} siste 90 dager.")
                    
                    # Vis topp-handler på tvers av alle aksjer (komprimert)
                    with st.expander("Alle insider-handler (siste 90 dager)", expanded=False):
                        st.dataframe(sammendrag.head(30), use_container_width=True, hide_index=True)
                else:
                    st.info("Kunne ikke laste insider-sammendrag.")
                
                st.caption("Beta-modul. Data hentes fra Oslo Børs Newsweb (meldepliktige handler for primærinnsidere). "
                           "Score-modellen vekter handelstype, rolle og aktualitet.")
        
        except ImportError:
            st.info("Insider-modul ikke tilgjengelig.")
        except Exception as e:
            st.warning(f"Insider-analyse feilet: {e}")
    
    # === POSISJONSKALKULATOR ===
    with st.expander("Posisjonskalkulator"):
        c1, c2, c3, c4 = st.columns(4)
        kap = c1.number_input("Kapital", value=config.DEFAULT_KAPITAL)
        ris = c2.number_input("Risiko %", value=config.DEFAULT_RISIKO_PROSENT)
        ent = c3.number_input("Inngang", value=float(df_full['Close'].iloc[-1]))
        stp = c4.number_input("Stop", value=float(df_full['Close'].iloc[-1]*0.95))
        
        if ent > stp:
            res = logic.beregn_risk_reward(ent, stp, kap, ris)
            if res:
                st.success(f"**{res['antall']} aksjer** @ {ent:.2f} = {res['total_investering']:,.0f} kr")
                st.caption(f"Risiko: {res['risiko_kr']:,.0f} kr | Target (2R): {res['target_2r']:.2f}")
    
    # === KELLY CRITERION ===
    with st.expander("Kelly Criterion - Hvor mye bør du investere?"):
        st.markdown("""
        ### Hva er dette?
        Kelly forteller deg **hvor stor andel av kapitalen** du bør satse per trade, 
        basert på hvor god du er til å velge aksjer.
        """)
        
        st.markdown("---")
        
        # Enkel input
        st.markdown("#### Din trading-statistikk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kelly_vinnrate = st.slider(
                "Hvor ofte vinner du?", 
                min_value=30, max_value=80, value=55, 
                format="%d%%",
                help="Av 10 handler, hvor mange er vinnere?"
            )
        
        with col2:
            kelly_gevinst = st.number_input(
                "Typisk gevinst (kr)", 
                min_value=500, value=3000, step=500,
                help="Når du vinner, hvor mye tjener du vanligvis?"
            )
        
        with col3:
            kelly_tap = st.number_input(
                "Typisk tap (kr)", 
                min_value=500, value=1500, step=500,
                help="Når du taper, hvor mye taper du vanligvis?"
            )
        
        kelly = logic.beregn_kelly_criterion(kelly_vinnrate/100, kelly_gevinst, kelly_tap)
        
        if kelly:
            st.markdown("---")
            
            if kelly['har_edge']:
                # Hovedresultat - stort og tydelig
                anbefalt_pct = kelly['kelly_half']
                anbefalt_kr = kap * (anbefalt_pct / 100)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); 
                            padding: 24px; border-radius: 16px; text-align: center; margin: 16px 0;">
                    <div style="font-size: 14px; color: rgba(255,255,255,0.7); margin-bottom: 8px;">
                        ANBEFALT INVESTERING PER TRADE
                    </div>
                    <div style="font-size: 42px; font-weight: 700; color: #4ade80;">
                        {anbefalt_kr:,.0f} kr
                    </div>
                    <div style="font-size: 16px; color: rgba(255,255,255,0.8); margin-top: 8px;">
                        {anbefalt_pct:.1f}% av {kap:,.0f} kr kapital
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Enkel forklaring
                st.success(f"**Du har en edge.** {kelly['anbefaling']}")
                
                # Detaljer i en liten boks
                with st.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Aggressiv", f"{kelly['kelly_full']}%", help="Full Kelly - høy risiko")
                    c2.metric("Anbefalt", f"{kelly['kelly_half']}%", help="Half Kelly - balansert")
                    c3.metric("Forsiktig", f"{kelly['kelly_quarter']}%", help="Quarter Kelly - lav risiko")
                
                # Praktisk eksempel
                aksjepris = float(df_full['Close'].iloc[-1])
                antall_aksjer = int(anbefalt_kr / aksjepris)
                
                st.markdown(f"""
                ---
                **Praktisk eksempel med {ticker}:**  
                Aksjepris: {aksjepris:.2f} kr → Kjøp **{antall_aksjer} aksjer** (= {antall_aksjer * aksjepris:,.0f} kr)
                """)
                
            else:
                # Ingen edge
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 100%); 
                            padding: 24px; border-radius: 16px; text-align: center; margin: 16px 0;">
                    <div style="font-size: 42px; font-weight: 700; color: #f87171;">
                        Ingen edge
                    </div>
                    <div style="font-size: 16px; color: rgba(255,255,255,0.8); margin-top: 8px;">
                        Med denne statistikken taper du penger over tid
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _vis_ai_teknisk_analyse(ticker: str, df_full: pd.DataFrame):
    """Vis algoritmisk teknisk analyse øverst på siden."""
    
    with st.expander("📊 Teknisk Analyse", expanded=False):
        if len(df_full) < 50:
            st.warning("Ikke nok data for analyse")
            return
        
        # Generer øyeblikkelig algoritmisk analyse
        analyse_resultat = _generer_algoritmisk_analyse(ticker, df_full)
        
        # Vis resultat
        _vis_analyse_resultat(
            analyse_resultat['anbefaling'], 
            analyse_resultat['tekst'], 
            ticker,
            analyse_resultat['score']
        )


def _generer_algoritmisk_analyse(ticker: str, df_full: pd.DataFrame) -> dict:
    """Generer regelbasert teknisk analyse inkludert signaler og mønstre."""
    
    if len(df_full) < 50:
        return {"anbefaling": "HOLD", "tekst": "Ikke nok data for analyse", "score": 50}
    
    siste = df_full.iloc[-1]
    signaler = []
    score = 50  # Start nøytral
    
    # === EKSISTERENDE STRATEGISIGNALER ===
    try:
        strategier = logic.sjekk_strategier(df_full)
        signal_keys = ["Kort_Sikt_RSI", "Momentum_Burst", "Golden_Cross", 
                      "Ichimoku_Breakout", "Wyckoff_Spring", "Bull_Race_Prep", 
                      "VCP_Pattern", "Pocket_Pivot"]
        
        aktive_signaler = []
        for key in signal_keys:
            if strategier.get(key, {}).get('signal', False):
                strategy_navn = {
                    "Kort_Sikt_RSI": "RSI Dip",
                    "Momentum_Burst": "Momentum Burst", 
                    "Golden_Cross": "Golden Cross",
                    "Ichimoku_Breakout": "Ichimoku Breakout",
                    "Wyckoff_Spring": "Wyckoff Spring",
                    "Bull_Race_Prep": "Bollinger Squeeze",
                    "VCP_Pattern": "VCP Mønster",
                    "Pocket_Pivot": "Pocket Pivot"
                }.get(key, key)
                
                kvalitet = strategier[key].get('kvalitet_klasse', 'Medium')
                if kvalitet == 'Høy':
                    signaler.append(f"🟢 **{strategy_navn}** (Høy kvalitet)")
                    score += 15
                elif kvalitet == 'Medium':
                    signaler.append(f"🟡 **{strategy_navn}** (Medium kvalitet)")
                    score += 8
                else:
                    signaler.append(f"🟠 **{strategy_navn}** (Lav kvalitet)")
                    score += 5
                aktive_signaler.append(strategy_navn)
        
        if aktive_signaler:
            signaler.insert(0, f"🎯 **Aktive strategier:** {', '.join(aktive_signaler[:3])}{'...' if len(aktive_signaler) > 3 else ''}")
            
    except Exception as e:
        logger.debug(f"Strategisjekk feilet: {e}")
    
    # === MØNSTERGJENKJENNING ===
    try:
        import pattern_logic
        mønstre = pattern_logic.skann_alle_mønstre(df_full, kun_siste_n_dager=60)
        
        if mønstre:
            for mønster in mønstre[:2]:  # Maks 2 nyeste mønstre
                styrke = mønster.get('styrke', 50)
                retning = mønster.get('retning', 'neutral')
                navn = mønster['mønster']
                
                if retning == 'bullish':
                    emoji = "📈"
                    if styrke >= 80:
                        signaler.append(f"{emoji} **{navn}** - Meget sterkt bullish mønster ({styrke}/100)")
                        score += 18
                    elif styrke >= 60:
                        signaler.append(f"{emoji} **{navn}** - Bullish mønster ({styrke}/100)")
                        score += 12
                    else:
                        signaler.append(f"{emoji} **{navn}** - Svakt bullish ({styrke}/100)")
                        score += 6
                        
                elif retning == 'bearish':
                    emoji = "📉"
                    if styrke >= 80:
                        signaler.append(f"{emoji} **{navn}** - Meget sterkt bearish mønster ({styrke}/100)")
                        score -= 18
                    elif styrke >= 60:
                        signaler.append(f"{emoji} **{navn}** - Bearish mønster ({styrke}/100)")
                        score -= 12
                    else:
                        signaler.append(f"{emoji} **{navn}** - Svakt bearish ({styrke}/100)")
                        score -= 6
                else:
                    signaler.append(f"⚪ **{navn}** - Nøytralt mønster ({styrke}/100)")
                    
    except Exception as e:
        logger.debug(f"Mønstersjekk feilet: {e}")
    
    # === TEKNISKE INDIKATORER ===
    # RSI analyse
    rsi = siste.get('RSI')
    if pd.notna(rsi):
        if rsi < 30:
            signaler.append("🟢 RSI oversolgt (kjøpsmulighet)")
            score += 12
        elif rsi > 70:
            signaler.append("🔴 RSI overkjøpt (salgssignal)")
            score -= 12
        elif 40 <= rsi <= 60:
            signaler.append("🟡 RSI nøytral")
    
    # MACD trend
    macd = siste.get('MACD', 0)
    macd_signal = siste.get('MACD_Signal', 0)
    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal:
            if macd > 0:
                signaler.append("🟢 MACD bullish over nullinje")
                score += 8
            else:
                signaler.append("🟡 MACD positiv, men under null")
                score += 4
        else:
            signaler.append("🔴 MACD bearish krysning")
            score -= 8
    
    # Moving Average posisjon og trend
    pris = siste['Close']
    sma20 = siste.get('SMA_20')
    sma50 = siste.get('SMA_50')
    
    if pd.notna(sma20):
        sma20_avvik = ((pris / sma20) - 1) * 100
        if sma20_avvik > 5:
            signaler.append(f"🟢 Pris {sma20_avvik:.1f}% over SMA20")
            score += 6
        elif sma20_avvik < -5:
            signaler.append(f"🔴 Pris {abs(sma20_avvik):.1f}% under SMA20")
            score -= 6
    
    # Trend (SMA20 vs SMA50)
    if pd.notna(sma20) and pd.notna(sma50):
        if sma20 > sma50:
            signaler.append("📈 Opptrend (SMA20 > SMA50)")
            score += 6
        else:
            signaler.append("📉 Nedtrend (SMA20 < SMA50)")
            score -= 6
    
    # Volum og momentum
    try:
        siste_vol = siste['Volume']
        avg_vol = df_full['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = siste_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio > 2:
            signaler.append("🔥 Ekstremt høyt volum")
            score += 4
        elif vol_ratio > 1.5:
            signaler.append("📊 Høyt volum")
            score += 2
        elif vol_ratio < 0.5:
            signaler.append("💤 Lavt volum")
            score -= 2
    except Exception:
        pass
    
    # Momentum (5-dagers endring)
    try:
        if len(df_full) >= 5:
            momentum_5d = ((pris / df_full['Close'].iloc[-6]) - 1) * 100
            if momentum_5d > 10:
                signaler.append(f"🚀 Sterkt momentum (+{momentum_5d:.1f}%)")
                score += 10
            elif momentum_5d > 3:
                signaler.append(f"📈 Positivt momentum (+{momentum_5d:.1f}%)")
                score += 4
            elif momentum_5d < -10:
                signaler.append(f"💥 Kraftig fall ({momentum_5d:.1f}%)")
                score -= 10
            elif momentum_5d < -3:
                signaler.append(f"📉 Negativ momentum ({momentum_5d:.1f}%)")
                score -= 4
    except Exception:
        pass
    
    # Insider aktivitet
    try:
        import insider_monitor
        handler = insider_monitor.hent_innsidehandler(dager=60)
        ins_result = insider_monitor.beregn_insider_score(ticker, handler)
        ins_score = ins_result.get('score', 0)
        
        if ins_score > 30:
            signaler.append("🏛️ Sterke insider-kjøp")
            score += 8
        elif ins_score > 10:
            signaler.append("🏛️ Insider-kjøp aktivitet")
            score += 4
        elif ins_score < -20:
            signaler.append("🏛️ Insider-salg aktivitet")
            score -= 6
    except Exception:
        pass
    
    # === STØTTE/MOTSTAND ===
    try:
        # Finn støtte/motstand nivåer
        recent_high = df_full['High'].iloc[-20:].max()
        recent_low = df_full['Low'].iloc[-20:].min()
        
        støtte = f"{recent_low:.2f}"
        motstand = f"{recent_high:.2f}"
        
        # Avstand til nivåer
        til_motstand = ((recent_high / pris) - 1) * 100
        til_støtte = ((pris / recent_low) - 1) * 100
        
    except Exception:
        støtte, motstand, til_motstand, til_støtte = "N/A", "N/A", 0, 0
    
    # === GENERER ANBEFALING ===
    # Juster grensene basert på signaler og mønstre
    if score >= 75:
        anbefaling = "KJØP"
        konklusion = "Flere positive tekniske signaler og/eller mønstre tyder på kjøpsmulighet."
    elif score >= 60:
        anbefaling = "SVAK KJØP" 
        konklusion = "Positive signaler, men vær forsiktig med posisjonsstørrelse."
    elif score <= 25:
        anbefaling = "SELG"
        konklusion = "Negative signaler og/eller mønstre tyder på press nedover."
    elif score <= 40:
        anbefaling = "SVAK SELG"
        konklusion = "Svake signaler, vurder redusert posisjon."
    else:
        anbefaling = "HOLD"
        konklusion = "Blandet teknisk bilde med motstridende signaler."
    
    # === BYGG TEKST ===
    tekst_deler = []
    
    # Hovedsignaler (prioriter strategier og mønstre først)
    if signaler:
        tekst_deler.append("**Tekniske signaler:**")
        
        # Vis strategier og mønstre først, deretter tekniske indikatorer
        strategi_signaler = [s for s in signaler if "**" in s]  # Strategier og mønstre har **
        teknisk_signaler = [s for s in signaler if "**" not in s]  # Vanlige tekniske
        
        # Maks 6 signaler totalt for å holde det kort
        alle_signaler = (strategi_signaler + teknisk_signaler)[:6]
        
        for signal in alle_signaler:
            tekst_deler.append(f"• {signal}")
        tekst_deler.append("")
    
    # Støtte/motstand
    if støtte != "N/A" and motstand != "N/A":
        tekst_deler.append(f"**Nøkkelnivåer:** Støtte {støtte} ({til_støtte:.1f}% ned) • Motstand {motstand} ({til_motstand:.1f}% opp)")
        tekst_deler.append("")
    
    # Konklusion
    tekst_deler.append(f"**Vurdering:** {konklusion}")
    if len(signaler) > 6:
        tekst_deler.append(f"*(Viser 6 av {len(signaler)} signaler - Score: {score}/100)*")
    else:
        tekst_deler.append(f"*(Teknisk score: {score}/100)*")
    
    return {
        "anbefaling": anbefaling,
        "tekst": "\n".join(tekst_deler),
        "score": score
    }


def _vis_analyse_resultat(anbefaling: str, tekst: str, ticker: str, score: int):
    """Vis algoritmisk analyse resultat."""
    
    # Fargekoding basert på anbefaling
    if "KJØP" in anbefaling:
        if "SVAK" in anbefaling:
            anb_color = "#4caf50"  # Lys grønn
        else:
            anb_color = "#00c853"  # Kraftig grønn
    elif "SELG" in anbefaling:
        if "SVAK" in anbefaling:
            anb_color = "#ff7043"  # Lys rød
        else:
            anb_color = "#ff5252"  # Kraftig rød
    else:
        anb_color = "#ffa726"  # Orange for HOLD
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 20px; margin: 16px 0;
                border-left: 4px solid {anb_color};">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="background: {anb_color}; color: #000; padding: 4px 12px;
                             border-radius: 8px; font-weight: 700; font-size: 14px;">
                    {anbefaling}
                </span>
                <span style="font-size: 18px; font-weight: 600; color: #fff;">
                    {ticker}
                </span>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 6px;">
                <span style="font-size: 12px; color: #ccc;">Score: {score}/100</span>
            </div>
        </div>
        <div style="color: #e0e0e0; line-height: 1.6; font-size: 14px; white-space: pre-line;">
            {tekst}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _samle_teknisk_analyse_data(ticker: str, df_full: pd.DataFrame) -> dict:
    """Samle all relevant data for AI teknisk analyse."""
    
    # Siste data
    siste = df_full.iloc[-1]
    
    # Sikker formatering av numeriske verdier
    def safe_format(value, format_spec, fallback='N/A'):
        try:
            if pd.isna(value) or value is None:
                return fallback
            return f"{float(value):{format_spec}}"
        except (ValueError, TypeError):
            return fallback
    
    # Tekniske indikatorer
    rsi = safe_format(siste.get('RSI'), '.1f')
    macd = safe_format(siste.get('MACD'), '.3f')
    macd_signal = safe_format(siste.get('MACD_Signal'), '.3f')
    sma20 = safe_format(siste.get('SMA_20'), '.2f')
    sma50 = safe_format(siste.get('SMA_50'), '.2f')
    bb_upper = safe_format(siste.get('BB_Upper'), '.2f')
    bb_lower = safe_format(siste.get('BB_Lower'), '.2f')
    
    try:
        vol_20d = f"{df_full['Volume'].rolling(20).mean().iloc[-1]:,.0f}"
    except Exception:
        vol_20d = "N/A"
    
    try:
        volatilitet = f"{(df_full['Close'].pct_change().rolling(20).std() * 100).iloc[-1]:.1f}"
    except Exception:
        volatilitet = "N/A"
    
    teknisk = f"""
RSI (14): {rsi}
MACD: {macd} | Signal: {macd_signal}
SMA20: {sma20} | SMA50: {sma50}
Bollinger: Øvre {bb_upper} | Nedre {bb_lower}
Volume (20d avg): {vol_20d}
Volatilitet (20d): {volatilitet}%
Siste pris: {siste['Close']:.2f} NOK
"""
    
    # Fundamental data
    fundamental = "Ikke tilgjengelig"
    try:
        import fundamental_data
        fund_data = fundamental_data.hent_fundamental_data(ticker.replace('.OL', ''))
        if fund_data:
            pe = fund_data.get('PE_ratio', 'N/A')
            ev_ebitda = fund_data.get('EV_EBITDA', 'N/A') 
            roe = fund_data.get('ROE', 'N/A')
            debt_equity = fund_data.get('debt_to_equity', 'N/A')
            
            fundamental = f"""
P/E: {pe} | EV/EBITDA: {ev_ebitda}
ROE: {roe}% | Debt/Equity: {debt_equity}
"""
    except Exception:
        pass
    
    # Insider data  
    insider = "Ingen data"
    try:
        import insider_monitor
        handler = insider_monitor.hent_innsidehandler(dager=60)
        ins_result = insider_monitor.beregn_insider_score(ticker, handler)
        insider = f"""
Insider-score: {ins_result.get('score', 0):+.0f}/100
Kjøp: {ins_result.get('antall_kjøp', 0)} | Salg: {ins_result.get('antall_salg', 0)}
Siste handel: {ins_result.get('siste_handel', 'N/A')}
"""
    except Exception:
        pass
    
    # Events (kommende utbytte etc)
    events = "Ingen planlagte events"
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if info.get('exDividendDate'):
            ex_date = info['exDividendDate']
            dividend = info.get('dividendYield', 0) * 100
            events = f"Ex-dividend: {ex_date} ({dividend:.2f}% yield)"
    except Exception:
        pass
    
    return {
        'teknisk': teknisk,
        'fundamental': fundamental, 
        'insider': insider,
        'events': events
    }


def _kjør_ml_beregning(ticker: str, df_full: pd.DataFrame, hist_dager: int = 60):
    """Kjør ML-modellen og lagre resultater i cache (uten visning).

    Args:
        ticker: Ticker-navn
        df_full: Full historisk DataFrame for tickeren
        hist_dager: Hvor mange dager tilbake å beregne historiske sannsynligheter for subchart
    """
    
    cache_key = f"ml_ta_{ticker}_{hist_dager}"
    cached = st.session_state.get(cache_key)
    
    # Bruk cache hvis allerede beregnet i dag OG har gyldige hist_probs
    if cached and cached.get('dato') == df_full.index[-1].strftime('%Y-%m-%d'):
        hp = cached.get('hist_probs', pd.Series())
        if not hp.empty and hp.index.max() >= df_full.index[-1] - pd.Timedelta(days=5):
            return  # Cache er fersk nok
    
    with st.spinner(f"🧠 Trener ML-modell for {ticker}... (kan ta 10-30 sek)"):
        try:
            predictor = beta_ml.EnsembleStockPredictor(
                horisont=10, target_pct=0.04, stop_pct=0.024
            )
            
            df_feat = beta_ml.beregn_avanserte_features(df_full)
            if df_feat.empty:
                st.warning("Ikke nok data for ML-analyse")
                return
            
            success = predictor.fit(df_feat)
            if not success:
                st.warning("Modelltrening feilet — for lite data eller for mange manglende features")
                return
            
            ml_score = predictor.predict_proba(df_feat)
            threshold = predictor.signal_threshold
            
            _, ci_low, ci_high = beta_ml.beregn_konfidensintervall(predictor, df_feat, n_bootstrap=15)
            confidence = (ci_low, ci_high)
            
            backtest = predictor.backtest(df_feat, lookback_signals=15)
            hist_probs = predictor.predict_historical(df_full, dager=hist_dager)
            
            # Cache resultater
            st.session_state[cache_key] = {
                'score': ml_score,
                'confidence': confidence,
                'backtest': backtest,
                'threshold': threshold,
                'hist_probs': hist_probs,
                'dato': df_full.index[-1].strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            st.error(f"ML-analyse feilet: {str(e)[:200]}")
            logger.error(f"ML-analyse feilet for {ticker}: {e}", exc_info=True)


def _vis_ml_resultat(ticker, ml_score, confidence, threshold, backtest, hist_probs, df_full):
    """Vis ML-resultater under chartet."""
    
    if ml_score >= 70:
        score_color = "#00c853"
        signal_txt = "KJØPSSIGNAL"
    elif ml_score >= threshold:
        score_color = "#66bb6a"
        signal_txt = "SVAKT KJØP"
    elif ml_score <= 30:
        score_color = "#ff5252"
        signal_txt = "UNNGÅ"
    else:
        score_color = "#ffa726"
        signal_txt = "NØYTRAL"
    
    is_signal = ml_score >= threshold
    
    siste_pris = df_full['Close'].iloc[-1]
    target_pris = siste_pris * 1.04
    stop_pris = siste_pris * 0.976
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
                border-radius: 12px; padding: 20px; margin: 16px 0;
                border: 1px solid {'#00c85340' if is_signal else '#ffffff15'};">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: #ccc;">🧠 ML Analyse</span>
                <span style="font-size: 11px; color: #666;">XGBoost + Random Forest + Gradient Boosting</span>
            </div>
            <span style="background: {score_color}20; color: {score_color}; padding: 4px 12px;
                         border-radius: 8px; font-weight: 700; font-size: 13px; border: 1px solid {score_color}40;">
                {signal_txt}
            </span>
        </div>
        <div style="display: flex; gap: 24px; align-items: flex-end;">
            <div style="text-align: center;">
                <div style="font-size: 42px; font-weight: 700; color: {score_color};">{ml_score:.0f}</div>
                <div style="font-size: 11px; color: #888;">Score / 100</div>
                <div style="font-size: 10px; color: #666;">Konfidens: {confidence[0]:.0f}–{confidence[1]:.0f}</div>
            </div>
            <div style="flex: 1; display: flex; gap: 16px;">
                <div style="background: #ffffff08; padding: 12px; border-radius: 8px; flex: 1; text-align: center;">
                    <div style="font-size: 10px; color: #888; text-transform: uppercase;">Target (+4%)</div>
                    <div style="font-size: 16px; font-weight: 600; color: #00c853;">{target_pris:.2f}</div>
                </div>
                <div style="background: #ffffff08; padding: 12px; border-radius: 8px; flex: 1; text-align: center;">
                    <div style="font-size: 10px; color: #888; text-transform: uppercase;">Stop (-2.4%)</div>
                    <div style="font-size: 16px; font-weight: 600; color: #ff5252;">{stop_pris:.2f}</div>
                </div>
                <div style="background: #ffffff08; padding: 12px; border-radius: 8px; flex: 1; text-align: center;">
                    <div style="font-size: 10px; color: #888; text-transform: uppercase;">Terskel</div>
                    <div style="font-size: 16px; font-weight: 600; color: #ccc;">{threshold:.0f}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Backtest-resultater
    if backtest and backtest.get('total_signals', 0) > 0:
        hits = backtest['hits']
        total = backtest['total_signals']
        hit_rate = backtest['hit_rate']
        avg_ret = backtest['avg_return']
        
        bt_color = "#00c853" if hit_rate >= 55 else "#ffa726" if hit_rate >= 40 else "#ff5252"
        
        st.markdown(f"""
        <div style="background: #ffffff06; border-radius: 8px; padding: 14px; margin-bottom: 12px;
                    border-left: 3px solid {bt_color};">
            <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                <span style="font-size: 12px; color: #aaa;">📊 Backtest ({total} signaler):</span>
                <span style="color: {bt_color}; font-weight: 600;">{hit_rate:.0f}% treffsikkerhet</span>
                <span style="color: #888;">({hits}/{total} treff)</span>
                <span style="color: {'#00c853' if avg_ret > 0 else '#ff5252'};">
                    Snitt: {avg_ret:+.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detaljert backtest-tabell
        if backtest.get('details'):
            with st.expander("📋 Siste signaler (backtest)", expanded=False):
                bt_rows = []
                for d in backtest['details']:
                    emoji = "✅" if d['treff'] else "❌"
                    bt_rows.append({
                        "": emoji,
                        "Dato": d['dato'],
                        "Score": f"{d['score']:.0f}",
                        "Avkastning": f"{d['avkastning']:+.1f}%",
                    })
                st.dataframe(pd.DataFrame(bt_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("Ingen historiske signaler over terskel for backtest.")
    
    # ML Score subchart vises nå direkte under pris-chartet (via chart_utils)
    if not hist_probs.empty and len(hist_probs) > 5:
        st.caption("📈 ML Score vises som subchart under pris-chartet.")
