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

logger = get_logger(__name__)

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
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        with feature_col1:
            show_sr = st.checkbox("📈 Støtte/Motstand", value=False, help="Automatisk beregnede nivåer")
        
        with feature_col2:
            show_exit = st.checkbox("🚨 Exit-signaler", value=False, help="Historiske exit-signaler (2+)")
        
        with feature_col3:
            show_insider = st.checkbox("💼 Insider-handler", value=False, help="Meldepliktige handler")
            
        with feature_col4:
            show_patterns = st.checkbox("📐 Mønstre", value=False, help="Tekniske prisformasjoner")
        
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
        )
        
        st.caption(f"Oppsett: {preset['name']} - {preset['description']}")
    
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
    
    # === SIGNALHISTORIKK ===
    st.markdown("---")
    st.markdown("### Signalhistorikk")
    
    # === MØNSTERGJENKJENNING (3.4) ===
    try:
        import pattern_logic
        with st.expander("📐 Mønstergjenkjenning — Prisformasjoner", expanded=False):
            mønstre = pattern_logic.skann_alle_mønstre(df_full, kun_siste_n_dager=120)
            if mønstre:
                for m in mønstre:
                    meta = pattern_logic.MØNSTER_METADATA.get(m['mønster'], {})
                    emoji = meta.get('emoji', '⚪')
                    retning_txt = {'bullish': '↑ Bullish', 'bearish': '↓ Bearish', 'neutral': '↔ Nøytral'}.get(m['retning'], '?')
                    retning_col = {'bullish': '#00c853', 'bearish': '#ff5252', 'neutral': '#ffa726'}.get(m['retning'], '#888')
                    st.markdown(
                        f"<div style='background:rgba({','.join(str(int(retning_col[i:i+2],16)) for i in (1,3,5))},0.12);"
                        f"border-left:4px solid {retning_col};border-radius:8px;padding:12px;margin-bottom:8px;'>"
                        f"<b style='font-size:15px;'>{emoji} {m['mønster']}</b>"
                        f"<span style='margin-left:12px;color:{retning_col};font-size:13px;'>{retning_txt}</span>"
                        f"<span style='margin-left:12px;color:rgba(255,255,255,0.5);font-size:12px;'>"
                        f"Styrke: {m['styrke']}/100 · {m['dato'].strftime('%d.%m.%Y')}</span>"
                        f"<br><span style='font-size:12px;color:rgba(255,255,255,0.6);'>{meta.get('kort', '')}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("Ingen prisformasjoner detektert i siste 120 dager.")
            st.caption("Ren prisdata-logikk: H&S, dobbel bunn/topp, kopp-og-hank, trekanter. Ingen ML.")
    except ImportError:
        pass
    
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
    
    rows = []
    for key in all_keys:
        if key not in signal_hist or signal_hist[key].empty:
            rows.append({"Strategi": all_names.get(key, key), "Signal": "-", "Kvalitet": "-", "Utvikling": "-"})
        else:
            sig_df = signal_hist[key]
            latest = sig_df.iloc[-1]
            rows.append({
                "Strategi": all_names.get(key, key),
                "Signal": sig_df.index[-1].strftime('%d.%m.%y'),
                "Kvalitet": latest.get('kvalitet_klasse', '-'),
                "Utvikling": f"{latest.get('utvikling_pst', 0):+.1f}%"
            })

        # Sorter etter dager siden signal (nyeste først)
        df_rows = pd.DataFrame(rows)
        def parse_signal_date(val):
            try:
                return pd.to_datetime(val, format='%d.%m.%y')
            except Exception:
                return pd.Timestamp.min
        if not df_rows.empty and 'Signal' in df_rows.columns:
            df_rows['Signal_dt'] = df_rows['Signal'].apply(parse_signal_date)
            df_rows = df_rows.sort_values('Signal_dt', ascending=False).drop(columns=['Signal_dt'])
        st.dataframe(df_rows, width="stretch", hide_index=True)
    
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
                    ins_color = "#888"
                    ins_status = "NØYTRAL"
                    ins_label = "Nøytral"
                
                st.markdown(f"""
                <div style="background: rgba({','.join(str(int(ins_color[i:i+2],16)) for i in (1,3,5))}, 0.15);
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
                
                st.error("Du trenger enten høyere vinnrate eller bedre gevinst/tap-forhold.")
                st.caption(f"Forventet tap per trade: {kelly['forventet_verdi']:,.0f} kr")
