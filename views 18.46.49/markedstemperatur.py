import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data
from shared_cache import cached_beregn_tekniske_indikatorer

# Import regime_model med try/except
regime_tilgjengelig = False
try:
    import regime_model
    regime_tilgjengelig = True
except Exception:
    regime_tilgjengelig = False


def render():
    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    unike_tickers = st.session_state.get('unike_tickers', [])

    st.title("Markedstemperatur")

    if regime_tilgjengelig:
        with st.spinner("Henter markedsdata..."):
            df_market = data.hent_markedsdata_df(force_refresh=False)
            st.session_state['df_market'] = df_market

        if not df_market.empty:
            # Kompakt header med datainfo
            col_info, col_slider = st.columns([2, 1])
            with col_info:
                try:
                    st.caption(f"Data: {df_market.index.min().strftime('%Y-%m-%d')} → {df_market.index.max().strftime('%Y-%m-%d')}")
                except:
                    st.caption(f"{len(df_market)} datapunkter")

            with col_slider:
                n_regimes = st.select_slider("Regimer", options=[2, 3, 4, 5], value=3)

            with st.spinner("Analyserer markedsregimer..."):
                regime_data = regime_model.full_regime_analyse(df_market, n_regimes=n_regimes)

            if regime_data:
                current = regime_data['current_info']

                # === HOVEDKORT: Nåværende regime ===
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {current['color']}15 0%, {current['color']}05 100%);
                            border: 2px solid {current['color']}; border-radius: 16px; padding: 30px; 
                            margin: 20px 0; text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 10px;">{current['emoji']}</div>
                    <h1 style="margin: 0; color: {current['color']}; font-size: 2.2rem; font-weight: 700;">
                        {current['name']}
                    </h1>
                    <p style="color: #a0a0a0; font-size: 1rem; margin: 10px 0 20px 0;">
                        {current['description']}
                    </p>
                    <div style="background: rgba(0,0,0,0.2); padding: 12px 20px; border-radius: 8px; 
                                display: inline-block; margin-bottom: 15px;">
                        <span style="color: #888; font-size: 0.8rem;">ANBEFALING:</span>
                        <span style="color: #fff; font-size: 1rem; font-weight: 600; margin-left: 8px;">
                            {current['action']}
                        </span>
                    </div>
                    <div style="color: #888; font-size: 0.9rem;">
                        {current.get('confidence_emoji', '')} Konfidens <strong style="color: {current['color']}">{current['probability']*100:.0f}%</strong>
                        (<em>{current.get('confidence', 'ukjent')}</em>)
                        &nbsp;•&nbsp; Varighet <strong>{current['streak_days']} dager</strong>
                        {f"&nbsp;•&nbsp; Forventet <strong>{current['expected_duration']:.0f} dager</strong>" if current.get('expected_duration') else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # === OVERGANGSVARSEL ===
                if current.get('transition_warning'):
                    tw = current['transition_warning']
                    st.warning(tw['message'])

                # === REGIME-SANNSYNLIGHETER (kun aktive regimer) ===
                st.markdown("#### Regime-sannsynligheter")
                prob_cols = st.columns(len(current['all_probs']))

                for i, (name, info) in enumerate(sorted(current['all_probs'].items(),
                                                        key=lambda x: x[1]['prob'], reverse=True)):
                    with prob_cols[i]:
                        is_current = name == current['name']
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; 
                                    background: {info['color']}{'20' if is_current else '08'}; 
                                    border-radius: 10px; 
                                    border: {'2px solid ' + info['color'] if is_current else '1px solid ' + info['color'] + '30'};">
                            <div style="font-size: 1.5rem;">{info['emoji']}</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: {info['color']};">
                                {info['prob']*100:.0f}%
                            </div>
                            <div style="color: #aaa; font-size: 0.8rem; margin-top: 5px;">{name}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # === NØKKELTALL ===
                st.markdown("#### Markedstilstand")
                m1, m2, m3 = st.columns(3)

                vol_delta = "Hoy" if current['volatility'] > 0.20 else "Normal" if current['volatility'] > 0.12 else "Lav"
                m1.metric("Volatilitet", f"{current['volatility']*100:.1f}%", delta=vol_delta)
                m2.metric("Avkastning (20d ann.)", f"{current['rolling_return']*100:.1f}%")
                m3.metric("Regime-varighet", f"{current['streak_days']} dager")

                # === REGIME-HISTORIE CHART ===
                st.markdown("#### Historikk")

                df_regimes = regime_data['df_regimes']
                periode = st.radio("Periode", ["3M", "6M", "1Y", "2Y", "Alt"], horizontal=True, index=2)
                dager_map = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "Alt": len(df_regimes)}
                df_plot = df_regimes.tail(dager_map[periode])

                # Enkel og ren chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.05, row_heights=[0.7, 0.3])

                # Pris med regime-farger (sammenhengende linje, farge per regime)
                for regime_id, label_info in regime_data['regime_labels'].items():
                    mask = df_plot['regime'] == regime_id
                    if mask.any():
                        fig.add_trace(go.Scatter(
                            x=df_plot[mask].index,
                            y=df_plot[mask]['Close'],
                            mode='markers+lines',
                            line=dict(color=label_info['color'], width=2),
                            marker=dict(size=3, color=label_info['color']),
                            name=label_info['name'],
                            hovertemplate=f"<b>{label_info['name']}</b><br>%{{x}}<br>Verdi: %{{y:.2f}}<extra></extra>"
                        ), row=1, col=1)

                # Sannsynligheter som stacked area
                for regime_id, label_info in regime_data['regime_labels'].items():
                    prob_col = f'prob_regime_{regime_id}'
                    if prob_col in df_plot.columns:
                        # Konverter hex til rgba for fillcolor
                        hex_color = label_info['color']
                        # Parse hex color
                        r = int(hex_color[1:3], 16)
                        g = int(hex_color[3:5], 16)
                        b = int(hex_color[5:7], 16)
                        fill_rgba = f'rgba({r},{g},{b},0.25)'

                        fig.add_trace(go.Scatter(
                            x=df_plot.index,
                            y=df_plot[prob_col],
                            mode='lines',
                            line=dict(color=label_info['color'], width=1),
                            fill='tozeroy',
                            fillcolor=fill_rgba,
                            name=f"P({label_info['name']})",
                            showlegend=False
                        ), row=2, col=1)

                fig.update_layout(
                    height=450,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                               font=dict(size=10)),
                    margin=dict(l=50, r=20, t=30, b=30),
                    hovermode='x unified'
                )
                fig.update_yaxes(title_text="Indeks", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
                fig.update_yaxes(title_text="Prob", row=2, col=1, range=[0, 1], gridcolor="rgba(255,255,255,0.05)")
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

                st.plotly_chart(fig, width="stretch")

                # === UTVIDBAR DETALJER ===
                with st.expander("Overgangsmatrise"):
                    st.caption("Sannsynlighet for å gå fra ett regime til et annet (rad → kolonne)")
                    transition_df = regime_data['transition_matrix'] * 100
                    st.dataframe(
                        transition_df.style.format("{:.0f}%").background_gradient(cmap='Blues', axis=1),
                        width="stretch"
                    )

            else:
                st.warning("Kunne ikke gjennomføre regimeanalyse.")
        else:
            st.warning("Ingen markedsdata. Klikk 'Oppdater Data'.")
    else:
        st.info("Regimeanalyse ikke tilgjengelig.")

    # === KLASSISK MARKEDSBREDDE ===
    st.markdown("---")
    st.markdown("### Markedsbredde — Oslo Børs")

    sma200_count = 0
    sma50_count = 0
    advances = 0
    declines = 0
    new_52w_high = 0
    new_52w_low = 0
    total_analyzed = 0

    # Historisk bredde — siste 6 mnd
    breadth_history = []

    pb = st.progress(0, text="Beregner markedsbredde...")

    # Cache technical indicators per ticker in session state
    if 'teknisk_cache' not in st.session_state:
        st.session_state['teknisk_cache'] = {}

    for i, ticker in enumerate(unike_tickers):
        df_t = df_clean[df_clean['Ticker'] == ticker]
        if len(df_t) > 200:
            if ticker not in st.session_state['teknisk_cache']:
                st.session_state['teknisk_cache'][ticker] = cached_beregn_tekniske_indikatorer(df_t)
            df_t = st.session_state['teknisk_cache'][ticker]
            last_close = df_t['Close'].iloc[-1]
            prev_close = df_t['Close'].iloc[-2]
            sma200 = df_t['SMA_200'].iloc[-1]
            total_analyzed += 1

            # SMA 200
            if last_close > sma200: sma200_count += 1
            
            # SMA 50
            if 'SMA_50' in df_t.columns:
                sma50 = df_t['SMA_50'].iloc[-1]
                if pd.notna(sma50) and last_close > sma50:
                    sma50_count += 1
            
            # Advances / Declines
            if last_close > prev_close: advances += 1
            elif last_close < prev_close: declines += 1
            
            # Nye 52-ukers høy/lav
            if len(df_t) >= 252:
                high_52w = df_t['High'].iloc[-252:].max()
                low_52w = df_t['Low'].iloc[-252:].min()
                if last_close >= high_52w * 0.99:  # Innen 1% av 52w high
                    new_52w_high += 1
                if last_close <= low_52w * 1.01:   # Innen 1% av 52w low
                    new_52w_low += 1
            
            # Samle historisk bredde-data (siste 130 handelsdager ≈ 6 mnd)
            if len(breadth_history) == 0 and len(df_t) > 130:
                # Bygg historikk for denne tickeren
                sma200_col = df_t['SMA_200']
                for j in range(min(130, len(df_t))):
                    idx = -(130 - j) if 130 - j < len(df_t) else 0
                    if len(breadth_history) <= j:
                        breadth_history.append({
                            'date': df_t.index[idx] if isinstance(df_t.index[idx], pd.Timestamp) else pd.Timestamp(df_t.index[idx]),
                            'over_sma200': 0, 'over_sma50': 0,
                            'adv': 0, 'dec': 0, 'total': 0
                        })
            
            # Oppdater historisk bredde for siste 130 dager
            lookback = min(130, len(df_t))
            sma200_series = df_t['SMA_200'].iloc[-lookback:]
            close_series = df_t['Close'].iloc[-lookback:]
            sma50_series = df_t['SMA_50'].iloc[-lookback:] if 'SMA_50' in df_t.columns else pd.Series([np.nan] * lookback)
            
            for j in range(lookback):
                if j < len(breadth_history):
                    if pd.notna(sma200_series.iloc[j]) and close_series.iloc[j] > sma200_series.iloc[j]:
                        breadth_history[j]['over_sma200'] += 1
                    if j < len(sma50_series) and pd.notna(sma50_series.iloc[j]) and close_series.iloc[j] > sma50_series.iloc[j]:
                        breadth_history[j]['over_sma50'] += 1
                    breadth_history[j]['total'] += 1
                    if j > 0 and close_series.iloc[j] > close_series.iloc[j-1]:
                        breadth_history[j]['adv'] += 1
                    elif j > 0 and close_series.iloc[j] < close_series.iloc[j-1]:
                        breadth_history[j]['dec'] += 1

        pb.progress((i+1)/len(unike_tickers))
    pb.empty()

    pct_sma200 = (sma200_count / total_analyzed) * 100 if total_analyzed else 0
    pct_sma50 = (sma50_count / total_analyzed) * 100 if total_analyzed else 0

    # === METRICS ROW 1 ===
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Over SMA 200", f"{pct_sma200:.1f}%", help="Aksjer med kurs over 200-dagers glidende snitt")
    c2.metric("Over SMA 50", f"{pct_sma50:.1f}%", help="Aksjer med kurs over 50-dagers glidende snitt — kortere trend")
    c3.metric("Positive i dag", f"{advances}")
    c4.metric("Negative i dag", f"{declines}")

    # === METRICS ROW 2 ===
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Nye 52u-høy", f"{new_52w_high}", help="Aksjer innen 1% av 52-ukers høyeste")
    c6.metric("Nye 52u-lav", f"{new_52w_low}", help="Aksjer innen 1% av 52-ukers laveste")
    
    # Advance/Decline-ratio
    ad_ratio = advances / max(declines, 1)
    ad_color = "normal" if 0.8 < ad_ratio < 1.3 else ("inverse" if ad_ratio < 0.8 else "off")
    c7.metric("A/D-ratio", f"{ad_ratio:.2f}", help="Advances / Declines. >1 = bullish, <1 = bearish")
    
    # McClellan Oscillator (basert på A/D-data)
    if breadth_history and len(breadth_history) > 38:
        ad_net = pd.Series([h['adv'] - h['dec'] for h in breadth_history])
        ema19 = ad_net.ewm(span=19, adjust=False).mean()
        ema39 = ad_net.ewm(span=39, adjust=False).mean()
        mcclellan = ema19.iloc[-1] - ema39.iloc[-1]
        mc_delta = "Bullish" if mcclellan > 0 else "Bearish"
        c8.metric("McClellan Osc.", f"{mcclellan:.1f}", delta=mc_delta, help="McClellan Oscillator = EMA19(A-D) − EMA39(A-D). >0 = positivt momentum")
    else:
        c8.metric("McClellan Osc.", "—", help="Utilstrekkelig data")

    # === HISTORISK BREDDE-CHART ===
    if breadth_history and len(breadth_history) > 5:
        with st.expander("Historisk markedsbredde (6 mnd)", expanded=True):
            hist_df = pd.DataFrame(breadth_history)
            hist_df['pct_sma200'] = (hist_df['over_sma200'] / hist_df['total'].replace(0, 1)) * 100
            hist_df['pct_sma50'] = (hist_df['over_sma50'] / hist_df['total'].replace(0, 1)) * 100
            
            # A/D-linje (kumulativ)
            hist_df['ad_net'] = hist_df['adv'] - hist_df['dec']
            hist_df['ad_line'] = hist_df['ad_net'].cumsum()
            
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.4, 0.3, 0.3],
                subplot_titles=("% over SMA", "A/D-linje (kumulativ)", "McClellan Oscillator"),
                vertical_spacing=0.08
            )
            
            # Panel 1: % over SMA 200 og SMA 50
            fig.add_trace(go.Scatter(
                x=hist_df['date'], y=hist_df['pct_sma200'],
                name='% > SMA 200', line=dict(color='#667eea', width=2),
                fill='tozeroy', fillcolor='rgba(102,126,234,0.1)'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=hist_df['date'], y=hist_df['pct_sma50'],
                name='% > SMA 50', line=dict(color='#f6ad55', width=2),
            ), row=1, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
            
            # Panel 2: A/D-linje
            fig.add_trace(go.Scatter(
                x=hist_df['date'], y=hist_df['ad_line'],
                name='A/D-linje', line=dict(color='#48bb78', width=2),
                fill='tozeroy', fillcolor='rgba(72,187,120,0.1)'
            ), row=2, col=1)
            
            # Panel 3: McClellan Oscillator
            ad_series = hist_df['ad_net']
            ema19_hist = ad_series.ewm(span=19, adjust=False).mean()
            ema39_hist = ad_series.ewm(span=39, adjust=False).mean()
            mcclellan_hist = ema19_hist - ema39_hist
            
            colors = ['#48bb78' if v >= 0 else '#f56565' for v in mcclellan_hist]
            fig.add_trace(go.Bar(
                x=hist_df['date'], y=mcclellan_hist,
                name='McClellan', marker_color=colors,
                showlegend=True
            ), row=3, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=3, col=1)
            
            fig.update_layout(
                height=600,
                template='plotly_dark',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=40, r=20, t=60, b=30)
            )
            fig.update_yaxes(title_text="%", row=1, col=1)
            fig.update_yaxes(title_text="Kumulativ", row=2, col=1)
            fig.update_yaxes(title_text="Oscillator", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tolkning
            st.markdown(f"""
            **Tolkning:**
            - **SMA 200:** {pct_sma200:.0f}% av aksjene er i langsiktig opptrend {'Sterkt marked' if pct_sma200 > 60 else 'Svakt marked' if pct_sma200 < 40 else 'Noytralt'}
            - **SMA 50:** {pct_sma50:.0f}% i kortsiktig opptrend {'Sterkt' if pct_sma50 > 60 else 'Svakt' if pct_sma50 < 40 else 'Noytralt'}
            - **Nye hoyder vs lav:** {new_52w_high} hoy / {new_52w_low} lav {'Ekspansjon' if new_52w_high > new_52w_low * 2 else 'Kontraksjon' if new_52w_low > new_52w_high else 'Balansert'}
            - **A/D-ratio:** {ad_ratio:.2f} {'Bred deltakelse' if ad_ratio > 1.2 else 'Smal ledelse' if ad_ratio < 0.8 else 'Normal'}
            """)
