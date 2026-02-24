import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logic
import data
import config
from utils import load_watchlist, add_to_watchlist, remove_from_watchlist

portfolio_tilgjengelig = False
try:
    import portfolio
    portfolio_tilgjengelig = True
except ImportError:
    pass

beta_tilgjengelig = False
try:
    import beta_ml
    beta_tilgjengelig = True
except ImportError:
    pass


def render():
    if not portfolio_tilgjengelig:
        st.error("Portefølje-modulen er ikke tilgjengelig. Sjekk at portfolio.py finnes og at alle avhengigheter er installert.")
        return

    df_clean = st.session_state.get('df_clean', pd.DataFrame())
    unike_tickers = st.session_state.get('unike_tickers', [])

    st.title("Min Portefølje")
    
    # Futuristisk styling for portefølje
    st.markdown("""
    <style>
    .portfolio-card {
        background: linear-gradient(135deg, rgba(78, 140, 255, 0.15) 0%, rgba(78, 140, 255, 0.05) 100%);
        border: 1px solid rgba(78, 140, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .position-row {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }
    .position-row:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateX(4px);
    }
    .position-profit { border-left-color: #26a69a; }
    .position-loss { border-left-color: #ef5350; }
    .position-critical { 
        border-left-color: #ff4444; 
        animation: pulse-critical 1.5s ease-in-out infinite;
    }
    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(255, 68, 68, 0); }
    }
    .alert-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .alert-critical { background: rgba(255, 68, 68, 0.2); color: #ff6b6b; }
    .alert-high { background: rgba(255, 165, 0, 0.2); color: #ffa500; }
    .alert-medium { background: rgba(255, 215, 0, 0.2); color: #ffd700; }
    .alert-info { background: rgba(78, 140, 255, 0.2); color: #4e8cff; }
    </style>
    """, unsafe_allow_html=True)
    
    # Last portefølje
    pf = portfolio.load_portfolio()
    positions = pf.get('positions', {})
    
    # --- TABS ---
    tab_oversikt, tab_monte_carlo, tab_ai, tab_legg_til, tab_selg, tab_historikk = st.tabs([
        "Oversikt", "Monte Carlo", "AI Analyse", "Legg til posisjon", "Selg/Juster", "Historikk"
    ])
    
    # ===========================================
    # TAB 1: OVERSIKT
    # ===========================================
    with tab_oversikt:
        if not positions:
            st.info("Du har ingen posisjoner ennå. Gå til 'Legg til posisjon' for å komme i gang!")
        else:
            # Bygg df_dict for analyse og oppdater trailing high
            df_dict = {}
            trailing_updated = []
            for ticker in positions.keys():
                df_t = df_clean[df_clean['Ticker'] == ticker]
                if not df_t.empty:
                    if ticker not in st.session_state.get('teknisk_cache', {}):
                        st.session_state['teknisk_cache'][ticker] = logic.beregn_tekniske_indikatorer(df_t.copy())
                    df_dict[ticker] = st.session_state['teknisk_cache'][ticker]
                    
                    # Auto-oppdater trailing high basert på historisk høyeste siden kjøp
                    pos_data = positions[ticker]
                    buy_date = pos_data.get('buy_date', '')
                    if buy_date:
                        try:
                            df_since_buy = df_t[df_t.index >= pd.to_datetime(buy_date)]
                            if not df_since_buy.empty:
                                historical_high = float(df_since_buy['High'].max())
                                current_trailing = pos_data.get('trailing_high', 0)
                                if historical_high > current_trailing:
                                    portfolio.update_trailing_high(ticker, historical_high)
                                    trailing_updated.append(ticker)
                        except:
                            pass
            
            # Reload portfolio if trailing highs were updated
            if trailing_updated:
                pf = portfolio.load_portfolio()
                positions = pf.get('positions', {})
            
            # Analyser portefølje
            analysis = portfolio.analyze_portfolio(df_dict)
            
            # === TOTAL OVERSIKT ===
            col1, col2, col3, col4 = st.columns(4)
            
            profit_color = "#26a69a" if analysis['total_profit'] >= 0 else "#ef5350"
            
            col1.metric(
                "Porteføljeverdi",
                f"{analysis['total_value']:,.0f} NOK"
            )
            col2.metric(
                "Total P/L",
                f"{analysis['total_profit']:+,.0f} NOK",
                f"{analysis['total_profit_pct']:+.1f}%"
            )
            col3.metric(
                "Antall posisjoner",
                f"{analysis['position_count']}"
            )
            col4.metric(
                "Varsler",
                f"{analysis['alert_count']}",
                "Krever handling" if analysis['alert_count'] > 0 else "Alt OK"
            )
            
            # === SEKTORFORDELING ===
            if len(positions) >= 2:
                st.markdown("### Sektorfordeling")
                
                sektor_data = {}
                for pos in analysis['positions']:
                    ticker = pos['ticker']
                    sektor = logic.hent_sektor(ticker)
                    if sektor not in sektor_data:
                        sektor_data[sektor] = {'verdi': 0, 'tickers': []}
                    sektor_data[sektor]['verdi'] += pos['value']
                    sektor_data[sektor]['tickers'].append(ticker)
                
                # Pie chart med Plotly
                import plotly.express as px
                
                sektor_df = pd.DataFrame([
                    {'Sektor': s, 'Verdi': d['verdi'], 'Aksjer': ', '.join(d['tickers'])}
                    for s, d in sektor_data.items()
                ])
                
                col_pie, col_table = st.columns([1, 1])
                
                with col_pie:
                    fig = px.pie(
                        sektor_df, 
                        values='Verdi', 
                        names='Sektor',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#d1d4dc',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                        margin=dict(t=20, b=20, l=20, r=20),
                        height=280
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, width="stretch")
                
                with col_table:
                    # Konsentrasjonsvarsel
                    total_verdi = sum(d['verdi'] for d in sektor_data.values())
                    for sektor, d in sektor_data.items():
                        pct = (d['verdi'] / total_verdi * 100) if total_verdi > 0 else 0
                        if pct > 40:
                            st.warning(f"Høy konsentrasjon i **{sektor}**: {pct:.0f}% av porteføljen")
                    
                    # Sektortabell
                    for sektor, d in sorted(sektor_data.items(), key=lambda x: -x[1]['verdi']):
                        pct = (d['verdi'] / total_verdi * 100) if total_verdi > 0 else 0
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <span>{sektor}</span>
                            <span style="color: #4e8cff;">{pct:.1f}% ({len(d['tickers'])} aksjer)</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # === ALERTS ===
            if analysis['alerts']:
                st.markdown("### Salgssignaler som krever handling")
                
                for alert in analysis['alerts']:
                    severity_class = f"alert-{alert['severity'].lower()}"
                    st.markdown(f"""
                    <div class="position-row position-critical" style="border-left-color: {'#ff4444' if alert['severity'] == 'CRITICAL' else '#ffa500' if alert['severity'] == 'HIGH' else '#ffd700'};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: 600; font-size: 1.1rem;">{alert['ticker']}</span>
                                <span class="alert-badge {severity_class}">{alert['severity']}</span>
                            </div>
                        </div>
                        <div style="color: #ccc; margin-top: 6px;">{alert['message']}</div>
                        <div style="color: #4e8cff; font-size: 0.9rem; margin-top: 4px;">{alert['action']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # === POSISJONER ===
            st.markdown("### Mine Posisjoner")
            
            for idx, pos in enumerate(analysis['positions']):
                ticker = pos['ticker']
                profit_class = "position-profit" if pos['profit'] >= 0 else "position-loss"
                
                # Sjekk om kritisk
                exit_info = pos.get('exit_analysis', {})
                if exit_info.get('should_exit'):
                    profit_class = "position-critical"
                
                profit_sign = "+" if pos['profit'] >= 0 else ""
                profit_color = "#26a69a" if pos['profit'] >= 0 else "#ef5350"
                
                # Exit-signaler for denne posisjonen
                exit_signals = exit_info.get('signals', [])
                signal_html = ""
                if exit_signals:
                    signal_html = "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);'>"
                    for sig in exit_signals[:2]:
                        sev_color = {'CRITICAL': '#ff6b6b', 'HIGH': '#ffa500', 'MEDIUM': '#ffd700'}.get(sig['severity'], '#888')
                        signal_html += f"<span style='color: {sev_color}; font-size: 0.85rem;'>{sig['message']}</span><br>"
                    signal_html += "</div>"
                
                # Sjekk aktive scanner-strategier for denne aksjen med kvalitet
                scanner_badges = ""
                if ticker in df_dict:
                    df_pos = df_dict[ticker]
                    try:
                        signaler = logic.sjekk_strategier(df_pos)
                        # Sjekk siste 5 dager for aktive signaler
                        aktive = []
                        strat_names = {
                            'Kort_Sikt_RSI': 'RSI', 'Momentum_Burst': 'MOM', 'Golden_Cross': 'GC',
                            'Ichimoku_Breakout': 'ICHI', 'Wyckoff_Spring': 'WYC', 
                            'Bull_Race_Prep': 'BB', 'VCP_Pattern': 'VCP', 'Pocket_Pivot': 'PP'
                        }
                        for strat, abbr in strat_names.items():
                            if strat in signaler.columns:
                                # Finn siste signal
                                signal_mask = signaler[strat].fillna(False).astype(bool)
                                last_5_signals = signal_mask.iloc[-5:]
                                if last_5_signals.any():
                                    # Finn dato for siste signal
                                    signal_dates = signaler.index[signal_mask]
                                    if len(signal_dates) > 0:
                                        siste_signal_dato = signal_dates[-1]
                                        dager_siden = (df_pos.index[-1] - siste_signal_dato).days
                                        
                                        # Beregn kvalitet hvis funksjonene finnes
                                        klasse = 'B'  # Default
                                        score = 50
                                        if hasattr(logic, 'beregn_signal_kvalitet') and hasattr(logic, 'klassifiser_signal_kvalitet'):
                                            kvalitet_info = logic.beregn_signal_kvalitet(df_pos, siste_signal_dato, strat)
                                            if kvalitet_info and isinstance(kvalitet_info, dict):
                                                score = kvalitet_info.get('score', 50)
                                                klasse = logic.klassifiser_signal_kvalitet(score)
                                        
                                        aktive.append({
                                            'abbr': abbr,
                                            'klasse': klasse,
                                            'score': score,
                                            'dager': dager_siden
                                        })
                        
                        if aktive:
                            # Kvalitetsfarger
                            kval_colors = {
                                'A': ('#26a69a', 'rgba(38,166,154,0.25)'),   # Grønn
                                'B': ('#4e8cff', 'rgba(78,140,255,0.25)'),   # Blå
                                'C': ('#ffd700', 'rgba(255,215,0,0.25)'),    # Gul
                                'D': ('#888', 'rgba(136,136,136,0.25)')      # Grå
                            }
                            scanner_badges = "<div style='margin-top: 6px;'>"
                            for sig in sorted(aktive, key=lambda x: x['score'], reverse=True):
                                fg, bg = kval_colors.get(sig['klasse'], ('#888', 'rgba(136,136,136,0.2)'))
                                dager_str = f"{sig['dager']}d" if sig['dager'] > 0 else "i dag"
                                scanner_badges += f"<span style='background: {bg}; color: {fg}; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; margin-right: 5px; font-weight: 500;'>{sig['abbr']} {sig['klasse']} ({dager_str})</span>"
                            scanner_badges += "</div>"
                    except Exception as e:
                        # Debug: vis feil
                        scanner_badges = f"<div style='color: #ff6b6b; font-size: 0.7rem;'>Scanner-feil: {str(e)[:50]}</div>"
                
                # Selskaps-navn og sektor
                selskap_raw = data.ticker_til_navn(ticker) if hasattr(data, "ticker_til_navn") else ticker.replace(".OL", "")
                # Escape HTML-spesialtegn
                import html as html_module
                selskap = html_module.escape(str(selskap_raw))
                sektor = html_module.escape(str(logic.hent_sektor(ticker)))
                
                # Posisjonskort med knapp ved siden
                col_info, col_btn = st.columns([6, 1])
                
                with col_info:
                    st.markdown(f'''<div class="position-row {profit_class}">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<div>
<span style="font-weight: 600; font-size: 1.1rem;">{ticker}</span>
<span style="color: #888; font-size: 0.9rem; margin-left: 8px;">{selskap}</span>
<span style="color: #666; font-size: 0.75rem; margin-left: 6px;">({sektor})</span>
<div style="color: #aaa; font-size: 0.8rem; margin-top: 4px;">
{pos['quantity']} aksjer @ {pos['avg_price']:.2f} | Holdt {pos['holding_days']} dager
</div>
{scanner_badges}
</div>
<div style="text-align: right;">
<div style="font-size: 1.2rem; font-weight: 600;">{pos['current_price']:.2f}</div>
<div style="color: {profit_color}; font-weight: 600;">
{profit_sign}{pos['profit']:,.0f} NOK ({profit_sign}{pos['profit_pct']:.1f}%)
</div>
<div style="color: #666; font-size: 0.8rem;">
Stop: {pos['stop_loss']:.2f} | Topp: {pos['trailing_high']:.2f}
</div>
</div>
</div>
{signal_html}
</div>''', unsafe_allow_html=True)
                
                with col_btn:
                    st.write("")  # Spacing
                    if st.button("Vis", key=f"goto_chart_{ticker}_{idx}", help=f"Åpne {ticker} i Teknisk Analyse"):
                        st.session_state['valgt_ticker'] = ticker
                        st.session_state['navigate_to'] = "Teknisk Analyse"
                        st.rerun()
            
            # === DAGLIG RAPPORT KNAPP ===
            st.markdown("---")
            with st.expander("Daglig Rapport (for fremtidig e-post)"):
                report = portfolio.generate_daily_report(df_dict, list(unike_tickers))
                report_text = portfolio.format_report_for_display(report)
                st.markdown(report_text)
                
                st.info("E-post-varsling kommer i en fremtidig oppdatering. Rapporten vil sendes daglig med A-kvalitet signaler og salgsvarsler.")
    
    # ===========================================
    # TAB: MONTE CARLO SIMULERING
    # ===========================================
    with tab_monte_carlo:
        if not positions:
            st.info("Legg til posisjoner for å se Monte Carlo-simulering.")
        else:
            st.markdown("### Monte Carlo Portefølje-simulering")
            st.caption("Simulerer 10 000 mulige porteføljescenarioer over 1 år basert på historisk volatilitet og korrelasjoner.")
            
            # Bygg df_dict for simulering
            mc_df_dict = {}
            mangler = []
            for ticker in positions.keys():
                df_t = df_clean[df_clean['Ticker'] == ticker]
                if not df_t.empty and len(df_t) >= 60:
                    mc_df_dict[ticker] = df_t
                else:
                    mangler.append(ticker)
            
            if mangler:
                st.warning(f"Mangler tilstrekkelig data for: {', '.join(mangler)} (krever min. 60 dager)")
            
            # Innstillinger
            with st.expander("Simuleringsinnstillinger", expanded=False):
                mc_col1, mc_col2, mc_col3 = st.columns(3)
                n_sim = mc_col1.select_slider(
                    "Antall simuleringer",
                    options=[1000, 5000, 10000, 25000, 50000],
                    value=10000,
                    key="mc_n_sim"
                )
                n_dager = mc_col2.select_slider(
                    "Horisont (handelsdager)",
                    options=[63, 126, 189, 252],
                    value=252,
                    format_func=lambda x: f"{x} dager (~{x//21} mnd)",
                    key="mc_n_dager"
                )
                mc_seed = mc_col3.number_input("Random seed", min_value=1, value=42, key="mc_seed")
            
            if len(mc_df_dict) == 0:
                st.error("Ingen posisjoner har nok data for simulering.")
            elif st.button("Kjør Monte Carlo-simulering", type="primary", key="mc_run_btn"):
                with st.spinner(f"Kjører {n_sim:,} simuleringer over {n_dager} dager..."):
                    mc_result = portfolio.monte_carlo_portefolje(
                        df_dict=mc_df_dict,
                        n_simuleringer=n_sim,
                        n_dager=n_dager,
                        seed=mc_seed
                    )
                
                if mc_result is None:
                    st.error("Simulering feilet. Sjekk at posisjonene har nok historisk data.")
                else:
                    st.session_state['mc_result'] = mc_result
                    st.success(f"{mc_result['n_simuleringer']:,} simuleringer fullført for {mc_result['n_posisjoner']} posisjoner!")
            
            # Vis resultater
            if 'mc_result' in st.session_state and st.session_state['mc_result']:
                mc = st.session_state['mc_result']
                
                st.markdown("---")
                
                # === NØKKELTALL ===
                st.markdown("#### Nøkkeltall (1-års horisont)")
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    "Startverdi",
                    f"{mc['startverdi']:,.0f} NOK"
                )
                k2.metric(
                    "Median utfall",
                    f"{mc['median_sluttverdi']:,.0f} NOK",
                    f"{mc['median_avkastning_pct']:+.1f}%"
                )
                k3.metric(
                    "Worst case (5%)",
                    f"{mc['worst_case_5']:,.0f} NOK",
                    f"{mc['worst_avkastning_pct']:+.1f}%"
                )
                k4.metric(
                    "Best case (95%)",
                    f"{mc['best_case_95']:,.0f} NOK",
                    f"{mc['best_avkastning_pct']:+.1f}%"
                )
                
                # === RISIKOMÅL ===
                st.markdown("#### Risikomål")
                r1, r2, r3, r4 = st.columns(4)
                
                # Fargekoding for VaR
                var_color = "#26a69a" if mc['var_95_pct'] < 15 else "#ffa500" if mc['var_95_pct'] < 25 else "#ef5350"
                
                r1.metric(
                    "VaR 95%",
                    f"{mc['var_95']:,.0f} NOK",
                    f"-{mc['var_95_pct']:.1f}% av portefølje"
                )
                r2.metric(
                    "CVaR 95%",
                    f"{mc['cvar_95']:,.0f} NOK",
                    f"-{mc['cvar_95_pct']:.1f}%"
                )
                r3.metric(
                    "Sharpe (est.)",
                    f"{mc['sharpe_estimate']:.2f}"
                )
                r4.metric(
                    "Snitt korrelasjon",
                    f"{mc['avg_korrelasjon']:.2f}"
                )
                
                # VaR-forklaring
                st.markdown(f"""
                <div style="background: rgba(78,140,255,0.1); border-left: 3px solid {var_color}; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;">
                    <strong>Value-at-Risk (VaR 95%):</strong> Med 95% sannsynlighet vil du <strong>ikke</strong> tape mer enn 
                    <span style="color: {var_color}; font-weight: 700;">{mc['var_95']:,.0f} NOK ({mc['var_95_pct']:.1f}%)</span> 
                    over {mc['n_dager']} handelsdager.<br>
                    <strong>CVaR (Expected Shortfall):</strong> Hvis det likevel går galt (verste 5%), er gjennomsnittlig tap 
                    <span style="color: #ef5350; font-weight: 700;">{mc['cvar_95']:,.0f} NOK ({mc['cvar_95_pct']:.1f}%)</span>.
                </div>
                """, unsafe_allow_html=True)
                
                # === FAN-CHART ===
                st.markdown("#### Porteføljeutvikling — Fan-chart")
                
                dager = list(range(len(mc['pct_50'])))
                
                fig_mc = go.Figure()
                
                # 5-95% bånd (lysest)
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_95'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_5'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(78, 140, 255, 0.08)',
                    name='5–95% intervall',
                    hoverinfo='skip'
                ))
                
                # 10-90% bånd
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_90'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_10'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(78, 140, 255, 0.15)',
                    name='10–90% intervall',
                    hoverinfo='skip'
                ))
                
                # 25-75% bånd (mørkest)
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_75'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_25'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(78, 140, 255, 0.25)',
                    name='25–75% intervall',
                    hoverinfo='skip'
                ))
                
                # Median-linje
                fig_mc.add_trace(go.Scatter(
                    x=dager, y=mc['pct_50'],
                    mode='lines',
                    line=dict(color='#4e8cff', width=2.5),
                    name='Median',
                    hovertemplate='Dag %{x}<br>Verdi: %{y:,.0f} NOK<extra></extra>'
                ))
                
                # Startverdi-linje
                fig_mc.add_hline(
                    y=mc['startverdi'],
                    line_dash="dash",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text=f"Start: {mc['startverdi']:,.0f}",
                    annotation_position="bottom right",
                    annotation_font_color="#888"
                )
                
                fig_mc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#d1d4dc',
                    xaxis=dict(
                        title='Handelsdager',
                        gridcolor='rgba(255,255,255,0.05)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title='Porteføljeverdi (NOK)',
                        gridcolor='rgba(255,255,255,0.05)',
                        showgrid=True,
                        tickformat=','
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(t=30, b=50, l=60, r=30),
                    height=450,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # === SANNSYNLIGHETER ===
                st.markdown("#### Sannsynligheter")
                
                p1, p2, p3 = st.columns(3)
                
                p1.metric("Sannsynlighet for tap", f"{mc['prob_tap_pct']:.1f}%")
                
                p2.metric("P(gevinst > 10%)", f"{mc['prob_gevinst_10_pct']:.1f}%")
                
                p3.metric("P(tap > 20%)", f"{mc['prob_tap_20_pct']:.1f}%")
                
                # === PORTEFØLJESAMMENSETNING ===
                with st.expander("Porteføljesammensetning & Korrelasjoner", expanded=False):
                    # Vekter
                    st.markdown("**Porteføljevekter:**")
                    vekt_df = pd.DataFrame([
                        {"Ticker": t, "Vekt (%)": w}
                        for t, w in mc['vekter'].items()
                    ]).sort_values("Vekt (%)", ascending=False)
                    st.dataframe(vekt_df, hide_index=True, use_container_width=True)
                    
                    # Korrelasjonsmatrise
                    if mc['n_posisjoner'] > 1:
                        st.markdown("**Korrelasjonsmatrise:**")
                        corr_data = mc['korrelasjon_matrise']
                        corr_df = pd.DataFrame(
                            corr_data['matrise'],
                            index=corr_data['tickers'],
                            columns=corr_data['tickers']
                        )
                        
                        # Heatmap
                        import plotly.express as px
                        fig_corr = px.imshow(
                            corr_df,
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            aspect='auto'
                        )
                        fig_corr.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#d1d4dc',
                            height=max(250, 50 * len(corr_data['tickers'])),
                            margin=dict(t=20, b=20, l=20, r=20)
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Diversifikasjonsvarsel
                        if mc['avg_korrelasjon'] > 0.6:
                            st.warning(f"Høy gjennomsnittlig korrelasjon ({mc['avg_korrelasjon']:.2f}). Porteføljen er dårlig diversifisert — vurder bredere sektorspredning.")
                        elif mc['avg_korrelasjon'] > 0.3:
                            st.info(f"Moderat korrelasjon ({mc['avg_korrelasjon']:.2f}). Diversifiseringen er OK.")
                        else:
                            st.success(f"Lav korrelasjon ({mc['avg_korrelasjon']:.2f}). God diversifisering!")
    
    # ===========================================
    # TAB 2: AI ANALYSE
    # ===========================================
    with tab_ai:
        if not positions:
            st.info("Legg til posisjoner for å se AI-analyse.")
        elif not beta_tilgjengelig:
            st.warning("AI Scanner (beta_ml) er ikke tilgjengelig. Sjekk at alle avhengigheter er installert.")
        else:
            st.markdown("### AI Scanner på dine posisjoner")
            st.caption("Kjører ML-modellen på hver posisjon for å vurdere fremtidig kurspotensial.")
            
            # Innstillinger
            with st.expander("AI-innstillinger", expanded=False):
                col1, col2, col3 = st.columns(3)
                ai_horisont = col1.selectbox("Horisont (dager)", [5, 10, 15, 20], index=1, key="pf_ai_horisont")
                ai_maal = col2.slider("Kursmål (%)", 2, 15, 5, key="pf_ai_maal")
                ai_stop = col3.slider("Stop loss (%)", 2, 10, 4, key="pf_ai_stop")
            
            # Kjør AI-analyse
            if st.button("Kjør AI-analyse på porteføljen", type="primary", key="pf_run_ai"):
                st.session_state['pf_ai_results'] = {}
                
                progress = st.progress(0, text="Starter AI-analyse...")
                tickers_list = list(positions.keys())
                
                for i, ticker in enumerate(tickers_list):
                    progress.progress((i + 1) / len(tickers_list), text=f"Analyserer {ticker}...")
                    
                    try:
                        df_t = df_clean[df_clean['Ticker'] == ticker].copy()
                        if len(df_t) < 300:
                            st.session_state['pf_ai_results'][ticker] = {
                                'score': None,
                                'error': 'For lite data'
                            }
                            continue
                        
                        # Beregn features
                        df_features = beta_ml.beregn_avanserte_features(df_t)
                        
                        if df_features.empty:
                            st.session_state['pf_ai_results'][ticker] = {
                                'score': None,
                                'error': 'Feature-beregning feilet'
                            }
                            continue
                        
                        # Opprett og tren modell
                        predictor = beta_ml.EnsembleStockPredictor(
                            horisont=ai_horisont,
                            target_pct=ai_maal / 100,
                            stop_pct=ai_stop / 100
                        )
                        
                        if predictor.fit(df_features, validate=True):
                            score = predictor.predict_proba(df_features)
                            vurdering, _ = beta_ml.gi_vurdering(score)
                            
                            # Beregn konfidensintervall
                            _, lower, upper = beta_ml.beregn_konfidensintervall(predictor, df_features)
                            
                            st.session_state['pf_ai_results'][ticker] = {
                                'score': score,
                                'vurdering': vurdering,
                                'konfidens': f"{lower:.0f}-{upper:.0f}%",
                                'rsi': df_features['RSI'].iloc[-1] if 'RSI' in df_features.columns else None,
                                'adx': df_features['ADX'].iloc[-1] if 'ADX' in df_features.columns else None,
                                'error': None
                            }
                        else:
                            st.session_state['pf_ai_results'][ticker] = {
                                'score': None,
                                'error': 'Modelltrening feilet'
                            }
                    
                    except Exception as e:
                        st.session_state['pf_ai_results'][ticker] = {
                            'score': None,
                            'error': str(e)[:50]
                        }
                
                progress.empty()
                st.success("AI-analyse fullført")
            
            # Vis resultater
            if 'pf_ai_results' in st.session_state and st.session_state['pf_ai_results']:
                st.markdown("---")
                st.markdown("### AI-resultater")
                
                # Sorter etter score (høyest først)
                results = st.session_state['pf_ai_results']
                sorted_tickers = sorted(
                    results.keys(),
                    key=lambda t: results[t].get('score', 0) or 0,
                    reverse=True
                )
                
                for ticker in sorted_tickers:
                    res = results[ticker]
                    pos = positions.get(ticker, {})
                    
                    if res.get('error'):
                        # Feil-kort
                        st.markdown(f"""
                        <div class="position-row" style="border-left-color: #666;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-weight: 600;">{ticker}</span>
                                    <span style="color: #888; margin-left: 8px;">{res['error']}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        score = res['score']
                        vurdering = res.get('vurdering', '')
                        konfidens = res.get('konfidens', '-')
                        
                        # Farge basert på score
                        if score >= 70:
                            score_color = "#26a69a"
                            score_bg = "rgba(38, 166, 154, 0.15)"
                            ai_signal = "BULLISH"
                        elif score >= 50:
                            score_color = "#ffd700"
                            score_bg = "rgba(255, 215, 0, 0.15)"
                            ai_signal = "NØYTRAL"
                        else:
                            score_color = "#ef5350"
                            score_bg = "rgba(239, 83, 80, 0.15)"
                            ai_signal = "BEARISH"
                        
                        # P/L for posisjonen
                        df_t = df_clean[df_clean['Ticker'] == ticker]
                        current_price = float(df_t['Close'].iloc[-1]) if not df_t.empty else pos.get('avg_price', 0)
                        avg_price = pos.get('avg_price', current_price)
                        profit_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
                        profit_color = "#26a69a" if profit_pct >= 0 else "#ef5350"
                        
                        # Anbefaling
                        if score >= 70 and profit_pct >= 0:
                            anbefaling = "Hold / Øk posisjon"
                            anb_color = "#26a69a"
                        elif score < 40 and profit_pct < -5:
                            anbefaling = "Vurder å selge"
                            anb_color = "#ef5350"
                        elif score < 50:
                            anbefaling = "Følg med - momentum avtar"
                            anb_color = "#ffa500"
                        else:
                            anbefaling = "Hold posisjon"
                            anb_color = "#4e8cff"
                        
                        st.markdown(f"""
                        <div class="position-row" style="border-left-color: {score_color}; background: {score_bg};">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <span style="font-weight: 600; font-size: 1.1rem;">{ticker}</span>
                                    <span style="color: {score_color}; font-weight: 600; margin-left: 12px;">{ai_signal}</span>
                                    <div style="color: #aaa; font-size: 0.85rem; margin-top: 6px;">
                                        RSI: {res.get('rsi', 0):.0f} | ADX: {res.get('adx', 0):.0f} | Konfidens: {konfidens}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.8rem; font-weight: 700; color: {score_color};">{score:.0f}</div>
                                    <div style="font-size: 0.8rem; color: #888;">AI Score</div>
                                    <div style="color: {profit_color}; font-size: 0.9rem; margin-top: 4px;">
                                        P/L: {profit_pct:+.1f}%
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
                                <span style="color: {anb_color}; font-weight: 500;">{anbefaling}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Oppsummering
                st.markdown("---")
                scores = [r['score'] for r in results.values() if r.get('score') is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    bullish = len([s for s in scores if s >= 60])
                    bearish = len([s for s in scores if s < 40])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Gjennomsnittlig AI Score", f"{avg_score:.0f}")
                    col2.metric("Bullish posisjoner", f"{bullish} av {len(scores)}")
                    col3.metric("Bearish posisjoner", f"{bearish} av {len(scores)}")
            else:
                st.info("Klikk knappen over for å kjøre AI-analyse på dine posisjoner.")
    
    # ===========================================
    # TAB 3: LEGG TIL POSISJON
    # ===========================================
    with tab_legg_til:
        st.markdown("### Legg til ny posisjon")
        
        # Initialiser session state for dynamisk oppdatering
        if 'portfolio_current_price' not in st.session_state:
            st.session_state['portfolio_current_price'] = None
        if 'portfolio_confirm_add' not in st.session_state:
            st.session_state['portfolio_confirm_add'] = False
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ny_ticker = st.selectbox(
                "Velg aksje",
                options=unike_tickers,
                format_func=lambda x: f"{x} - {data.ticker_til_navn(x) if hasattr(data, 'ticker_til_navn') else x.replace('.OL', '')}",
                key="portfolio_add_ticker"
            )
            
            # Hent aktuell kurs for valgt ticker
            df_ticker = df_clean[df_clean['Ticker'] == ny_ticker]
            if not df_ticker.empty:
                current_price = float(df_ticker['Close'].iloc[-1])
            else:
                current_price = 100.0
            
            # Oppdater session state når ticker endres
            st.session_state['portfolio_current_price'] = current_price
            
            c1, c2 = st.columns(2)
            antall = c1.number_input("Antall aksjer", min_value=1, value=100, key="portfolio_add_qty")
            # Bruk current_price som default - oppdateres automatisk når ticker endres
            kjopskurs = c2.number_input(
                "Kjøpskurs", 
                min_value=0.01, 
                value=current_price,  # Alltid sist kjente kurs
                format="%.2f", 
                key=f"portfolio_add_price_{ny_ticker}"  # Unik key per ticker for auto-oppdatering
            )
            
            c3, c4 = st.columns(2)
            kjopsdato = c3.date_input("Kjøpsdato", value=pd.Timestamp.now(), key="portfolio_add_date")
            stop_loss = c4.number_input("Stop Loss", min_value=0.01, value=kjopskurs * 0.92, format="%.2f", key=f"portfolio_add_stop_{ny_ticker}")
            
            strategi = st.selectbox(
                "Strategi som trigget kjøpet (valgfritt)",
                options=["Manuelt", "Scanner", "AI Scanner", "Intradag", "Annet"],
                key="portfolio_add_strategy"
            )
            
            notater = st.text_area("Notater (valgfritt)", key="portfolio_add_notes", max_chars=500)
        
        with col2:
            st.markdown("#### Beregning")
            total_investering = antall * kjopskurs
            risiko_kr = antall * (kjopskurs - stop_loss)
            risiko_pct = ((kjopskurs - stop_loss) / kjopskurs) * 100
            
            st.metric("Total investering", f"{total_investering:,.0f} NOK")
            st.metric("Risiko ved stop", f"{risiko_kr:,.0f} NOK", f"-{risiko_pct:.1f}%")
            
            # Vis trailing stop-nivå
            trailing_stop_pct = getattr(config, 'EXIT_TRAILING_STOP_PCT', 8.0)
            trailing_stop_price = kjopskurs * (1 - trailing_stop_pct / 100)
            st.metric("Trailing stop ved", f"{trailing_stop_price:.2f}", f"-{trailing_stop_pct}%")
        
        st.markdown("---")
        
        # Bekreftelsesflyt
        if not st.session_state['portfolio_confirm_add']:
            # Steg 1: Vis "Legg til" knapp
            if st.button("Legg til posisjon", type="primary", key="portfolio_add_btn"):
                st.session_state['portfolio_confirm_add'] = True
                st.session_state['portfolio_pending'] = {
                    'ticker': ny_ticker,
                    'quantity': antall,
                    'buy_price': kjopskurs,
                    'buy_date': kjopsdato.strftime('%Y-%m-%d'),
                    'stop_loss': stop_loss,
                    'notes': notater,
                    'strategy': strategi
                }
                st.rerun()
        else:
            # Steg 2: Vis bekreftelsesdialog
            pending = st.session_state.get('portfolio_pending', {})
            selskap = data.ticker_til_navn(pending['ticker']) if hasattr(data, 'ticker_til_navn') else pending['ticker'].replace('.OL', '')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(38, 166, 154, 0.2) 0%, rgba(38, 166, 154, 0.05) 100%);
                        border: 2px solid rgba(38, 166, 154, 0.5); border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #26a69a; margin-top: 0;">Bekreft posisjon</h4>
                <table style="width: 100%; color: #fff;">
                    <tr><td style="color: #888;">Aksje:</td><td><strong>{pending['ticker']}</strong> ({selskap})</td></tr>
                    <tr><td style="color: #888;">Antall:</td><td><strong>{pending['quantity']}</strong> aksjer</td></tr>
                    <tr><td style="color: #888;">Kjøpskurs:</td><td><strong>{pending['buy_price']:.2f}</strong> NOK</td></tr>
                    <tr><td style="color: #888;">Total:</td><td><strong>{pending['quantity'] * pending['buy_price']:,.0f}</strong> NOK</td></tr>
                    <tr><td style="color: #888;">Stop Loss:</td><td><strong>{pending['stop_loss']:.2f}</strong> NOK</td></tr>
                    <tr><td style="color: #888;">Dato:</td><td>{pending['buy_date']}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            col_confirm, col_cancel = st.columns(2)
            
            with col_confirm:
                if st.button("Bekreft kjøp", type="primary", key="portfolio_confirm_btn", width="stretch"):
                    success, message = portfolio.add_position(
                        ticker=pending['ticker'],
                        quantity=pending['quantity'],
                        buy_price=pending['buy_price'],
                        buy_date=pending['buy_date'],
                        stop_loss=pending['stop_loss'],
                        notes=pending.get('notes', ''),
                        strategy=pending.get('strategy', '')
                    )
                    
                    st.session_state['portfolio_confirm_add'] = False
                    st.session_state['portfolio_pending'] = None
                    
                    if success:
                        st.success(f"{message}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"{message}")
            
            with col_cancel:
                if st.button("Avbryt", key="portfolio_cancel_btn", width="stretch"):
                    st.session_state['portfolio_confirm_add'] = False
                    st.session_state['portfolio_pending'] = None
                    st.rerun()
    
    # ===========================================
    # TAB 4: SELG / JUSTER
    # ===========================================
    with tab_selg:
        if not positions:
            st.info("Du har ingen posisjoner å selge.")
        else:
            st.markdown("### Selg eller juster posisjon")
            
            # Håndter quick-actions fra oversikt
            default_ticker_idx = 0
            default_action_idx = 0
            ticker_list = list(positions.keys())
            
            if st.session_state.get('portfolio_goto_sell'):
                pre_ticker = st.session_state.get('portfolio_sell_ticker')
                if pre_ticker in ticker_list:
                    default_ticker_idx = ticker_list.index(pre_ticker)
                default_action_idx = 0  # Selg posisjon
                st.session_state['portfolio_goto_sell'] = False
            elif st.session_state.get('portfolio_goto_adjust'):
                pre_ticker = st.session_state.get('portfolio_adjust_ticker')
                if pre_ticker in ticker_list:
                    default_ticker_idx = ticker_list.index(pre_ticker)
                default_action_idx = 1  # Juster stop loss
                st.session_state['portfolio_goto_adjust'] = False
            
            selg_ticker = st.selectbox(
                "Velg posisjon",
                options=ticker_list,
                index=default_ticker_idx,
                format_func=lambda x: f"{x} - {positions[x]['quantity']} aksjer @ {positions[x]['avg_price']:.2f}",
                key="portfolio_selg_select"
            )
            
            pos = positions[selg_ticker]
            
            # Hent aktuell kurs
            df_ticker = df_clean[df_clean['Ticker'] == selg_ticker]
            if not df_ticker.empty:
                current_price = float(df_ticker['Close'].iloc[-1])
            else:
                current_price = pos['avg_price']
            
            # Vis posisjoninfo
            profit_pct = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
            profit_total = (current_price - pos['avg_price']) * pos['quantity']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Nåværende kurs", f"{current_price:.2f}")
            col2.metric("Urealisert P/L", f"{profit_total:+,.0f} NOK", f"{profit_pct:+.1f}%")
            col3.metric("Antall", f"{pos['quantity']} aksjer")
            
            st.markdown("---")
            
            # Valg: Selg eller Juster
            action_options = ["Selg posisjon", "Juster stop loss"]
            action = st.radio("Hva vil du gjøre?", action_options, index=default_action_idx, key="portfolio_action", horizontal=True)
            
            if action == "Selg posisjon":
                col1, col2 = st.columns(2)
                
                selg_antall = col1.number_input(
                    "Antall å selge",
                    min_value=1,
                    max_value=pos['quantity'],
                    value=pos['quantity'],
                    key="portfolio_sell_qty"
                )
                
                selg_kurs = col2.number_input(
                    "Salgskurs",
                    min_value=0.01,
                    value=current_price,
                    format="%.2f",
                    key="portfolio_sell_price"
                )
                
                selg_grunn = st.selectbox(
                    "Grunn for salg",
                    options=[
                        "Manuelt salg",
                        "Stop loss trigget",
                        "Trailing stop trigget", 
                        "Profittmål nådd",
                        "Exit-signal fra systemet",
                        "Rebalansering",
                        "Annet"
                    ],
                    key="portfolio_sell_reason"
                )
                
                # Beregn resultat
                selg_resultat = (selg_kurs - pos['avg_price']) * selg_antall
                selg_pct = ((selg_kurs - pos['avg_price']) / pos['avg_price']) * 100
                
                if selg_resultat >= 0:
                    st.success(f"Forventet gevinst: {selg_resultat:+,.0f} NOK ({selg_pct:+.1f}%)")
                else:
                    st.warning(f"Forventet tap: {selg_resultat:,.0f} NOK ({selg_pct:.1f}%)")
                
                if st.button("Bekreft salg", type="primary", key="portfolio_sell_confirm"):
                    success, message, trade_result = portfolio.sell_position(
                        ticker=selg_ticker,
                        quantity=selg_antall,
                        sell_price=selg_kurs,
                        reason=selg_grunn
                    )
                    
                    if success:
                        st.success(f"{message}")
                        if trade_result.get('profit', 0) >= 0:
                            st.balloons()
                        st.rerun()
                    else:
                        st.error(f"{message}")
            
            else:  # Juster stop loss
                st.markdown("#### Juster Stop Loss")
                
                col1, col2 = st.columns(2)
                
                current_stop = pos.get('stop_loss', pos['avg_price'] * 0.92)
                col1.metric("Nåværende stop", f"{current_stop:.2f}")
                
                ny_stop = col2.number_input(
                    "Ny stop loss",
                    min_value=0.01,
                    value=current_stop,
                    format="%.2f",
                    key="portfolio_new_stop"
                )
                
                # Anbefalinger
                st.markdown("##### Anbefalte nivåer")
                c1, c2, c3 = st.columns(3)
                
                trailing_stop_price = pos.get('trailing_high', current_price) * 0.92
                breakeven = pos['avg_price']
                profit_lock = pos.get('trailing_high', current_price) * 0.95
                
                c1.metric("Trailing (8%)", f"{trailing_stop_price:.2f}")
                c2.metric("Breakeven", f"{breakeven:.2f}")
                c3.metric("Lås 5% gevinst", f"{profit_lock:.2f}")
                
                if st.button("Oppdater stop loss", key="portfolio_update_stop"):
                    success, message = portfolio.update_stop_loss(selg_ticker, ny_stop)
                    if success:
                        st.success(f"{message}")
                        st.rerun()
                    else:
                        st.error(f"{message}")
    
    # ===========================================
    # TAB 5: HISTORIKK
    # ===========================================
    with tab_historikk:
        st.markdown("### Transaksjonshistorikk")
        
        history = portfolio.load_transaction_history()
        
        if not history:
            st.info("Ingen transaksjoner registrert ennå.")
        else:
            # Konverter til DataFrame
            hist_df = pd.DataFrame(history)
            
            # Beregn statistikk
            sells = [t for t in history if t.get('type') == 'SELL']
            if sells:
                wins = [t for t in sells if t.get('profit', 0) > 0]
                win_rate = len(wins) / len(sells) * 100 if sells else 0
                total_profit = sum(t.get('profit', 0) for t in sells)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Antall handler", f"{len(sells)}")
                col2.metric("Vinnrate", f"{win_rate:.0f}%")
                col3.metric("Total P/L", f"{total_profit:+,.0f} NOK")
                
                st.markdown("---")
            
            # Vis historikk
            for tx in reversed(history[-20:]):  # Siste 20
                tx_type = tx.get('type', 'UNKNOWN')
                tx_label = "KJOP" if tx_type == "BUY" else "SALG"
                tx_color = "#26a69a" if tx_type == "BUY" else "#ef5350"
                
                profit_str = ""
                if tx_type == "SELL" and 'profit' in tx:
                    profit_color = "#26a69a" if tx['profit'] >= 0 else "#ef5350"
                    profit_str = f"<span style='color: {profit_color}; font-weight: 600;'> → {tx['profit']:+,.0f} NOK ({tx.get('profit_pct', 0):+.1f}%)</span>"
                
                # Formater dato til norsk format
                ts_raw = tx.get('timestamp', '')[:10]
                try:
                    ts_date = pd.to_datetime(ts_raw)
                    ts_formatted = ts_date.strftime('%d.%m.%Y')
                except:
                    ts_formatted = ts_raw
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 10px 15px; border-radius: 8px; margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 600; color: {tx_color};">{tx_label}</span>
                    <span style="font-weight: 600;">{tx.get('ticker', 'N/A')}</span>
                    <span style="color: #888;">|</span>
                    <span>{tx.get('quantity', 0)} @ {tx.get('price', 0):.2f}</span>
                    {profit_str}
                    <span style="color: #666; float: right; font-size: 0.85rem;">{ts_formatted}</span>
                </div>
                """, unsafe_allow_html=True)
