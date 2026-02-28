# components.py
"""
Gjenbrukbare UI-komponenter og styling-funksjoner for InveStock Pro.
Sentraliserer felles HTML/CSS-blokker og DataFrame-styling som
brukes av flere views.
"""

import streamlit as st


# =============================================================================
# DATAFRAME HIGHLIGHT-FUNKSJONER (brukes av scanner etc.)
# =============================================================================

def highlight_kvalitet(val):
    """Fargekoder kvalitets-kolonnen (🟢A, 🟡B, 🟠C)."""
    if '🟢' in str(val):
        return 'background-color: rgba(0, 200, 5, 0.2)'
    elif '🟡' in str(val):
        return 'background-color: rgba(255, 193, 7, 0.2)'
    elif '🟠' in str(val):
        return 'background-color: rgba(255, 152, 0, 0.2)'
    return ''


def highlight_rs(val):
    """Fargekoder RS-kolonnen (80+ grønn, 60-79 gul, <40 rød)."""
    try:
        v = int(val)
        if v >= 80:
            return 'background-color: rgba(0, 200, 5, 0.2); font-weight: bold'
        elif v >= 60:
            return 'background-color: rgba(255, 193, 7, 0.1)'
        elif v < 40:
            return 'color: #FF5252'
    except (ValueError, TypeError):
        pass
    return ''


def highlight_exit(val):
    """Fargekoder exit-kolonnen (⚠️ = rød bakgrunn)."""
    if val == '⚠️':
        return 'background-color: rgba(255, 82, 82, 0.3); font-weight: bold'
    return ''


def highlight_false_breakout(val):
    """Fargekoder false breakout-kolonnen (❌ = oransje bakgrunn)."""
    if val == '❌':
        return 'background-color: rgba(255, 152, 0, 0.35); font-weight: bold'
    return ''


def highlight_utvikling(val):
    """Fargekoder utvikling-% (grønn positiv, rød negativ)."""
    try:
        v = float(val)
        if v > 0:
            return 'color: #00E676; font-weight: bold'
        elif v < 0:
            return 'color: #FF5252; font-weight: bold'
    except (ValueError, TypeError):
        pass
    return ''


def highlight_rr(val):
    """Fargekoder R:R-ratio (grønn ≥2, gul 1-2, rød <1)."""
    try:
        v = float(val)
        if v >= 3.0:
            return 'color: #00E676; font-weight: bold'
        elif v >= 2.0:
            return 'color: #4caf50'
        elif v >= 1.0:
            return 'color: #ffc107'
        elif v > 0:
            return 'color: #FF5252'
    except (ValueError, TypeError):
        pass
    return ''


def highlight_star(val):
    """Fargekoder watchlist-stjernen."""
    if val == '⭐':
        return 'color: #FFD700; font-weight: bold'
    return 'color: #666'


# =============================================================================
# HTML-RENDERERE (gjenbrukbare markdown-blokker)
# =============================================================================

def render_neon_divider():
    """Renderer en neon-divider (brukes i intradag-modulen)."""
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)


def render_cyber_scan_header(antall: int, min_score: int):
    """Renderer 'SKANN FULLFØRT'-header for intradag-scanner."""
    st.markdown(f"""
    <div class="cyber-header">
        <h3 style="margin:0; color:#00ff88;">🎯 SKANN FULLFØRT</h3>
        <p style="margin:5px 0 0 0; color:#aaa;">Fant {antall} kandidater med score ≥ {min_score}</p>
    </div>
    """, unsafe_allow_html=True)


def render_cyber_stat(value, label: str, color: str = "#00ff88"):
    """Renderer ett cyber-stat-kort (verdi + label)."""
    return f"""
    <div class="cyber-stat">
        <div class="cyber-stat-value" style="color:{color}">{value}</div>
        <div class="cyber-stat-label">{label}</div>
    </div>
    """


def render_score_card(emoji: str, label: str, value, color: str = "#667eea",
                      sublabel: str = "", subvalue: str = ""):
    """Renderer et stilisert stat-kort for resultater."""
    sub_html = ""
    if sublabel:
        sub_html = f"""
        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
            <span style="font-size: 11px; color: rgba(255,255,255,0.6);">{subvalue}</span>
        </div>
        """

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; 
                border: 1px solid rgba(255,255,255,0.1); text-align: center;">
        <div style="font-size: 24px; margin-bottom: 4px;">{emoji}</div>
        <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px;">{label}</div>
        <div style="font-size: 28px; font-weight: 700; color: {color};">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_daily_strat_css(accent: str = "#00ff88"):
    """Renderer CSS for daglige strategi-kort. Kall kun én gang per side.
    
    Args:
        accent: Primærfarge ('#00ff88' for scanner-tab, '#00aaff' for utforsk-tab)
    """
    st.markdown(f"""
    <style>
    .daily-strat-card {{
        background: linear-gradient(135deg, rgba(10,20,40,0.9), rgba(20,30,50,0.9));
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }}
    .daily-strat-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }}
    .strat-trigger {{ 
        color: {accent}; 
        font-weight: bold;
        text-shadow: 0 0 10px {accent}80;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    .strat-no-trigger {{ color: #666; }}
    .strat-criteria {{ font-size: 0.85em; color: #888; margin-top: 8px; }}
    .criteria-ok {{ color: {accent}; text-shadow: 0 0 5px {accent}50; }}
    .criteria-fail {{ color: #ff4444; }}
    .live-indicator {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: {accent};
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1s infinite;
    }}
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.3; }}
    }}
    </style>
    """, unsafe_allow_html=True)


def render_daily_strat_card(strat_info: dict, is_trigger: bool, status: dict, accent_color: str):
    """Renderer ett daglig strategi-kort.
    
    Args:
        strat_info: Dict med 'name', 'description', 'color'
        is_trigger: Om strategien har trigget
        status: Dict med kriterier og verdier
        accent_color: Aksentfarge for kortet
    """
    trigger_class = "strat-trigger" if is_trigger else "strat-no-trigger"
    trigger_icon = "🔥" if is_trigger else "⏳"
    border_color = strat_info['color'] if is_trigger else "#333"

    # Bygg kriterier-tekst
    criteria_html = ""
    for crit_key, crit_val in status.items():
        if isinstance(crit_val, bool):
            icon = "✓" if crit_val else "✗"
            color_class = "criteria-ok" if crit_val else "criteria-fail"
            criteria_html += f"<span class='{color_class}'>{icon} {crit_key}</span> | "
        elif crit_val is not None:
            criteria_html += f"<span style='color:#aaa'>{crit_key}: {crit_val}</span> | "

    st.markdown(f"""
    <div class="daily-strat-card" style="border-left-color: {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: {strat_info['color']}; font-weight: bold;">
                {strat_info['name']}
            </span>
            <span class="{trigger_class}">
                {trigger_icon} {'TRIGGER!' if is_trigger else 'Venter...'}
            </span>
        </div>
        <div style="font-size: 0.8em; color: #666; margin-top: 3px;">
            {strat_info['description']}
        </div>
        <div class="strat-criteria">
            {criteria_html.rstrip(' | ')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_live_indicator(accent: str = "#00ff88"):
    """Renderer LIVE-indikator."""
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span class="live-indicator"></span>
        <span style="color: {accent}; font-weight: bold;">LIVE</span>
        <span style="color: #666; margin-left: 10px;">Sjekker daglige strategisignaler i sanntid...</span>
    </div>
    """, unsafe_allow_html=True)
