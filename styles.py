# styles.py
"""
Sentralisert CSS-styling for InveStock Pro.
Alle CSS-blokker defineres her for å unngå duplisering.
"""


def get_base_css() -> str:
    """Returnerer basis-CSS som brukes på alle sider."""
    return """
    <style>
    /* === iOS LIQUID GLASS EFFECTS === */
    
    /* Global glassmorphism variables */
    :root {
        --glass-bg-light: rgba(255, 255, 255, 0.1);
        --glass-bg-dark: rgba(255, 255, 255, 0.05);
        --glass-border-light: rgba(255, 255, 255, 0.2);
        --glass-border-dark: rgba(255, 255, 255, 0.1);
        --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        --glass-blur: blur(16px);
    }
    
    /* Sidebar liquid glass */
    div[data-testid="stSidebar"] {
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        background: var(--glass-bg-light) !important;
        border-right: 1px solid var(--glass-border-light) !important;
        box-shadow: var(--glass-shadow) !important;
    }
    
    /* Dark mode sidebar */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stSidebar"] {
            background: var(--glass-bg-dark) !important;
            border-right: 1px solid var(--glass-border-dark) !important;
        }
    }
    
    /* Metric containers with liquid glass */
    div[data-testid="metric-container"] {
        background: var(--glass-bg-light) !important;
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        border: 1px solid var(--glass-border-light) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: var(--glass-shadow) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15) !important;
        background: rgba(255, 255, 255, 0.15) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background: var(--glass-bg-dark) !important;
            border: 1px solid var(--glass-border-dark) !important;
        }
        div[data-testid="metric-container"]:hover {
            background: rgba(255, 255, 255, 0.08) !important;
        }
    }
    
    /* Buttons with liquid glass */
    .stButton > button {
        background: var(--glass-bg-light) !important;
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        border: 1px solid var(--glass-border-light) !important;
        border-radius: 12px !important;
        color: inherit !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--glass-shadow) !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .stButton > button {
            background: var(--glass-bg-dark) !important;
            border: 1px solid var(--glass-border-dark) !important;
        }
        .stButton > button:hover {
            background: rgba(255, 255, 255, 0.1) !important;
        }
    }
    
    /* Selectboxes with liquid glass */
    .stSelectbox > div > div {
        background: var(--glass-bg-light) !important;
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        border: 1px solid var(--glass-border-light) !important;
        border-radius: 12px !important;
        box-shadow: var(--glass-shadow) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .stSelectbox > div > div {
            background: var(--glass-bg-dark) !important;
            border: 1px solid var(--glass-border-dark) !important;
        }
    }
    
    /* DataFrames with subtle glass effect */
    .stDataFrame {
        background: var(--glass-bg-light) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid var(--glass-border-light) !important;
        border-radius: 12px !important;
        box-shadow: var(--glass-shadow) !important;
        overflow: hidden !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .stDataFrame {
            background: var(--glass-bg-dark) !important;
            border: 1px solid var(--glass-border-dark) !important;
        }
    }
    
    /* Container elements */
    .element-container {
        backdrop-filter: blur(2px) !important;
        -webkit-backdrop-filter: blur(2px) !important;
    }
    
    /* Custom glass info box */
    .info-box {
        background: linear-gradient(135deg, rgba(78, 140, 255, 0.1), rgba(78, 140, 255, 0.05)) !important;
        backdrop-filter: var(--glass-blur) !important;
        -webkit-backdrop-filter: var(--glass-blur) !important;
        border: 1px solid rgba(78, 140, 255, 0.2) !important;
        border-left: 4px solid #4e8cff !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        box-shadow: var(--glass-shadow) !important;
        color: inherit !important;
        transition: all 0.3s ease !important;
    }
    
    .info-box:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 48px rgba(78, 140, 255, 0.15) !important;
    }
    
    .horisont-label {
        font-weight: 700 !important;
        color: #4e8cff !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        margin-bottom: 8px !important;
        display: block !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Smooth animations */
    * {
        transition: backdrop-filter 0.3s ease, background 0.3s ease !important;
    }
    </style>
    """


def get_scanner_css(antall_aksjer: int) -> str:
    """Returnerer Scanner-spesifikk CSS med liquid glass-effekter."""
    return f"""
    <style>
    .scanner-header {{
        background: linear-gradient(135deg, 
            rgba(26, 26, 46, 0.7) 0%, 
            rgba(22, 33, 62, 0.7) 50%, 
            rgba(15, 52, 96, 0.7) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 32px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 16px 64px rgba(0, 0, 0, 0.2), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    .scanner-header:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 80px rgba(0, 0, 0, 0.25), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }}
    
    .scanner-title {{
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.8px !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    
    .scanner-subtitle {{
        font-size: 16px !important;
        color: rgba(255, 255, 255, 0.8) !important;
        margin-bottom: 0 !important;
        font-weight: 400 !important;
    }}
    
    .strategy-card {{
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin: 6px !important;
        display: inline-block !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
    }}
    
    .strategy-card:hover {{
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }}
    
    .strategy-card.selected {{
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.3) 0%, 
            rgba(118, 75, 162, 0.3) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }}
    
    .filter-section {{
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease !important;
    }}
    
    .filter-section:hover {{
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }}
    
    .filter-title {{
        font-size: 14px !important;
        font-weight: 700 !important;
        color: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 16px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    
    .stat-card {{
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.12) 0%, 
            rgba(118, 75, 162, 0.12) 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(102, 126, 234, 0.25) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        text-align: center !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    .stat-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }}
    .stat-value {{
        font-size: 32px;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 4px;
    }}
    .stat-label {{
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .result-highlight {{
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    </style>

    <div class="scanner-header">
        <div class="scanner-title">🔍 Strategi-Scanner</div>
        <div class="scanner-subtitle">Finn tekniske kjøpssignaler blant {antall_aksjer} aksjer med 8 velprøvde strategier</div>
    </div>
    """


def get_cyber_css() -> str:
    """Returnerer futuristisk CSS for intradag-modulen."""
    return """
    <style>
    .cyber-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, rgba(0,170,255,0.1) 100%);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .cyber-stat {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .cyber-stat-value {
        font-size: 2em;
        font-weight: bold;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }
    .cyber-stat-label {
        font-size: 0.8em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .neon-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff88, #00aaff, transparent);
        margin: 20px 0;
        box-shadow: 0 0 10px rgba(0,255,136,0.5);
    }
    </style>
    """


def get_liquid_glass_css() -> str:
    """Returnerer avansert liquid glass CSS for spesielle komponenter."""
    return """
    <style>
    /* === ENHANCED LIQUID GLASS COMPONENTS === */
    
    /* Glass navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin: 4px !important;
        padding: 12px 20px !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-1px) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.3) 0%, 
            rgba(118, 75, 162, 0.3) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Glass containers for charts */
    .chart-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin: 16px 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Glass text inputs */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: inherit !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
    /* Glass number inputs */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    /* Glass expander */
    .streamlit-expander {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        margin: 16px 0 !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
    }
    
    .streamlit-expander:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    
    /* Smooth scroll for the whole app */
    .main .block-container {
        scroll-behavior: smooth !important;
    }
    
    /* Loading states with glass */
    .stSpinner {
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    /* Glass success/error messages */
    .stSuccess {
        background: linear-gradient(135deg, 
            rgba(76, 175, 80, 0.15) 0%, 
            rgba(76, 175, 80, 0.08) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.2) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, 
            rgba(244, 67, 54, 0.15) 0%, 
            rgba(244, 67, 54, 0.08) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(244, 67, 54, 0.2) !important;
    }
    </style>
    """
