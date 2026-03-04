"""
views/signal_optimizer.py — OsloKjøp Signal Optimizer (Streamlit side).

Viser:
  1. Kjør optimalisering (grid-søk + walk-forward validering)
  2. Beste parametere + out-of-sample ytelse
  3. Egenkapitalkurve (default vs. optimalisert)
  4. Parameter-sensitivitets-tabell
  5. Aktive kjøpssignaler med optimaliserte parametere
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import signal_optimizer as so
from shared_cache import cached_beregn_tekniske_indikatorer
import data
import logic
from log_config import get_logger

_logger = get_logger(__name__)


# ─── Fargekonstanter ────────────────────────────────────────────────────────

_GRØNN  = "#00c853"
_GUL    = "#ffa726"
_RØD    = "#ef5350"
_BLÅ    = "#42a5f5"
_LILLA  = "#ab47bc"


# ─── UI-hjelpere ────────────────────────────────────────────────────────────

def _farge_win_rate(wr: float) -> str:
    if wr >= 60: return _GRØNN
    if wr >= 50: return _GUL
    return _RØD


def _farge_pf(pf: float) -> str:
    if pf >= 2.0: return _GRØNN
    if pf >= 1.0: return _GUL
    return _RØD


def _hl_score(val) -> str:
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ''
    if v >= 55: return 'color: #00c853; font-weight: 700'
    if v >= 40: return 'color: #66bb6a'
    if v >= 25: return 'color: #ffa726'
    return 'color: #ef5350'


def _hl_wr(val) -> str:
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ''
    if v >= 60: return 'color: #00c853; font-weight: 700'
    if v >= 50: return 'color: #ffa726'
    return 'color: #ef5350'


def _hl_ret(val) -> str:
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ''
    if v > 0: return 'color: #66bb6a'
    return 'color: #ef5350'


# ─── Equity-kurve ───────────────────────────────────────────────────────────

def _tegn_equity_kurve(
    eq_default: pd.DataFrame,
    eq_opt: pd.DataFrame,
    title: str = "Kumulativ avkastning",
) -> go.Figure:
    fig = go.Figure()

    if not eq_default.empty:
        fig.add_trace(go.Scatter(
            x=eq_default['dato'],
            y=eq_default['avkastning_pct'],
            mode='lines',
            name='Standard parametere',
            line={'color': _BLÅ, 'width': 2},
        ))

    if not eq_opt.empty:
        fig.add_trace(go.Scatter(
            x=eq_opt['dato'],
            y=eq_opt['avkastning_pct'],
            mode='lines',
            name='Optimaliserte parametere',
            line={'color': _GRØNN, 'width': 2.5},
        ))

    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.3)')

    fig.update_layout(
        title=title,
        xaxis_title='Dato',
        yaxis_title='Kumulativ avkastning (%)',
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': '#fff'},
        legend={'bgcolor': 'rgba(0,0,0,0)'},
        margin={'t': 40, 'b': 30, 'l': 50, 'r': 20},
    )
    return fig


# ─── Metrics-kort ───────────────────────────────────────────────────────────

def _vis_metrics_kort(metrics: dict, tittel: str = "") -> None:
    """Viser nøkkelmetrikk i fire kolonner."""
    if not metrics:
        st.info("Ingen metrikk tilgjengelig.")
        return

    if tittel:
        st.markdown(f"**{tittel}**")

    c1, c2, c3, c4 = st.columns(4)
    wr  = metrics.get('win_rate', 0)
    pf  = metrics.get('profit_factor', 0)
    ret = metrics.get('avg_return', 0)
    n   = metrics.get('n', 0)

    wr_farge  = _farge_win_rate(wr)
    pf_farge  = _farge_pf(pf)
    ret_farge = _GRØNN if ret > 0 else _RØD

    with c1:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px;
                    padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: rgba(255,255,255,0.5); text-transform: uppercase; margin-bottom: 4px;">Win Rate</div>
            <div style="font-size: 26px; font-weight: 700; color: {wr_farge};">{wr:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px;
                    padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: rgba(255,255,255,0.5); text-transform: uppercase; margin-bottom: 4px;">Profit Factor</div>
            <div style="font-size: 26px; font-weight: 700; color: {pf_farge};">{pf:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        sign = '+' if ret > 0 else ''
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px;
                    padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: rgba(255,255,255,0.5); text-transform: uppercase; margin-bottom: 4px;">Snitt avkastning</div>
            <div style="font-size: 26px; font-weight: 700; color: {ret_farge};">{sign}{ret:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border-radius: 10px;
                    padding: 14px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: rgba(255,255,255,0.5); text-transform: uppercase; margin-bottom: 4px;">Antall handler</div>
            <div style="font-size: 26px; font-weight: 700; color: rgba(255,255,255,0.9);">{n}</div>
        </div>""", unsafe_allow_html=True)

    # Ekstra metrikk
    st.caption(
        f"Sharpe: {metrics.get('sharpe', 0):.2f}  •  "
        f"Snitt drawdown: {metrics.get('avg_drawdown', 0):.1f}%  •  "
        f"Snitt holdt: {metrics.get('avg_dager', 0):.0f} dager"
    )


# ─── Aktive signaler ─────────────────────────────────────────────────────────

def _vis_aktive_signaler(
    df_clean: pd.DataFrame,
    unike_tickers: list,
    params: so.OsloKjopParams,
) -> None:
    """Viser aksjer med aktive OsloKjøp-signal i dag."""
    st.markdown("### 📡 Aksjer med aktivt OsloKjøp-signal nå")
    st.caption(f"Parametere: {params}")

    aktive: list[dict] = []

    _hent_navn = data.ticker_til_navn if hasattr(data, 'ticker_til_navn') else lambda t: t.replace('.OL', '')

    progress = st.progress(0.0, text="Søker etter aktive signaler…")
    for i, ticker in enumerate(unike_tickers):
        progress.progress((i + 1) / max(len(unike_tickers), 1),
                          text=f"Sjekker {ticker}…")
        try:
            df_t = df_clean[df_clean['Ticker'] == ticker].copy()
            if len(df_t) < 80:
                continue
            signal = so.generer_oslo_kjop_signal(df_t, params)
            if signal.empty or not signal.iloc[-1]:
                continue
            siste = df_t.iloc[-1]
            aktive.append({
                'Ticker':   ticker,
                'Selskap':  _hent_navn(ticker) or ticker.replace('.OL', ''),
                'Kurs':     round(siste['Close'], 2),
                'RSI':      round(so._beregn_rsi(df_t['Close'], params.rsi_period).iloc[-1], 1),
                'Volum x':  round(siste['Volume'] / df_t['Volume'].rolling(20).mean().iloc[-1], 2)
                            if df_t['Volume'].rolling(20).mean().iloc[-1] > 0 else 0,
            })
        except Exception as exc:
            _logger.debug("Signal-sjekk feilet for %s: %s", ticker, exc)

    progress.empty()

    if aktive:
        st.success(f"**{len(aktive)} aksjer** med aktivt OsloKjøp-signal!")
        df_aktive = pd.DataFrame(aktive)

        def _hl_rsi(val):
            try:
                v = float(val)
                return 'color: #66bb6a' if v < 40 else ''
            except:
                return ''

        st.dataframe(
            df_aktive.style.map(_hl_rsi, subset=['RSI'])
                           .format({'Kurs': '{:.2f}', 'RSI': '{:.1f}', 'Volum x': '{:.2f}'}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Ingen aktive OsloKjøp-signaler akkurat nå.")


# ─── Parameter-sensitivitetstabell ──────────────────────────────────────────

def _vis_sensitivitet(alle_komboer: list[dict]) -> None:
    """Viser de beste parameter-kombinasjonene som tabell."""
    if not alle_komboer:
        return

    st.markdown("### 📊 Parameter-sensitivitet (topp 20)")
    df_k = pd.DataFrame(alle_komboer).head(20)
    df_k = df_k.rename(columns={
        'params':        'Parametere',
        'train_score':   'Score',
        'win_rate':      'Win Rate %',
        'profit_factor': 'Profit Factor',
        'avg_return':    'Snitt Avk %',
        'n_trades':      'Handler',
    })

    st.dataframe(
        df_k.style
            .map(_hl_score,  subset=['Score'])
            .map(_hl_wr,     subset=['Win Rate %'])
            .map(_hl_ret,    subset=['Snitt Avk %'])
            .format({
                'Score':         '{:.1f}',
                'Win Rate %':    '{:.0f}',
                'Profit Factor': '{:.2f}',
                'Snitt Avk %':   '{:+.1f}',
            }),
        hide_index=True,
        use_container_width=True,
    )


# ─── Metodikk-forklaring ─────────────────────────────────────────────────────

def _vis_metodikk() -> None:
    with st.expander("ℹ️ Metodikk — OsloKjøp-optimalisering", expanded=False):
        st.markdown("""
### OsloKjøp — Signallogikk

**OsloKjøp** er et *RSI-pullback i opptrend*-signal med tre bekreftende betingelser:

| Betingelse | Beskrivelse |
|-----------|-------------|
| **Trend** | Pris > trend-SMA (langsiktig opptrend aktiv) |
| **SMA stigende** | trend-SMA > trend-SMA for 20 dager siden (trending opp) |
| **RSI oversold** | RSI(14) < terskel (lokal tilbakefall) |
| **RSI var sunn** | RSI var > 50 i løpet av siste 20 dager (ikke breakdown) |
| **Volum** | Volum > N × 20d gjennomsnitt (institusjonell interesse) |

---

### Optimaliserings-prosessen

```
1. DATA-SPLIT (70/30 kronologisk)
   ─ Trening: første 70% av prishistorien per ticker
   ─ Test:    siste 30% (out-of-sample, ukjent under optimalisering)

2. GRID-SØK på treningsdelen
   ─ 72 kombinasjoner (RSI-terskel × trend-SMA × volum × stop × target)
   ─ Composite score = win_rate×40% + profit_factor×35% + avg_avk×25%

3. VALIDERING på testdelen
   ─ Topp-3 kombinasjoner fra trening testes mot ukjente data
   ─ Vinneren er den med best kombinert train+test score

4. PERSISTENS
   ─ Beste parametere lagres til data_storage/oslo_kjop_params.json
```

### Exit-regler

| Regel | Utløser |
|-------|---------|
| **Profit target** | Close ≥ entry + atr_target × ATR14 |
| **Trailing stop** | Close ≤ peak − atr_stop × ATR14 |
| **Maks tid**      | Etter `holding`-dager |

> Alle kostnader inkludert: 0.05% kurtasje + 0.10% spread/slippage (round-trip).
        """)


# ─── Hoved-render ────────────────────────────────────────────────────────────

def render() -> None:
    """Hoved-render for Signal Optimizer-siden."""

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
                border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;
                border: 1px solid rgba(102,126,234,0.25);">
        <div style="font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 6px;">
            🎯 OsloKjøp Signal Optimizer
        </div>
        <div style="font-size: 13px; color: rgba(255,255,255,0.6); line-height: 1.6;">
            Utvikler, backtester og optimaliserer ett enkelt kjøpssignal for Oslo Børs.
            Grid-søk over 72 parameter-kombinasjoner med walk-forward validering (70/30 split).
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_clean     = st.session_state.get('df_clean', pd.DataFrame())
    unike_tickers = st.session_state.get('unike_tickers', [])

    if df_clean.empty:
        st.error("Ingen data tilgjengelig. Oppdater data fra sidebar.")
        return

    # ── Kontroller ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Innstillinger")
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

    with ctrl1:
        train_frac = st.slider(
            "Treningsandel", 0.55, 0.80, 0.70, 0.05,
            help="Andel historisk data brukt til trening (resten brukes til out-of-sample test)",
        )
    with ctrl2:
        max_tickers = st.slider(
            "Maks tickers", 5, len(unike_tickers), min(40, len(unike_tickers)), 5,
            help="Begrens antall tickers for raskere kjøring",
        )
    with ctrl3:
        st.markdown("""
        <div style="background: rgba(102,126,234,0.08); border-radius: 8px;
                    padding: 10px 14px; margin-top: 4px; font-size: 12px;
                    color: rgba(255,255,255,0.65); border: 1px solid rgba(102,126,234,0.2);">
            <b>72 kombinasjoner</b> testes: RSI-terskel (30/35/40) × Trend-SMA (50/100/200)
            × Volum-faktor (1.5/2.0) × ATR-stop (1.0/1.5) × ATR-target (2.0/3.0)
        </div>
        """, unsafe_allow_html=True)

    # ── Kjør-knapp ───────────────────────────────────────────────────────
    col_knapp, col_status = st.columns([1, 3])
    with col_knapp:
        kjør_opt = st.button(
            "🚀 Kjør optimalisering",
            type="primary",
            use_container_width=True,
        )
    with col_status:
        lagret = so.last_optimale_params()
        if lagret:
            st.caption(
                f"✅ Lagret optimalisering finnes: {lagret[0]}  "
                f"(test-score {lagret[1].get('test_score', '?')})"
            )

    st.markdown("---")

    # ── Kjør grid-søk ────────────────────────────────────────────────────
    if kjør_opt:
        utvalg_tickers = unike_tickers[:max_tickers]
        df_lister: list[pd.DataFrame] = []

        progress = st.progress(0.0, text="Forbereder data…")
        for i, ticker in enumerate(utvalg_tickers):
            progress.progress((i + 0.5) / len(utvalg_tickers),
                              text=f"Laster {ticker}…")
            df_t = df_clean[df_clean['Ticker'] == ticker].copy()
            if len(df_t) >= 100:
                df_lister.append(df_t)

        progress.progress(0.9, text="Kjører grid-søk (kan ta 10–30 sekunder)…")

        try:
            resultat = so.grid_søk(
                df_lister,
                param_grid=so.STANDARD_PARAM_GRID,
                train_frac=train_frac,
            )

            # Lagre
            if resultat['antall_komboer'] > 0:
                so.lagre_optimale_params(
                    resultat['beste_params'],
                    {
                        **resultat['test_metrics'],
                        'train_score': resultat['train_score'],
                        'test_score':  resultat['test_score'],
                        'antall_tickers': resultat['antall_tickers'],
                    },
                )
            st.session_state['opt_resultat'] = resultat
        except Exception as exc:
            st.error(f"Feil under optimalisering: {exc}")
            _logger.exception("Optimalisering feilet")
            progress.empty()
            return

        progress.empty()
        st.success(
            f"✅ Optimalisering fullført!  "
            f"{resultat['antall_komboer']} kombinasjoner testet  •  "
            f"{resultat['antall_tickers']} tickers"
        )

    # ── Vis resultater ───────────────────────────────────────────────────
    opt_resultat = st.session_state.get('opt_resultat')

    # Last fra fil dersom ingen kjøring er gjort ennå
    if opt_resultat is None and lagret:
        params_lagret, metrics_lagret = lagret
        opt_resultat = {
            'beste_params':        params_lagret,
            'train_score':         metrics_lagret.get('train_score', 0),
            'test_score':          metrics_lagret.get('test_score', 0),
            'train_metrics':       {},
            'test_metrics':        metrics_lagret,
            'alle_kombinasjoner':  [],
        }

    if opt_resultat is None:
        st.info("Trykk «Kjør optimalisering» for å starte analysen.")
        _vis_metodikk()
        return

    beste_params = opt_resultat['beste_params']

    # ── 1. Beste parametere ──────────────────────────────────────────────
    st.markdown("### 🏆 Optimale parametere")
    p = beste_params
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(0,200,83,0.08) 0%, rgba(0,230,118,0.04) 100%);
                border: 1px solid rgba(0,200,83,0.25); border-radius: 12px;
                padding: 18px 22px; margin-bottom: 16px;">
        <div style="font-size: 14px; color: rgba(255,255,255,0.9); line-height: 2.0;">
            <b>RSI({p.rsi_period})</b> &lt; <b>{p.rsi_threshold}</b>
            &nbsp;|&nbsp; Trend-SMA: <b>{p.trend_sma}</b>
            &nbsp;|&nbsp; Volum: <b>×{p.vol_factor}</b>
            &nbsp;|&nbsp; Trailing stop: <b>{p.atr_stop}×ATR</b>
            &nbsp;|&nbsp; Profit target: <b>{p.atr_target}×ATR</b>
            &nbsp;|&nbsp; Maks holdt: <b>{p.holding} dager</b>
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: rgba(255,255,255,0.5);">
            Train-score: <b>{opt_resultat['train_score']:.1f}</b>
            &nbsp;&nbsp;•&nbsp;&nbsp;
            Test-score (out-of-sample): <b>{opt_resultat['test_score']:.1f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Ytelsesmetrikk (trening vs. test) ─────────────────────────────
    fane_trening, fane_test = st.tabs(["📚 Treningsresultat", "🧪 Test (out-of-sample)"])

    with fane_trening:
        _vis_metrics_kort(opt_resultat.get('train_metrics', {}), "Treningsperiode (in-sample)")

    with fane_test:
        _vis_metrics_kort(opt_resultat.get('test_metrics', {}), "Testperiode (out-of-sample)")
        if opt_resultat.get('test_metrics'):
            sharpe = opt_resultat['test_metrics'].get('sharpe', 0)
            dd     = opt_resultat['test_metrics'].get('avg_drawdown', 0)
            if sharpe > 0.5:
                st.success(f"✅ Positiv Sharpe ratio: {sharpe:.2f}")
            elif sharpe > 0:
                st.warning(f"⚠️ Lav Sharpe ratio: {sharpe:.2f}")
            else:
                st.error(f"❌ Negativ Sharpe ratio: {sharpe:.2f}")

    st.markdown("---")

    # ── 3. Egenkapitalkurve ──────────────────────────────────────────────
    st.markdown("### 📈 Egenkapitalkurve")

    @st.cache_data(ttl=600, show_spinner=False)
    def _hent_equity_kurver(tickers_tuple, params_str, train_frac_val):
        tickers = list(tickers_tuple)
        df_lister_eq = [
            df_clean[df_clean['Ticker'] == t].copy()
            for t in tickers
            if len(df_clean[df_clean['Ticker'] == t]) >= 80
        ]
        opt_p   = beste_params
        std_p   = so.standard_params()
        eq_opt  = so.beregn_equity_kurve(df_lister_eq, opt_p)
        eq_std  = so.beregn_equity_kurve(df_lister_eq, std_p)
        return eq_opt, eq_std

    with st.spinner("Beregner egenkapitalkurve…"):
        eq_opt, eq_std = _hent_equity_kurver(
            tuple(unike_tickers[:max_tickers]),
            str(beste_params),
            train_frac,
        )

    if not eq_opt.empty or not eq_std.empty:
        fig = _tegn_equity_kurve(eq_std, eq_opt, "OsloKjøp — kumulativ avkastning per trade")
        st.plotly_chart(fig, use_container_width=True)

        kol1, kol2 = st.columns(2)
        if not eq_opt.empty:
            slutt_opt = eq_opt['avkastning_pct'].iloc[-1]
            sign = '+' if slutt_opt > 0 else ''
            with kol1:
                st.metric("🟢 Optimalisert total-avkastning", f"{sign}{slutt_opt:.1f}%")
        if not eq_std.empty:
            slutt_std = eq_std['avkastning_pct'].iloc[-1]
            sign = '+' if slutt_std > 0 else ''
            with kol2:
                st.metric("🔵 Standard parametere", f"{sign}{slutt_std:.1f}%")
    else:
        st.info("Ikke nok handler for egenkapitalkurve.")

    st.markdown("---")

    # ── 4. Parameter-sensitivitet ────────────────────────────────────────
    if opt_resultat.get('alle_kombinasjoner'):
        _vis_sensitivitet(opt_resultat['alle_kombinasjoner'])
        st.markdown("---")

    # ── 5. Aktive signaler nå ────────────────────────────────────────────
    _vis_aktive_signaler(df_clean, unike_tickers, beste_params)

    # ── 6. Metodikk ──────────────────────────────────────────────────────
    _vis_metodikk()
