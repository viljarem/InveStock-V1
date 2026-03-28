"""
pine_signal.py — Pine Script-inspirert kjøpssignal-motor for Oslo Børs.

Analyserer tekniske indikatorer og gir ett samlet handelssignal, tilsvarende
TradingView sin «Technical Analysis»-sammendrag.

Signal-logikk
─────────────
Hvert subsignal gir +1 (Kjøp), 0 (Nøytral) eller -1 (Selg).
Samlet score = sum(alle subsignaler) / antall subsignaler  →  [-1, +1]

Terskelgrenser (tilsvarende TradingView):
  ≥  0.60  →  Sterkt Kjøp
  ≥  0.20  →  Kjøp
  ≤ -0.60  →  Sterkt Selg
  ≤ -0.20  →  Selg
  ellers   →  Nøytral

Oscillatorer (7):  RSI(14), MACD(12,26,9), Stochastic %K/%D(14,3),
                   CCI(20), Momentum(10), Awesome Oscillator, Bull/Bear Power

Glidende gjennomsnitt (opptil 13):
  EMA 10/20/30/50/100/200, SMA 10/20/30/50/100/150/200
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from log_config import get_logger

logger = get_logger(__name__)

# ── Signal-konstanter ───────────────────────────────────────────────────────

STERKT_KJOP = "Sterkt Kjøp"
KJOP = "Kjøp"
NØYTRAL = "Nøytral"
SELG = "Selg"
STERKT_SELG = "Sterkt Selg"

# Hex-farger for UI-visning
SIGNAL_FARGE: dict[str, str] = {
    STERKT_KJOP: "#00c853",
    KJOP:        "#66bb6a",
    NØYTRAL:     "#ffa726",
    SELG:        "#ef5350",
    STERKT_SELG: "#b71c1c",
}

# Emoji for signal-badges
SIGNAL_EMOJI: dict[str, str] = {
    STERKT_KJOP: "🟢",
    KJOP:        "🟩",
    NØYTRAL:     "🟡",
    SELG:        "🔴",
    STERKT_SELG: "⛔",
}


# ── Subsignal-funksjoner (returnerer +1, 0 eller -1) ───────────────────────

def _rsi_signal(rsi: float) -> int:
    """RSI(14): oversold (<30) = Kjøp, overkjøpt (>70) = Selg."""
    if pd.isna(rsi):
        return 0
    if rsi < 30:
        return 1
    if rsi > 70:
        return -1
    return 0


def _macd_signal(macd: float, signal_line: float) -> int:
    """MACD: macd > signal-linje = Kjøp."""
    if pd.isna(macd) or pd.isna(signal_line):
        return 0
    return 1 if macd > signal_line else -1


def _stoch_signal(k: float, d: float) -> int:
    """Stochastic %K/%D: begge < 20 = Kjøp, begge > 80 = Selg, ellers K vs D."""
    if pd.isna(k) or pd.isna(d):
        return 0
    if k < 20 and d < 20:
        return 1
    if k > 80 and d > 80:
        return -1
    return 1 if k > d else -1 if k < d else 0


def _cci_signal(cci: float) -> int:
    """CCI(20): < -100 = Kjøp, > 100 = Selg."""
    if pd.isna(cci):
        return 0
    if cci < -100:
        return 1
    if cci > 100:
        return -1
    return 0


def _momentum_signal(mom: float) -> int:
    """Momentum(10): positivt = Kjøp, negativt = Selg."""
    if pd.isna(mom):
        return 0
    return 1 if mom > 0 else -1 if mom < 0 else 0


def _ao_signal(ao: float, ao_prev: float) -> int:
    """Awesome Oscillator: AO > 0 og stigende = Kjøp, AO < 0 og fallende = Selg."""
    if pd.isna(ao) or pd.isna(ao_prev):
        return 0
    if ao > 0 and ao > ao_prev:
        return 1
    if ao < 0 and ao < ao_prev:
        return -1
    return 0


def _bull_bear_power_signal(ema13: float, high: float, low: float) -> int:
    """Bull/Bear Power: bull=High-EMA13, bear=Low-EMA13; begge positive = Kjøp."""
    if pd.isna(ema13) or pd.isna(high) or pd.isna(low):
        return 0
    bull_power = high - ema13
    bear_power = low - ema13
    if bull_power > 0 and bear_power > 0:
        return 1
    if bull_power < 0 and bear_power < 0:
        return -1
    return 0


def _ma_signal(close: float, ma: float) -> int:
    """Glidende gjennomsnitt: pris > MA = Kjøp, pris < MA = Selg."""
    if pd.isna(close) or pd.isna(ma):
        return 0
    return 1 if close > ma else -1 if close < ma else 0


# ── Score → signal-streng ───────────────────────────────────────────────────

def _score_til_signal(score: float) -> str:
    """Konverterer normalisert score [-1, +1] til signal-streng."""
    if score >= 0.60:
        return STERKT_KJOP
    if score >= 0.20:
        return KJOP
    if score <= -0.60:
        return STERKT_SELG
    if score <= -0.20:
        return SELG
    return NØYTRAL


# ── Indikator-utvidelse ─────────────────────────────────────────────────────

def utvid_indikatorer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legger til ekstra indikatorer som trengs av ``beregn_pine_signal``
    men som ikke er i ``logic.beregn_tekniske_indikatorer()``.

    Nye kolonner som legges til (kun hvis de ikke allerede finnes):
    - ``Stoch_K``, ``Stoch_D``   — Stochastic %K/%D (14, 3)
    - ``CCI``                    — Commodity Channel Index (20)
    - ``Momentum``               — Close-diff over 10 perioder
    - ``AO``                     — Awesome Oscillator (5 vs 34 midpoint SMA)
    - ``EMA_13``                 — For Bull/Bear Power
    - ``EMA_10/20/30/50/100/200``
    - ``SMA_10/20/30/100``

    Args:
        df: OHLCV DataFrame (allerede prosessert av beregn_tekniske_indikatorer).

    Returns:
        Kopi av df med ekstra indikatorer.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Stochastic %K / %D (14, 3)
    if 'Stoch_K' not in df.columns or 'Stoch_D' not in df.columns:
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        rng = (high14 - low14).replace(0, np.nan)
        df['Stoch_K'] = 100 * (df['Close'] - low14) / rng
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # CCI (20)
    if 'CCI' not in df.columns:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        tp_ma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df['CCI'] = (tp - tp_ma) / (0.015 * mad.replace(0, np.nan))

    # Momentum (10)
    if 'Momentum' not in df.columns:
        df['Momentum'] = df['Close'].diff(10)

    # Awesome Oscillator
    if 'AO' not in df.columns:
        midpoint = (df['High'] + df['Low']) / 2
        df['AO'] = midpoint.rolling(5).mean() - midpoint.rolling(34).mean()

    # EMA-kolonner
    for p in [13, 10, 20, 30, 50, 100, 200]:
        col = f'EMA_{p}'
        if col not in df.columns:
            df[col] = df['Close'].ewm(span=p, adjust=False).mean()

    # SMA-kolonner
    for p in [10, 20, 30, 100]:
        col = f'SMA_{p}'
        if col not in df.columns:
            df[col] = df['Close'].rolling(window=p).mean()

    return df


# ── Hoved-signal-funksjon ───────────────────────────────────────────────────

def beregn_pine_signal(df: pd.DataFrame) -> dict:
    """
    Beregner Pine Script-inspirert sammensatt handelssignal.

    Kombinerer 7 oscillatorer og opptil 13 glidende gjennomsnitt til ett
    signal med kjøp-/selg-teller, delscorer per kategori og detaljert
    subsignal-breakdown.

    Args:
        df: OHLCV DataFrame med indikatorer (fra ``beregn_tekniske_indikatorer``
            + ``utvid_indikatorer``).

    Returns:
        dict med nøklene::

            signal            str   — "Sterkt Kjøp" / "Kjøp" / "Nøytral" / …
            score             float — normalisert [-1, +1]
            kjop              int   — antall subsignaler med verdi +1
            nøytral           int   — antall subsignaler med verdi 0
            selg              int   — antall subsignaler med verdi -1
            oscillator_signal str
            oscillator_score  float
            ma_signal         str
            ma_score          float
            detaljer          dict  — {'oscillatorer': {navn: int}, 'ma': {navn: int}}
    """
    if df is None or df.empty or len(df) < 5:
        return _tomt_signal()

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row

    close = row['Close']

    # ── Oscillatorer (7 subsignaler) ─────────────────────────────────────

    osc_signals: list[int] = []
    detaljer_osc: dict[str, int] = {}

    # 1. RSI(14)
    s = _rsi_signal(row.get('RSI', np.nan))
    osc_signals.append(s); detaljer_osc['RSI(14)'] = s

    # 2. MACD(12,26,9)
    s = _macd_signal(row.get('MACD', np.nan), row.get('MACD_Signal', np.nan))
    osc_signals.append(s); detaljer_osc['MACD(12,26,9)'] = s

    # 3. Stochastic %K/%D(14,3)
    s = _stoch_signal(row.get('Stoch_K', np.nan), row.get('Stoch_D', np.nan))
    osc_signals.append(s); detaljer_osc['Stoch %K/%D(14,3)'] = s

    # 4. CCI(20)
    s = _cci_signal(row.get('CCI', np.nan))
    osc_signals.append(s); detaljer_osc['CCI(20)'] = s

    # 5. Momentum(10)
    s = _momentum_signal(row.get('Momentum', np.nan))
    osc_signals.append(s); detaljer_osc['Momentum(10)'] = s

    # 6. Awesome Oscillator
    s = _ao_signal(row.get('AO', np.nan), prev.get('AO', np.nan))
    osc_signals.append(s); detaljer_osc['Awesome Oscillator'] = s

    # 7. Bull/Bear Power
    s = _bull_bear_power_signal(
        row.get('EMA_13', np.nan), row.get('High', np.nan), row.get('Low', np.nan)
    )
    osc_signals.append(s); detaljer_osc['Bull/Bear Power'] = s

    # ── Glidende gjennomsnitt (opptil 13 subsignaler) ────────────────────

    ma_signals: list[int] = []
    detaljer_ma: dict[str, int] = {}

    _ma_kolonner = [
        'EMA_10', 'EMA_20', 'EMA_30', 'EMA_50', 'EMA_100', 'EMA_200',
        'SMA_10', 'SMA_20', 'SMA_30', 'SMA_50', 'SMA_100', 'SMA_150', 'SMA_200',
    ]
    for col in _ma_kolonner:
        if col in df.columns:
            ma_val = row.get(col, np.nan)
            s = _ma_signal(close, ma_val)
            ma_signals.append(s)
            detaljer_ma[col] = s

    # ── Sammensatte scores ────────────────────────────────────────────────

    def _snitt(sigs: list[int]) -> float:
        return sum(sigs) / len(sigs) if sigs else 0.0

    osc_score = _snitt(osc_signals)
    ma_score  = _snitt(ma_signals)

    all_signals = osc_signals + ma_signals
    total_score = _snitt(all_signals)

    kjop_n    = sum(1 for s in all_signals if s == 1)
    selg_n    = sum(1 for s in all_signals if s == -1)
    nøytral_n = sum(1 for s in all_signals if s == 0)

    return {
        'signal':            _score_til_signal(total_score),
        'score':             round(total_score, 3),
        'kjop':              kjop_n,
        'nøytral':           nøytral_n,
        'selg':              selg_n,
        'oscillator_signal': _score_til_signal(osc_score),
        'oscillator_score':  round(osc_score, 3),
        'ma_signal':         _score_til_signal(ma_score),
        'ma_score':          round(ma_score, 3),
        'detaljer':          {'oscillatorer': detaljer_osc, 'ma': detaljer_ma},
    }


def _tomt_signal() -> dict:
    """Returnerer et tomt/nøytralt signal-dict."""
    return {
        'signal':            NØYTRAL,
        'score':             0.0,
        'kjop':              0,
        'nøytral':           0,
        'selg':              0,
        'oscillator_signal': NØYTRAL,
        'oscillator_score':  0.0,
        'ma_signal':         NØYTRAL,
        'ma_score':          0.0,
        'detaljer':          {'oscillatorer': {}, 'ma': {}},
    }


# ── Batch-skanner ───────────────────────────────────────────────────────────

def skann_pine_signaler(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Skanner alle tickers og returnerer en DataFrame med Pine-signaler.

    Args:
        df_dict: ``{ticker: DataFrame}`` der DataFramene allerede har tekniske
                 indikatorer beregnet (f.eks. via ``beregn_tekniske_indikatorer``).
                 ``utvid_indikatorer`` kjøres automatisk her.

    Returns:
        DataFrame med kolonner::

            Ticker | Signal | Score | Kjøp | Nøytral | Selg
            Oscillator | Osc Score | MA | MA Score

        Sortert etter Score (høyest → lavest).
        Returnerer tom DataFrame hvis ingen tickers.
    """
    resultater: list[dict] = []

    for ticker, df in df_dict.items():
        try:
            df_utv = utvid_indikatorer(df)
            pine = beregn_pine_signal(df_utv)
            resultater.append({
                'Ticker':     ticker,
                'Signal':     pine['signal'],
                'Score':      pine['score'],
                'Kjøp':       pine['kjop'],
                'Nøytral':    pine['nøytral'],
                'Selg':       pine['selg'],
                'Oscillator': pine['oscillator_signal'],
                'Osc Score':  pine['oscillator_score'],
                'MA':         pine['ma_signal'],
                'MA Score':   pine['ma_score'],
            })
        except Exception as exc:  # pragma: no cover
            logger.warning("Pine signal feil for %s: %s", ticker, exc)

    if not resultater:
        return pd.DataFrame()

    result_df = pd.DataFrame(resultater)
    return result_df.sort_values('Score', ascending=False).reset_index(drop=True)
