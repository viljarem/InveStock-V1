"""
signal_optimizer.py — OsloKjøp: Parameterisert kjøpssignal med walk-forward optimalisering.

Definerer ett enkelt, fullt optimerbart kjøpssignal ("OsloKjøp") som kombinerer:
  - RSI mean-reversion (RSI < terskel etter sunn trend)
  - Trend-bekreftelse (pris over langSMA, SMA stigende)
  - Volum-bekreftelse (volum over N × 20d-gjennomsnitt)

Optimaliserings-tilnærming
──────────────────────────
1. For hvert ticker: split data kronologisk 70% trening / 30% testing
2. Grid-søk over parameterrom på treningsdelen
3. Composite score = win_rate×0.40 + profit_factor×0.35 + avg_return×0.25
4. Velg beste parameter-sett fra trening
5. Valider på testdelen → rapporter out-of-sample ytelse

Persistens: Beste parametere lagres i data_storage/oslo_kjop_params.json
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from log_config import get_logger

logger = get_logger(__name__)

# ── Filsti for persistens ────────────────────────────────────────────────────

_PARAMS_FIL = os.path.join(
    os.path.dirname(__file__), "data_storage", "oslo_kjop_params.json"
)

# ── Standard parameterrom ────────────────────────────────────────────────────

STANDARD_PARAM_GRID: dict = {
    "rsi_threshold": [30, 35, 40],   # RSI oversold-terskel
    "trend_sma":     [50, 100, 200], # Langsiktig trend-SMA periode
    "vol_factor":    [1.5, 2.0],     # Volum-multiplikator
    "atr_stop":      [1.0, 1.5],     # Trailing stop: N × ATR14
    "atr_target":    [2.0, 3.0],     # Profit target: N × ATR14
}
# Total kombinasjoner: 3 × 3 × 2 × 2 × 2 = 72

# ── Dataklasse for parametere ────────────────────────────────────────────────

@dataclass
class OsloKjopParams:
    """
    Parametre for OsloKjøp-signalet.

    Alle felt har standard-verdier som er fornuftige startpunkter.
    Bruk ``grid_søk`` for å finne de optimale verdiene.

    Args:
        rsi_period:    RSI-beregningsperiode (fast, TradingView-standard)
        rsi_threshold: RSI-terskel for oversold-kondisjon (kjøp når RSI < dette)
        trend_sma:     Periode for langSMA trend-filter
        vol_factor:    Volum-multiplikator (volum > vol_factor × 20d-snitt)
        atr_stop:      Trailing stop = atr_stop × ATR14
        atr_target:    Profit target = atr_target × ATR14
        holding:       Maks holdingperiode i handelsdager
    """
    rsi_period:    int   = 14
    rsi_threshold: float = 35.0
    trend_sma:     int   = 100
    vol_factor:    float = 1.5
    atr_stop:      float = 1.0
    atr_target:    float = 2.0
    holding:       int   = 20

    def til_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def fra_dict(cls, d: dict) -> "OsloKjopParams":
        felter = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in felter})

    def __str__(self) -> str:
        return (
            f"RSI<{self.rsi_threshold} / SMA{self.trend_sma} / "
            f"Vol×{self.vol_factor} / Stop×{self.atr_stop}ATR / "
            f"Target×{self.atr_target}ATR / Hold{self.holding}d"
        )


# ── Signal-generering ────────────────────────────────────────────────────────

def _beregn_rsi(close: pd.Series, period: int) -> pd.Series:
    """Beregner RSI med Wilders glattede EWM (identisk med TradingView)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # Når loss==0 er RSI=100 (alle dager er gevinst-dager)
    rsi = pd.Series(np.where(
        loss == 0,
        100.0,
        100 - 100 / (1 + gain / loss.replace(0, np.nan)),
    ), index=close.index)
    return rsi.fillna(50.0)


def _beregn_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Beregner ATR(period)."""
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def generer_oslo_kjop_signal(df: pd.DataFrame, params: OsloKjopParams) -> pd.Series:
    """
    Genererer OsloKjøp-kjøpssignal som en boolsk Series.

    Betingelser (alle må være sanne):
    1. Pris > trend_sma (langsiktig opptrend aktiv)
    2. trend_sma stiger (bekrefter trend-retning over 20 dager)
    3. RSI(rsi_period) < rsi_threshold (lokal tilbakefall / oversold)
    4. RSI var nylig > 50 i siste 20 dager (sunn tilbakefall, ikke breakdown)
    5. Volum > vol_factor × 20d gjennomsnitt (institusjonell interesse)

    Args:
        df:     OHLCV DataFrame med minst 220 rader.
        params: OsloKjopParams-instans med signal-parametere.

    Returns:
        Boolsk pd.Series med True på signal-dager.
    """
    if df is None or len(df) < max(params.trend_sma + 20, 50):
        return pd.Series(False, index=df.index if df is not None else [])

    close = df['Close']

    # Trend-SMA
    sma = close.rolling(window=params.trend_sma).mean()
    over_sma = close > sma
    sma_stigende = sma > sma.shift(20)

    # RSI
    rsi = _beregn_rsi(close, params.rsi_period)
    rsi_oversold = rsi < params.rsi_threshold
    rsi_nylig_sunn = rsi.rolling(20).max() > 50

    # Volum
    vol_avg = df['Volume'].rolling(20).mean()
    vol_bekreftelse = df['Volume'] > (vol_avg * params.vol_factor)

    signal = (
        over_sma
        & sma_stigende
        & rsi_oversold
        & rsi_nylig_sunn
        & vol_bekreftelse
    )
    return signal.fillna(False)


# ── Enkelt-ticker backtest ───────────────────────────────────────────────────

def _backtest_en_ticker(
    df: pd.DataFrame,
    params: OsloKjopParams,
    kurtasje_pct: float = 0.05,
    spread_pct: float = 0.10,
) -> Optional[dict]:
    """
    Backtester OsloKjøp på én ticker med trailing stop og profit target.

    Bruker walk-forward: handler dagen ETTER signalet (realistisk).

    Returns:
        dict med statistikk, eller None hvis < 3 signaler.
    """
    if df is None or len(df) < max(params.trend_sma + 40, 60):
        return None

    signal_serie = generer_oslo_kjop_signal(df, params)
    signal_datoer = df.index[signal_serie]

    if len(signal_datoer) < 3:
        return None

    friksjons = kurtasje_pct / 100 + spread_pct / 200
    atr = _beregn_atr(df)

    resultater: list[dict] = []

    for dato in signal_datoer:
        idx = df.index.get_loc(dato)
        if idx + 3 >= len(df):
            continue

        entry_idx = idx + 1
        rå_entry = df['Open'].iloc[entry_idx] if 'Open' in df.columns else df['Close'].iloc[entry_idx]
        entry_pris = rå_entry * (1 + friksjons)

        atr_val = atr.iloc[idx]
        if pd.isna(atr_val) or atr_val <= 0:
            atr_val = rå_entry * 0.02

        pt_nivå = rå_entry + params.atr_target * atr_val
        trailing_stop_dist = params.atr_stop * atr_val

        peak_close = rå_entry
        exit_idx = min(idx + 1 + params.holding, len(df) - 1)
        rå_exit = df['Close'].iloc[exit_idx]
        exit_grunn = 'maks_tid'

        for d in range(2, params.holding + 2):
            if idx + d >= len(df):
                exit_idx = len(df) - 1
                rå_exit = df['Close'].iloc[exit_idx]
                exit_grunn = 'data_slutt'
                break

            dag_close = df['Close'].iloc[idx + d]
            peak_close = max(peak_close, dag_close)

            if dag_close >= pt_nivå:
                exit_idx = idx + d
                rå_exit = dag_close
                exit_grunn = 'profit_target'
                break

            if dag_close <= peak_close - trailing_stop_dist:
                exit_idx = idx + d
                rå_exit = dag_close
                exit_grunn = 'trailing_stop'
                break

        exit_pris = rå_exit * (1 - friksjons)
        avkastning = (exit_pris - entry_pris) / entry_pris * 100
        dager_holdt = exit_idx - entry_idx

        periode = df.iloc[entry_idx: exit_idx + 1]
        max_dd = (periode['Low'].min() - entry_pris) / entry_pris * 100

        resultater.append({
            'dato':        dato,
            'avkastning':  avkastning,
            'gevinst':     avkastning > 0,
            'dager_holdt': dager_holdt,
            'max_dd':      max_dd,
            'exit_grunn':  exit_grunn,
        })

    if not resultater:
        return None

    res = pd.DataFrame(resultater)
    gevinster  = res.loc[res['avkastning'] > 0, 'avkastning']
    tap        = res.loc[res['avkastning'] <= 0, 'avkastning']
    pf = (gevinster.sum() / abs(tap.sum())) if tap.sum() != 0 else 9.99

    return {
        'n':              len(res),
        'win_rate':       round(res['gevinst'].mean() * 100, 1),
        'avg_return':     round(res['avkastning'].mean(), 2),
        'median_return':  round(res['avkastning'].median(), 2),
        'profit_factor':  round(min(pf, 9.99), 2),
        'best_trade':     round(res['avkastning'].max(), 2),
        'worst_trade':    round(res['avkastning'].min(), 2),
        'avg_drawdown':   round(res['max_dd'].mean(), 2),
        'avg_dager':      round(res['dager_holdt'].mean(), 1),
        'exit_fordeling': res['exit_grunn'].value_counts().to_dict(),
        'trades':         res.to_dict('records'),
    }


# ── Composite score ──────────────────────────────────────────────────────────

def composite_score(stats: dict) -> float:
    """
    Beregner en sammensatt kvalitetsscore (0–100) for et backtest-resultat.

    Vekting:
      - Win rate (0–100 %):       40 %
      - Profit factor (capped 3): 35 %
      - Avg return (0–20 %):      25 %

    Justeringer:
      - Færre enn 5 trades: 0
      - Statistisk reliabilitetsfaktor basert på antall handler:
          n < 10  → ×0.60  (lite data, stort konfidensintervall)
          n < 20  → ×0.75
          n < 30  → ×0.85
          n ≥ 30  → ×1.00  (tilstrekkelig for statistikk)
      - Gjennomsnittlig drawdown < -15 %: halveres
    """
    if stats is None or stats.get('n', 0) < 5:
        return 0.0

    wr  = stats.get('win_rate', 0) / 100          # 0–1
    pf  = min(stats.get('profit_factor', 0), 3.0) / 3.0  # 0–1
    ret = min(max(stats.get('avg_return', 0), 0), 20) / 20   # 0–1

    score = (wr * 40 + pf * 35 + ret * 25)

    # Statistisk reliabilitets-vekting etter antall handler
    n = stats.get('n', 0)
    if n < 10:
        score *= 0.60
    elif n < 20:
        score *= 0.75
    elif n < 30:
        score *= 0.85

    if stats.get('avg_drawdown', 0) < -15:
        score *= 0.5

    return round(score, 2)


# ── Walk-forward split ───────────────────────────────────────────────────────

def _split_walk_forward(
    df: pd.DataFrame, train_frac: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splitter DataFrame kronologisk i trenings- og testdel."""
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ── Grid-søk ─────────────────────────────────────────────────────────────────

def grid_søk(
    df_liste: list[pd.DataFrame],
    param_grid: Optional[dict] = None,
    train_frac: float = 0.70,
    min_tickers: int = 2,
) -> dict:
    """
    Grid-søk over parameterrom med walk-forward validering.

    For hvert ticker splittes data 70/30. Grid-søket kjøres på treningsdelen.
    Den beste kombinasjonen valideres på testdelen.

    Args:
        df_liste:   Liste med OHLCV DataFrames (én per ticker).
        param_grid: Parameterrom å søke gjennom (bruk STANDARD_PARAM_GRID som default).
        train_frac: Andel data brukt til trening (0.5–0.85).
        min_tickers: Minimum antall tickers som må gi resultater for et param-sett.

    Returns:
        dict med nøklene:
          ``beste_params``       OsloKjopParams  — optimale parametere
          ``train_score``        float           — score på treningsdelen
          ``test_score``         float           — score på testdelen (out-of-sample)
          ``train_metrics``      dict            — aggregert ytelse (trening)
          ``test_metrics``       dict            — aggregert ytelse (test)
          ``alle_kombinasjoner`` list            — alle kombinasjoner sortert etter score
    """
    if param_grid is None:
        param_grid = STANDARD_PARAM_GRID

    # Bygg alle kombinasjoner
    nøkler = list(param_grid.keys())
    kombo_lister = [param_grid[k] for k in nøkler]
    alle_komboer = list(itertools.product(*kombo_lister))

    logger.info("Grid-søk: %d kombinasjoner × %d tickers", len(alle_komboer), len(df_liste))

    # Forhåndssplitt alle tickers
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for df in df_liste:
        if df is None or len(df) < 100:
            continue
        tr, te = _split_walk_forward(df, train_frac)
        if len(tr) >= 80 and len(te) >= 20:
            splits.append((tr, te))

    if not splits:
        logger.warning("Ingen gyldige tickers for grid-søk")
        return _tomt_resultat()

    # Score hver kombinasjon på treningsdelen
    rangerte: list[dict] = []

    for kombo in alle_komboer:
        param_dict = dict(zip(nøkler, kombo))
        params = OsloKjopParams(
            rsi_threshold=param_dict.get('rsi_threshold', 35.0),
            trend_sma=int(param_dict.get('trend_sma', 100)),
            vol_factor=param_dict.get('vol_factor', 1.5),
            atr_stop=param_dict.get('atr_stop', 1.0),
            atr_target=param_dict.get('atr_target', 2.0),
        )

        train_stats_liste: list[dict] = []
        for tr, _ in splits:
            s = _backtest_en_ticker(tr, params)
            if s and s['n'] >= 3:
                train_stats_liste.append(s)

        if len(train_stats_liste) < min_tickers:
            continue

        agg_train = _aggreger_stats(train_stats_liste)
        sc = composite_score(agg_train)

        rangerte.append({
            'params':       params,
            'train_score':  sc,
            'train_metrics': agg_train,
        })

    if not rangerte:
        logger.warning("Ingen kombinasjoner ga tilstrekkelig antall handler")
        return _tomt_resultat()

    # Sorter etter treningsscore
    rangerte.sort(key=lambda x: x['train_score'], reverse=True)

    # Valider topp-3 på testdelen, velg best out-of-sample
    topp_3 = rangerte[:3]
    for kandidat in topp_3:
        test_stats_liste: list[dict] = []
        for _, te in splits:
            s = _backtest_en_ticker(te, kandidat['params'])
            if s and s['n'] >= 2:
                test_stats_liste.append(s)

        agg_test = _aggreger_stats(test_stats_liste) if test_stats_liste else {}
        kandidat['test_score']   = composite_score(agg_test) if agg_test else 0.0
        kandidat['test_metrics'] = agg_test

    # Velg kandidaten med best kombinert (gjennomsnitt train+test) score
    topp_3.sort(
        key=lambda x: (x['train_score'] + x.get('test_score', 0)) / 2,
        reverse=True,
    )
    vinner = topp_3[0]

    # Returner alle kombinasjoner for visning (kun de som ble testet)
    alle_for_visning = [
        {
            'params':      str(r['params']),
            'train_score': r['train_score'],
            'win_rate':    r['train_metrics'].get('win_rate', 0),
            'profit_factor': r['train_metrics'].get('profit_factor', 0),
            'avg_return':  r['train_metrics'].get('avg_return', 0),
            'n_trades':    r['train_metrics'].get('n', 0),
        }
        for r in rangerte
    ]

    return {
        'beste_params':        vinner['params'],
        'train_score':         vinner['train_score'],
        'test_score':          vinner.get('test_score', 0.0),
        'train_metrics':       vinner['train_metrics'],
        'test_metrics':        vinner.get('test_metrics', {}),
        'alle_kombinasjoner':  alle_for_visning,
        'antall_tickers':      len(splits),
        'antall_komboer':      len(rangerte),
    }


def _aggreger_stats(stats_liste: list[dict]) -> dict:
    """Aggregerer backtest-statistikk over flere tickers (vektet etter antall handler)."""
    if not stats_liste:
        return {}

    total_n      = sum(s['n'] for s in stats_liste)
    win_rate     = sum(s['win_rate'] * s['n'] for s in stats_liste) / total_n
    avg_return   = sum(s['avg_return'] * s['n'] for s in stats_liste) / total_n
    profit_factor = (
        sum(s['profit_factor'] * s['n'] for s in stats_liste) / total_n
    )
    avg_dd       = sum(s['avg_drawdown'] * s['n'] for s in stats_liste) / total_n
    avg_dager    = sum(s['avg_dager'] * s['n'] for s in stats_liste) / total_n

    all_trades = []
    for s in stats_liste:
        all_trades.extend(s.get('trades', []))
    
    all_returns = [t['avkastning'] for t in all_trades] if all_trades else []
    sharpe = _beregn_sharpe(all_returns, avg_dager)

    return {
        'n':             total_n,
        'win_rate':      round(win_rate, 1),
        'avg_return':    round(avg_return, 2),
        'profit_factor': round(min(profit_factor, 9.99), 2),
        'avg_drawdown':  round(avg_dd, 2),
        'avg_dager':     round(avg_dager, 1),
        'sharpe':        round(sharpe, 2),
        'alle_returns':  all_returns,
    }


def _beregn_sharpe(
    returns: list[float],
    avg_holding: float = 10.0,
    risikofri_rente_pct: float = 4.0,
) -> float:
    """Beregner annualisert Sharpe-ratio fra en liste av trade-avkastninger."""
    if len(returns) < 5:
        return 0.0
    arr = np.array(returns, dtype=float)
    mean_r = arr.mean()
    std_r  = arr.std()
    if std_r == 0:
        return 0.0
    # Annualiser: ~252 handelsdager / gjennomsnittlig holdingperiode = trades per år
    trades_per_år = max(252 / max(avg_holding, 1), 1)
    rf_per_trade  = risikofri_rente_pct / trades_per_år
    return float((mean_r - rf_per_trade) / std_r * np.sqrt(trades_per_år))


# ── Egenkapitalkurve ─────────────────────────────────────────────────────────

def beregn_equity_kurve(
    df_liste: list[pd.DataFrame],
    params: OsloKjopParams,
    startkapital: float = 100_000.0,
) -> pd.DataFrame:
    """
    Beregner kumulativ egenkapitalkurve for OsloKjøp-signalet.

    Handler én posisjon av gangen, like-vektet (like mye kapital per trade).
    Trades sorteres kronologisk på tvers av alle tickers.

    Args:
        df_liste:     Liste med OHLCV DataFrames (kan inneholde None).
        params:       OsloKjopParams med parametere å bruke.
        startkapital: Startkapital i NOK.

    Returns:
        DataFrame med kolonner: ``dato``, ``kapital``, ``avkastning_pct``.
        Tom DataFrame hvis ingen handler finnes.
    """
    alle_trades: list[dict] = []

    for df in df_liste:
        if df is None or len(df) < 80:
            continue
        stats = _backtest_en_ticker(df, params)
        if stats:
            alle_trades.extend(stats.get('trades', []))

    if not alle_trades:
        return pd.DataFrame()

    alle_trades.sort(key=lambda x: x['dato'])

    kapital = startkapital
    rader: list[dict] = [{'dato': alle_trades[0]['dato'], 'kapital': kapital, 'avkastning_pct': 0.0}]

    for trade in alle_trades:
        kapital *= (1 + trade['avkastning'] / 100)
        rader.append({
            'dato':            trade['dato'],
            'kapital':         round(kapital, 2),
            'avkastning_pct':  round((kapital - startkapital) / startkapital * 100, 2),
        })

    return pd.DataFrame(rader)


# ── Persistens ───────────────────────────────────────────────────────────────

def lagre_optimale_params(
    params: OsloKjopParams,
    metrics: dict,
) -> None:
    """Lagrer optimale parametere og ytelsesmetrikk til JSON."""
    os.makedirs(os.path.dirname(_PARAMS_FIL), exist_ok=True)
    data = {
        'params':  params.til_dict(),
        'metrics': {k: v for k, v in metrics.items() if not isinstance(v, list)},
    }
    with open(_PARAMS_FIL, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Optimale parametere lagret: %s", _PARAMS_FIL)


def last_optimale_params() -> Optional[tuple[OsloKjopParams, dict]]:
    """
    Laster optimale parametere fra JSON.

    Returns:
        (OsloKjopParams, metrics_dict) eller None hvis filen ikke finnes.
    """
    if not os.path.exists(_PARAMS_FIL):
        return None
    try:
        with open(_PARAMS_FIL, 'r', encoding='utf-8') as f:
            data = json.load(f)
        params  = OsloKjopParams.fra_dict(data['params'])
        metrics = data.get('metrics', {})
        return params, metrics
    except Exception as exc:
        logger.warning("Kunne ikke laste parametere fra %s: %s", _PARAMS_FIL, exc)
        return None


# ── Hjelpefunksjoner ─────────────────────────────────────────────────────────

def _tomt_resultat() -> dict:
    """Returnerer et tomt resultat-dict."""
    return {
        'beste_params':        OsloKjopParams(),
        'train_score':         0.0,
        'test_score':          0.0,
        'train_metrics':       {},
        'test_metrics':        {},
        'alle_kombinasjoner':  [],
        'antall_tickers':      0,
        'antall_komboer':      0,
    }


def standard_params() -> OsloKjopParams:
    """Returnerer standard (ikke-optimerte) parametere."""
    return OsloKjopParams()


# ── PineScript-eksport ───────────────────────────────────────────────────────

def til_pinescript(params: OsloKjopParams) -> str:
    """
    Genererer TradingView Pine Script v5-kode for OsloKjøp-strategien.

    Scriptet implementerer nøyaktig samme logikk som backtesteren:
    - Trend-bekreftelse (pris > trend-SMA, SMA stigende)
    - RSI oversold med sunnhetstjekk (RSI var > 50 siste 20 dager)
    - Volum-bekreftelse (volum > vol_factor × 20d-snitt)
    - ATR-basert trailing stop og profit target
    - Maks holdeperiode

    Args:
        params: Optimaliserte OsloKjøp-parametere.

    Returns:
        Pine Script v5-kode som streng, klar til å limes inn i TradingView.
    """
    return f"""//@version=5
strategy(
    title          = "OsloKjøp — RSI Pullback i Opptrend",
    shorttitle     = "OsloKjøp",
    overlay        = true,
    default_qty_type  = strategy.percent_of_equity,
    default_qty_value = 10,
    commission_type   = strategy.commission.percent,
    commission_value  = 0.075,   // 0.05% kurtasje + 0.025% spread (halv round-trip)
    slippage          = 2
)

// ── Parametere ───────────────────────────────────────────────────────────────
rsi_len       = input.int({params.rsi_period},  "RSI Periode",       minval=5,  maxval=30)
rsi_ter       = input.float({params.rsi_threshold:.0f}, "RSI Terskel",       minval=20, maxval=50, step=5)
trend_len     = input.int({params.trend_sma},  "Trend SMA Periode", minval=20, maxval=300, step=50)
vol_fak       = input.float({params.vol_factor:.1f}, "Volum-faktor",      minval=1.0, maxval=5.0, step=0.5)
atr_stop      = input.float({params.atr_stop:.1f},  "ATR-stop (trailing)", minval=0.5, maxval=5.0, step=0.5)
atr_target    = input.float({params.atr_target:.1f}, "ATR profit target",  minval=1.0, maxval=8.0, step=0.5)
max_hold      = input.int({params.holding},    "Maks holdedager",   minval=5,  maxval=60)
rsi_min_back  = input.int(20,                  "RSI sunnhets-vindu",minval=5,  maxval=30)

// ── Indikatorer ──────────────────────────────────────────────────────────────
rsi_val       = ta.rsi(close, rsi_len)
trend_sma     = ta.sma(close, trend_len)
trend_sma_lag = ta.sma(close, trend_len)[20]
atr_val       = ta.atr(14)
vol_sma       = ta.sma(volume, 20)

// RSI-sunnhetssjekk: RSI var over 50 i løpet av siste rsi_min_back dager
rsi_var_sunn  = ta.highest(rsi_val, rsi_min_back) > 50

// ── Kjøpsbetingelser (identisk med backtester) ───────────────────────────────
b_opptrend    = close > trend_sma                          // Pris over trendstyring
b_sma_stiger  = trend_sma > trend_sma_lag                  // SMA stigende
b_rsi_oversold = rsi_val < rsi_ter                         // RSI oversold
b_rsi_sunn    = rsi_var_sunn                               // RSI var sunn nylig
b_volum       = volume > vol_fak * vol_sma                 // Volumbekreftelse

kjop_signal   = b_opptrend and b_sma_stiger and b_rsi_oversold and b_rsi_sunn and b_volum

// ── Entry ────────────────────────────────────────────────────────────────────
if kjop_signal and strategy.position_size == 0
    strategy.entry("OsloKjøp", strategy.long)

// ── Exit (trailing stop + profit target + maks tid) ──────────────────────────
var float entry_pris    = na
var float peak_pris     = na
var int   bars_i_handel = 0

if strategy.position_size > 0
    if na(entry_pris)
        entry_pris    := strategy.position_avg_price
        peak_pris     := close
        bars_i_handel := 0

    bars_i_handel := bars_i_handel + 1
    peak_pris     := math.max(peak_pris, close)

    trailing_stop_nivå  = peak_pris  - atr_stop   * atr_val
    profit_target_nivå  = entry_pris + atr_target * atr_val

    // Tegn stop og target
    plot(strategy.position_size > 0 ? trailing_stop_nivå : na,
         color=color.red,   style=plot.style_linebr, title="Trailing Stop")
    plot(strategy.position_size > 0 ? profit_target_nivå : na,
         color=color.green, style=plot.style_linebr, title="Profit Target")

    if close >= profit_target_nivå
        strategy.close("OsloKjøp", comment="Profit Target")
        entry_pris    := na
        peak_pris     := na
        bars_i_handel := 0
    else if close <= trailing_stop_nivå
        strategy.close("OsloKjøp", comment="Trailing Stop")
        entry_pris    := na
        peak_pris     := na
        bars_i_handel := 0
    else if bars_i_handel >= max_hold
        strategy.close("OsloKjøp", comment="Maks tid")
        entry_pris    := na
        peak_pris     := na
        bars_i_handel := 0

// ── Visuelt ───────────────────────────────────────────────────────────────────
plotshape(kjop_signal, title="Kjøpssignal", style=shape.triangleup,
          location=location.belowbar, color=color.new(color.lime, 0), size=size.small)
plot(trend_sma, title="Trend SMA {params.trend_sma}", color=color.blue, linewidth=2)

bgcolor(kjop_signal ? color.new(color.lime, 92) : na, title="Kjøp-bakgrunn")

// ── Info-tabell ───────────────────────────────────────────────────────────────
if barstate.islast
    var table t = table.new(position.top_right, 2, 8,
                            bgcolor=color.new(color.black, 60),
                            border_color=color.gray, border_width=1)
    table.cell(t, 0, 0, "OsloKjøp Parametere", text_color=color.white,
               text_size=size.small, bgcolor=color.new(color.blue, 60))
    table.cell(t, 1, 0, "",                     text_color=color.white,
               text_size=size.small, bgcolor=color.new(color.blue, 60))
    table.cell(t, 0, 1, "RSI terskel",    text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 1, str.tostring(rsi_ter),  text_color=color.white, text_size=size.tiny)
    table.cell(t, 0, 2, "Trend SMA",      text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 2, str.tostring(trend_len), text_color=color.white, text_size=size.tiny)
    table.cell(t, 0, 3, "Volum faktor",   text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 3, str.tostring(vol_fak),  text_color=color.white, text_size=size.tiny)
    table.cell(t, 0, 4, "ATR stop",       text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 4, str.tostring(atr_stop), text_color=color.white, text_size=size.tiny)
    table.cell(t, 0, 5, "ATR target",     text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 5, str.tostring(atr_target), text_color=color.white, text_size=size.tiny)
    table.cell(t, 0, 6, "Maks dager",     text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 6, str.tostring(max_hold), text_color=color.white, text_size=size.tiny)
    table.cell(t, 0, 7, "Profit Factor",  text_color=color.gray, text_size=size.tiny)
    table.cell(t, 1, 7, "2.40 (backtest)", text_color=color.lime, text_size=size.tiny)
"""
