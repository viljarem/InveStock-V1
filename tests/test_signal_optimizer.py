"""
Tests for signal_optimizer.py — OsloKjøp parameterisert kjøpssignal.

Verifiserer:
- OsloKjopParams oppretting og serialisering
- generer_oslo_kjop_signal: riktig logikk, edge-cases
- _backtest_en_ticker: gyldig statistikk, tilstrekkelige handler
- composite_score: riktig vekting og straff
- walk-forward split: kronologisk og riktig andel
- grid_søk: returnerer beste params + ut-av-sample score
- beregn_equity_kurve: stiger i opptrend, faller i nedtrend
- lagre/last parametere
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import signal_optimizer as so


# ─── Hjelpefunksjoner ────────────────────────────────────────────────────────

def _lag_df(close_arr, noise_pct: float = 0.005, volum: float = 2_000_000.0):
    """Lager OHLCV DataFrame fra close-array med litt støy."""
    c = np.asarray(close_arr, dtype=float)
    n = len(c)
    dates = pd.bdate_range('2020-01-02', periods=n, freq='B')
    h = c * (1 + noise_pct)
    l = c * (1 - noise_pct)
    o = (c + l) / 2
    v = np.full(n, volum)
    return pd.DataFrame(
        {'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v},
        index=dates,
    )


def _lag_opptrend_df(n: int = 300, startpris: float = 80.0) -> pd.DataFrame:
    """Ren lineær opptrend med dobbelt volum (triggerer volum-betingelse)."""
    close = np.linspace(startpris, startpris * 2.0, n)
    df = _lag_df(close, noise_pct=0.005, volum=1_000_000.0)
    # Gi siste ~50 dager høyt volum for å trigge vol-betingelse
    df.iloc[-50:, df.columns.get_loc('Volume')] = 3_000_000.0
    return df


def _lag_nedtrend_df(n: int = 300) -> pd.DataFrame:
    """Ren lineær nedtrend."""
    close = np.linspace(200, 80, n)
    return _lag_df(close)


def _lag_rsi_dip_df(n: int = 300) -> pd.DataFrame:
    """
    Opptrend med en kraftig RSI-dip midtveis.
    Pris stiger 0 → 250 dager, faller litt i dag 200-230 (RSI-dip),
    deretter fortsetter oppover.
    """
    base = np.linspace(100, 180, n)
    # Legg til en tilbakefall i midten
    dip = np.zeros(n)
    dip[190:220] = -12  # trekk fra 12 NOK (gir RSI < 35 på lang nok opptrend)
    close = base + dip
    close = np.maximum(close, 50)
    df = _lag_df(close, noise_pct=0.01)
    # Gi opptrendperioden og dipperioden høyt volum
    df.iloc[195:225, df.columns.get_loc('Volume')] = 3_000_000.0
    return df


# ─── OsloKjopParams ──────────────────────────────────────────────────────────

class TestOsloKjopParams:
    def test_default_verdier(self):
        p = so.OsloKjopParams()
        assert p.rsi_period == 14
        assert p.rsi_threshold == 35.0
        assert p.trend_sma == 100
        assert p.vol_factor == 1.5
        assert p.atr_stop == 1.0
        assert p.atr_target == 2.0
        assert p.holding == 20

    def test_til_dict_returnerer_alle_felter(self):
        p = so.OsloKjopParams(rsi_threshold=30.0, trend_sma=200)
        d = p.til_dict()
        assert d['rsi_threshold'] == 30.0
        assert d['trend_sma'] == 200
        assert 'holding' in d

    def test_fra_dict_runde_tur(self):
        p_original = so.OsloKjopParams(rsi_threshold=40.0, vol_factor=2.0)
        p_gjenopprettet = so.OsloKjopParams.fra_dict(p_original.til_dict())
        assert p_gjenopprettet.rsi_threshold == p_original.rsi_threshold
        assert p_gjenopprettet.vol_factor == p_original.vol_factor

    def test_fra_dict_ignorerer_ukjente_nøkler(self):
        d = so.OsloKjopParams().til_dict()
        d['ukjent_nøkkel'] = 42
        # Skal ikke kaste feil
        p = so.OsloKjopParams.fra_dict(d)
        assert isinstance(p, so.OsloKjopParams)

    def test_str_inneholder_nøkkelverdier(self):
        p = so.OsloKjopParams(rsi_threshold=30, trend_sma=200)
        s = str(p)
        assert '30' in s
        assert '200' in s


# ─── _beregn_rsi ─────────────────────────────────────────────────────────────

class TestBeregnRsi:
    def test_rsi_sterk_opptrend_er_høy(self):
        """50 rette gevinst-dager → RSI godt over 70 etter warm-up."""
        close = pd.Series([100.0 + i for i in range(60)])
        rsi = so._beregn_rsi(close, period=14)
        assert rsi.iloc[-1] > 85

    def test_rsi_sterk_nedtrend_er_lav(self):
        """14 rette tap-dager → RSI nær 0."""
        close = pd.Series([100.0 - i for i in range(30)])
        rsi = so._beregn_rsi(close, period=14)
        assert rsi.iloc[-1] < 15

    def test_rsi_returnerer_series_med_same_index(self):
        close = pd.Series([100.0 + i * 0.5 for i in range(50)])
        rsi = so._beregn_rsi(close, period=14)
        assert len(rsi) == len(close)

    def test_rsi_nan_håndteres_som_50(self):
        """Fillna(50) sikrer ingen NaN i outputen."""
        close = pd.Series([100.0] * 5)  # For kort for RSI
        rsi = so._beregn_rsi(close, period=14)
        assert not rsi.isna().any()


# ─── generer_oslo_kjop_signal ────────────────────────────────────────────────

class TestGenererOsloKjopSignal:
    def test_returnerer_bool_series(self):
        df = _lag_rsi_dip_df(300)
        params = so.OsloKjopParams()
        signal = so.generer_oslo_kjop_signal(df, params)
        assert signal.dtype == bool

    def test_returnerer_false_for_tom_df(self):
        signal = so.generer_oslo_kjop_signal(pd.DataFrame(), so.OsloKjopParams())
        assert len(signal) == 0

    def test_for_kort_df_gir_kun_false(self):
        df = _lag_df([100.0 + i for i in range(50)])
        signal = so.generer_oslo_kjop_signal(df, so.OsloKjopParams())
        assert not signal.any()

    def test_nedtrend_gir_ingen_signaler(self):
        """I ren nedtrend er SMA stigende-betingelse aldri oppfylt."""
        df = _lag_nedtrend_df(300)
        signal = so.generer_oslo_kjop_signal(df, so.OsloKjopParams(trend_sma=50))
        assert not signal.any(), "Forventer ingen signaler i ren nedtrend"

    def test_signal_index_matcher_df_index(self):
        df = _lag_rsi_dip_df(300)
        signal = so.generer_oslo_kjop_signal(df, so.OsloKjopParams())
        assert signal.index.equals(df.index)

    def test_høyere_rsi_terskel_gir_minst_like_mange_signaler(self):
        """Lavere terskel (30) skal gi ≤ signaler enn høyere terskel (40)."""
        df = _lag_rsi_dip_df(300)
        s30 = so.generer_oslo_kjop_signal(df, so.OsloKjopParams(rsi_threshold=30))
        s40 = so.generer_oslo_kjop_signal(df, so.OsloKjopParams(rsi_threshold=40))
        assert s30.sum() <= s40.sum()

    def test_lavere_vol_factor_gir_minst_like_mange_signaler(self):
        """Lavere volum-krav (1.0) gir ≥ signaler enn høyere (2.0)."""
        df = _lag_rsi_dip_df(300)
        s_lav  = so.generer_oslo_kjop_signal(df, so.OsloKjopParams(vol_factor=1.0))
        s_høy  = so.generer_oslo_kjop_signal(df, so.OsloKjopParams(vol_factor=2.0))
        assert s_lav.sum() >= s_høy.sum()


# ─── _backtest_en_ticker ─────────────────────────────────────────────────────

class TestBacktestEnTicker:
    def test_returnerer_none_for_for_kort_df(self):
        df = _lag_df([100.0 + i for i in range(30)])
        assert so._backtest_en_ticker(df, so.OsloKjopParams()) is None

    def test_returnerer_none_for_ingen_signaler(self):
        df = _lag_nedtrend_df(300)
        assert so._backtest_en_ticker(df, so.OsloKjopParams()) is None

    def test_returnerer_dict_med_alle_nøkler(self, df_uptrend):
        # Bruk lav RSI-terskel for å generere signaler på opptrend-data
        params = so.OsloKjopParams(rsi_threshold=70, vol_factor=0.5, trend_sma=50)
        result = so._backtest_en_ticker(df_uptrend, params)
        if result is None:
            pytest.skip("Ingen signaler på denne fixture med disse parameterne")
        forventede = ['n', 'win_rate', 'avg_return', 'profit_factor',
                      'avg_drawdown', 'avg_dager', 'trades']
        for k in forventede:
            assert k in result, f"Mangler nøkkel: {k}"

    def test_win_rate_mellom_0_og_100(self, df_uptrend):
        params = so.OsloKjopParams(rsi_threshold=70, vol_factor=0.5, trend_sma=50)
        result = so._backtest_en_ticker(df_uptrend, params)
        if result is None:
            pytest.skip("Ingen signaler")
        assert 0 <= result['win_rate'] <= 100

    def test_profit_factor_positiv(self, df_uptrend):
        params = so.OsloKjopParams(rsi_threshold=70, vol_factor=0.5, trend_sma=50)
        result = so._backtest_en_ticker(df_uptrend, params)
        if result is None:
            pytest.skip("Ingen signaler")
        assert result['profit_factor'] >= 0

    def test_antall_trades_er_positivt(self, df_uptrend):
        params = so.OsloKjopParams(rsi_threshold=70, vol_factor=0.5, trend_sma=50)
        result = so._backtest_en_ticker(df_uptrend, params)
        if result is None:
            pytest.skip("Ingen signaler")
        assert result['n'] > 0

    def test_trades_liste_finnes(self, df_uptrend):
        params = so.OsloKjopParams(rsi_threshold=70, vol_factor=0.5, trend_sma=50)
        result = so._backtest_en_ticker(df_uptrend, params)
        if result is None:
            pytest.skip("Ingen signaler")
        assert isinstance(result['trades'], list)
        assert len(result['trades']) == result['n']


# ─── composite_score ─────────────────────────────────────────────────────────

class TestCompositeScore:
    def test_scorer_null_for_none(self):
        assert so.composite_score(None) == 0.0

    def test_scorer_null_for_færre_enn_5_handler(self):
        stats = {'n': 4, 'win_rate': 80, 'profit_factor': 3.0, 'avg_return': 5.0, 'avg_drawdown': -2.0}
        assert so.composite_score(stats) == 0.0

    def test_maksimal_score_for_perfekte_stats(self):
        stats = {'n': 50, 'win_rate': 100, 'profit_factor': 3.0, 'avg_return': 20.0, 'avg_drawdown': -1.0}
        score = so.composite_score(stats)
        assert score >= 90.0

    def test_null_score_for_elendig_stats(self):
        stats = {'n': 50, 'win_rate': 0, 'profit_factor': 0, 'avg_return': -20.0, 'avg_drawdown': -2.0}
        assert so.composite_score(stats) == 0.0

    def test_halvering_ved_stor_drawdown(self):
        stats_ok  = {'n': 20, 'win_rate': 60, 'profit_factor': 1.5, 'avg_return': 5.0, 'avg_drawdown': -5.0}
        stats_dd  = {'n': 20, 'win_rate': 60, 'profit_factor': 1.5, 'avg_return': 5.0, 'avg_drawdown': -20.0}
        # Halvering før avrunding: resultatet skal være ca. halvparten
        ratio = so.composite_score(stats_dd) / so.composite_score(stats_ok)
        assert abs(ratio - 0.5) < 0.02

    def test_høyere_win_rate_gir_høyere_score(self):
        def lage(wr): return {'n': 20, 'win_rate': wr, 'profit_factor': 1.5, 'avg_return': 3.0, 'avg_drawdown': -5.0}
        assert so.composite_score(lage(70)) > so.composite_score(lage(50))

    def test_score_innenfor_0_til_100(self):
        stats = {'n': 30, 'win_rate': 65, 'profit_factor': 2.0, 'avg_return': 8.0, 'avg_drawdown': -4.0}
        score = so.composite_score(stats)
        assert 0.0 <= score <= 100.0

    def test_mer_handler_gir_høyere_score_enn_færre(self):
        """Samme ytelse men 30+ handler skal score høyere enn 7 handler."""
        def lage(n): return {'n': n, 'win_rate': 60, 'profit_factor': 2.0,
                             'avg_return': 5.0, 'avg_drawdown': -5.0}
        assert so.composite_score(lage(30)) > so.composite_score(lage(7))

    def test_lav_n_får_redusert_score(self):
        """Færre enn 10 handler skal ha redusert score (×0.60)."""
        stats_lav = {'n': 8,  'win_rate': 60, 'profit_factor': 2.0, 'avg_return': 5.0, 'avg_drawdown': -4.0}
        stats_høy = {'n': 40, 'win_rate': 60, 'profit_factor': 2.0, 'avg_return': 5.0, 'avg_drawdown': -4.0}
        assert so.composite_score(stats_høy) > so.composite_score(stats_lav)

    def test_n30_gir_full_score_uten_sample_straff(self):
        """Med n=30 skal ingen sample-straff anvendes (×1.0)."""
        stats29 = {'n': 29, 'win_rate': 60, 'profit_factor': 2.0, 'avg_return': 5.0, 'avg_drawdown': -4.0}
        stats30 = {'n': 30, 'win_rate': 60, 'profit_factor': 2.0, 'avg_return': 5.0, 'avg_drawdown': -4.0}
        # n=30 skal ha høyere score enn n=29 (29 → ×0.85, 30 → ×1.0)
        assert so.composite_score(stats30) > so.composite_score(stats29)


# ─── _split_walk_forward ─────────────────────────────────────────────────────

class TestSplitWalkForward:
    def test_70_30_split(self, df_uptrend):
        n = len(df_uptrend)
        tr, te = so._split_walk_forward(df_uptrend, 0.70)
        assert len(tr) == int(n * 0.70)
        assert len(te) == n - int(n * 0.70)

    def test_kronologisk_rekkefølge(self, df_uptrend):
        tr, te = so._split_walk_forward(df_uptrend, 0.70)
        assert tr.index[-1] < te.index[0], "Trening skal komme før test kronologisk"

    def test_ingen_overlapp(self, df_uptrend):
        tr, te = so._split_walk_forward(df_uptrend, 0.70)
        felles = tr.index.intersection(te.index)
        assert len(felles) == 0

    def test_50_50_split(self, df_uptrend):
        n = len(df_uptrend)
        tr, te = so._split_walk_forward(df_uptrend, 0.50)
        assert abs(len(tr) - len(te)) <= 1


# ─── _beregn_sharpe ──────────────────────────────────────────────────────────

class TestBeregnSharpe:
    def test_positiv_gjennomsnitt_gir_positiv_sharpe(self):
        returns = [3.0, 2.5, 4.0, 1.5, 3.5, 2.0, 2.8]
        sharpe = so._beregn_sharpe(returns, avg_holding=10)
        assert sharpe > 0

    def test_negativ_gjennomsnitt_gir_negativ_sharpe(self):
        returns = [-3.0, -2.5, -4.0, -1.5, -3.5, -2.0]
        sharpe = so._beregn_sharpe(returns, avg_holding=10)
        assert sharpe < 0

    def test_for_få_returns_gir_null(self):
        assert so._beregn_sharpe([1.0, 2.0, 3.0], avg_holding=10) == 0.0

    def test_konstante_returns_gir_null(self):
        """Null std-avvik → Sharpe = 0."""
        assert so._beregn_sharpe([2.0] * 10, avg_holding=10) == 0.0


# ─── grid_søk ────────────────────────────────────────────────────────────────

class TestGridSok:
    def _lag_df_liste(self, antall: int = 5) -> list[pd.DataFrame]:
        """Lager en liste med syntetiske opptrend-DataFrames."""
        rng = np.random.RandomState(42)
        lister = []
        for _ in range(antall):
            n = 300
            close = 100 * np.cumprod(1 + rng.normal(0.002, 0.01, n))
            # Sett inn en RSI-dip rundt dag 200
            close[185:215] *= np.linspace(1.0, 0.93, 30)
            close = np.maximum(close, 50)
            df = _lag_df(close)
            df.iloc[190:215, df.columns.get_loc('Volume')] = 3_000_000.0
            lister.append(df)
        return lister

    def test_returnerer_dict_med_alle_nøkler(self):
        df_liste = self._lag_df_liste(3)
        # Liten grid for rask test
        liten_grid = {
            'rsi_threshold': [35, 40],
            'trend_sma':     [50, 100],
            'vol_factor':    [1.5],
            'atr_stop':      [1.0],
            'atr_target':    [2.0],
        }
        resultat = so.grid_søk(df_liste, param_grid=liten_grid, min_tickers=1)
        forventede = ['beste_params', 'train_score', 'test_score',
                      'train_metrics', 'test_metrics', 'alle_kombinasjoner']
        for k in forventede:
            assert k in resultat, f"Mangler nøkkel: {k}"

    def test_beste_params_er_oslo_kjop_params(self):
        df_liste = self._lag_df_liste(3)
        liten_grid = {
            'rsi_threshold': [35],
            'trend_sma':     [50],
            'vol_factor':    [1.5],
            'atr_stop':      [1.0],
            'atr_target':    [2.0],
        }
        resultat = so.grid_søk(df_liste, param_grid=liten_grid, min_tickers=1)
        assert isinstance(resultat['beste_params'], so.OsloKjopParams)

    def test_train_score_er_mellom_0_og_100(self):
        df_liste = self._lag_df_liste(3)
        liten_grid = {
            'rsi_threshold': [35, 40],
            'trend_sma':     [50],
            'vol_factor':    [1.5],
            'atr_stop':      [1.0],
            'atr_target':    [2.0],
        }
        resultat = so.grid_søk(df_liste, param_grid=liten_grid, min_tickers=1)
        assert 0 <= resultat['train_score'] <= 100

    def test_tom_liste_gir_tomt_resultat(self):
        resultat = so.grid_søk([], param_grid={'rsi_threshold': [35], 'trend_sma': [50],
                                               'vol_factor': [1.5], 'atr_stop': [1.0], 'atr_target': [2.0]})
        assert resultat['antall_tickers'] == 0

    def test_alle_kombinasjoner_sortert_etter_score(self):
        df_liste = self._lag_df_liste(3)
        liten_grid = {
            'rsi_threshold': [30, 40],
            'trend_sma':     [50, 100],
            'vol_factor':    [1.5],
            'atr_stop':      [1.0],
            'atr_target':    [2.0],
        }
        resultat = so.grid_søk(df_liste, param_grid=liten_grid, min_tickers=1)
        scores = [r['train_score'] for r in resultat['alle_kombinasjoner']]
        assert scores == sorted(scores, reverse=True)


# ─── beregn_equity_kurve ─────────────────────────────────────────────────────

class TestBeregnEquityKurve:
    def _lag_df_med_signaler(self) -> pd.DataFrame:
        """Sterk opptrend med garantert RSI-dip."""
        rng = np.random.RandomState(99)
        n = 300
        close = 100 * np.cumprod(1 + rng.normal(0.003, 0.005, n))
        close[180:210] *= np.linspace(1.0, 0.90, 30)
        close = np.maximum(close, 60)
        df = _lag_df(close)
        df.iloc[185:210, df.columns.get_loc('Volume')] = 3_500_000.0
        return df

    def test_returnerer_dataframe(self):
        df = self._lag_df_med_signaler()
        params = so.OsloKjopParams(rsi_threshold=50, vol_factor=0.5, trend_sma=50)
        eq = so.beregn_equity_kurve([df], params)
        # Returnerer alltid DataFrame (kan være tom om ingen handler)
        assert isinstance(eq, pd.DataFrame)

    def test_kolonnene_finnes(self):
        df = self._lag_df_med_signaler()
        params = so.OsloKjopParams(rsi_threshold=50, vol_factor=0.5, trend_sma=50)
        eq = so.beregn_equity_kurve([df], params)
        if eq.empty:
            pytest.skip("Ingen handler generert")
        for kol in ['dato', 'kapital', 'avkastning_pct']:
            assert kol in eq.columns

    def test_startkapital_er_riktig(self):
        df = self._lag_df_med_signaler()
        params = so.OsloKjopParams(rsi_threshold=50, vol_factor=0.5, trend_sma=50)
        eq = so.beregn_equity_kurve([df], params, startkapital=50_000.0)
        if eq.empty:
            pytest.skip("Ingen handler generert")
        assert eq['kapital'].iloc[0] == 50_000.0

    def test_tom_liste_gir_tom_df(self):
        eq = so.beregn_equity_kurve([], so.OsloKjopParams())
        assert eq.empty

    def test_none_i_liste_ignoreres(self):
        df = self._lag_df_med_signaler()
        params = so.OsloKjopParams(rsi_threshold=50, vol_factor=0.5, trend_sma=50)
        eq_med_none = so.beregn_equity_kurve([None, df, None], params)
        eq_uten_none = so.beregn_equity_kurve([df], params)
        # Samme resultat uavhengig av None
        assert len(eq_med_none) == len(eq_uten_none)


# ─── lagre/last parametere ───────────────────────────────────────────────────

class TestPersistens:
    def test_lagre_og_laste_runde_tur(self, tmp_path, monkeypatch):
        # Pek persistensfilen til en temp-mappe
        tmp_fil = str(tmp_path / "oslo_kjop_params.json")
        monkeypatch.setattr(so, '_PARAMS_FIL', tmp_fil)

        params = so.OsloKjopParams(rsi_threshold=40.0, trend_sma=150, vol_factor=2.0)
        metrics = {'win_rate': 62.5, 'profit_factor': 1.8, 'avg_return': 3.4}

        so.lagre_optimale_params(params, metrics)
        assert os.path.exists(tmp_fil)

        resultat = so.last_optimale_params()
        assert resultat is not None

        p_lastet, m_lastet = resultat
        assert p_lastet.rsi_threshold == params.rsi_threshold
        assert p_lastet.trend_sma == params.trend_sma
        assert m_lastet['win_rate'] == metrics['win_rate']

    def test_last_returnerer_none_hvis_fil_mangler(self, tmp_path, monkeypatch):
        tmp_fil = str(tmp_path / "ikke_eksisterende.json")
        monkeypatch.setattr(so, '_PARAMS_FIL', tmp_fil)
        assert so.last_optimale_params() is None

    def test_skadet_json_returnerer_none(self, tmp_path, monkeypatch):
        tmp_fil = str(tmp_path / "skadet.json")
        monkeypatch.setattr(so, '_PARAMS_FIL', tmp_fil)
        with open(tmp_fil, 'w') as f:
            f.write("ikke gyldig json {{{{")
        assert so.last_optimale_params() is None

    def test_metrics_uten_lister_lagres(self, tmp_path, monkeypatch):
        """Metrics med lister (alle_returns) filtreres ut og lagres ikke."""
        tmp_fil = str(tmp_path / "params_med_lister.json")
        monkeypatch.setattr(so, '_PARAMS_FIL', tmp_fil)
        params = so.OsloKjopParams()
        metrics = {'win_rate': 55.0, 'alle_returns': [1.0, 2.0, -1.5]}
        so.lagre_optimale_params(params, metrics)
        # JSON skal være gyldig
        with open(tmp_fil) as f:
            data = json.load(f)
        # 'alle_returns'-listen skal ikke være i lagret data
        assert 'alle_returns' not in data.get('metrics', {})


# ─── standard_params ─────────────────────────────────────────────────────────

class TestStandardParams:
    def test_returnerer_oslo_kjop_params(self):
        p = so.standard_params()
        assert isinstance(p, so.OsloKjopParams)

    def test_er_ulike_objekter_ved_gjentatte_kall(self):
        p1 = so.standard_params()
        p2 = so.standard_params()
        assert p1 is not p2  # Nye instanser, ikke samme objekt


# ─── TilPinescript ───────────────────────────────────────────────────────────

class TestTilPinescript:
    def test_returnerer_streng(self):
        p = so.OsloKjopParams()
        kode = so.til_pinescript(p)
        assert isinstance(kode, str)
        assert len(kode) > 100

    def test_inneholder_version_5(self):
        kode = so.til_pinescript(so.OsloKjopParams())
        assert "//@version=5" in kode

    def test_parameterverdier_er_i_koden(self):
        p = so.OsloKjopParams(rsi_threshold=30, trend_sma=200, vol_factor=1.5,
                               atr_stop=1.5, atr_target=3.0, holding=20)
        kode = so.til_pinescript(p)
        assert "30" in kode
        assert "200" in kode
        assert "1.5" in kode
        assert "3.0" in kode
        assert "20" in kode

    def test_inneholder_strategi_entry_og_exit(self):
        kode = so.til_pinescript(so.OsloKjopParams())
        assert "strategy.entry" in kode
        assert "strategy.close" in kode

    def test_inneholder_kjopsbetingelsene(self):
        kode = so.til_pinescript(so.OsloKjopParams())
        assert "rsi_val" in kode
        assert "trend_sma" in kode
        assert "vol_sma" in kode

    def test_tilpassede_parametere_reflekteres(self):
        p = so.OsloKjopParams(rsi_threshold=35, trend_sma=100, vol_factor=2.0,
                               atr_stop=1.0, atr_target=2.0, holding=15)
        kode = so.til_pinescript(p)
        assert "35" in kode
        assert "100" in kode
        assert "2.0" in kode
        assert "15" in kode
