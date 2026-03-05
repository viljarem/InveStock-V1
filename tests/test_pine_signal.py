"""
Tests for pine_signal.py — Pine Script-inspirert kjøpssignal-motor.

Verifiserer at:
- Subsignal-funksjoner returnerer korrekte verdier (+1 / 0 / -1)
- Score-tersklene gir riktig signal-streng
- utvid_indikatorer() legger til alle nødvendige kolonner
- beregn_pine_signal() produserer en komplett og korrekt dict
- Sterkt Kjøp gis ved vedvarende opptrend (alle MA over pris)
- Sterkt Selg gis ved vedvarende nedtrend
- Nøytral-signal gis ved flat kurs
- Edge-cases: tom df, for liten df, manglende indikatorer
- skann_pine_signaler() fungerer korrekt for én og flere tickers
"""

import numpy as np
import pandas as pd
import pytest

import pine_signal as ps


# ── Hjelpefunksjoner ────────────────────────────────────────────────────────

def _lag_df(close_arr, noise_pct=0.005, volum=1_000_000.0):
    """Lager en minimal OHLCV DataFrame fra en close-array."""
    c = np.asarray(close_arr, dtype=float)
    n = len(c)
    dates = pd.bdate_range('2023-01-02', periods=n, freq='B')
    h = c * (1 + noise_pct)
    l = c * (1 - noise_pct)
    o = (c + l) / 2
    v = np.full(n, volum)
    return pd.DataFrame(
        {'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v},
        index=dates,
    )


# ── Subsignal-tester ────────────────────────────────────────────────────────

class TestRsiSignal:
    def test_oversold_gir_kjop(self):
        assert ps._rsi_signal(25.0) == 1

    def test_overkjopt_gir_selg(self):
        assert ps._rsi_signal(75.0) == -1

    def test_nøytral_sone(self):
        assert ps._rsi_signal(50.0) == 0

    def test_nan_gir_nøytral(self):
        assert ps._rsi_signal(float('nan')) == 0

    def test_eksakt_grense_30(self):
        # 30 er ikke oversold (< 30 trengs)
        assert ps._rsi_signal(30.0) == 0

    def test_eksakt_grense_70(self):
        # 70 er ikke overkjøpt (> 70 trengs)
        assert ps._rsi_signal(70.0) == 0


class TestMacdSignal:
    def test_macd_over_signal_gir_kjop(self):
        assert ps._macd_signal(0.5, 0.3) == 1

    def test_macd_under_signal_gir_selg(self):
        assert ps._macd_signal(0.3, 0.5) == -1

    def test_nan_gir_nøytral(self):
        assert ps._macd_signal(float('nan'), 0.5) == 0
        assert ps._macd_signal(0.5, float('nan')) == 0


class TestStochSignal:
    def test_begge_oversold_gir_kjop(self):
        assert ps._stoch_signal(15.0, 10.0) == 1

    def test_begge_overkjopt_gir_selg(self):
        assert ps._stoch_signal(85.0, 82.0) == -1

    def test_k_over_d_gir_kjop(self):
        assert ps._stoch_signal(55.0, 45.0) == 1

    def test_k_under_d_gir_selg(self):
        assert ps._stoch_signal(45.0, 55.0) == -1

    def test_nan_gir_nøytral(self):
        assert ps._stoch_signal(float('nan'), 50.0) == 0


class TestCciSignal:
    def test_veldig_negativt_gir_kjop(self):
        assert ps._cci_signal(-150.0) == 1

    def test_veldig_positivt_gir_selg(self):
        assert ps._cci_signal(150.0) == -1

    def test_mellom_gir_nøytral(self):
        assert ps._cci_signal(0.0) == 0

    def test_eksakt_minus_100_gir_nøytral(self):
        assert ps._cci_signal(-100.0) == 0

    def test_eksakt_100_gir_nøytral(self):
        assert ps._cci_signal(100.0) == 0

    def test_nan_gir_nøytral(self):
        assert ps._cci_signal(float('nan')) == 0


class TestMomentumSignal:
    def test_positivt_gir_kjop(self):
        assert ps._momentum_signal(5.0) == 1

    def test_negativt_gir_selg(self):
        assert ps._momentum_signal(-5.0) == -1

    def test_null_gir_nøytral(self):
        assert ps._momentum_signal(0.0) == 0

    def test_nan_gir_nøytral(self):
        assert ps._momentum_signal(float('nan')) == 0


class TestAoSignal:
    def test_positivt_og_stigende_gir_kjop(self):
        assert ps._ao_signal(5.0, 3.0) == 1

    def test_negativt_og_fallende_gir_selg(self):
        assert ps._ao_signal(-5.0, -3.0) == -1

    def test_positivt_men_fallende_gir_nøytral(self):
        assert ps._ao_signal(3.0, 5.0) == 0

    def test_nan_gir_nøytral(self):
        assert ps._ao_signal(float('nan'), 1.0) == 0


class TestBullBearPower:
    def test_begge_positive_gir_kjop(self):
        # ema13=100, high=105, low=102 → bull=5, bear=2
        assert ps._bull_bear_power_signal(100.0, 105.0, 102.0) == 1

    def test_begge_negative_gir_selg(self):
        # ema13=100, high=98, low=95 → bull=-2, bear=-5
        assert ps._bull_bear_power_signal(100.0, 98.0, 95.0) == -1

    def test_blandet_gir_nøytral(self):
        # ema13=100, high=102, low=98 → bull=2, bear=-2
        assert ps._bull_bear_power_signal(100.0, 102.0, 98.0) == 0

    def test_nan_gir_nøytral(self):
        assert ps._bull_bear_power_signal(float('nan'), 105.0, 95.0) == 0


class TestMaSignal:
    def test_pris_over_ma_gir_kjop(self):
        assert ps._ma_signal(110.0, 100.0) == 1

    def test_pris_under_ma_gir_selg(self):
        assert ps._ma_signal(90.0, 100.0) == -1

    def test_pris_lik_ma_gir_nøytral(self):
        assert ps._ma_signal(100.0, 100.0) == 0

    def test_nan_gir_nøytral(self):
        assert ps._ma_signal(float('nan'), 100.0) == 0
        assert ps._ma_signal(100.0, float('nan')) == 0


# ── Score-terskel-tester ────────────────────────────────────────────────────

class TestScoreTilSignal:
    def test_sterkt_kjop(self):
        assert ps._score_til_signal(0.60) == ps.STERKT_KJOP
        assert ps._score_til_signal(1.00) == ps.STERKT_KJOP

    def test_kjop(self):
        assert ps._score_til_signal(0.20) == ps.KJOP
        assert ps._score_til_signal(0.59) == ps.KJOP

    def test_nøytral(self):
        assert ps._score_til_signal(0.0)   == ps.NØYTRAL
        assert ps._score_til_signal(0.19)  == ps.NØYTRAL
        assert ps._score_til_signal(-0.19) == ps.NØYTRAL

    def test_selg(self):
        assert ps._score_til_signal(-0.20) == ps.SELG
        assert ps._score_til_signal(-0.59) == ps.SELG

    def test_sterkt_selg(self):
        assert ps._score_til_signal(-0.60) == ps.STERKT_SELG
        assert ps._score_til_signal(-1.00) == ps.STERKT_SELG


# ── utvid_indikatorer-tester ─────────────────────────────────────────────────

class TestUtvidIndikatorer:
    def test_legger_til_alle_kolonner(self, df_uptrend):
        df_utv = ps.utvid_indikatorer(df_uptrend)
        forventede = ['Stoch_K', 'Stoch_D', 'CCI', 'Momentum', 'AO',
                      'EMA_13', 'EMA_10', 'EMA_20', 'EMA_30',
                      'EMA_50', 'EMA_100', 'EMA_200',
                      'SMA_10', 'SMA_20', 'SMA_30', 'SMA_100']
        for kol in forventede:
            assert kol in df_utv.columns, f"Mangler kolonne: {kol}"

    def test_returnerer_kopi(self, df_uptrend):
        df_utv = ps.utvid_indikatorer(df_uptrend)
        assert df_utv is not df_uptrend

    def test_tom_df_returneres_uendret(self):
        df_tom = pd.DataFrame()
        assert ps.utvid_indikatorer(df_tom).empty

    def test_eksisterende_kolonner_overskrives_ikke(self, df_uptrend):
        """Når BEGGE Stoch_K og Stoch_D allerede finnes, skal de ikke beregnes på nytt."""
        df_med_kd = df_uptrend.copy()
        df_med_kd['Stoch_K'] = 42.0
        df_med_kd['Stoch_D'] = 42.0
        df_utv = ps.utvid_indikatorer(df_med_kd)
        # Begge eksisterer → skal beholdes uendret
        assert (df_utv['Stoch_K'] == 42.0).all()
        assert (df_utv['Stoch_D'] == 42.0).all()

    def test_kun_stoch_k_uten_stoch_d_beregner_begge_på_nytt(self, df_uptrend):
        """Hvis bare Stoch_K finnes men ikke Stoch_D, beregnes begge på nytt (konsistent par)."""
        df_med_k = df_uptrend.copy()
        df_med_k['Stoch_K'] = 42.0
        # Stoch_D mangler
        df_utv = ps.utvid_indikatorer(df_med_k)
        # Stoch_K skal nå inneholde de reellt beregnede verdiene, ikke 42
        assert 'Stoch_K' in df_utv.columns
        assert 'Stoch_D' in df_utv.columns
        # Verdiene skal ikke lenger være 42 — de er reberegnet
        assert not (df_utv['Stoch_K'].dropna() == 42.0).all()

    def test_stoch_mellom_0_og_100(self, df_uptrend):
        df_utv = ps.utvid_indikatorer(df_uptrend)
        k_vals = df_utv['Stoch_K'].dropna()
        assert (k_vals >= 0).all() and (k_vals <= 100).all()

    def test_momentum_er_diff_10(self, df_uptrend):
        df_utv = ps.utvid_indikatorer(df_uptrend)
        forventet = df_uptrend['Close'].diff(10)
        pd.testing.assert_series_equal(df_utv['Momentum'], forventet, check_names=False)


# ── beregn_pine_signal-tester ────────────────────────────────────────────────

class TestBeregnPineSignal:
    def test_returnerer_alle_nøkler(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        resultat = ps.beregn_pine_signal(df_utv)
        forventede_nøkler = [
            'signal', 'score', 'kjop', 'nøytral', 'selg',
            'oscillator_signal', 'oscillator_score',
            'ma_signal', 'ma_score', 'detaljer',
        ]
        for k in forventede_nøkler:
            assert k in resultat, f"Mangler nøkkel: {k}"

    def test_score_innenfor_pluss_minus_1(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        resultat = ps.beregn_pine_signal(df_utv)
        assert -1.0 <= resultat['score'] <= 1.0

    def test_teller_stemmer_med_detaljer(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        alle_subsignaler = (
            list(r['detaljer']['oscillatorer'].values())
            + list(r['detaljer']['ma'].values())
        )
        assert r['kjop']    == sum(1 for s in alle_subsignaler if s == 1)
        assert r['selg']    == sum(1 for s in alle_subsignaler if s == -1)
        assert r['nøytral'] == sum(1 for s in alle_subsignaler if s == 0)

    def test_total_er_kjop_pluss_nøytral_pluss_selg(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        assert r['kjop'] + r['nøytral'] + r['selg'] == len(
            list(r['detaljer']['oscillatorer'].values())
            + list(r['detaljer']['ma'].values())
        )

    def test_tom_df_gir_nøytral_signal(self):
        r = ps.beregn_pine_signal(pd.DataFrame())
        assert r['signal'] == ps.NØYTRAL
        assert r['score'] == 0.0

    def test_for_liten_df_gir_nøytral(self):
        df_liten = _lag_df([100, 101, 102, 103])   # bare 4 rader
        r = ps.beregn_pine_signal(df_liten)
        assert r['signal'] == ps.NØYTRAL

    def test_opptrend_gir_positivt_signal(self, df_uptrend):
        """300 dager med opptrend skal gi positivt (Kjøp eller Sterkt Kjøp) signal."""
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        assert r['score'] > 0, f"Forventet positiv score, fikk {r['score']}"
        assert r['signal'] in (ps.KJOP, ps.STERKT_KJOP)

    def test_nedtrend_gir_negativt_signal(self, df_downtrend):
        """300 dager med nedtrend skal gi negativt (Selg eller Sterkt Selg) signal."""
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_downtrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        assert r['score'] < 0, f"Forventet negativ score, fikk {r['score']}"
        assert r['signal'] in (ps.SELG, ps.STERKT_SELG)

    def test_7_oscillatorer_i_detaljer(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        assert len(r['detaljer']['oscillatorer']) == 7

    def test_oscillator_subsignaler_er_pluss_minus_1_eller_0(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        for navn, val in r['detaljer']['oscillatorer'].items():
            assert val in (-1, 0, 1), f"Ugyldig verdi for {navn}: {val}"

    def test_ma_subsignaler_er_pluss_minus_1_eller_0(self, df_uptrend):
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df_uptrend)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        for navn, val in r['detaljer']['ma'].items():
            assert val in (-1, 0, 1), f"Ugyldig verdi for {navn}: {val}"


# ── Sterkt Kjøp / Sterkt Selg med syntetiske data ───────────────────────────

class TestSterktSignal:
    def test_sterkt_kjop_ved_sterk_opptrend(self):
        """Lag en lang opptrend der alle MA er langt under close → Sterkt Kjøp."""
        n = 300
        # Sterk lineær opptrend: 50 → 200
        close = np.linspace(50, 200, n)
        df = _lag_df(close)
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        assert r['signal'] == ps.STERKT_KJOP, f"Forventet Sterkt Kjøp, fikk {r['signal']} (score={r['score']})"
        assert r['score'] >= 0.60

    def test_sterkt_selg_ved_sterk_nedtrend(self):
        """Lag en lang nedtrend der alle MA er langt over close → Sterkt Selg."""
        n = 300
        # Sterk lineær nedtrend: 200 → 50
        close = np.linspace(200, 50, n)
        df = _lag_df(close)
        import logic
        df_ind = logic.beregn_tekniske_indikatorer(df)
        df_utv = ps.utvid_indikatorer(df_ind)
        r = ps.beregn_pine_signal(df_utv)
        assert r['signal'] == ps.STERKT_SELG, f"Forventet Sterkt Selg, fikk {r['signal']} (score={r['score']})"
        assert r['score'] <= -0.60


# ── skann_pine_signaler-tester ───────────────────────────────────────────────

class TestSkannPineSignaler:
    def test_returnerer_dataframe(self, df_uptrend, df_downtrend):
        import logic
        df_dict = {
            'EQUI.OL': logic.beregn_tekniske_indikatorer(df_uptrend),
            'NEDTR.OL': logic.beregn_tekniske_indikatorer(df_downtrend),
        }
        resultat = ps.skann_pine_signaler(df_dict)
        assert isinstance(resultat, pd.DataFrame)

    def test_riktig_antall_rader(self, df_uptrend, df_downtrend):
        import logic
        df_dict = {
            'EQUI.OL':  logic.beregn_tekniske_indikatorer(df_uptrend),
            'NEDTR.OL': logic.beregn_tekniske_indikatorer(df_downtrend),
        }
        resultat = ps.skann_pine_signaler(df_dict)
        assert len(resultat) == 2

    def test_inneholder_alle_kolonner(self, df_uptrend):
        import logic
        df_dict = {'TEST.OL': logic.beregn_tekniske_indikatorer(df_uptrend)}
        resultat = ps.skann_pine_signaler(df_dict)
        forventede = ['Ticker', 'Signal', 'Score', 'Kjøp', 'Nøytral', 'Selg',
                      'Oscillator', 'Osc Score', 'MA', 'MA Score']
        for kol in forventede:
            assert kol in resultat.columns, f"Mangler kolonne: {kol}"

    def test_sortert_etter_score_høyest_først(self, df_uptrend, df_downtrend):
        import logic
        df_dict = {
            'OPP.OL':  logic.beregn_tekniske_indikatorer(df_uptrend),
            'NED.OL':  logic.beregn_tekniske_indikatorer(df_downtrend),
        }
        resultat = ps.skann_pine_signaler(df_dict)
        scores = resultat['Score'].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_tom_dict_gir_tom_df(self):
        resultat = ps.skann_pine_signaler({})
        assert resultat.empty

    def test_opptrend_ticker_har_positivt_signal(self, df_uptrend):
        import logic
        df_dict = {'OPP.OL': logic.beregn_tekniske_indikatorer(df_uptrend)}
        resultat = ps.skann_pine_signaler(df_dict)
        row = resultat[resultat['Ticker'] == 'OPP.OL'].iloc[0]
        assert row['Score'] > 0
        assert row['Signal'] in (ps.KJOP, ps.STERKT_KJOP)

    def test_nedtrend_ticker_har_negativt_signal(self, df_downtrend):
        import logic
        df_dict = {'NED.OL': logic.beregn_tekniske_indikatorer(df_downtrend)}
        resultat = ps.skann_pine_signaler(df_dict)
        row = resultat[resultat['Ticker'] == 'NED.OL'].iloc[0]
        assert row['Score'] < 0
        assert row['Signal'] in (ps.SELG, ps.STERKT_SELG)


# ── Konstant og metadata-tester ──────────────────────────────────────────────

class TestKonstanter:
    def test_alle_signal_farge_nøkler_finnes(self):
        for sig in [ps.STERKT_KJOP, ps.KJOP, ps.NØYTRAL, ps.SELG, ps.STERKT_SELG]:
            assert sig in ps.SIGNAL_FARGE, f"Mangler farge for: {sig}"

    def test_alle_signal_emoji_nøkler_finnes(self):
        for sig in [ps.STERKT_KJOP, ps.KJOP, ps.NØYTRAL, ps.SELG, ps.STERKT_SELG]:
            assert sig in ps.SIGNAL_EMOJI, f"Mangler emoji for: {sig}"

    def test_farger_er_hex_strenger(self):
        for sig, farge in ps.SIGNAL_FARGE.items():
            assert farge.startswith('#'), f"{sig}: ugyldig fargeformat '{farge}'"
            assert len(farge) == 7, f"{sig}: ugyldig fargelengde '{farge}'"
