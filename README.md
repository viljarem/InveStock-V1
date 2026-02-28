# InveStock Pro v2.0

**Avansert teknisk analyse for Oslo Bors**

InveStock Pro er et profesjonelt Streamlit-basert analyseverktoy for aksjer pa Oslo Bors. Appen kombinerer klassiske tekniske strategier med maskinlaering, markedsregime-analyse (HMM), Smart Money Flow og innsidehandel-overvaking.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/Tests-319_passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Hurtigstart

```bash
cd "InveStock DEV"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Appen apnes automatisk i nettleseren pa `http://localhost:8501`.

Ved oppstart lastes data automatisk fra Yahoo Finance. Ga til **Scanner** for a finne aksjer med aktive signaler, eller klikk pa en ticker for detaljert **Teknisk Analyse**.

---

## Oversikt over faner

### Markedstemperatur

Overordnet markedsbilde med regime-analyse og bredde-indikatorer.

- **Regime-analyse (Hidden Markov Model):** Klassifiserer markedet i regimer (Bull, Bear, Volatilt, Stabilt, Korreksjon) med sanntids-sannsynligheter og handlingsanbefaling.
- **Markedsbredde:** Andel aksjer over SMA 200/50, nye 52-ukers hoy/lav, Advance/Decline-ratio, kumulativ A/D-linje, McClellan Oscillator.
- **Makroindikatorer:** Brent crude-korrelasjon med OSEBX, USD/NOK-paring.
- **Historikk:** 6-maneders bredde-chart med tre paneler (SMA-bredde, A/D-linje, McClellan).

### Scanner

Skanner Oslo Bors for aksjer som oppfyller kriteriene til 8 tekniske strategier.

**Strategier:** RSI Mean Reversion, Momentum Burst, Golden Cross, Ichimoku Breakout, Wyckoff Spring, Bull Race Prep, VCP (Minervini), Pocket Pivot.

**Resultat-kolonner:**

| Kolonne | Beskrivelse |
|---------|-------------|
| Ticker  | Aksjens ticker-symbol |
| Signal  | Dato for siste kjopssignal |
| Dager   | Handelsdager siden signalet |
| Pris    | Siste sluttkurs |
| Utv%    | Kursutvikling siden signalet |
| Score   | Signalkvalitet (A-D) basert pa walk-forward backtesting |
| RS      | IBD Relativ Styrke vs. OSEBX (1-99) |
| MTF     | Multi-timeframe konvergens |
| Exit    | Markeres hvis exit-signal er aktivt |

**Valgfrie tillegg:**
- Fundamentalanalyse (P/E, gjeld, inntjening)
- Smart Money Flow (institusjonell akkumulering/distribusjon)
- Innsidehandel (kjop/salg fra primaerinsidere, beta)
- Pattern Vision (AI-basert monsterdetektion med YOLO)
- Regime-filter (tilpasser signalkrav til markedsregimet)
- Sektor-konsentrasjonsfilter

**To faner i resultatvisningen:**
1. Alle treff med klikkbar navigering til Teknisk Analyse
2. Watchlist med P/L-oversikt og gjennomsnittlig avkastning

**Historisk treffsikkerhet:** Walk-forward backtesting av hver strategi med win rate og snittavkastning.

### Teknisk Analyse

Detaljert analyse av enkeltticker med TradingView-inspirerte charts.

**Chart-funksjoner:**
- Graftyper: Candlestick, OHLC, Linje, Area
- Tidsperioder: 3 maneder til 5 ar
- Morkt og lyst tema
- Forhandsdefinerte indikatoroppsett (Swing, Trend, Intraday m.fl.)

**Tilgjengelige indikatorer:**

| Kategori         | Indikatorer |
|------------------|-------------|
| Glidende snitt   | SMA (10, 20, 50, 100, 150, 200), EMA (9, 12, 21, 26) |
| Volatilitet      | Bollinger Bands, Keltner Channel, Donchian Channel |
| Oscillatorer     | RSI, Stochastic, MACD, CCI, OBV |
| Volum            | Volumsoyler med glidende snitt |
| Andre            | VWAP, ATR, stotte/motstand-linjer |

**Utvidede analysemoduler (ekspanderbare):**
- Smart Money Flow -- CMF, OBV, intradag-SMI, divergensanalyse
- Innsidehandel -- siste handler, roller, belop, score med markorer pa chart
- Pattern Vision -- YOLO-basert monsterdetektion og formasjonsscanner
- Kelly Criterion -- optimal posisjonsstorrelse basert pa din trading-statistikk
- Posisjonskalkulator med ATR-basert stop-loss

### Beta-moduler

Eksperimentelle funksjoner under aktiv utvikling.

**Intradagmodul:**
- Scanner for samme-dag momentum basert pa intradag-data (5m, 15m, 30m, 60m)
- Scorer aksjer pa trend, VWAP, breakout og volum
- Utforsk-fane for detaljert intradag-chart
- Daglige strategier med live triggere

**AI Scanner:**
- XGBoost ensemble-modell med 30+ features
- Regime-justert scoring, fundamental- og insider-integrasjon
- Prediktiv signalanalyse med konfidensnivaer

**Pattern Vision:**
- YOLO-basert monsterdetektion pa kursgrafer (15+ monstre)
- Formation Scanner -- finn monstre under utvikling for entry for breakout
- Historisk monstertreffsikkerhet
- Full skanning av hele universet

---

## Handelsstrategier

| Horisont | Strategi | Beskrivelse |
| :--- | :--- | :--- |
| Kort (1-5 d) | RSI Mean Reversion | Pris over SMA 200, RSI under 30, var over 50 siste 10 dager |
| Kort (1-10 d) | Momentum Burst | Breakout over 20-dagers hoy med volum 2x+ snitt |
| Kort (1-10 d) | Bull Race Prep | Bollinger squeeze + breakout over ovre band |
| Kort-mellom (3-20 d) | Wyckoff Spring | Falskt brudd under stotte som lukker over |
| Kort-mellom (5-20 d) | Pocket Pivot | Volum pa opp-dag overstiger hoyeste ned-dag volum |
| Mellom (2-8 u) | Ichimoku Breakout | Pris bryter opp gjennom Kumo-skyen |
| Mellom (2-8 u) | VCP (Minervini) | Stage 2 opptrend med krympende volatilitet |
| Lang (3-12 mnd) | Golden Cross | SMA 50 krysser over SMA 200 med stigende snitt |

---

## Exit-signaler

Et exit-signal utloses nar **2 eller flere** av folgende kriterier er oppfylt:

| Signal | Kriterium |
|--------|-----------|
| Death Cross | SMA 50 krysser under SMA 200 |
| RSI reversal | RSI faller fra overkjopt niva (>70) |
| SMA 50 break | Pris bryter under SMA 50 med hoyt volum |
| MACD bearish | MACD krysser under signallinjen |
| Drawdown | Ned mer enn 7 % fra 20-dagers topp |
| Bollinger break | Pris under nedre Bollinger Band |

---

## Signalkvalitet

Signaler scores basert pa walk-forward backtesting:

| Grade | Score | Betydning |
|-------|-------|-----------|
| A | 60+ | Hoy kvalitet -- historisk god treffsikkerhet |
| B | 45-59 | Middels-hoy kvalitet |
| C | 35-44 | Lav kvalitet -- vaer forsiktig |
| D | <35 | Unnga -- historisk svak |

---

## Prosjektstruktur

```
InveStock DEV/
|-- app.py                    Hovedapplikasjon (Streamlit)
|-- data.py                   Datahenting (Yahoo Finance) og validering
|-- logic.py                  Tekniske strategier, indikatorer og exit-signaler
|-- chart_utils.py            TradingView-inspirerte charts (Plotly)
|-- config.py                 Konfigurasjon og parametere
|-- regime_model.py           Markedsregime-analyse (HMM)
|-- beta_ml.py                AI/ML-scanner (XGBoost ensemble)
|-- smart_money.py            Smart Money Flow-analyse (CMF, OBV, SMI)
|-- insider_monitor.py        Innsidehandel-overvaking (Oslo Bors Newsweb)
|-- fundamental_data.py       Fundamental datainnhenting og scoring
|-- shared_cache.py           Delt caching for ytelse
|-- shared_state.py           Delt tilstand pa tvers av moduler
|-- log_config.py             Logging-konfigurasjon
|-- utils.py                  Hjelpefunksjoner
|-- requirements.txt          Python-avhengigheter
|-- mine_tickere.txt          Egendefinerte aksjer (valgfri)
|-- views/                    UI-moduler (Streamlit-sider)
|   |-- hjem.py               Hjemmeside / dokumentasjon
|   |-- markedstemperatur.py   Markedsregime og bredde
|   |-- scanner.py            Strategi-scanner
|   |-- teknisk_analyse.py    Detaljert teknisk analyse
|   |-- beta_intradag.py      Intradag-scanner
|   |-- beta_ai_scanner.py    AI-scanner
|   |-- beta_pattern_vision.py Pattern Vision (YOLO)
|-- pattern_vision/           YOLO-basert monsterdetektion
|   |-- chart_generator.py
|   |-- fast_detector.py
|   |-- formation_scanner.py
|   |-- pattern_detector.py
|   |-- signal_fusion.py
|   |-- models/model.pt
|-- tests/                    Testsuite (319 tester)
|-- data_storage/             Cachede data og watchlist
    |-- tickers.txt
    |-- watchlist.json
    |-- fundamental_cache/
    |-- ml_cache/
    |-- ml_models/
```

---

## Konfigurasjon

### Egendefinerte aksjer

Opprett `mine_tickere.txt` i rotmappen med en ticker per linje:

```
EQNR.OL
DNB.OL
NHY.OL
MOWI.OL
```

Standard-universet hentes fra `data_storage/tickers.txt`.

### Risiko-innstillinger (config.py)

```python
DEFAULT_KAPITAL = 100000      # Standard kapital (NOK)
DEFAULT_RISIKO_PROSENT = 1.0  # Risiko per handel (%)
EXIT_TRAILING_STOP_PCT = 8.0  # Trailing stop (%)
```

### Sidebar-innstillinger

- **Oppdater Data** -- Last ned ferske kursdata fra Yahoo Finance
- **Min. dagsomsetning** -- Filtrer bort illikvide aksjer (standard: 500 000 NOK)

---

## Datakilder

| Kilde | Data |
|-------|------|
| Yahoo Finance | Historiske kursdata, intradag-data, fundamental-data |
| Oslo Bors Newsweb | Meldepliktige innsidehandler |
| Beregnet | Tekniske indikatorer, signaler, regimer, Smart Money |

---

## Testing

Prosjektet har 319 automatiserte tester:

```bash
source .venv/bin/activate
python -m pytest tests/ -v --tb=short
```

---

## Feilsoking

**Appen kjorer sakte:**
1. Reduser antall aksjer i `mine_tickere.txt`
2. Ok minimum dagsomsetning i sidebar
3. Bruk kortere tidsperioder

**Data lastes ikke:**
1. Sjekk internettforbindelse
2. Yahoo Finance kan vaere ustabil -- prov igjen
3. Appen bruker cachede data hvis tilgjengelig

---

## Viktig

**Dette er ikke investeringsrad.** InveStock Pro er et analyseverktoy for informasjonsforml.

- All handel innebarer risiko for tap
- Gjor alltid egen research (DYOR)
- Vurder a konsultere en finansiell radgiver
- Bruk alltid stop-loss og diversifiser
- Test strategier med paper trading forst

---

## Lisens

MIT License -- Apen kildekode for laering og personlig bruk.

---

<div align="center">

**InveStock Pro v2.0**

</div>
