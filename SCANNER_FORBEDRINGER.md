# InveStock — Forbedringer & Roadmap

> Systematisk arbeidsplan. Rekkefølge: rydd først, forbedre kjernen, bygg nytt til slutt.
> ⚠️ Scanner fungerer godt i dag — alt under er *forbedringer*, ikke omskrivinger.

---

## Fase 1 — Opprydding (fjern død kode)
> Gjøres først. Reduserer kompleksitet og fjerner forvirrende moduler.
> Risiko: Lav. Sletter kun isolerte moduler + referanser.

- [x] **1.1** Fjern Intradag-modulen
  - Slett `views/beta_intradag.py`
  - Fjern menyvalg + import i `app.py`
  - Fjern intraday-funksjoner i `chart_utils.py`

- [x] **1.2** Fjern Pattern Vision
  - Slett `views/beta_pattern_vision.py`
  - Slett hele `pattern_vision/`-mappen
  - Fjern PV-toggle, PV-kolonne, PV-cache i `views/scanner.py`
  - Fjern PV-referanser i `app.py`
  - Rydd `views/beta_ai_scanner.py` — fjern PV-input, behold resten

---

## Fase 2 — Scanner-presisjon (forbedre kjernen)
> Gjøres én om gangen, med test mellom hvert steg.
> Endrer scoring og filtrering — scanner skal gi *færre, bedre* signaler.

### Trinn A: Lav risiko (additive endringer)
> Legger til ny info / justerer score uten å endre eksisterende signallogikk.

- [x] **2.1** Exit-signal påvirker score
  - Automatisk -20 poeng hvis ≥ 2 exit-signaler
  - Enkel endring i `_analyse_en_ticker()`, ingen signallogikk berøres

- [x] **2.2** Tidsvektet utvikling / maks drawdown
  - Vis peak-avkastning + nåværende i Utv%-kolonnen (f.eks. "↓+2% (peak +8%)")
  - Kun visningsendring — scorer berøres ikke

- [x] **2.3** Sektor-RS + individuell RS-kobling
  - Bonus kun for aksjer der RS > 60 OG sektor-RS > 50
  - Additivt — eksisterende RS-logikk beholdes

### Trinn B: Medium risiko (justerer scoring)
> Endrer hvordan kvalitetsscore beregnes. Kan påvirke hvilke signaler som vises.

- [x] **2.4** False breakout-filter
  - Sjekk om pris < signalets close OG under SMA 10 → marker/fjern
  - Toggle i filter-panelet så bruker kan slå av/på

- [x] **2.5** Minimumskrav per kvalitetsfaktor
  - A-kvalitet krever trend ≥ 15 OG volum ≥ 10 (ikke bare totalsum ≥ 75)
  - Endrer `klassifiser_signal_kvalitet()` — totalsum beholdes som sekundært krav

- [x] **2.6** Volum-akselerasjon
  - Erstatt ren volum-ratio med akselererende volum over 3-5 dager
  - Endrer volum-score i `beregn_signal_kvalitet()`

- [x] **2.7** Konvergens: timing + logisk konsistens
  - Bonus for signaler klynget på 1-2 dager (vs. spredt over 5)
  - Definer strategipar: VCP+Momentum = sterk, RSI+Golden Cross = svak

### Trinn C: Avansert (kobler flere systemer)
> Krever at regime-modell og backtest fungerer. Bygger på trinn A+B.

- [x] **2.8** Adaptiv kvalitetsvekting per regime
  - Bear → trend/MTF vektes tyngre. Bull → momentum/volum tyngre
  - Avhenger av `regime_model.py`

- [x] **2.9** Backtest: trailing stop / profit target exit
  - Trailing stop (-1×ATR fra topp) + profit target (+2×ATR)
  - Dag 20 kun som fallback. Gir realistisk win rate.

- [x] **2.10** Backtest-validert strategi-vekting
  - Strategier med høyere win rate i nåværende regime vises høyere
  - Avhenger av 2.8 + 2.9

---

## Fase 3 — Nye funksjoner (bygger videre)
> Gjøres etter at kjernen er forbedret. Hver funksjon er uavhengig.

- [x] **3.1** Innstillingsfil (brukerpreferanser)
  - `user_settings.py` med DEFAULT_SETTINGS, load/save/get/set/reset
  - Lagres til `data_storage/user_settings.json` med deep merge mot defaults
  - Integrert i scanner (filter-defaults, 💾 Lagre filter-knapp) og chart (indikator-prefs)

- [x] **3.2** Forbedret chart-opplevelse
  - Klikk ticker-rad → popup med all info + konfigurerbart chart
  - Ekstra nøkkeltall: Score, RS, Peak%, Exit-status
  - Chart bruker brukerens indikator-preferanser fra 3.1
  - Signalhistorikk i popup med kvalitet og utvikling

- [x] **3.3** Risk/Reward-filter
  - R:R-ratio beregnet fra støtte/motstand-nivåer, vist som kolonne med fargekoding
  - Min R:R-slider i filterpanelet (filtrerer bort signaler med dårlig R:R)
  - Sortering etter R:R via toggle og dropdown
  - `highlight_rr()` i components.py (grønn ≥2, gul 1-2, rød <1)

- [x] **3.4** Algoritmisk mønstergjenkjenning
  - `pattern_logic.py`: Ren prisdata-logikk (ingen ML) med 6 detektorer
  - Mønstre: Dobbel bunn/topp, Head & Shoulders / Inv. H&S, Kopp & Hank, trekanter (ascending/descending/symmetrisk)
  - Hjelpefunksjoner: lokale ekstremer, horisontalitetssjekk, trend-retning, styrke-scoring
  - Scanner: Mønsterfilter-dropdown (Ingen / Kun bullish / Kun bearish / Alle) + Mønster-kolonne
  - Teknisk Analyse: Expander med detekterte mønstre (fargekodede kort med retning og styrke)

- [x] **3.5** Autogenerert anbefalt portefølje
  - `anbefalt_portefolje.py`: Komplett anbefalingsmotor med realistiske begrensninger
  - Bygger på eksisterende posisjoner (ikke kjøp/selg alt daglig)
  - Regime-tilpasset allokering (bull=90%, bear=50%, nøytral=70%)
  - Sektor-diversifisering (maks 3 per sektor), maks handler/dag, kontant-buffer
  - Long-posisjoner: basert på score + R:R + RS med dynamisk posisjonsstørrelse
  - Short-kandidater: kun i bear, basert på exit-signaler + lav RS + negativ utvikling
  - Selg/reduser: exit-signaler, stor drawdown, dårlig R:R
  - Kurtasje og transaksjonskostnader inkludert i beregning
  - Scanner-integrert: Expander med kapital-input, maks handler, kurtasje, generer-knapp
  - Visuell output: kjøp/selg/hold/short med detaljerte beløp og porteføljeallokering

---

### ~~Likviditetsfilter~~ ✅ Allerede implementert
> Dekkes av sidebar `min_volum` → `data.filtrer_likvide_aksjer()`

---

*Sist oppdatert: 21. februar 2026*
