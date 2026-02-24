"""
Anbefalt Portefølje — Autogenerert porteføljeanbefaling.

Bygger en realistisk porteføljeanbefaling som tar hensyn til:
- Eksisterende posisjoner (bygger på det du har)
- Kontantbeholdning (handler bare det du har råd til)
- Kurtasje og transaksjonskostnader
- Sektor-diversifisering
- Markedsregime (aggressiv i bull, defensiv i bear)
- Langsiktige og kortsiktige posisjoner
- Long OG short (bear-marked = short-kandidater)
- Maks antall handler per dag (realistisk)

Bruker scanner-data + regime + sektor + R:R for å foreslå:
1. NYE KJØP: Beste signaler etter score, R:R, diversifisering
2. ØK POSISJON: Eksisterende signaler med god utvikling (add)
3. REDUSER/SELG: Posisjoner med exit-signaler eller dårlig R:R
4. SHORT-KANDIDATER: I bear — aksjer med bearish mønstre/exit
"""

import pandas as pd
import numpy as np
from datetime import datetime
from log_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASJON
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Portefølje-begrensninger
    'maks_posisjoner': 12,           # Maks antall samtidige posisjoner
    'maks_handler_per_dag': 3,       # Realistisk: 1-3 handler per dag
    'maks_per_sektor': 3,            # Maks 3 aksjer i samme sektor
    'min_posisjon_pct': 3.0,         # Minste posisjon: 3% av kapital
    'maks_posisjon_pct': 15.0,       # Største posisjon: 15% av kapital
    'kontant_buffer_pct': 10.0,      # Alltid ha 10% kontant
    
    # Transaksjonskostnader
    'kurtasje_pct': 0.05,            # Per side
    'spread_slippage_pct': 0.10,     # Estimert
    
    # Score-krav
    'min_score_kjøp': 55,            # Minimum score for kjøpsanbefaling
    'min_rr_kjøp': 1.5,              # Minimum R:R for kjøp
    'min_rs_kjøp': 40,               # Minimum RS-rating
    
    # Short-krav (strengere)
    'min_score_short': 60,           # Høyere krav for short
    'aktivér_short': True,           # Kan slås av
    
    # Regime-justering
    'bull_target_investert_pct': 90,  # Maks investert i bull
    'bear_target_investert_pct': 50,  # Maks investert i bear
    'neutral_target_investert_pct': 70,
}


def generer_anbefaling(
    scanner_resultater,
    eksisterende_posisjoner=None,
    kapital=100_000,
    kontant=None,
    regime=None,
    sektor_mapping=None,
    config=None,
):
    """
    Genererer en komplett porteføljeanbefaling.
    
    Args:
        scanner_resultater: Liste med dicts fra scanner (Ticker, Score, RS, R:R, etc.)
        eksisterende_posisjoner: Dict fra portfolio.load_portfolio()['positions']
        kapital: Total porteføljeverdi (investert + kontant)
        kontant: Ledig kontantbeholdning (None = beregnes)
        regime: Nåværende markedsregime str eller None
        sektor_mapping: {ticker: sektor} eller None
        config: Override dict for DEFAULT_CONFIG
        
    Returns:
        Dict med:
            - 'kjøp': liste med kjøpsanbefalinger
            - 'selg': liste med salgsanbefalinger  
            - 'short': liste med short-kandidater
            - 'hold': liste med hold-posisjoner
            - 'oppsummering': kort tekstoppsummering
            - 'allokering': porteføljeallokering
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    if eksisterende_posisjoner is None:
        eksisterende_posisjoner = {}
    
    if kontant is None:
        # Estimer kontant fra kapital minus investert
        investert = sum(
            pos.get('quantity', 0) * pos.get('average_price', 0)
            for pos in eksisterende_posisjoner.values()
        )
        kontant = max(0, kapital - investert)
    
    # Parse scanner-resultater til DataFrame
    if isinstance(scanner_resultater, list):
        df_scan = pd.DataFrame(scanner_resultater)
    elif isinstance(scanner_resultater, pd.DataFrame):
        df_scan = scanner_resultater.copy()
    else:
        df_scan = pd.DataFrame()
    
    if df_scan.empty:
        return _tom_anbefaling("Ingen scanner-data tilgjengelig")
    
    # --- STEG 1: Kategoriser eksisterende posisjoner ---
    selg_anbefalinger = []
    hold_anbefalinger = []
    
    for ticker, pos in eksisterende_posisjoner.items():
        ticker_i_scan = df_scan[df_scan['Ticker'] == ticker]
        
        har_exit = False
        utv_pct = 0.0
        score = 0
        rr = 0.0
        
        if not ticker_i_scan.empty:
            row = ticker_i_scan.iloc[0]
            har_exit = str(row.get('Exit⚠️', '')) == '⚠️'
            utv_pct = float(row.get('Utv%', 0))
            score = int(row.get('Score', 0))
            rr = float(row.get('R:R', 0))
        
        antall = pos.get('quantity', 0)
        snitt_pris = pos.get('average_price', 0)
        posisjon_verdi = antall * snitt_pris
        
        # Bestem handling
        if har_exit and utv_pct < -3:
            selg_anbefalinger.append({
                'ticker': ticker,
                'handling': 'SELG',
                'grunn': f"Exit-signal + {utv_pct:+.1f}% utvikling",
                'prioritet': 1,  # Høyeste prioritet
                'verdi': posisjon_verdi,
                'antall': antall,
            })
        elif utv_pct < -10:
            selg_anbefalinger.append({
                'ticker': ticker,
                'handling': 'SELG',
                'grunn': f"Stor drawdown: {utv_pct:+.1f}%",
                'prioritet': 2,
                'verdi': posisjon_verdi,
                'antall': antall,
            })
        elif har_exit:
            selg_anbefalinger.append({
                'ticker': ticker,
                'handling': 'REDUSER',
                'grunn': f"Exit-signal, men {utv_pct:+.1f}% utvikling — vurder å ta gevinst",
                'prioritet': 3,
                'verdi': posisjon_verdi,
                'antall': antall,
            })
        else:
            hold_anbefalinger.append({
                'ticker': ticker,
                'handling': 'HOLD',
                'grunn': f"Score: {score}, R:R: {rr:.1f}, Utv: {utv_pct:+.1f}%",
                'verdi': posisjon_verdi,
                'antall': antall,
            })
    
    # --- STEG 2: Regime-tilpasset allokering ---
    regime_lower = (regime or '').lower()
    if 'bull' in regime_lower:
        target_investert_pct = cfg['bull_target_investert_pct']
        aktivér_short = False
    elif 'bear' in regime_lower:
        target_investert_pct = cfg['bear_target_investert_pct']
        aktivér_short = cfg['aktivér_short']
    else:
        target_investert_pct = cfg['neutral_target_investert_pct']
        aktivér_short = False
    
    kontant_buffer = kapital * cfg['kontant_buffer_pct'] / 100
    maks_investering = kapital * target_investert_pct / 100
    tilgjengelig_for_kjøp = max(0, min(kontant - kontant_buffer, maks_investering))
    
    # --- STEG 3: Finn kjøpskandidater ---
    # Ekskluder tickers vi allerede eier
    eksisterende_tickers = set(eksisterende_posisjoner.keys())
    
    kjøpskandidater = df_scan[~df_scan['Ticker'].isin(eksisterende_tickers)].copy()
    
    # Filtrer etter minimumskrav
    if 'Score' in kjøpskandidater.columns:
        kjøpskandidater = kjøpskandidater[kjøpskandidater['Score'] >= cfg['min_score_kjøp']]
    if 'R:R' in kjøpskandidater.columns:
        kjøpskandidater = kjøpskandidater[kjøpskandidater['R:R'] >= cfg['min_rr_kjøp']]
    if 'RS' in kjøpskandidater.columns:
        kjøpskandidater = kjøpskandidater[kjøpskandidater['RS'] >= cfg['min_rs_kjøp']]
    
    # Ekskluder exit-signaler
    if 'Exit⚠️' in kjøpskandidater.columns:
        kjøpskandidater = kjøpskandidater[kjøpskandidater['Exit⚠️'] != '⚠️']
    
    # Ekskluder false breakouts
    if 'FB' in kjøpskandidater.columns:
        kjøpskandidater = kjøpskandidater[kjøpskandidater['FB'] != '❌']
    
    # Sorter etter score + R:R (vektet)
    if 'R:R' in kjøpskandidater.columns and 'Score' in kjøpskandidater.columns:
        kjøpskandidater['_rank'] = kjøpskandidater['Score'] * 0.6 + kjøpskandidater['R:R'] * 20 * 0.4
        kjøpskandidater = kjøpskandidater.sort_values('_rank', ascending=False)
    elif 'Score' in kjøpskandidater.columns:
        kjøpskandidater = kjøpskandidater.sort_values('Score', ascending=False)
    
    # Diversifisering: max per sektor
    sektor_count = {}
    for _, pos in eksisterende_posisjoner.items():
        sek = (sektor_mapping or {}).get(_, 'Annet')
        sektor_count[sek] = sektor_count.get(sek, 0) + 1
    
    kjøp_anbefalinger = []
    antall_eksisterende = len(eksisterende_posisjoner)
    
    for _, row in kjøpskandidater.iterrows():
        if len(kjøp_anbefalinger) >= cfg['maks_handler_per_dag']:
            break
        
        if antall_eksisterende + len(kjøp_anbefalinger) >= cfg['maks_posisjoner']:
            break
        
        ticker = row['Ticker']
        sektor = (sektor_mapping or {}).get(ticker, str(row.get('Sektor', 'Annet')))
        
        # Sjekk sektor-konsentrasjon
        current_sek = sektor_count.get(sektor, 0)
        if current_sek >= cfg['maks_per_sektor']:
            continue
        
        # Beregn posisjonsstørrelse
        score = int(row.get('Score', 50))
        rr = float(row.get('R:R', 1.0))
        
        # Dynamisk posisjonsstørrelse: høyere score/R:R → større posisjon
        base_pct = (cfg['min_posisjon_pct'] + cfg['maks_posisjon_pct']) / 2
        if score >= 80 and rr >= 2.5:
            posisjon_pct = cfg['maks_posisjon_pct']
        elif score >= 65 and rr >= 2.0:
            posisjon_pct = base_pct * 1.2
        elif score >= 55:
            posisjon_pct = base_pct
        else:
            posisjon_pct = cfg['min_posisjon_pct']
        
        posisjon_verdi = kapital * posisjon_pct / 100
        
        # Sjekk at vi har nok kontanter
        kurtasje = posisjon_verdi * (cfg['kurtasje_pct'] + cfg['spread_slippage_pct']) / 100
        total_kostnad = posisjon_verdi + kurtasje
        
        if total_kostnad > tilgjengelig_for_kjøp:
            # Prøv mindre posisjon
            posisjon_verdi = tilgjengelig_for_kjøp * 0.95
            if posisjon_verdi < kapital * cfg['min_posisjon_pct'] / 100:
                continue  # For lite kontanter igjen
        
        try:
            pris_str = str(row.get('Pris', '0')).replace(',', '.').strip()
            pris = float(pris_str)
        except (ValueError, TypeError):
            continue
        
        if pris <= 0:
            continue
        
        antall_aksjer = int(posisjon_verdi / pris)
        if antall_aksjer < 1:
            continue
        
        faktisk_verdi = antall_aksjer * pris
        kurtasje = faktisk_verdi * (cfg['kurtasje_pct'] + cfg['spread_slippage_pct']) / 100
        
        kjøp_anbefalinger.append({
            'ticker': ticker,
            'handling': 'KJØP',
            'grunn': f"Score {score}, R:R {rr:.1f}, RS {row.get('RS', '?')}",
            'sektor': sektor,
            'antall': antall_aksjer,
            'pris': pris,
            'verdi': round(faktisk_verdi, 0),
            'kurtasje': round(kurtasje, 0),
            'posisjon_pct': round(faktisk_verdi / kapital * 100, 1),
            'score': score,
            'rr': rr,
            'strategi': str(row.get('Strategi', '—')),
        })
        
        tilgjengelig_for_kjøp -= (faktisk_verdi + kurtasje)
        sektor_count[sektor] = current_sek + 1
    
    # --- STEG 4: Short-kandidater (kun i bear) ---
    short_anbefalinger = []
    if aktivér_short and not df_scan.empty:
        # Short-kandidater: lav RS, exit-signaler, bearish mønstre
        short_df = df_scan.copy()
        
        # Finn aksjer med bearish indikatorer
        short_kandidater = []
        if 'Exit⚠️' in short_df.columns:
            exit_aksjer = short_df[short_df['Exit⚠️'] == '⚠️']
            for _, row in exit_aksjer.iterrows():
                rs = int(row.get('RS', 50))
                utv = float(row.get('Utv%', 0))
                if rs < 30 and utv < -5:
                    short_kandidater.append({
                        'ticker': row['Ticker'],
                        'handling': 'SHORT',
                        'grunn': f"Exit-signal, RS {rs}, Utv {utv:+.1f}%",
                        'rs': rs,
                        'utv_pct': utv,
                        'score': abs(utv) + (100 - rs) / 5,  # Sorterings-score
                    })
        
        # Sorter og ta topp N
        short_kandidater.sort(key=lambda x: x['score'], reverse=True)
        short_anbefalinger = short_kandidater[:3]
    
    # --- STEG 5: Oppsummering ---
    total_kjøp_verdi = sum(k['verdi'] for k in kjøp_anbefalinger)
    total_selg_verdi = sum(s['verdi'] for s in selg_anbefalinger)
    total_kurtasje = sum(k.get('kurtasje', 0) for k in kjøp_anbefalinger)
    
    ny_kontant = kontant - total_kjøp_verdi - total_kurtasje + total_selg_verdi
    
    oppsummering_deler = []
    if kjøp_anbefalinger:
        oppsummering_deler.append(f"{len(kjøp_anbefalinger)} kjøp ({total_kjøp_verdi:,.0f} kr)")
    if selg_anbefalinger:
        oppsummering_deler.append(f"{len(selg_anbefalinger)} selg/reduser")
    if short_anbefalinger:
        oppsummering_deler.append(f"{len(short_anbefalinger)} short-kandidater")
    oppsummering_deler.append(f"Kontant etter: {ny_kontant:,.0f} kr ({ny_kontant/kapital*100:.0f}%)")
    
    regime_tekst = regime if regime else "Ukjent"
    
    return {
        'kjøp': kjøp_anbefalinger,
        'selg': selg_anbefalinger,
        'short': short_anbefalinger,
        'hold': hold_anbefalinger,
        'oppsummering': ' · '.join(oppsummering_deler),
        'allokering': {
            'kapital': kapital,
            'kontant_før': kontant,
            'kontant_etter': ny_kontant,
            'investert_pct': (1 - ny_kontant / kapital) * 100 if kapital > 0 else 0,
            'target_investert_pct': target_investert_pct,
            'total_kurtasje': total_kurtasje,
            'regime': regime_tekst,
            'antall_posisjoner': antall_eksisterende + len(kjøp_anbefalinger),
        },
        'dato': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }


def _tom_anbefaling(grunn):
    """Returnerer en tom anbefaling med forklaring."""
    return {
        'kjøp': [],
        'selg': [],
        'short': [],
        'hold': [],
        'oppsummering': grunn,
        'allokering': {
            'kapital': 0, 'kontant_før': 0, 'kontant_etter': 0,
            'investert_pct': 0, 'target_investert_pct': 0,
            'total_kurtasje': 0, 'regime': 'Ukjent',
            'antall_posisjoner': 0,
        },
        'dato': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
