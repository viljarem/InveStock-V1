"""
Algoritmisk mønstergjenkjenning — ren prisdata-logikk.

Erstatter det gamle Pattern Vision med lette, vektoriserte detektorer:
- Head & Shoulders (bearish) / Inverse Head & Shoulders (bullish)
- Dobbel bunn (bullish) / Dobbel topp (bearish)
- Kopp-og-hank (bullish)
- Trekanter: ascending, descending, symmetrisk

Alle funksjoner opererer på standard OHLCV DataFrames med DatetimeIndex.
Returformat er konsistent: liste med dicts {dato, mønster, retning, styrke, detaljer}.
"""

import pandas as pd
import numpy as np
from log_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HJELPEFUNKSJONER
# ─────────────────────────────────────────────────────────────────────────────

def _finn_lokale_ekstremer(series, vindu=5):
    """Finner lokale topper og bunner i en prisserie.
    
    Returnerer (topper_idx, bunner_idx) som lister av integer-posisjoner.
    """
    if len(series) < vindu * 2 + 1:
        return [], []
    
    topper = []
    bunner = []
    
    for i in range(vindu, len(series) - vindu):
        vindu_verdier = series.iloc[i - vindu:i + vindu + 1]
        if series.iloc[i] == vindu_verdier.max():
            topper.append(i)
        if series.iloc[i] == vindu_verdier.min():
            bunner.append(i)
    
    return topper, bunner


def _er_horisontalt(verdier, toleranse_pct=2.0):
    """Sjekker om en serie verdier er ca. horisontale (innenfor toleranse)."""
    if len(verdier) < 2:
        return False
    snitt = np.mean(verdier)
    if snitt == 0:
        return True
    avvik = (max(verdier) - min(verdier)) / snitt * 100
    return avvik <= toleranse_pct


def _trend_retning(verdier):
    """Beregner lineær trend: positiv = stigende, negativ = fallende."""
    if len(verdier) < 2:
        return 0.0
    x = np.arange(len(verdier))
    slope = np.polyfit(x, verdier, 1)[0]
    return slope


def _beregn_mønster_styrke(volum_bekreftelse=False, trend_alignment=False,
                            nærhet_pct=100.0, dager_i_mønster=30):
    """Beregner en styrke-score 0-100 for et detektert mønster."""
    score = 40  # Basis for detektert mønster
    
    if volum_bekreftelse:
        score += 20
    if trend_alignment:
        score += 15
    
    # Nærhet bonus: nyere mønstre er mer relevante
    if nærhet_pct <= 10:  # Innenfor siste 10% av data
        score += 15
    elif nærhet_pct <= 25:
        score += 10
    elif nærhet_pct <= 50:
        score += 5
    
    # Dager-i-mønster: for korte = støy, for lange = utdatert
    if 15 <= dager_i_mønster <= 60:
        score += 10
    elif 10 <= dager_i_mønster <= 90:
        score += 5
    
    return min(100, score)


# ─────────────────────────────────────────────────────────────────────────────
# MØNSTERDETEKTORER
# ─────────────────────────────────────────────────────────────────────────────

def finn_dobbel_bunn(df, vindu=10, toleranse_pct=3.0):
    """
    Detekterer dobbel bunn (W-mønster) — bullish reversal.
    
    Kriterier:
    1. To bunner innenfor toleranse_pct av hverandre
    2. En topp mellom dem (neckline)
    3. Avstand mellom bunner: 15-60 dager
    4. Volum helst høyere ved andre bunn
    """
    if df.empty or len(df) < 50:
        return []
    
    resultater = []
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    _, bunner = _finn_lokale_ekstremer(close, vindu=vindu)
    topper, _ = _finn_lokale_ekstremer(close, vindu=vindu)
    
    for i in range(len(bunner) - 1):
        b1_idx = bunner[i]
        b2_idx = bunner[i + 1]
        
        avstand = b2_idx - b1_idx
        if avstand < 15 or avstand > 60:
            continue
        
        b1_pris = close.iloc[b1_idx]
        b2_pris = close.iloc[b2_idx]
        
        # Sjekk at bunnene er nær hverandre
        snitt_bunn = (b1_pris + b2_pris) / 2
        if snitt_bunn == 0:
            continue
        diff_pct = abs(b1_pris - b2_pris) / snitt_bunn * 100
        if diff_pct > toleranse_pct:
            continue
        
        # Finn topp mellom bunnene (neckline)
        mellom_topper = [t for t in topper if b1_idx < t < b2_idx]
        if not mellom_topper:
            continue
        
        neckline_idx = max(mellom_topper, key=lambda t: close.iloc[t])
        neckline_pris = close.iloc[neckline_idx]
        
        # Neckline må være tydelig over bunnene
        if neckline_pris < snitt_bunn * 1.02:
            continue
        
        # Volum-bekreftelse
        vol_bekreftet = False
        if volume.iloc[b2_idx] > volume.iloc[b1_idx] * 0.8:
            vol_bekreftet = True
        
        # Nærhet: hvor nært slutten av data
        nærhet = (len(df) - b2_idx) / len(df) * 100
        
        styrke = _beregn_mønster_styrke(
            volum_bekreftelse=vol_bekreftet,
            trend_alignment=True,
            nærhet_pct=nærhet,
            dager_i_mønster=avstand
        )
        
        resultater.append({
            'dato': df.index[b2_idx],
            'mønster': 'Dobbel Bunn',
            'retning': 'bullish',
            'styrke': styrke,
            'detaljer': {
                'bunn1': {'idx': b1_idx, 'pris': round(b1_pris, 2), 'dato': df.index[b1_idx]},
                'bunn2': {'idx': b2_idx, 'pris': round(b2_pris, 2), 'dato': df.index[b2_idx]},
                'neckline': round(neckline_pris, 2),
                'avstand_dager': avstand,
                'volum_bekreftet': vol_bekreftet,
            }
        })
    
    return resultater


def finn_dobbel_topp(df, vindu=10, toleranse_pct=3.0):
    """
    Detekterer dobbel topp (M-mønster) — bearish reversal.
    
    Kriterier:
    1. To topper innenfor toleranse_pct av hverandre
    2. En bunn mellom dem (neckline)
    3. Avstand mellom topper: 15-60 dager
    """
    if df.empty or len(df) < 50:
        return []
    
    resultater = []
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    topper, _ = _finn_lokale_ekstremer(close, vindu=vindu)
    _, bunner = _finn_lokale_ekstremer(close, vindu=vindu)
    
    for i in range(len(topper) - 1):
        t1_idx = topper[i]
        t2_idx = topper[i + 1]
        
        avstand = t2_idx - t1_idx
        if avstand < 15 or avstand > 60:
            continue
        
        t1_pris = close.iloc[t1_idx]
        t2_pris = close.iloc[t2_idx]
        
        snitt_topp = (t1_pris + t2_pris) / 2
        if snitt_topp == 0:
            continue
        diff_pct = abs(t1_pris - t2_pris) / snitt_topp * 100
        if diff_pct > toleranse_pct:
            continue
        
        mellom_bunner = [b for b in bunner if t1_idx < b < t2_idx]
        if not mellom_bunner:
            continue
        
        neckline_idx = min(mellom_bunner, key=lambda b: close.iloc[b])
        neckline_pris = close.iloc[neckline_idx]
        
        if neckline_pris > snitt_topp * 0.98:
            continue
        
        vol_bekreftet = volume.iloc[t2_idx] > volume.iloc[t1_idx] * 0.8
        nærhet = (len(df) - t2_idx) / len(df) * 100
        
        styrke = _beregn_mønster_styrke(
            volum_bekreftelse=vol_bekreftet,
            trend_alignment=True,
            nærhet_pct=nærhet,
            dager_i_mønster=avstand
        )
        
        resultater.append({
            'dato': df.index[t2_idx],
            'mønster': 'Dobbel Topp',
            'retning': 'bearish',
            'styrke': styrke,
            'detaljer': {
                'topp1': {'idx': t1_idx, 'pris': round(t1_pris, 2), 'dato': df.index[t1_idx]},
                'topp2': {'idx': t2_idx, 'pris': round(t2_pris, 2), 'dato': df.index[t2_idx]},
                'neckline': round(neckline_pris, 2),
                'avstand_dager': avstand,
                'volum_bekreftet': vol_bekreftet,
            }
        })
    
    return resultater


def finn_hode_skuldre(df, vindu=8):
    """
    Detekterer Head & Shoulders (bearish) og Inverse H&S (bullish).
    
    H&S: Tre topper der midterste (hodet) er høyest, skuldrene omtrent like.
    IH&S: Tre bunner der midterste er lavest, skuldrene omtrent like.
    """
    if df.empty or len(df) < 60:
        return []
    
    resultater = []
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    topper, bunner = _finn_lokale_ekstremer(close, vindu=vindu)
    
    # --- HEAD & SHOULDERS (bearish) ---
    for i in range(len(topper) - 2):
        ls_idx = topper[i]      # Venstre skulder
        h_idx = topper[i + 1]   # Hode
        rs_idx = topper[i + 2]  # Høyre skulder
        
        ls_pris = close.iloc[ls_idx]
        h_pris = close.iloc[h_idx]
        rs_pris = close.iloc[rs_idx]
        
        # Hodet må være høyest
        if h_pris <= ls_pris or h_pris <= rs_pris:
            continue
        
        # Skuldrene skal være omtrent like (innenfor 5%)
        skulder_snitt = (ls_pris + rs_pris) / 2
        if skulder_snitt == 0:
            continue
        skulder_diff = abs(ls_pris - rs_pris) / skulder_snitt * 100
        if skulder_diff > 5.0:
            continue
        
        # Hodet skal være minst 2% over skuldrene
        if h_pris < skulder_snitt * 1.02:
            continue
        
        # Maks avstand
        total_avstand = rs_idx - ls_idx
        if total_avstand < 20 or total_avstand > 90:
            continue
        
        # Neckline: finn bunnene mellom skulder og hode
        nl_bunner = [b for b in bunner if ls_idx < b < rs_idx]
        if len(nl_bunner) < 2:
            continue
        neckline = np.mean([close.iloc[b] for b in nl_bunner])
        
        vol_bekreftet = volume.iloc[rs_idx] < volume.iloc[h_idx]
        nærhet = (len(df) - rs_idx) / len(df) * 100
        
        styrke = _beregn_mønster_styrke(
            volum_bekreftelse=vol_bekreftet,
            trend_alignment=True,
            nærhet_pct=nærhet,
            dager_i_mønster=total_avstand
        )
        
        resultater.append({
            'dato': df.index[rs_idx],
            'mønster': 'Head & Shoulders',
            'retning': 'bearish',
            'styrke': styrke,
            'detaljer': {
                'venstre_skulder': round(ls_pris, 2),
                'hode': round(h_pris, 2),
                'høyre_skulder': round(rs_pris, 2),
                'neckline': round(neckline, 2),
                'avstand_dager': total_avstand,
            }
        })
    
    # --- INVERSE HEAD & SHOULDERS (bullish) ---
    for i in range(len(bunner) - 2):
        ls_idx = bunner[i]
        h_idx = bunner[i + 1]
        rs_idx = bunner[i + 2]
        
        ls_pris = close.iloc[ls_idx]
        h_pris = close.iloc[h_idx]
        rs_pris = close.iloc[rs_idx]
        
        # Hodet (midterste bunnen) må være lavest
        if h_pris >= ls_pris or h_pris >= rs_pris:
            continue
        
        skulder_snitt = (ls_pris + rs_pris) / 2
        if skulder_snitt == 0:
            continue
        skulder_diff = abs(ls_pris - rs_pris) / skulder_snitt * 100
        if skulder_diff > 5.0:
            continue
        
        if h_pris > skulder_snitt * 0.98:
            continue
        
        total_avstand = rs_idx - ls_idx
        if total_avstand < 20 or total_avstand > 90:
            continue
        
        nl_topper = [t for t in topper if ls_idx < t < rs_idx]
        if len(nl_topper) < 2:
            continue
        neckline = np.mean([close.iloc[t] for t in nl_topper])
        
        vol_bekreftet = volume.iloc[rs_idx] > volume.iloc[h_idx]
        nærhet = (len(df) - rs_idx) / len(df) * 100
        
        styrke = _beregn_mønster_styrke(
            volum_bekreftelse=vol_bekreftet,
            trend_alignment=True,
            nærhet_pct=nærhet,
            dager_i_mønster=total_avstand
        )
        
        resultater.append({
            'dato': df.index[rs_idx],
            'mønster': 'Inv. Head & Shoulders',
            'retning': 'bullish',
            'styrke': styrke,
            'detaljer': {
                'venstre_skulder': round(ls_pris, 2),
                'hode': round(h_pris, 2),
                'høyre_skulder': round(rs_pris, 2),
                'neckline': round(neckline, 2),
                'avstand_dager': total_avstand,
            }
        })
    
    return resultater


def finn_kopp_og_hank(df, vindu=8, min_kopp_dager=30, maks_kopp_dager=120):
    """
    Detekterer kopp-og-hank (Cup & Handle) — bullish continuation.
    
    Kriterier:
    1. Koppen: U-formet nedgang og oppgang til ca. samme nivå
    2. Hanken: liten konsolidering (5-15% av koppens dybde) nær toppen
    3. Breakout over koppens høyde
    """
    if df.empty or len(df) < min_kopp_dager + 20:
        return []
    
    resultater = []
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    topper, bunner = _finn_lokale_ekstremer(close, vindu=vindu)
    
    for i in range(len(topper) - 1):
        lip_left_idx = topper[i]
        
        # Finn potensielle høyre lipp-topper
        for j in range(i + 1, len(topper)):
            lip_right_idx = topper[j]
            
            avstand = lip_right_idx - lip_left_idx
            if avstand < min_kopp_dager or avstand > maks_kopp_dager:
                continue
            
            lip_left = close.iloc[lip_left_idx]
            lip_right = close.iloc[lip_right_idx]
            
            # Lippene skal være ca. like høye (innenfor 5%)
            lip_snitt = (lip_left + lip_right) / 2
            if lip_snitt == 0:
                continue
            lip_diff = abs(lip_left - lip_right) / lip_snitt * 100
            if lip_diff > 5.0:
                continue
            
            # Finn bunnen av koppen mellom lippene
            mellom_bunner = [b for b in bunner if lip_left_idx < b < lip_right_idx]
            if not mellom_bunner:
                continue
            
            kopp_bunn_idx = min(mellom_bunner, key=lambda b: close.iloc[b])
            kopp_bunn = close.iloc[kopp_bunn_idx]
            
            # Koppen må ha tydelig dybde (minst 10%)
            kopp_dybde_pct = (lip_snitt - kopp_bunn) / lip_snitt * 100
            if kopp_dybde_pct < 10 or kopp_dybde_pct > 50:
                continue
            
            # U-form: bunn bør være ca. midt i koppen
            bunn_posisjon = (kopp_bunn_idx - lip_left_idx) / avstand
            if bunn_posisjon < 0.3 or bunn_posisjon > 0.7:
                continue
            
            # Sjekk for hank (liten pullback etter høyre lipp)
            hank_start = lip_right_idx
            hank_end = min(lip_right_idx + 20, len(df) - 1)
            if hank_end <= hank_start + 3:
                continue
            
            hank_segment = close.iloc[hank_start:hank_end + 1]
            hank_min = hank_segment.min()
            hank_dybde_pct = (lip_right - hank_min) / lip_right * 100 if lip_right > 0 else 0
            
            # Hanken: 3-15% dybde
            if hank_dybde_pct < 2 or hank_dybde_pct > 15:
                continue
            
            vol_bekreftet = False
            if len(volume) > hank_end:
                # Volum bør øke ved breakout
                snitt_vol = volume.iloc[lip_left_idx:lip_right_idx].mean()
                siste_vol = volume.iloc[hank_end]
                if siste_vol > snitt_vol * 1.3:
                    vol_bekreftet = True
            
            nærhet = (len(df) - hank_end) / len(df) * 100
            
            styrke = _beregn_mønster_styrke(
                volum_bekreftelse=vol_bekreftet,
                trend_alignment=True,
                nærhet_pct=nærhet,
                dager_i_mønster=avstand
            )
            
            resultater.append({
                'dato': df.index[min(hank_end, len(df) - 1)],
                'mønster': 'Kopp & Hank',
                'retning': 'bullish',
                'styrke': styrke,
                'detaljer': {
                    'lipp_pris': round(lip_snitt, 2),
                    'kopp_bunn': round(kopp_bunn, 2),
                    'kopp_dybde_pct': round(kopp_dybde_pct, 1),
                    'hank_dybde_pct': round(hank_dybde_pct, 1),
                    'avstand_dager': avstand,
                    'volum_bekreftet': vol_bekreftet,
                }
            })
            break  # Kun beste match per venstre lipp
    
    return resultater


def finn_trekanter(df, vindu=7, min_dager=15, maks_dager=60):
    """
    Detekterer trekant-mønstre:
    - Ascending triangle (bullish): flat motstand, stigende støtte
    - Descending triangle (bearish): flat støtte, fallende motstand
    - Symmetrisk triangle (nøytral): konvergerende trendlinjer
    """
    if df.empty or len(df) < 50:
        return []
    
    resultater = []
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    topper, bunner = _finn_lokale_ekstremer(close, vindu=vindu)
    
    # Analyser siste 3-4 måneder i rullerende vinduer
    for lookback_start in range(max(0, len(df) - maks_dager * 2), len(df) - min_dager, 10):
        segment_end = min(lookback_start + maks_dager, len(df))
        
        # Finn topper og bunner i segmentet
        seg_topper = [t for t in topper if lookback_start <= t < segment_end]
        seg_bunner = [b for b in bunner if lookback_start <= b < segment_end]
        
        if len(seg_topper) < 2 or len(seg_bunner) < 2:
            continue
        
        # Toppverdier og bunnverdier
        topp_priser = [close.iloc[t] for t in seg_topper]
        bunn_priser = [close.iloc[b] for b in seg_bunner]
        
        topp_slope = _trend_retning(topp_priser)
        bunn_slope = _trend_retning(bunn_priser)
        
        # Normaliser slope relativt til prisnivå
        pris_nivå = close.iloc[seg_topper[0]]
        if pris_nivå == 0:
            continue
        rel_topp_slope = topp_slope / pris_nivå * 100
        rel_bunn_slope = bunn_slope / pris_nivå * 100
        
        mønster = None
        retning = None
        
        # Ascending: flat motstand (topper), stigende støtte (bunner)
        if abs(rel_topp_slope) < 0.3 and rel_bunn_slope > 0.1:
            mønster = 'Ascending Triangle'
            retning = 'bullish'
        
        # Descending: fallende motstand (topper), flat støtte (bunner)
        elif rel_topp_slope < -0.1 and abs(rel_bunn_slope) < 0.3:
            mønster = 'Descending Triangle'
            retning = 'bearish'
        
        # Symmetrisk: konvergerende (topper fallende, bunner stigende)
        elif rel_topp_slope < -0.05 and rel_bunn_slope > 0.05:
            mønster = 'Symmetrisk Triangle'
            retning = 'neutral'
        
        if mønster is None:
            continue
        
        # Sjekk at vi faktisk konvergerer (range minsker)
        første_range = close.iloc[seg_topper[0]] - close.iloc[seg_bunner[0]]
        siste_range = close.iloc[seg_topper[-1]] - close.iloc[seg_bunner[-1]]
        if siste_range >= første_range:
            continue
        
        avstand = segment_end - lookback_start
        nærhet = (len(df) - segment_end) / len(df) * 100
        
        # Volumkontraksjon bekrefter trekant
        vol_start = volume.iloc[lookback_start:lookback_start + 10].mean() if len(volume) > lookback_start + 10 else 0
        vol_end = volume.iloc[max(0, segment_end - 10):segment_end].mean()
        vol_bekreftet = vol_end < vol_start * 0.8 if vol_start > 0 else False
        
        styrke = _beregn_mønster_styrke(
            volum_bekreftelse=vol_bekreftet,
            trend_alignment=(retning != 'neutral'),
            nærhet_pct=nærhet,
            dager_i_mønster=avstand
        )
        
        resultater.append({
            'dato': df.index[min(segment_end - 1, len(df) - 1)],
            'mønster': mønster,
            'retning': retning,
            'styrke': styrke,
            'detaljer': {
                'topp_slope': round(rel_topp_slope, 3),
                'bunn_slope': round(rel_bunn_slope, 3),
                'avstand_dager': avstand,
                'konvergens_pct': round((1 - siste_range / første_range) * 100, 1) if første_range > 0 else 0,
                'volum_kontraksjon': vol_bekreftet,
            }
        })
    
    # Dedupliser: behold sterkeste per mønstertype
    unik = {}
    for r in resultater:
        key = r['mønster']
        if key not in unik or r['styrke'] > unik[key]['styrke']:
            unik[key] = r
    
    return list(unik.values())


# ─────────────────────────────────────────────────────────────────────────────
# SAMLET SKANNING
# ─────────────────────────────────────────────────────────────────────────────

# Mønster-metadata (for UI)
MØNSTER_METADATA = {
    'Dobbel Bunn':           {'emoji': '🔵', 'retning': 'bullish', 'kort': 'W-formet reversal'},
    'Dobbel Topp':           {'emoji': '🔴', 'retning': 'bearish', 'kort': 'M-formet reversal'},
    'Head & Shoulders':      {'emoji': '🔴', 'retning': 'bearish', 'kort': 'Klassisk topp-mønster'},
    'Inv. Head & Shoulders': {'emoji': '🔵', 'retning': 'bullish', 'kort': 'Klassisk bunn-mønster'},
    'Kopp & Hank':           {'emoji': '🟢', 'retning': 'bullish', 'kort': 'Cup & Handle continuation'},
    'Ascending Triangle':    {'emoji': '🟢', 'retning': 'bullish', 'kort': 'Flat motstand, stigende støtte'},
    'Descending Triangle':   {'emoji': '🔴', 'retning': 'bearish', 'kort': 'Fallende motstand, flat støtte'},
    'Symmetrisk Triangle':   {'emoji': '🟡', 'retning': 'neutral', 'kort': 'Konvergerende trendlinjer'},
}


def skann_alle_mønstre(df, kun_siste_n_dager=90):
    """
    Kjører alle mønsterdetektorer på en DataFrame.
    
    Args:
        df: OHLCV DataFrame med DatetimeIndex
        kun_siste_n_dager: Bare rapporter mønstre innenfor siste N dager
        
    Returns:
        Liste med alle detekterte mønstre, sortert etter styrke (sterkeste først).
    """
    if df.empty or len(df) < 50:
        return []
    
    alle = []
    
    try:
        alle.extend(finn_dobbel_bunn(df))
    except Exception as e:
        logger.debug(f"Dobbel bunn feilet: {e}")
    
    try:
        alle.extend(finn_dobbel_topp(df))
    except Exception as e:
        logger.debug(f"Dobbel topp feilet: {e}")
    
    try:
        alle.extend(finn_hode_skuldre(df))
    except Exception as e:
        logger.debug(f"H&S feilet: {e}")
    
    try:
        alle.extend(finn_kopp_og_hank(df))
    except Exception as e:
        logger.debug(f"Kopp & Hank feilet: {e}")
    
    try:
        alle.extend(finn_trekanter(df))
    except Exception as e:
        logger.debug(f"Trekanter feilet: {e}")
    
    # Filtrer til siste N dager
    if kun_siste_n_dager and len(df) > kun_siste_n_dager:
        cutoff = df.index[-kun_siste_n_dager]
        alle = [m for m in alle if m['dato'] >= cutoff]
    
    # Sorter etter styrke
    alle.sort(key=lambda x: x['styrke'], reverse=True)
    
    return alle


def skann_for_scanner(df, ticker=''):
    """
    Kompakt mønsterskanning for bruk i scanner-tabellen.
    
    Returnerer dict med:
        - 'mønstre': liste med korte beskrivelser
        - 'beste': det sterkeste mønsteret (eller None)
        - 'emoji': visuell indikator
        - 'antall': antall detekterte mønstre
        - 'retning': 'bullish' / 'bearish' / 'mixed' / None
    """
    mønstre = skann_alle_mønstre(df, kun_siste_n_dager=60)
    
    if not mønstre:
        return {
            'mønstre': [],
            'beste': None,
            'emoji': '',
            'antall': 0,
            'retning': None,
            'tekst': '',
        }
    
    beste = mønstre[0]
    meta = MØNSTER_METADATA.get(beste['mønster'], {})
    
    retninger = set(m['retning'] for m in mønstre)
    if len(retninger) == 1:
        samlet_retning = retninger.pop()
    else:
        samlet_retning = 'mixed'
    
    korte_navn = [m['mønster'] for m in mønstre[:3]]
    
    return {
        'mønstre': mønstre,
        'beste': beste,
        'emoji': meta.get('emoji', '⚪'),
        'antall': len(mønstre),
        'retning': samlet_retning,
        'tekst': ', '.join(korte_navn),
    }
