"""
Gaussian Mixture Model for Market Regime Detection
Inspirert av Two Sigma: https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/

Regimer basert på avkastning og volatilitet:
- "Steady Bull": Høy avkastning, lav volatilitet (ideelt marked)
- "Volatile Rally": Høy avkastning, høy volatilitet (oppgang med usikkerhet)
- "Walking on Ice": Lav avkastning, lav volatilitet (rolig, men retningsløst)
- "Correction": Negativ avkastning, moderat volatilitet
- "Crisis": Negativ avkastning, høy volatilitet (panikk/krasj)
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Regime-definisjoner med beskrivende navn, emoji og farger
REGIME_DEFINITIONS = {
    'steady_bull': {
        'name': 'Steady Bull',
        'description': 'Stabil oppgang med lav volatilitet - ideelle forhold',
        'emoji': '🚀',
        'color': '#00C805',
        'action': 'Aggressiv posisjonering'
    },
    'volatile_bull': {
        'name': 'Volatile Rally',
        'description': 'Oppgang med høy usikkerhet - vær forberedt på svingninger',
        'emoji': '🎢',
        'color': '#90EE90',
        'action': 'Moderat posisjonering med tett stop-loss'
    },
    'walking_on_ice': {
        'name': 'Walking on Ice',
        'description': 'Rolig marked uten klar retning - avvent bedre muligheter',
        'emoji': '🧊',
        'color': '#87CEEB',
        'action': 'Reduser eksponering, vent på signal'
    },
    'correction': {
        'name': 'Correction',
        'description': 'Moderat nedgang - normal markedskorreksjon',
        'emoji': '📉',
        'color': '#FFA500',
        'action': 'Defensiv, vurder hedging'
    },
    'crisis': {
        'name': 'Crisis Mode',
        'description': 'Høy volatilitet og negativ avkastning - beskyttelsesmodus',
        'emoji': '🔥',
        'color': '#FF5252',
        'action': 'Maksimer kontanter, strengt risikostyring'
    }
}


def beregn_regime_features(df_market: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Beregner features for regimemodellering fra markedsindeks-data.
    """
    df = df_market.copy()
    
    # Håndter MultiIndex kolonner (fra yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Finn Close-kolonnen
    if 'Close' not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in str(c).lower()]
        if close_candidates:
            df['Close'] = df[close_candidates[0]]
        else:
            raise ValueError(f"Finner ikke 'Close'-kolonnen. Tilgjengelige: {df.columns.tolist()}")
    
    # Konverter og rens data
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.dropna(subset=['Close'])
    
    if len(df) < lookback * 3:
        raise ValueError(f"For lite data: {len(df)} rader, trenger minst {lookback * 3}")
    
    # Beregn features
    df['returns'] = df['Close'].pct_change()
    
    min_periods = max(5, lookback // 2)
    
    # Annualisert volatilitet (standardavvik * sqrt(252))
    df['volatility'] = df['returns'].rolling(lookback, min_periods=min_periods).std() * np.sqrt(252)
    
    # Annualisert rullerende avkastning (gjennomsnitt * 252)
    df['rolling_return'] = df['returns'].rolling(lookback, min_periods=min_periods).mean() * 252
    
    # Ekstra features for bedre regime-separasjon
    df['vol_of_vol'] = df['volatility'].rolling(lookback, min_periods=min_periods).std()
    df['momentum'] = df['Close'].pct_change(lookback)
    df['drawdown'] = df['Close'] / df['Close'].rolling(lookback * 2, min_periods=lookback).max() - 1
    
    # Fjern NaN - kun i de kritiske kolonnene
    df = df.dropna(subset=['volatility', 'rolling_return'])
    
    return df


def klassifiser_regime_type(avg_return: float, avg_vol: float, 
                            all_returns: np.ndarray, all_vols: np.ndarray) -> str:
    """
    Klassifiserer et regime basert på gjennomsnittlig avkastning og volatilitet.
    Bruker percentiler fra datasettet for dynamiske terskler.
    """
    # Beregn percentiler for dynamiske terskler
    ret_median = np.median(all_returns)
    vol_median = np.median(all_vols)
    ret_75 = np.percentile(all_returns, 75)
    ret_25 = np.percentile(all_returns, 25)
    vol_75 = np.percentile(all_vols, 75)
    vol_25 = np.percentile(all_vols, 25)
    
    high_return = avg_return > ret_75
    positive_return = avg_return > ret_median
    negative_return = avg_return < ret_25
    high_vol = avg_vol > vol_75
    low_vol = avg_vol < vol_25
    
    # Klassifiser basert på kombinasjoner
    if high_return and low_vol:
        return 'steady_bull'
    elif high_return and high_vol:
        return 'volatile_bull'
    elif positive_return and not high_vol:
        return 'walking_on_ice'  # Moderat positiv, ikke høy vol
    elif negative_return and high_vol:
        return 'crisis'
    elif negative_return:
        return 'correction'
    else:
        return 'walking_on_ice'  # Default for nøytrale tilstander


def tren_gmm_model(df_features: pd.DataFrame, n_regimes: int = 3, 
                   feature_cols: list = None) -> tuple:
    """
    Trener en Gaussian Mixture Model for å identifisere markedsregimer.
    """
    if feature_cols is None:
        feature_cols = ['rolling_return', 'volatility']
    
    # Verifiser kolonner
    missing_cols = [col for col in feature_cols if col not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Manglende kolonner: {missing_cols}")
    
    # Hent data og fjern NaN/Inf
    X = df_features[feature_cols].copy()
    valid_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    X = X[valid_mask].values
    
    if len(X) < n_regimes * 20:
        raise ValueError(f"For lite gyldig data for {n_regimes} regimer: {len(X)} rader")
    
    # Standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Tren GMM med flere initialiseringer for stabilitet
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        n_init=20,
        random_state=42,
        max_iter=300,
        tol=1e-4
    )
    gmm.fit(X_scaled)
    
    return gmm, scaler, feature_cols


def klassifiser_regimer(gmm, scaler, feature_cols: list, 
                        df_features: pd.DataFrame) -> tuple:
    """
    Klassifiserer hver dag inn i et regime med beskrivende navn.
    """
    # Forbered data
    X = df_features[feature_cols].copy()
    valid_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    
    # Lag resultat-DataFrame
    df_result = df_features.copy()
    df_result['regime'] = -1  # Default verdi
    
    # Bare prosesser gyldige rader
    X_valid = X[valid_mask].values
    X_scaled = scaler.transform(X_valid)
    
    # Prediker for gyldige rader
    regimes_valid = gmm.predict(X_scaled)
    probs_valid = gmm.predict_proba(X_scaled)
    
    # Sett verdier for gyldige rader
    df_result.loc[valid_mask, 'regime'] = regimes_valid
    
    # Legg til sannsynligheter
    for i in range(gmm.n_components):
        df_result[f'prob_regime_{i}'] = 0.0
        df_result.loc[valid_mask, f'prob_regime_{i}'] = probs_valid[:, i]
    
    # Fjern rader uten gyldig regime
    df_result = df_result[df_result['regime'] >= 0]
    
    # Beregn statistikk per regime
    regime_stats = df_result.groupby('regime').agg({
        'rolling_return': 'mean',
        'volatility': 'mean'
    })
    
    # Hent alle returns og vols for dynamisk klassifisering
    all_returns = regime_stats['rolling_return'].values
    all_vols = regime_stats['volatility'].values
    
    # Klassifiser hvert regime og lag labels
    regime_labels = {}
    used_types = set()
    
    for regime_id in regime_stats.index:
        avg_ret = regime_stats.loc[regime_id, 'rolling_return']
        avg_vol = regime_stats.loc[regime_id, 'volatility']
        
        # Finn regime-type
        regime_type = klassifiser_regime_type(avg_ret, avg_vol, all_returns, all_vols)
        
        # Unngå duplikater - velg neste beste type hvis allerede brukt
        if regime_type in used_types:
            # Sorter alle typer etter hvor godt de passer
            type_scores = []
            for rtype in REGIME_DEFINITIONS.keys():
                if rtype not in used_types:
                    type_scores.append((rtype, 0))  # Kan forbedres med bedre scoring
            if type_scores:
                regime_type = type_scores[0][0]
        
        used_types.add(regime_type)
        
        regime_def = REGIME_DEFINITIONS[regime_type]
        regime_labels[regime_id] = (
            regime_def['name'],
            regime_def['emoji'],
            regime_def['color'],
            regime_def['description'],
            regime_def['action']
        )
    
    # Map til DataFrame
    df_result['regime_name'] = df_result['regime'].map(
        lambda x: regime_labels.get(x, ('Ukjent', '❓', '#808080', '', ''))[0]
    )
    df_result['regime_emoji'] = df_result['regime'].map(
        lambda x: regime_labels.get(x, ('Ukjent', '❓', '#808080', '', ''))[1]
    )
    df_result['regime_color'] = df_result['regime'].map(
        lambda x: regime_labels.get(x, ('Ukjent', '❓', '#808080', '', ''))[2]
    )
    df_result['regime_description'] = df_result['regime'].map(
        lambda x: regime_labels.get(x, ('Ukjent', '❓', '#808080', '', ''))[3]
    )
    df_result['regime_action'] = df_result['regime'].map(
        lambda x: regime_labels.get(x, ('Ukjent', '❓', '#808080', '', ''))[4]
    )
    
    return df_result, regime_labels, regime_stats


def get_current_regime_info(df_regimes: pd.DataFrame, regime_labels: dict) -> dict:
    """
    Henter detaljert informasjon om nåværende regime.
    """
    if df_regimes.empty:
        return None
    
    latest = df_regimes.iloc[-1]
    current_regime = int(latest['regime'])
    
    # Sannsynlighet
    prob_col = f'prob_regime_{current_regime}'
    current_prob = float(latest[prob_col]) if prob_col in df_regimes.columns else 0.0
    
    # Regime-streak (hvor mange dager på rad med samme regime)
    regime_streak = 1
    for i in range(len(df_regimes) - 2, -1, -1):
        if int(df_regimes.iloc[i]['regime']) == current_regime:
            regime_streak += 1
        else:
            break
    
    # Historisk fordeling
    regime_distribution = df_regimes['regime_name'].value_counts(normalize=True).to_dict()
    
    # Alle sannsynligheter for nåværende tidspunkt
    all_probs = {}
    for regime_id, label_info in regime_labels.items():
        prob_col = f'prob_regime_{regime_id}'
        if prob_col in df_regimes.columns:
            all_probs[label_info[0]] = float(latest[prob_col])
    
    # Hent volatilitet og avkastning med fallback
    volatility = float(latest['volatility']) if pd.notna(latest['volatility']) else 0.0
    rolling_return = float(latest['rolling_return']) if pd.notna(latest['rolling_return']) else 0.0
    
    return {
        'regime': current_regime,
        'name': str(latest['regime_name']),
        'emoji': str(latest['regime_emoji']),
        'color': str(latest['regime_color']),
        'description': str(latest.get('regime_description', '')),
        'action': str(latest.get('regime_action', '')),
        'probability': current_prob,
        'streak_days': regime_streak,
        'volatility': volatility,
        'rolling_return': rolling_return,
        'distribution': regime_distribution,
        'all_probs': all_probs
    }


def beregn_regime_transitions(df_regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Beregner overgangsmatrise mellom regimer.
    """
    if df_regimes.empty or len(df_regimes) < 2:
        return pd.DataFrame()
    
    regimes = df_regimes['regime_name'].values
    transitions = {}
    
    for i in range(len(regimes) - 1):
        from_regime = regimes[i]
        to_regime = regimes[i + 1]
        key = (from_regime, to_regime)
        transitions[key] = transitions.get(key, 0) + 1
    
    unique_regimes = list(df_regimes['regime_name'].unique())
    matrix = pd.DataFrame(0.0, index=unique_regimes, columns=unique_regimes)
    
    for (from_r, to_r), count in transitions.items():
        if from_r in matrix.index and to_r in matrix.columns:
            matrix.loc[from_r, to_r] = count
    
    # Normaliser rad-vis (sannsynlighet for å gå FRA et regime TIL et annet)
    row_sums = matrix.sum(axis=1)
    # Unngå divisjon med null
    row_sums = row_sums.replace(0, 1)
    matrix = matrix.div(row_sums, axis=0)
    
    return matrix


def full_regime_analyse(df_market: pd.DataFrame, n_regimes: int = 3) -> dict:
    """
    Kjører full regimeanalyse og returnerer alle resultater.
    """
    try:
        if df_market is None or len(df_market) == 0:
            print("[REGIME] Ingen markedsdata mottatt")
            return None
        
        # Beregn features
        df_features = beregn_regime_features(df_market)
        
        if len(df_features) < 50:
            print(f"[REGIME] For få datapunkter etter feature-beregning: {len(df_features)}")
            return None
        
        # Tren modell
        gmm, scaler, feature_cols = tren_gmm_model(df_features, n_regimes=n_regimes)
        
        # Klassifiser
        df_regimes, regime_labels, regime_stats = klassifiser_regimer(
            gmm, scaler, feature_cols, df_features
        )
        
        if df_regimes.empty:
            print("[REGIME] Ingen gyldige regimer funnet")
            return None
        
        # Nåværende info
        current_info = get_current_regime_info(df_regimes, regime_labels)
        
        if current_info is None:
            print("[REGIME] Kunne ikke hente current regime info")
            return None
        
        # Transisjoner
        transition_matrix = beregn_regime_transitions(df_regimes)
        
        return {
            'df_regimes': df_regimes,
            'regime_labels': regime_labels,
            'regime_stats': regime_stats,
            'current_info': current_info,
            'transition_matrix': transition_matrix,
            'gmm': gmm,
            'scaler': scaler
        }
        
    except Exception as e:
        import traceback
        print(f"[REGIME] Feil i regimeanalyse: {e}")
        print(traceback.format_exc())
        return None
