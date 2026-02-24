"""
Hidden Markov Model for Market Regime Detection
Inspirert av Two Sigma: https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/

HMM er bedre enn GMM for regimedeteksjon fordi:
- Modellerer tidsavhengighet mellom regimer
- Gir naturlige overgangsmatriser
- Unngår urealistisk hyppige regimeskifter

Features (utvidet v2):
- rolling_return: annualisert 20-dagers avkastning
- volatility: annualisert 20-dagers volatilitet
- volume_trend: volum vs 50-dagers snitt (institusjonell aktivitet)
- momentum_roc: 60-dagers rate of change (trendstyrke)
- breadth: % av daglige avkastninger > 0 siste 20d (proxy for bredde)

Regimer basert på avkastning og volatilitet:
- "Steady Bull": Høy avkastning, lav volatilitet (ideelt marked)
- "Volatile Rally": Høy avkastning, høy volatilitet (oppgang med usikkerhet)
- "Walking on Ice": Lav avkastning, lav volatilitet (rolig, men retningsløst)
- "Correction": Negativ avkastning, moderat volatilitet
- "Crisis": Negativ avkastning, høy volatilitet (panikk/krasj)
"""
from log_config import get_logger
logger = get_logger(__name__)

import os
import json
import hashlib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config

# Cache-sti for modellparametere
MODEL_CACHE_DIR = os.path.join(config.DATA_DIR, "regime_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# Regime-definisjoner - kun disse 5 finnes
REGIME_DEFINITIONS = {
    'bull': {
        'name': 'Bull Market',
        'description': 'Sterk oppgang - ideelle forhold for aksjer',
        'emoji': '🚀',
        'color': '#00C805',
        'action': 'Aggressiv posisjonering'
    },
    'mild_bull': {
        'name': 'Mild Bull',
        'description': 'Moderat oppgang med lav risiko',
        'emoji': '📈',
        'color': '#90EE90',
        'action': 'Normal eksponering'
    },
    'neutral': {
        'name': 'Nøytral',
        'description': 'Sidelengs marked - avvent bedre muligheter',
        'emoji': '➡️',
        'color': '#87CEEB',
        'action': 'Reduser eksponering, vent på signal'
    },
    'mild_bear': {
        'name': 'Mild Bear',
        'description': 'Moderat nedgang - vær forsiktig',
        'emoji': '📉',
        'color': '#FFA500',
        'action': 'Defensiv posisjonering'
    },
    'bear': {
        'name': 'Bear Market',
        'description': 'Sterk nedgang - beskyttelsesmodus',
        'emoji': '🔥',
        'color': '#FF5252',
        'action': 'Maksimer kontanter, unngå risiko'
    }
}


def beregn_regime_features(df_market: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Beregner utvidede features for regimemodellering.
    
    Nye features (v2):
    - volume_trend: relativ volum vs 50d snitt (fanger institusjonell aktivitet)
    - momentum_roc: 60-dagers Rate of Change (trendstyrke)
    - breadth: andel positive daglige avkastninger siste 20d (proxy for markedsbredde)
    """
    df = df_market.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if 'Close' not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in str(c).lower()]
        if close_candidates:
            df['Close'] = df[close_candidates[0]]
        else:
            raise ValueError(f"Finner ikke 'Close'-kolonnen. Tilgjengelige: {df.columns.tolist()}")
    
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.dropna(subset=['Close'])
    
    if len(df) < lookback * 3:
        raise ValueError(f"For lite data: {len(df)} rader, trenger minst {lookback * 3}")
    
    # Originale features
    df['returns'] = df['Close'].pct_change()
    min_periods = max(5, lookback // 2)
    df['volatility'] = df['returns'].rolling(lookback, min_periods=min_periods).std() * np.sqrt(252)
    df['rolling_return'] = df['returns'].rolling(lookback, min_periods=min_periods).mean() * 252
    
    # Nye features v2
    # 1. Volum-trend: relativ volum vs 50-dagers snitt
    if 'Volume' in df.columns:
        vol_ma50 = df['Volume'].rolling(50, min_periods=20).mean()
        df['volume_trend'] = (df['Volume'].rolling(lookback).mean() / vol_ma50).fillna(1.0)
        # Clip ekstremverdier
        df['volume_trend'] = df['volume_trend'].clip(0.2, 5.0)
    else:
        df['volume_trend'] = 1.0
    
    # 2. Momentum: 60-dagers Rate of Change
    roc_period = 60
    df['momentum_roc'] = df['Close'].pct_change(roc_period) * 100  # i prosent
    df['momentum_roc'] = df['momentum_roc'].clip(-50, 50)  # Clip ekstremverdier
    
    # 3. Bredde-proxy: andel positive dager siste 20d
    df['breadth'] = (df['returns'] > 0).rolling(lookback, min_periods=min_periods).mean()
    
    df = df.dropna(subset=['volatility', 'rolling_return'])
    
    return df


def get_regime_config(n_regimes: int) -> list:
    """Returnerer regime-typer basert på antall valgte regimer."""
    configs = {
        2: ['bull', 'bear'],
        3: ['bull', 'neutral', 'bear'],
        4: ['bull', 'mild_bull', 'mild_bear', 'bear'],
        5: ['bull', 'mild_bull', 'neutral', 'mild_bear', 'bear']
    }
    return configs.get(n_regimes, configs[3])


def _data_hash(X: np.ndarray) -> str:
    """Genererer hash av feature-data for cache-nøkkel."""
    return hashlib.md5(X.tobytes()[:4096]).hexdigest()[:12]


def _cache_path(data_hash: str, n_regimes: int) -> str:
    """Returnerer filsti for cached modell."""
    return os.path.join(MODEL_CACHE_DIR, f"hmm_{n_regimes}r_{data_hash}.json")


def _save_model_cache(path: str, model, scaler, feature_cols: list, bic_score: float):
    """Lagrer modellparametere til disk (JSON-serialiserbart)."""
    try:
        cache_data = {
            'means': model.means_.tolist(),
            'covars': model.covars_.tolist(),
            'transmat': model.transmat_.tolist(),
            'startprob': model.startprob_.tolist(),
            'n_components': model.n_components,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'feature_cols': feature_cols,
            'bic_score': bic_score
        }
        with open(path, 'w') as f:
            json.dump(cache_data, f)
        logger.info(f"Regime-modell cached: {path}")
    except Exception as e:
        logger.warning(f"Kunne ikke cache regime-modell: {e}")


def _load_model_cache(path: str):
    """Laster modellparametere fra disk. Returnerer (model, scaler, feature_cols, bic) eller None."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            cache_data = json.load(f)
        
        n = cache_data['n_components']
        model = hmm.GaussianHMM(n_components=n, covariance_type='full')
        model.means_ = np.array(cache_data['means'])
        model.covars_ = np.array(cache_data['covars'])
        model.transmat_ = np.array(cache_data['transmat'])
        model.startprob_ = np.array(cache_data['startprob'])
        model.n_features = len(cache_data['feature_cols'])
        
        scaler = StandardScaler()
        scaler.mean_ = np.array(cache_data['scaler_mean'])
        scaler.scale_ = np.array(cache_data['scaler_scale'])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(cache_data['feature_cols'])
        
        logger.info(f"Regime-modell lastet fra cache: {path}")
        return model, scaler, cache_data['feature_cols'], cache_data.get('bic_score', 0)
    except Exception as e:
        logger.warning(f"Kunne ikke laste regime-cache: {e}")
        return None


def _beregn_bic(model, X_scaled: np.ndarray) -> float:
    """
    Beregner Bayesian Information Criterion (BIC) for modellvalg.
    BIC = -2 * log_likelihood + k * ln(n)
    Lavere BIC = bedre modell (balanserer fit vs kompleksitet).
    """
    n_samples = X_scaled.shape[0]
    n_features = X_scaled.shape[1]
    n_components = model.n_components
    
    log_likelihood = model.score(X_scaled) * n_samples
    
    # Antall frie parametere i GaussianHMM med full covariance
    k = (n_components - 1)  # startprob
    k += n_components * (n_components - 1)  # transmat
    k += n_components * n_features  # means
    k += n_components * n_features * (n_features + 1) // 2  # full covariance
    
    bic = -2 * log_likelihood + k * np.log(n_samples)
    return bic


def velg_optimalt_antall_regimer(df_features: pd.DataFrame,
                                  feature_cols: list = None,
                                  max_regimer: int = 6) -> dict:
    """
    Evaluerer BIC for 2-max_regimer og returnerer optimal modell.
    
    Returns:
        dict med 'optimal_n', 'bic_scores', 'modeller'
    """
    if feature_cols is None:
        feature_cols = _get_available_features(df_features)
    
    X = df_features[feature_cols].copy()
    valid_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    X = X[valid_mask].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    bic_scores = {}
    modeller = {}
    
    for n in range(2, max_regimer + 1):
        if len(X) < n * 20:
            continue
        try:
            model = hmm.GaussianHMM(
                n_components=n, covariance_type='full',
                n_iter=200, random_state=42, tol=1e-4, verbose=False
            )
            model.fit(X_scaled)
            bic = _beregn_bic(model, X_scaled)
            bic_scores[n] = bic
            modeller[n] = model
            logger.info(f"BIC for {n} regimer: {bic:.0f}")
        except Exception as e:
            logger.warning(f"BIC-eval feilet for {n} regimer: {e}")
    
    if not bic_scores:
        return {'optimal_n': 3, 'bic_scores': {}, 'modeller': {}}
    
    optimal_n = min(bic_scores, key=bic_scores.get)
    logger.info(f"Optimalt antall regimer (BIC): {optimal_n}")
    
    return {
        'optimal_n': optimal_n,
        'bic_scores': bic_scores,
        'modeller': modeller
    }


def _get_available_features(df: pd.DataFrame) -> list:
    """Returnerer tilgjengelige feature-kolonner fra DataFrame."""
    all_features = ['rolling_return', 'volatility', 'volume_trend', 'momentum_roc', 'breadth']
    return [f for f in all_features if f in df.columns and df[f].notna().sum() > 50]


def tren_hmm_model(df_features: pd.DataFrame, n_regimes: int = 3, 
                   feature_cols: list = None, use_cache: bool = True) -> tuple:
    """
    Trener HMM for regimedeteksjon.
    
    Forbedringer v2:
    - Bruker utvidede features (volum, momentum, bredde)
    - Cacher modellparametere til disk
    - Returnerer BIC-score
    """
    if feature_cols is None:
        feature_cols = _get_available_features(df_features)
    
    X = df_features[feature_cols].copy()
    valid_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    X = X[valid_mask].values
    
    if len(X) < n_regimes * 20:
        raise ValueError(f"For lite data for {n_regimes} regimer: {len(X)} rader")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sjekk cache
    data_hash = _data_hash(X_scaled)
    cache_file = _cache_path(data_hash, n_regimes)
    
    if use_cache:
        cached = _load_model_cache(cache_file)
        if cached is not None:
            model, cached_scaler, cached_cols, bic = cached
            return model, cached_scaler, cached_cols, bic
    
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        n_iter=200,
        random_state=42,
        tol=1e-4,
        verbose=False
    )
    model.fit(X_scaled)
    
    bic = _beregn_bic(model, X_scaled)
    
    # Cache til disk
    _save_model_cache(cache_file, model, scaler, feature_cols, bic)
    
    return model, scaler, feature_cols, bic


def klassifiser_regimer(hmm_model, scaler, feature_cols: list, 
                        df_features: pd.DataFrame) -> tuple:
    """Klassifiserer regimer med konsistente navn."""
    n_regimes = hmm_model.n_components
    regime_types = get_regime_config(n_regimes)
    
    X = df_features[feature_cols].copy()
    valid_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    
    df_result = df_features.copy()
    df_result['regime'] = -1
    
    X_valid = X[valid_mask].values
    X_scaled = scaler.transform(X_valid)
    
    regimes_valid = hmm_model.predict(X_scaled)
    probs_valid = hmm_model.predict_proba(X_scaled)
    
    df_result.loc[valid_mask, 'regime'] = regimes_valid
    
    for i in range(n_regimes):
        df_result[f'prob_regime_{i}'] = 0.0
        df_result.loc[valid_mask, f'prob_regime_{i}'] = probs_valid[:, i]
    
    df_result = df_result[df_result['regime'] >= 0]
    
    # Beregn snitt per regime og ranger fra best til verst
    regime_stats = df_result.groupby('regime').agg({
        'rolling_return': 'mean',
        'volatility': 'mean'
    })
    
    # Score: høy avkastning bra, høy volatilitet dårlig
    regime_stats['score'] = regime_stats['rolling_return'] - 0.3 * regime_stats['volatility']
    sorted_regimes = regime_stats.sort_values('score', ascending=False).index.tolist()
    
    # Map HMM regime-ID til våre regime-typer
    regime_labels = {}
    for rank, regime_id in enumerate(sorted_regimes):
        regime_type = regime_types[rank] if rank < len(regime_types) else 'neutral'
        regime_def = REGIME_DEFINITIONS[regime_type]
        regime_labels[regime_id] = {
            'type': regime_type,
            'name': regime_def['name'],
            'emoji': regime_def['emoji'],
            'color': regime_def['color'],
            'description': regime_def['description'],
            'action': regime_def['action']
        }
    
    # Legg til labels i DataFrame
    df_result['regime_name'] = df_result['regime'].map(lambda x: regime_labels.get(x, {}).get('name', 'Ukjent'))
    df_result['regime_emoji'] = df_result['regime'].map(lambda x: regime_labels.get(x, {}).get('emoji', '❓'))
    df_result['regime_color'] = df_result['regime'].map(lambda x: regime_labels.get(x, {}).get('color', '#808080'))
    df_result['regime_description'] = df_result['regime'].map(lambda x: regime_labels.get(x, {}).get('description', ''))
    df_result['regime_action'] = df_result['regime'].map(lambda x: regime_labels.get(x, {}).get('action', ''))
    
    return df_result, regime_labels, regime_stats


def get_current_regime_info(df_regimes: pd.DataFrame, regime_labels: dict, 
                            hmm_model=None, confidence_threshold: float = 0.60) -> dict:
    """
    Henter info om nåværende regime med confidence og overgangsvarsler.
    
    Forbedringer v2:
    - confidence_threshold: kun vis regime-bytte hvis P(nytt regime) > terskel
    - overgangsvarsel: «gult lys» hvis P(nåværende regime) < 50%
    - regime_confidence: 'high'/'medium'/'low' basert på sannsynlighet
    """
    if df_regimes.empty:
        return None
    
    latest = df_regimes.iloc[-1]
    current_regime = int(latest['regime'])
    
    prob_col = f'prob_regime_{current_regime}'
    current_prob = float(latest[prob_col]) if prob_col in df_regimes.columns else 0.0
    
    # Regime-streak
    regime_streak = 1
    for i in range(len(df_regimes) - 2, -1, -1):
        if int(df_regimes.iloc[i]['regime']) == current_regime:
            regime_streak += 1
        else:
            break
    
    # Samle alle sannsynligheter (kun for aktive regimer)
    all_probs = {}
    for regime_id, label_info in regime_labels.items():
        prob_col = f'prob_regime_{regime_id}'
        if prob_col in df_regimes.columns:
            all_probs[label_info['name']] = {
                'prob': float(latest[prob_col]),
                'color': label_info['color'],
                'emoji': label_info['emoji']
            }
    
    volatility = float(latest['volatility']) if pd.notna(latest['volatility']) else 0.0
    rolling_return = float(latest['rolling_return']) if pd.notna(latest['rolling_return']) else 0.0
    
    # Forventet varighet
    expected_duration = None
    if hmm_model is not None:
        try:
            self_prob = hmm_model.transmat_[current_regime, current_regime]
            if self_prob < 1.0:
                expected_duration = 1.0 / (1.0 - self_prob)
        except Exception:
            pass
    
    current_label = regime_labels.get(current_regime, {})
    
    # Confidence-klassifisering
    if current_prob >= confidence_threshold:
        regime_confidence = 'high'
        confidence_emoji = '🟢'
    elif current_prob >= 0.40:
        regime_confidence = 'medium'
        confidence_emoji = '🟡'
    else:
        regime_confidence = 'low'
        confidence_emoji = '🔴'
    
    # Overgangsvarsel: sjekk om et ANNET regime har høyere sannsynlighet
    transition_warning = None
    for regime_id, label_info in regime_labels.items():
        if regime_id == current_regime:
            continue
        prob_col_other = f'prob_regime_{regime_id}'
        if prob_col_other in df_regimes.columns:
            other_prob = float(latest[prob_col_other])
            if other_prob > current_prob and other_prob > 0.30:
                transition_warning = {
                    'target_regime': label_info['name'],
                    'target_emoji': label_info['emoji'],
                    'target_prob': other_prob,
                    'current_prob': current_prob,
                    'message': f"⚠️ Mulig overgang til {label_info['emoji']} {label_info['name']} "
                               f"(P={other_prob:.0%} vs nåværende P={current_prob:.0%})"
                }
                break
    
    return {
        'regime': current_regime,
        'name': current_label.get('name', 'Ukjent'),
        'emoji': current_label.get('emoji', '❓'),
        'color': current_label.get('color', '#808080'),
        'description': current_label.get('description', ''),
        'action': current_label.get('action', ''),
        'probability': current_prob,
        'streak_days': regime_streak,
        'volatility': volatility,
        'rolling_return': rolling_return,
        'all_probs': all_probs,
        'expected_duration': expected_duration,
        'confidence': regime_confidence,
        'confidence_emoji': confidence_emoji,
        'transition_warning': transition_warning
    }


def beregn_regime_transitions(df_regimes: pd.DataFrame, regime_labels: dict, 
                              hmm_model=None) -> pd.DataFrame:
    """Returnerer overgangsmatrise med lesbare navn."""
    if hmm_model is None:
        return pd.DataFrame()
    
    n_states = hmm_model.n_components
    names = [regime_labels.get(i, {}).get('name', f'Regime {i}') for i in range(n_states)]
    
    transition_matrix = pd.DataFrame(
        hmm_model.transmat_,
        index=names,
        columns=names
    )
    return transition_matrix


def full_regime_analyse(df_market: pd.DataFrame, n_regimes: int = 3,
                        auto_select_regimes: bool = False) -> dict:
    """
    Kjører full HMM-basert regimeanalyse.
    
    Forbedringer v2:
    - Utvidede features (volum-trend, momentum, bredde)
    - Modellcaching til disk
    - BIC-basert optimal regime-valg (auto_select_regimes=True)
    - Confidence-threshold og overgangsvarsler
    """
    try:
        if df_market is None or len(df_market) == 0:
            return None
        
        df_features = beregn_regime_features(df_market)
        
        if len(df_features) < 50:
            return None
        
        # BIC-basert regime-valg
        bic_info = None
        if auto_select_regimes:
            bic_result = velg_optimalt_antall_regimer(df_features)
            n_regimes = bic_result['optimal_n']
            bic_info = bic_result['bic_scores']
            logger.info(f"Auto-valgt {n_regimes} regimer basert på BIC")
        
        hmm_model, scaler, feature_cols, bic_score = tren_hmm_model(
            df_features, n_regimes=n_regimes
        )
        df_regimes, regime_labels, regime_stats = klassifiser_regimer(
            hmm_model, scaler, feature_cols, df_features
        )
        
        if df_regimes.empty:
            return None
        
        current_info = get_current_regime_info(df_regimes, regime_labels, hmm_model)
        
        if current_info is None:
            return None
        
        transition_matrix = beregn_regime_transitions(df_regimes, regime_labels, hmm_model)
        
        return {
            'df_regimes': df_regimes,
            'regime_labels': regime_labels,
            'regime_stats': regime_stats,
            'current_info': current_info,
            'transition_matrix': transition_matrix,
            'hmm_model': hmm_model,
            'scaler': scaler,
            'n_regimes': n_regimes,
            'bic_score': bic_score,
            'bic_info': bic_info,
            'feature_cols': feature_cols
        }
        
    except Exception as e:
        logger.error(f"Regime-analyse feilet: {e}", exc_info=True)
        return None
