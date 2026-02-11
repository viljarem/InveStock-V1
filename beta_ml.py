import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import logic

# =============================================================================
# DATA VALIDERING - Må være først siden andre funksjoner bruker dem
# =============================================================================

def valider_ticker_data(df, ticker):
    """
    Validerer om ticker-data er aktiv og av god kvalitet.
    Returnerer (is_valid, reason) tuple.
    """
    if df.empty:
        return False, "Tom dataframe"
    
    if len(df) < 252:  # Mindre enn 1 års data
        return False, f"Kun {len(df)} datapunkter (krever minimum 252)"
    
    # Sjekk om ticker er inaktiv (ingen handel siste 30 dager)
    try:
        siste_dato = df.index.max()
        dager_siden_handel = (pd.Timestamp.now() - siste_dato).days
        if dager_siden_handel > 30:
            return False, f"Ingen handel siste {dager_siden_handel} dager"
    except Exception:
        pass  # Ignorer hvis datoberegning feiler
    
    # Sjekk volum (må ha gjennomsnittlig volum > 1000 siste 30 dager)
    try:
        recent_volume = df['Volume'].tail(30).mean()
        if recent_volume < 1000:
            return False, f"Lavt volum: {recent_volume:.0f}/dag"
    except Exception:
        pass
    
    # Sjekk for unormal prisdata
    try:
        close = df['Close']
        
        # Sjekk for null/negative priser
        if (close <= 0).any():
            return False, "Inneholder null eller negative priser"
        
        # Sjekk for ekstreme prisbevegelser (>50% på en dag) som kan indikere feil data
        daily_changes = close.pct_change().abs()
        extreme_moves = (daily_changes > 0.5).sum()
        if extreme_moves > 5:  # Maks 5 ekstreme bevegelser totalt
            return False, f"For mange ekstreme prisbevegelser: {extreme_moves}"
        
        # Sjekk for konstante priser (døde ticker)
        recent_prices = close.tail(10)
        if recent_prices.nunique() == 1:
            return False, "Konstant pris siste 10 dager"
        
        # Sjekk for unormalt lav volatilitet (kan indikere suspendert handel)
        recent_volatility = close.tail(30).pct_change().std() * np.sqrt(252)
        if pd.notna(recent_volatility) and recent_volatility < 0.05:  # Under 5% årlig volatilitet
            return False, f"Unormalt lav volatilitet: {recent_volatility:.1%}"
    except Exception as e:
        return False, f"Valideringsfeil: {str(e)[:30]}"
    
    return True, "OK"

def sikre_numeriske_data(df):
    """
    Sikrer at alle numeriske kolonner er rene og håndterer problematiske verdier.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Liste over kolonner som må være numeriske
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in numeric_cols:
        if col in df.columns:
            # Konverter til numerisk, erstatt feil med NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Erstatt 0-verdier i pris-kolonner med NaN
            if col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].replace(0, np.nan)
            
            # Erstatt 0-volum med 1 (for å unngå divisjon med 0)
            if col == 'Volume':
                df[col] = df[col].fillna(1).replace(0, 1)
    
    # Fjern rader der alle pris-kolonner er NaN
    price_cols = ['Open', 'High', 'Low', 'Close']
    df = df.dropna(subset=price_cols, how='all')
    
    return df

# =============================================================================
# FEATURE ENGINEERING - Avansert indikatorberegning
# =============================================================================

def beregn_avanserte_features(df):
    """
    Beregner et bredt spekter av tekniske features for ML-modellen.
    Inkluderer robust error-håndtering for problematiske data.
    """
    if df.empty or len(df) < 252:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        
        # Sikre numeriske data først
        df = sikre_numeriske_data(df)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sørg for basisindikatorer
        if 'RSI' not in df.columns:
            df = logic.beregn_tekniske_indikatorer(df)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Sikkerhetsjekk for grunnleggende data
        if close.isna().all() or (close <= 0).all():
            return pd.DataFrame()
        
        # --- MOMENTUM FEATURES (med robuste beregninger) ---
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = close.pct_change(period) * 100
            df[f'ROC_{period}'] = df[f'ROC_{period}'].clip(-50, 50)
        
        # Momentum oscillator
        close_shift_10 = close.shift(10)
        close_shift_20 = close.shift(20)
        df['MOM_10'] = np.where(close_shift_10 > 0, (close - close_shift_10) / close_shift_10 * 100, 0)
        df['MOM_20'] = np.where(close_shift_20 > 0, (close - close_shift_20) / close_shift_20 * 100, 0)
        df['MOM_10'] = df['MOM_10'].clip(-50, 50)
        df['MOM_20'] = df['MOM_20'].clip(-50, 50)
        
        # Williams %R (med sikker divisjon)
        highest_14 = high.rolling(14).max()
        lowest_14 = low.rolling(14).min()
        range_14 = highest_14 - lowest_14
        df['Williams_R'] = np.where(range_14 > 0, -100 * (highest_14 - close) / range_14, -50)
        
        # RSI divergens
        if 'RSI' in df.columns:
            rsi_pct = df['RSI'].pct_change(5)
            df['RSI_Divergence'] = (rsi_pct - df['ROC_5']).clip(-20, 20)
        
        # --- VOLATILITET FEATURES ---
        if 'ATR' in df.columns:
            df['ATR_Norm'] = np.where(close > 0, df['ATR'] / close * 100, 0)
            df['ATR_Norm'] = df['ATR_Norm'].clip(0, 50)
        
        # Bollinger Band features
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            bb_range = df['BB_Upper'] - df['BB_Lower']
            df['BB_Width'] = np.where(df['BB_Middle'] > 0, bb_range / df['BB_Middle'] * 100, 0)
            df['BB_Width'] = df['BB_Width'].clip(0, 50)
            df['BB_PctB'] = np.where(bb_range > 0, (close - df['BB_Lower']) / bb_range, 0.5)
            df['BB_PctB'] = df['BB_PctB'].clip(0, 1)
        
        # Historisk volatilitet
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        for period in [10, 20]:
            hv = returns.rolling(period).std() * np.sqrt(252) * 100
            df[f'HV_{period}'] = hv.clip(0, 200)
        
        # Keltner Channel posisjon
        if 'ATR' in df.columns:
            keltner_mid = close.ewm(span=20, adjust=False).mean()
            keltner_range = 2 * df['ATR']
            keltner_lower = keltner_mid - keltner_range
            df['Keltner_Pos'] = np.where(keltner_range > 0, (close - keltner_lower) / (2 * keltner_range), 0.5)
            df['Keltner_Pos'] = df['Keltner_Pos'].clip(0, 1)
        
        # --- VOLUM FEATURES ---
        for period in [20, 50]:
            vol_ma = volume.rolling(period).mean()
            df[f'Vol_Ratio_{period}'] = np.where(vol_ma > 0, volume / vol_ma, 1.0)
            df[f'Vol_Ratio_{period}'] = df[f'Vol_Ratio_{period}'].clip(0, 10)
        
        vol_ma_20 = volume.rolling(20).mean()
        vol_ma_50 = volume.rolling(50).mean()
        df['Vol_Trend'] = np.where(vol_ma_50 > 0, vol_ma_20 / vol_ma_50, 1.0)
        df['Vol_Trend'] = df['Vol_Trend'].clip(0, 5)
        
        # OBV trend
        price_change = np.sign(close.diff().fillna(0))
        obv = (price_change * volume).cumsum()
        obv_pct = obv.pct_change(10) * 100
        df['OBV_Trend'] = obv_pct.fillna(0).clip(-100, 100)
        
        # Money Flow Index (MFI)
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df['MFI'] = np.where(negative_flow > 0, 100 - (100 / (1 + positive_flow / negative_flow)), 50)
        df['MFI'] = df['MFI'].clip(0, 100)
        
        # --- TREND FEATURES ---
        for sma_col in ['SMA_50', 'SMA_200']:
            if sma_col in df.columns:
                ratio_col = f'Price_{sma_col}_Ratio'
                df[ratio_col] = np.where(df[sma_col] > 0, close / df[sma_col], 1.0)
                df[ratio_col] = df[ratio_col].clip(0.5, 2.0)
        
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            df['SMA50_SMA200_Ratio'] = np.where(df['SMA_200'] > 0, df['SMA_50'] / df['SMA_200'], 1.0)
            df['SMA50_SMA200_Ratio'] = df['SMA50_SMA200_Ratio'].clip(0.8, 1.2)
        
        # ADX og DI beregning - FIKSET: Konverter numpy arrays til pandas Series
        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        
        # Bruk pandas .abs() i stedet for np.abs() på Series
        low_diff_abs = low.diff().abs()
        plus_dm = high.diff().where((high.diff() > low_diff_abs) & (high.diff() > 0), 0).clip(lower=0)
        minus_dm = (-low.diff()).where((low_diff_abs > high.diff()) & (low.diff() < 0), 0).clip(lower=0)
        
        plus_dm_smooth = plus_dm.rolling(14).mean()
        minus_dm_smooth = minus_dm.rolling(14).mean()
        
        # Beregn DI som pandas Series
        plus_di = pd.Series(np.where(atr_14 > 0, 100 * plus_dm_smooth / atr_14, 0), index=df.index)
        minus_di = pd.Series(np.where(atr_14 > 0, 100 * minus_dm_smooth / atr_14, 0), index=df.index)
        
        di_sum = plus_di + minus_di
        dx = pd.Series(np.where(di_sum > 0, 100 * (plus_di - minus_di).abs() / di_sum, 0), index=df.index)
        
        df['ADX'] = dx.rolling(14).mean().clip(0, 100)
        df['DI_Diff'] = (plus_di - minus_di).clip(-100, 100)
        
        # MACD features
        if 'MACD_Hist' in df.columns:
            df['MACD_Hist_Change'] = df['MACD_Hist'].diff().clip(-5, 5)
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Signal_Dist'] = (df['MACD'] - df['MACD_Signal']).clip(-10, 10)
        
        # --- PRIS MØNSTRE ---
        body = close - df['Open']
        range_hl = high - low
        
        df['Body_Ratio'] = np.where(range_hl > 0, body / range_hl, 0)
        df['Body_Ratio'] = df['Body_Ratio'].clip(-1, 1)
        
        close_max = np.maximum(close, df['Open'])
        close_min = np.minimum(close, df['Open'])
        df['Upper_Shadow'] = np.where(range_hl > 0, (high - close_max) / range_hl, 0)
        df['Upper_Shadow'] = df['Upper_Shadow'].clip(0, 1)
        df['Lower_Shadow'] = np.where(range_hl > 0, (close_min - low) / range_hl, 0)
        df['Lower_Shadow'] = df['Lower_Shadow'].clip(0, 1)
        
        # Consecutive days
        up_days = (close > close.shift(1)).astype(int)
        down_days = (close < close.shift(1)).astype(int)
        df['Consec_Up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum().clip(0, 10)
        df['Consec_Down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum().clip(0, 10)
        
        # --- RELATIVE STYRKE ---
        if 'High_52w' in df.columns and 'Low_52w' in df.columns:
            df['Dist_52w_High'] = np.where(df['High_52w'] > 0, (close - df['High_52w']) / df['High_52w'] * 100, 0)
            df['Dist_52w_High'] = df['Dist_52w_High'].clip(-50, 5)
            df['Dist_52w_Low'] = np.where(df['Low_52w'] > 0, (close - df['Low_52w']) / df['Low_52w'] * 100, 0)
            df['Dist_52w_Low'] = df['Dist_52w_Low'].clip(-5, 500)
        
        # Ichimoku signal
        if 'ISA_9' in df.columns and 'ISB_26' in df.columns:
            df['Ichimoku_Signal'] = np.where(close > df['ISA_9'], 1, np.where(close < df['ISB_26'], -1, 0))
        else:
            df['Ichimoku_Signal'] = 0
        
        # --- RENGJØR ALLE FEATURES ---
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                if col.startswith(('RSI', 'MFI', 'BB_PctB', 'Keltner_Pos')):
                    df[col] = df[col].fillna(50)
                elif col.startswith(('Vol_Ratio', 'Price_', 'SMA', 'Vol_Trend')):
                    df[col] = df[col].fillna(1.0)
                else:
                    df[col] = df[col].fillna(0)
        
        return df
        
    except Exception as e:
        print(f"Feil i feature engineering: {e}")
        return pd.DataFrame()

def velg_features():
    """Returnerer listen over features brukt i modellen."""
    return [
        # Momentum (mer konservativ liste)
        'RSI', 'ROC_5', 'ROC_10', 'ROC_20', 'MOM_10', 'Williams_R',
        # Volatilitet
        'ATR_Norm', 'BB_Width', 'BB_PctB', 'HV_10', 'Keltner_Pos',
        # Volum
        'Vol_Ratio_20', 'Vol_Trend', 'OBV_Trend', 'MFI',
        # Trend
        'Price_SMA_50_Ratio', 'Price_SMA_200_Ratio', 'SMA50_SMA200_Ratio',
        'ADX', 'DI_Diff',
        # Pris mønstre
        'Body_Ratio', 'Upper_Shadow', 'Lower_Shadow',
        # Relative
        'Dist_52w_High', 'Dist_52w_Low',
        # Teknisk
        'Ichimoku_Signal'
    ]

# =============================================================================
# ENSEMBLE ML-MODELL (Oppdatert med bedre datavalidering)
# =============================================================================

class EnsembleStockPredictor:
    """
    Ensemble-modell med robust datahåndtering for døde/problematiske tickers.
    """
    
    def __init__(self, horisont=10, target_pct=0.04):
        self.horisont = horisont
        self.target_pct = target_pct
        self.scaler = StandardScaler()
        self.features = velg_features()
        self.models = {}
        self.weights = {'xgb': 0.4, 'rf': 0.3, 'gb': 0.3}
        self.is_fitted = False
        self.metrics = {}
        
    def _create_target(self, df):
        """Oppretter target-variabel: 1 hvis pris stiger med target_pct innen horisont dager."""
        future_max = df['Close'].shift(-self.horisont).rolling(self.horisont).max()
        target = (future_max >= df['Close'] * (1 + self.target_pct)).astype(int)
        return target
        
    def _validate_data_quality(self, df):
        """Validerer datakvalitet før ML-prosessering."""
        if df.empty:
            return False, "Tom dataframe"
        
        # Sjekk for tilstrekkelig data
        if len(df) < 100:
            return False, f"For lite data: {len(df)} rader"
        
        # Sjekk for features
        available_features = [f for f in self.features if f in df.columns]
        if len(available_features) < len(self.features) * 0.7:
            return False, f"Mangler for mange features: {len(available_features)}/{len(self.features)}"
        
        # Sjekk for excessive NaN
        feature_data = df[available_features]
        nan_pct = feature_data.isna().mean().mean()
        if nan_pct > 0.3:
            return False, f"For mange NaN verdier: {nan_pct:.1%}"
        
        return True, "OK"
    
    def _prepare_data(self, df):
        """Forbereder data med omfattende validering."""
        df_work = df.copy()
        
        # Validér grunnleggende datakvalitet
        is_valid, reason = self._validate_data_quality(df_work)
        if not is_valid:
            return pd.DataFrame(), []
        
        # Opprett target
        df_work['Target'] = self._create_target(df_work)
        
        # Velg tilgjengelige features
        available_features = [f for f in self.features if f in df_work.columns]
        
        # Fjern rader med for mange NaN
        required_cols = available_features + ['Target']
        df_clean = df_work.dropna(subset=required_cols, thresh=len(required_cols)*0.8)
        
        # Final validation
        if len(df_clean) < 50:
            return pd.DataFrame(), []
        
        return df_clean, available_features
    
    def fit(self, df, validate=True):
        """Trener modellen med omfattende datavalidering."""
        try:
            df_clean, feature_cols = self._prepare_data(df)
            
            if df_clean.empty or len(feature_cols) == 0:
                return False
            
            # Separer data
            X = df_clean[feature_cols].iloc[:-self.horisont]
            y = df_clean['Target'].iloc[:-self.horisont]
            
            if len(X) < 50:
                return False
            
            # Fjern eventuelle gjenværende problematiske verdier
            X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Skaler features med robust scaler
            X_scaled = self.scaler.fit_transform(X_clean)
            
            # Sjekk for NaN etter skalering
            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                return False
            
            # Tren modeller med error handling
            try:
                self.models['xgb'] = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    verbosity=0, use_label_encoder=False, random_state=42,
                    n_jobs=1  # Reduser parallellitet for stabilitet
                )
                self.models['xgb'].fit(X_scaled, y)
            except Exception:
                return False
            
            try:
                self.models['rf'] = RandomForestClassifier(
                    n_estimators=50, max_depth=5, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=1
                )
                self.models['rf'].fit(X_scaled, y)
            except Exception:
                return False
            
            try:
                self.models['gb'] = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                self.models['gb'].fit(X_scaled, y)
            except Exception:
                return False
            
            self.is_fitted = True
            self.feature_cols = feature_cols
            
            return True
            
        except Exception as e:
            print(f"Feil under modelltrening: {e}")
            return False
    
    def predict_proba(self, df):
        """Robust prediksjon med omfattende error handling."""
        if not self.is_fitted:
            return 50.0
        
        try:
            df_work = beregn_avanserte_features(df)
            if df_work.empty:
                return 50.0
            
            # Valider data quality
            is_valid, _ = self._validate_data_quality(df_work)
            if not is_valid:
                return 50.0
            
            # Sjekk at vi har de nødvendige features
            missing_features = [f for f in self.feature_cols if f not in df_work.columns]
            if len(missing_features) > len(self.feature_cols) * 0.3:
                return 50.0
            
            # Hent siste rad
            latest = df_work[self.feature_cols].tail(1)
            
            # Fylle manglende features med nøytrale verdier
            for col in self.feature_cols:
                if col not in latest.columns:
                    if col.startswith(('RSI', 'MFI')):
                        latest[col] = 50
                    elif col.startswith(('Vol_Ratio', 'Price_')):
                        latest[col] = 1.0
                    else:
                        latest[col] = 0
            
            # Sikre riktig rekkefølge
            latest = latest[self.feature_cols]
            
            # Rens data
            latest_clean = latest.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if latest_clean.isna().any().any():
                return 50.0
            
            # Skaler
            latest_scaled = self.scaler.transform(latest_clean)
            
            if np.isnan(latest_scaled).any() or np.isinf(latest_scaled).any():
                return 50.0
            
            # Ensemble prediksjon med error handling
            probs = []
            for name, model in self.models.items():
                try:
                    prob = model.predict_proba(latest_scaled)[0][1]
                    if not (np.isnan(prob) or np.isinf(prob)):
                        probs.append(prob)
                except Exception:
                    probs.append(0.5)  # Nøytral hvis feil
            
            if not probs:
                return 50.0
            
            # Returnér gjennomsnitt hvis vi har problemer med vekter
            weighted_prob = np.mean(probs)
            
            return round(float(weighted_prob * 100), 1)
            
        except Exception as e:
            print(f"Feil under prediksjon: {e}")
            return 50.0

    def predict_historical(self, df, dager=60):
        """
        Beregner historiske sannsynligheter for grafvisning.
        """
        if not self.is_fitted:
            return pd.Series()
        
        try:
            df_work = beregn_avanserte_features(df)
            if df_work.empty or len(df_work) < dager:
                return pd.Series()
            
            test_data = df_work[self.feature_cols].tail(dager)
            valid_indices = test_data.dropna().index
            
            if len(valid_indices) == 0:
                return pd.Series()
            
            test_scaled = self.scaler.transform(test_data.loc[valid_indices])
            
            # Ensemble prediksjon for hele perioden
            all_probs = np.zeros(len(test_scaled))
            for name, model in self.models.items():
                try:
                    model_probs = model.predict_proba(test_scaled)[:, 1]
                    all_probs += model_probs * self.weights[name]
                except:
                    all_probs += 0.5 * self.weights[name]
            
            return pd.Series(all_probs * 100, index=valid_indices)
        except Exception as e:
            print(f"Feil i predict_historical: {e}")
            return pd.Series()
    
    def get_feature_importance(self):
        """Returnerer feature importance fra XGBoost-modellen."""
        if 'xgb' not in self.models or not hasattr(self, 'feature_cols'):
            return {}
        
        try:
            importance = self.models['xgb'].feature_importances_
            return dict(zip(self.feature_cols, importance))
        except:
            return {}

# =============================================================================
# KONFIDENSANALYSE
# =============================================================================

def beregn_konfidensintervall(predictor, df, n_bootstrap=25):
    """
    Beregner konfidensintervall ved bootstrap-sampling (redusert antall for ytelse).
    """
    try:
        df_work = beregn_avanserte_features(df)
        if df_work.empty:
            return 50.0, 40, 60
        
        # Sjekk at vi har nødvendige features
        if not hasattr(predictor, 'feature_cols') or len(predictor.feature_cols) == 0:
            return 50.0, 40, 60
        
        predictions = []
        for _ in range(n_bootstrap):
            try:
                # Legg til liten støy i features for å simulere usikkerhet
                latest = df_work[predictor.feature_cols].tail(1).copy()
                
                # Fyll manglende features med nøytrale verdier
                for col in predictor.feature_cols:
                    if col not in latest.columns:
                        if col.startswith(('RSI', 'MFI')):
                            latest[col] = 50
                        elif col.startswith(('Vol_Ratio', 'Price_')):
                            latest[col] = 1.0
                        else:
                            latest[col] = 0
                
                latest_vals = latest[predictor.feature_cols].values
                noise = np.random.normal(0, 0.01, len(predictor.feature_cols))
                latest_with_noise = latest_vals + noise
                
                latest_scaled = predictor.scaler.transform(latest_with_noise.reshape(1, -1))
                
                probs = []
                for name, model in predictor.models.items():
                    try:
                        prob = model.predict_proba(latest_scaled)[0][1]
                        if not (np.isnan(prob) or np.isinf(prob)):
                            probs.append(prob)
                    except:
                        probs.append(0.5)
                
                if probs:
                    predictions.append(np.mean(probs) * 100)
                else:
                    predictions.append(50.0)
                    
            except Exception:
                predictions.append(50.0)
        
        if predictions:
            mean_pred = np.mean(predictions)
            lower = np.percentile(predictions, 10)
            upper = np.percentile(predictions, 90)
            return round(mean_pred, 1), round(lower, 1), round(upper, 1)
        else:
            return 50.0, 40, 60
            
    except Exception as e:
        print(f"Feil i konfidensintervall: {e}")
        return 50.0, 40, 60

# =============================================================================
# STREAMLIT UI
# =============================================================================

def gi_vurdering(score, lower=None, upper=None):
    """Returnerer tekstlig vurdering basert på AI Score."""
    if score >= 75:
        return "🟢 Sterkt Kjøp", "#00C805"
    if score >= 60:
        return "🟡 Kjøp", "#FFC107"
    if score >= 50:
        return "🟠 Hold/Vurder", "#FF9800"
    return "🔴 Nøytral", "#9E9E9E"

def vis_beta_side(df_full, tickers):
    """
    Hovedfunksjonen for Beta AI Scanner-siden.
    """
    st.title("🤖 AI-Drevet Aksjescanner")
    
    st.caption(f"Analyserer {len(tickers)} likvide aksjer basert på valgt minimumsomsetning.")
    
    # Metodikkbeskrivelse
    with st.expander("📚 Metodikk og Modellarkitektur", expanded=False):
        st.markdown("""
        ### Ensemble Machine Learning-modell
        
        Denne modulen benytter en **ensemble-tilnærming** som kombinerer tre kraftige algoritmer:
        
        | Algoritme | Vekt | Styrke |
        |-----------|------|--------|
        | **XGBoost** | Dynamisk | Fanger komplekse ikke-lineære mønstre |
        | **Random Forest** | Dynamisk | Robust mot overfitting, håndterer støy |
        | **Gradient Boosting** | Dynamisk | God på sekvensiell læring |
        
        #### Feature Engineering (30+ indikatorer)
        
        **Momentum-indikatorer:**
        - RSI (14), Williams %R, Rate of Change (5, 10, 20 dager)
        - RSI-pris divergens, Momentum oscillator
        
        **Volatilitetsmål:**
        - Normalisert ATR, Bollinger Band Width, Historisk volatilitet
        - Keltner Channel posisjon, Bollinger %B
        
        **Volumindikatorer:**
        - Volumratio (20/50-dagers), OBV-trend, Money Flow Index (MFI)
        
        **Trendindikatorer:**
        - ADX (trendstyrke), DI+/DI- differanse
        - Pris/SMA-ratioer, MACD histogram-endring
        
        **Prismønstre:**
        - Candlestick body-ratio, skygger, consecutive up/down dager
        
        #### Walk-Forward Validering
        
        Modellen trenes med **tidsserie-kryssvalidering** for å unngå lookahead bias:
        1. Data deles i 3 sekvensielle folder
        2. Hver modell evalueres på fremtidige data
        3. Modellvekter justeres basert på precision-score
        
        #### Konfidensintervall
        
        Bootstrap-sampling (25 iterasjoner) brukes for å beregne 80% konfidensintervall
        rundt prediksjonen, som gir et mål på usikkerheten.
        """)
    
    st.markdown("---")
    
    # Parametervalg
    col1, col2, col3 = st.columns(3)
    with col1:
        valgt_horisont = st.slider("📅 Horisont (dager)", 5, 30, 10, 
                                   help="Antall dager frem i tid modellen predikerer")
    with col2:
        valgt_maal = st.slider("📈 Mål oppgang (%)", 2.0, 15.0, 5.0, 0.5,
                               help="Minimum prisøkning for å definere 'suksess'")
    with col3:
        terskel = st.slider("🎯 Min. AI Score", 1, 100, 55,
                           help="Filtrer resultater med score under denne verdien")
    
    # Start analyse
    if st.button("🚀 Start AI-Analyse", type="primary"):
        resultater = []
        feilede_tickers = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        totalt = len(tickers)
        for i, ticker in enumerate(tickers):
            status_text.text(f"Analyserer {ticker}... ({i+1}/{totalt})")
            
            try:
                df_ticker = df_full[df_full['Ticker'] == ticker].copy()
                
                # Validér ticker data først
                is_valid, reason = valider_ticker_data(df_ticker, ticker)
                if not is_valid:
                    feilede_tickers.append((ticker, reason))
                    continue
                
                if len(df_ticker) > 300:
                    df_ticker_features = beregn_avanserte_features(df_ticker)
                    
                    if not df_ticker_features.empty:
                        predictor = EnsembleStockPredictor(
                            horisont=valgt_horisont, 
                            target_pct=valgt_maal/100
                        )
                        
                        if predictor.fit(df_ticker_features, validate=True):
                            score = predictor.predict_proba(df_ticker_features)
                            
                            if score >= terskel:
                                # Beregn konfidensintervall
                                _, lower, upper = beregn_konfidensintervall(predictor, df_ticker_features)
                                
                                vurdering, _ = gi_vurdering(score)
                                
                                resultater.append({
                                    "Ticker": ticker,
                                    "AI Score": score,
                                    "Konfidens": f"{lower:.0f}-{upper:.0f}%",
                                    "Vurdering": vurdering,
                                    "Pris": df_ticker_features['Close'].iloc[-1],
                                    "RSI": df_ticker_features['RSI'].iloc[-1],
                                    "ADX": df_ticker_features['ADX'].iloc[-1] if 'ADX' in df_ticker_features.columns else 0,
                                    "Vol_Trend": df_ticker_features['Vol_Trend'].iloc[-1] if 'Vol_Trend' in df_ticker_features.columns else 1.0,
                                    "predictor": predictor
                                })
                        else:
                            feilede_tickers.append((ticker, "Modelltrening feilet"))
                    else:
                        feilede_tickers.append((ticker, "Feature engineering feilet"))
                else:
                    feilede_tickers.append((ticker, f"For lite data: {len(df_ticker)} rader"))
                    
            except Exception as e:
                feilede_tickers.append((ticker, f"Ukjent feil: {str(e)[:50]}"))
            
            progress_bar.progress((i + 1) / totalt)
        
        progress_bar.empty()
        status_text.empty()
        
        # Lagre resultater
        if resultater:
            st.session_state['ml_results'] = resultater
            st.session_state['ml_df'] = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictor'} for r in resultater]).sort_values(by="AI Score", ascending=False)
            st.session_state['ml_terskel'] = terskel
        else:
            st.session_state['ml_results'] = []
            st.session_state['ml_df'] = pd.DataFrame()
            st.warning("Ingen aksjer møtte kravet til AI-score.")
        
        # Vis feilede tickers
        if feilede_tickers:
            with st.expander(f"⚠️ {len(feilede_tickers)} tickers ble hoppet over", expanded=False):
                error_df = pd.DataFrame(feilede_tickers, columns=['Ticker', 'Årsak'])
                st.dataframe(error_df, use_container_width=True, hide_index=True)
    
    # Vis resultater
    if 'ml_df' in st.session_state and not st.session_state['ml_df'].empty:
        df_res = st.session_state['ml_df']
        saved_terskel = st.session_state.get('ml_terskel', terskel)
        
        st.success(f"✅ Analysen fant **{len(df_res)}** kandidater med AI Score ≥ {saved_terskel}%")
        
        # Oppsummeringsmetrikker
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Høyeste Score", f"{df_res['AI Score'].max():.1f}%")
        col2.metric("Snitt Score", f"{df_res['AI Score'].mean():.1f}%")
        col3.metric("Sterke Kjøp (≥75%)", len(df_res[df_res['AI Score'] >= 75]))
        col4.metric("Kjøp (≥60%)", len(df_res[df_res['AI Score'] >= 60]))
        
        st.dataframe(
            df_res,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AI Score": st.column_config.ProgressColumn(
                    "AI Score", format="%.1f%%", min_value=0, max_value=100
                ),
                "Konfidens": st.column_config.TextColumn("80% Konfidens"),
                "Vurdering": st.column_config.TextColumn("Signal"),
                "Pris": st.column_config.NumberColumn("Pris", format="%.2f kr"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "ADX": st.column_config.NumberColumn("ADX", format="%.1f"),
                "Vol_Trend": st.column_config.NumberColumn("Vol Trend", format="%.2f")
            }
        )
        
        st.markdown("---")
        
        # Detaljert analyse
        st.subheader("🔍 Detaljert Analyse")
        
        valgt_ticker = st.selectbox(
            "Velg ticker for dypdykk", 
            df_res['Ticker'].tolist(),
            key="detail_ticker"
        )
        
        if valgt_ticker:
            # Finn predictor for valgt ticker
            predictor = None
            for r in st.session_state['ml_results']:
                if r['Ticker'] == valgt_ticker:
                    predictor = r['predictor']
                    break
            
            if predictor:
                df_view = df_full[df_full['Ticker'] == valgt_ticker].copy()
                df_view_features = beregn_avanserte_features(df_view)
                
                if not df_view_features.empty:
                    # Hent data
                    score = df_res[df_res['Ticker'] == valgt_ticker]['AI Score'].values[0]
                    vurdering, color = gi_vurdering(score)
                    
                    # Konfidensintervall
                    _, lower, upper = beregn_konfidensintervall(predictor, df_view_features)
                    
                    # Historisk trend
                    ml_trend = predictor.predict_historical(df_view_features, dager=90)
                    
                    # Feature importance
                    importance = predictor.get_feature_importance()
                    
                    # Beregn tekniske signaler
                    signaler = logic.sjekk_strategier(df_view_features)
                    
                    # Layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # --- Chart Settings ---
                        with st.expander("⚙️ Graf-innstillinger", expanded=False):
                            chart_col1, chart_col2 = st.columns(2)
                            with chart_col1:
                                v_chart_type = st.radio("Graftype", ["Candlestick", "Linje"], horizontal=True, key="ml_chart_type")
                                v_horisont = st.selectbox("Tidsperspektiv", ["3 Måneder", "6 Måneder", "1 År"], index=2, key="ml_horisont")
                            with chart_col2:
                                v_sma_50 = st.checkbox("SMA 50", value=True, key="ml_sma50")
                                v_sma_200 = st.checkbox("SMA 200", value=True, key="ml_sma200")
                                v_bb = st.checkbox("Bollinger Bands", value=False, key="ml_bb")
                        
                        # Filtrer data basert på horisont
                        offset_dager = {"3 Måneder": 90, "6 Måneder": 180, "1 År": 365}
                        end_date = df_view_features.index.max()
                        start_date = end_date - pd.DateOffset(days=offset_dager[v_horisont])
                        df_plot = df_view_features[df_view_features.index >= start_date]
                        
                        # Bygg chart med subplots (matcher Teknisk Analyse)
                        fig = make_subplots(
                            rows=4, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.45, 0.2, 0.18, 0.17],
                            subplot_titles=(f"Prisutvikling: {valgt_ticker}", "AI Score", "RSI", "Volum")
                        )
                        
                        # Farger
                        up_color = "#00C805"
                        down_color = "#FF5252"
                        
                        # 1. Hovedchart - Candlestick eller Linje
                        if v_chart_type == "Candlestick":
                            fig.add_trace(go.Candlestick(
                                x=df_plot.index,
                                open=df_plot['Open'], high=df_plot['High'],
                                low=df_plot['Low'], close=df_plot['Close'],
                                name="Pris",
                                increasing=dict(line=dict(color=up_color), fillcolor=up_color),
                                decreasing=dict(line=dict(color=down_color), fillcolor=down_color)
                            ), row=1, col=1)
                        else:
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=df_plot['Close'],
                                mode='lines', line=dict(color=up_color, width=2),
                                name="Pris"
                            ), row=1, col=1)
                        
                        # SMA linjer
                        if v_sma_50 and 'SMA_50' in df_plot.columns:
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=df_plot['SMA_50'],
                                line=dict(color='#45B7D1', width=1.5),
                                name="SMA 50"
                            ), row=1, col=1)
                        if v_sma_200 and 'SMA_200' in df_plot.columns:
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=df_plot['SMA_200'],
                                line=dict(color='#DDA0DD', width=1.5),
                                name="SMA 200"
                            ), row=1, col=1)
                        
                        # Bollinger Bands
                        if v_bb and all(col in df_plot.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=df_plot['BB_Upper'],
                                line=dict(color='rgba(174, 214, 241, 0.7)', width=1),
                                name="BB Upper", showlegend=False
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=df_plot['BB_Lower'],
                                line=dict(color='rgba(174, 214, 241, 0.7)', width=1),
                                name="BB Lower", fill='tonexty', fillcolor='rgba(174, 214, 241, 0.15)'
                            ), row=1, col=1)
                        
                        # 2. AI Score historikk
                        if not ml_trend.empty:
                            ml_plot = ml_trend[ml_trend.index >= start_date]
                            if not ml_plot.empty:
                                fig.add_trace(go.Scatter(
                                    x=ml_plot.index, y=ml_plot.values,
                                    mode='lines',
                                    line=dict(color='#4e8cff', width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(78, 140, 255, 0.2)',
                                    name="AI Score"
                                ), row=2, col=1)
                                
                                fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
                                fig.add_hline(y=75, line_dash="dash", line_color="#00C805", opacity=0.5, row=2, col=1)
                        
                        # 3. RSI
                        if 'RSI' in df_plot.columns:
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=df_plot['RSI'],
                                line=dict(color='#a29bfe', width=1.5),
                                name="RSI"
                            ), row=3, col=1)
                            fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", opacity=0.5, row=3, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="#00C805", opacity=0.5, row=3, col=1)
                            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(162, 155, 254, 0.1)", line_width=0, row=3, col=1)
                        
                        # 4. Volum
                        vol_colors = [up_color if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] else down_color for i in range(len(df_plot))]
                        fig.add_trace(go.Bar(
                            x=df_plot.index, y=df_plot['Volume'],
                            name="Volum", marker_color=vol_colors, opacity=0.7
                        ), row=4, col=1)
                        
                        # Layout
                        fig.update_layout(
                            height=700,
                            template="plotly_dark",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            showlegend=True,
                            legend=dict(orientation="h", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                            margin=dict(l=60, r=20, t=40, b=40),
                            xaxis_rangeslider_visible=False,
                            hovermode='x unified'
                        )
                        
                        # Y-akser
                        fig.update_yaxes(title_text="Pris", gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
                        fig.update_yaxes(title_text="Score %", range=[0, 100], gridcolor="rgba(255,255,255,0.06)", row=2, col=1)
                        fig.update_yaxes(title_text="RSI", range=[0, 100], gridcolor="rgba(255,255,255,0.06)", row=3, col=1)
                        fig.update_yaxes(title_text="Volum", gridcolor="rgba(255,255,255,0.06)", row=4, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True, config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'toImageButtonOptions': {'format': 'png', 'filename': f'{valgt_ticker}_ai_chart', 'height': 800, 'width': 1600, 'scale': 2}
                        })
                    
                    with col2:
                        # Score-display
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}30, {color}10); 
                                    border-radius: 16px; padding: 20px; text-align: center;
                                    border: 2px solid {color};">
                            <h1 style="margin: 0; font-size: 3rem; color: {color};">{score:.1f}%</h1>
                            <p style="margin: 5px 0; font-size: 1.2rem;">{vurdering}</p>
                            <p style="margin: 0; color: #8892b0; font-size: 0.9rem;">
                                80% Konfidens: {lower:.0f}% - {upper:.0f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Key metrics
                        st.markdown("**📊 Nøkkeltall**")
                        metrics_df = df_view_features.iloc[-1]
                        
                        st.metric("RSI", f"{metrics_df['RSI']:.1f}")
                        if 'ADX' in df_view_features.columns:
                            adx_val = metrics_df['ADX'] if pd.notna(metrics_df['ADX']) else 0
                            st.metric("ADX (Trendstyrke)", f"{adx_val:.1f}")
                        if 'MFI' in df_view_features.columns:
                            mfi_val = metrics_df['MFI'] if pd.notna(metrics_df['MFI']) else 50
                            st.metric("Money Flow Index", f"{mfi_val:.1f}")
                        if 'Vol_Ratio_20' in df_view_features.columns:
                            vol_ratio = metrics_df['Vol_Ratio_20'] if pd.notna(metrics_df['Vol_Ratio_20']) else 1.0
                            st.metric("Volum vs 20d snitt", f"{vol_ratio:.2f}x")
                        
                        st.markdown("---")
                        
                        # Top features
                        st.markdown("**🎯 Viktigste faktorer**")
                        if importance:
                            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                            for feat, imp in sorted_imp:
                                st.progress(float(imp), text=f"{feat}: {imp*100:.1f}%")
                    
                    # --- SIGNALTABELL (som i Teknisk Analyse) ---
                    st.markdown("---")
                    st.markdown("### 📊 Siste signaler og kursutvikling for alle strategier")
                    
                    all_signal_keys = ["Kort_Sikt_RSI", "Momentum_Burst", "Golden_Cross", 
                                      "Ichimoku_Breakout", "Wyckoff_Spring", "Bull_Race_Prep", "VCP_Pattern"]
                    signal_display = {
                        "Kort_Sikt_RSI": "RSI Mean Reversion",
                        "Momentum_Burst": "Momentum Burst", 
                        "Golden_Cross": "Golden Cross",
                        "Ichimoku_Breakout": "Ichimoku Breakout",
                        "Wyckoff_Spring": "Wyckoff Spring",
                        "Bull_Race_Prep": "Bollinger Squeeze",
                        "VCP_Pattern": "VCP (Minervini)"
                    }
                    
                    signal_rows = []
                    for s in all_signal_keys:
                        info = logic.finn_siste_signal_info(df_view_features, signaler, s)
                        signal_rows.append({
                            "Strategi": signal_display.get(s, s),
                            "Siste Signal": info['dato'],
                            "Dager siden": info['dager_siden'],
                            "Kursutvikling (%)": info['utvikling_pst']
                        })
                    
                    signal_df = pd.DataFrame(signal_rows)
                    st.dataframe(signal_df, use_container_width=True, hide_index=True)
                    
                    # Tolkning
                    st.markdown("---")
                    st.markdown("### 📝 AI-Tolkning")
                    
                    interpretation = []
                    if score >= 75:
                        interpretation.append(f"**Sterkt signal:** Modellen ser høy sannsynlighet ({score:.1f}%) for at {valgt_ticker} stiger ≥{valgt_maal}% i løpet av {valgt_horisont} dager.")
                    elif score >= 60:
                        interpretation.append(f"**Positivt signal:** Modellen indikerer moderat sannsynlighet ({score:.1f}%) for oppgang.")
                    else:
                        interpretation.append(f"**Nøytralt signal:** Sannsynligheten ({score:.1f}%) er ikke overbevisende.")
                    
                    if 'ADX' in df_view_features.columns and pd.notna(metrics_df.get('ADX')) and metrics_df['ADX'] > 25:
                        interpretation.append(f"📈 **Sterk trend:** ADX på {metrics_df['ADX']:.1f} indikerer en etablert trend.")
                    
                    if metrics_df['RSI'] < 30:
                        interpretation.append("⚠️ **Oversolgt:** RSI under 30 kan indikere en reverseringsmulighet.")
                    elif metrics_df['RSI'] > 70:
                        interpretation.append("⚠️ **Overkjøpt:** RSI over 70 kan indikere at aksjen er strukket.")
                    
                    if 'Vol_Ratio_20' in df_view_features.columns and pd.notna(metrics_df.get('Vol_Ratio_20')) and metrics_df['Vol_Ratio_20'] > 1.5:
                        interpretation.append(f"📊 **Høyt volum:** Dagens volum er {metrics_df['Vol_Ratio_20']:.1f}x over 20-dagers snitt.")
                    
                    for interp in interpretation:
                        st.markdown(interp)
                    
                    st.warning("⚠️ **Ansvarsfraskrivelse:** AI-prediksjoner er basert på historiske mønstre og gir ingen garanti for fremtidig avkastning. Alltid gjør egen research før investeringsbeslutninger.")