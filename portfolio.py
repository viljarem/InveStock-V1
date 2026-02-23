# portfolio.py
"""
Portefølje-modul for InveStock Pro.
Håndterer:
- Persistent lagring av posisjoner (JSON)
- Exit-signaler med prioritering
- Trailing stop beregning
- Portefølje-analyse og daglig rapport
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import config
from log_config import get_logger

logger = get_logger(__name__)

# Maks antall backup-filer som beholdes
_MAX_BACKUPS = 5

# =============================================================================
# PORTFOLIO STORAGE - JSON-basert persistent lagring
# =============================================================================

def _get_portfolio_path() -> str:
    """Returnerer filsti for portefølje-lagring."""
    return getattr(config, 'PORTFOLIO_FILE', os.path.join(config.DATA_DIR, 'portfolio.json'))

def _get_history_path() -> str:
    """Returnerer filsti for transaksjonshistorikk."""
    return getattr(config, 'PORTFOLIO_HISTORY_FILE', os.path.join(config.DATA_DIR, 'portfolio_history.json'))


def _create_backup(path: str) -> None:
    """Lager roterende backup av portefølje-filen (.bak, .bak.1, .bak.2, ...)."""
    try:
        # Roter eksisterende backups
        for i in range(_MAX_BACKUPS - 1, 0, -1):
            old = f"{path}.bak.{i}"
            new = f"{path}.bak.{i + 1}"
            if os.path.exists(old):
                if i + 1 >= _MAX_BACKUPS:
                    os.remove(old)
                else:
                    shutil.move(old, new)
        
        # Flytt nåværende .bak til .bak.1
        bak = f"{path}.bak"
        if os.path.exists(bak):
            shutil.move(bak, f"{path}.bak.1")
        
        # Kopier original til .bak
        shutil.copy2(path, bak)
        logger.debug(f"Backup opprettet: {bak}")
    except Exception as e:
        logger.warning(f"Kunne ikke lage backup: {e}")


def _restore_from_backup(path: str) -> Dict:
    """Forsøker å gjenopprette portefølje fra backup-filer."""
    backup_paths = [f"{path}.bak"] + [f"{path}.bak.{i}" for i in range(1, _MAX_BACKUPS + 1)]
    
    for bak_path in backup_paths:
        if os.path.exists(bak_path):
            try:
                with open(bak_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'positions' in data:
                    logger.warning(f"Gjenopprettet portefølje fra backup: {bak_path}")
                    # Skriv tilbake til original
                    shutil.copy2(bak_path, path)
                    return data
            except (json.JSONDecodeError, Exception):
                continue
    
    logger.error("Ingen gyldig backup funnet, returnerer tom portefølje")
    return {
        'positions': {},
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'version': '1.0'
    }

def load_portfolio() -> Dict:
    """
    Laster portefølje fra JSON-fil.
    Returnerer dict med posisjoner og metadata.
    """
    path = _get_portfolio_path()
    
    if not os.path.exists(path):
        return {
            'positions': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        if not raw.strip():
            logger.warning("Portefølje-filen er tom, returnerer default")
            return {
                'positions': {},
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
        data = json.loads(raw)
        # Valider minimumstruktur
        if not isinstance(data, dict) or 'positions' not in data:
            logger.error("Portefølje-filen har ugyldig struktur, prøver backup")
            return _restore_from_backup(path)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Korrupt portefølje-JSON: {e}. Prøver backup.")
        return _restore_from_backup(path)
    except Exception as e:
        logger.error(f"Feil ved lasting av portefølje: {e}")
        return {
            'positions': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }

def save_portfolio(portfolio: Dict) -> bool:
    """
    Lagrer portefølje til JSON-fil.
    Lager backup av eksisterende fil før overskriving.
    Returnerer True hvis vellykket.
    """
    path = _get_portfolio_path()
    portfolio['last_updated'] = datetime.now().isoformat()
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Lag backup av eksisterende fil
        if os.path.exists(path):
            _create_backup(path)
        
        # Skriv til temp-fil først, deretter rename (atomisk)
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)
        
        # Verifiser at tmp-filen er gyldig JSON
        with open(tmp_path, 'r', encoding='utf-8') as f:
            json.load(f)  # Kaster feil hvis ugyldig
        
        # Erstatt original med tmp (tilnærmet atomisk)
        shutil.move(tmp_path, path)
        logger.debug(f"Portefølje lagret: {len(portfolio.get('positions', {}))} posisjoner")
        return True
    except Exception as e:
        logger.error(f"Feil ved lagring av portefølje: {e}")
        # Rydd opp tmp-fil
        tmp_path = path + '.tmp'
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def load_transaction_history() -> List[Dict]:
    """Laster transaksjonshistorikk."""
    path = _get_history_path()
    
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_transaction(transaction: Dict) -> bool:
    """Legger til en transaksjon i historikken."""
    history = load_transaction_history()
    transaction['timestamp'] = datetime.now().isoformat()
    transaction['id'] = len(history) + 1
    history.append(transaction)
    
    path = _get_history_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

# =============================================================================
# POSITION MANAGEMENT - Legg til, selg, juster
# =============================================================================

def add_position(
    ticker: str,
    quantity: int,
    buy_price: float,
    buy_date: str = None,
    stop_loss: float = None,
    notes: str = "",
    strategy: str = None
) -> Tuple[bool, str]:
    """
    Legger til eller øker en posisjon.
    
    Args:
        ticker: Aksje-ticker (f.eks. 'VAR.OL')
        quantity: Antall aksjer
        buy_price: Kjøpskurs
        buy_date: Kjøpsdato (YYYY-MM-DD), default i dag
        stop_loss: Initial stop loss (default: buy_price * 0.92)
        notes: Notater om posisjonen
        strategy: Strategi som trigget kjøpet
    
    Returns:
        (success, message)
    """
    # === INPUT-VALIDERING ===
    if not ticker or not isinstance(ticker, str):
        return False, "Ugyldig ticker: tom eller feil type"
    
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        return False, f"Ugyldig antall: {quantity}. Må være et positivt tall."
    quantity = int(quantity)
    
    if not isinstance(buy_price, (int, float)) or buy_price <= 0:
        return False, f"Ugyldig kjøpskurs: {buy_price}. Må være et positivt tall."
    if buy_price > 100_000:
        return False, f"Urimelig høy kjøpskurs: {buy_price}. Maks 100 000 NOK."
    
    portfolio = load_portfolio()
    
    if buy_date is None:
        buy_date = datetime.now().strftime('%Y-%m-%d')
    else:
        # Valider datoformat
        try:
            datetime.strptime(buy_date, '%Y-%m-%d')
        except ValueError:
            return False, f"Ugyldig datoformat: '{buy_date}'. Bruk YYYY-MM-DD."
    
    if stop_loss is not None:
        if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
            return False, f"Ugyldig stop loss: {stop_loss}. Må være et positivt tall."
        if stop_loss >= buy_price:
            return False, f"Stop loss ({stop_loss:.2f}) kan ikke være ≥ kjøpskurs ({buy_price:.2f})."
    else:
        stop_loss = buy_price * 0.92  # 8% under kjøpskurs
    
    ticker = ticker.upper()
    logger.info(f"Legger til posisjon: {ticker} × {quantity} @ {buy_price:.2f}")
    
    if ticker in portfolio['positions']:
        # Øk eksisterende posisjon (gjennomsnittlig kjøpskurs)
        pos = portfolio['positions'][ticker]
        old_qty = pos['quantity']
        old_avg = pos['avg_price']
        
        new_qty = old_qty + quantity
        new_avg = ((old_qty * old_avg) + (quantity * buy_price)) / new_qty
        
        pos['quantity'] = new_qty
        pos['avg_price'] = round(new_avg, 2)
        pos['last_added'] = buy_date
        
        # Oppdater trailing high hvis ny pris er høyere
        if buy_price > pos.get('trailing_high', 0):
            pos['trailing_high'] = buy_price
        
        message = f"Økte posisjon i {ticker}: +{quantity} aksjer, ny snitt: {new_avg:.2f}"
    else:
        # Ny posisjon
        portfolio['positions'][ticker] = {
            'quantity': quantity,
            'avg_price': buy_price,
            'buy_date': buy_date,
            'last_added': buy_date,
            'stop_loss': round(stop_loss, 2),
            'trailing_high': buy_price,
            'notes': notes,
            'strategy': strategy
        }
        message = f"Lagt til {ticker}: {quantity} aksjer @ {buy_price:.2f}"
    
    # Logg transaksjon
    save_transaction({
        'type': 'BUY',
        'ticker': ticker,
        'quantity': quantity,
        'price': buy_price,
        'date': buy_date,
        'strategy': strategy
    })
    
    success = save_portfolio(portfolio)
    return success, message

def sell_position(
    ticker: str,
    quantity: int = None,
    sell_price: float = None,
    reason: str = "Manuelt salg"
) -> Tuple[bool, str, Dict]:
    """
    Selger hele eller deler av en posisjon.
    
    Args:
        ticker: Aksje-ticker
        quantity: Antall å selge (None = hele posisjonen)
        sell_price: Salgskurs (None = må oppgis)
        reason: Grunn for salget
    
    Returns:
        (success, message, trade_result)
    """
    portfolio = load_portfolio()
    ticker = ticker.upper()
    
    if not ticker or not isinstance(ticker, str):
        return False, "Ugyldig ticker", {}
    
    if ticker not in portfolio['positions']:
        return False, f"{ticker} finnes ikke i porteføljen", {}
    
    pos = portfolio['positions'][ticker]
    
    if quantity is None:
        quantity = pos['quantity']
    elif not isinstance(quantity, (int, float)) or quantity <= 0:
        return False, f"Ugyldig antall: {quantity}. Må være et positivt tall.", {}
    else:
        quantity = int(quantity)
    
    if quantity > pos['quantity']:
        return False, f"Kan ikke selge {quantity} aksjer, du har bare {pos['quantity']}", {}
    
    if sell_price is None:
        return False, "Salgskurs må oppgis", {}
    if not isinstance(sell_price, (int, float)) or sell_price <= 0:
        return False, f"Ugyldig salgskurs: {sell_price}. Må være et positivt tall.", {}
    
    logger.info(f"Selger {quantity}× {ticker} @ {sell_price:.2f} — {reason}")
    
    # Beregn resultat
    avg_price = pos['avg_price']
    profit_per_share = sell_price - avg_price
    total_profit = profit_per_share * quantity
    profit_pct = (profit_per_share / avg_price) * 100
    
    trade_result = {
        'ticker': ticker,
        'quantity': quantity,
        'buy_price': avg_price,
        'sell_price': sell_price,
        'profit': round(total_profit, 2),
        'profit_pct': round(profit_pct, 2),
        'reason': reason,
        'holding_days': _calculate_holding_days(pos['buy_date'])
    }
    
    # Oppdater eller fjern posisjon
    remaining = pos['quantity'] - quantity
    
    if remaining > 0:
        pos['quantity'] = remaining
        message = f"Solgte {quantity} av {ticker} @ {sell_price:.2f} ({profit_pct:+.1f}%)"
    else:
        del portfolio['positions'][ticker]
        message = f"Solgte hele posisjonen i {ticker} @ {sell_price:.2f} ({profit_pct:+.1f}%)"
    
    # Logg transaksjon
    save_transaction({
        'type': 'SELL',
        'ticker': ticker,
        'quantity': quantity,
        'price': sell_price,
        'profit': total_profit,
        'profit_pct': profit_pct,
        'reason': reason
    })
    
    success = save_portfolio(portfolio)
    return success, message, trade_result

def update_stop_loss(ticker: str, new_stop: float) -> Tuple[bool, str]:
    """Oppdaterer stop loss for en posisjon."""
    portfolio = load_portfolio()
    ticker = ticker.upper()
    
    if ticker not in portfolio['positions']:
        return False, f"{ticker} finnes ikke i porteføljen"
    
    old_stop = portfolio['positions'][ticker].get('stop_loss', 0)
    portfolio['positions'][ticker]['stop_loss'] = round(new_stop, 2)
    
    save_portfolio(portfolio)
    return True, f"Stop loss for {ticker} endret: {old_stop:.2f} → {new_stop:.2f}"

def update_trailing_high(ticker: str, current_price: float) -> bool:
    """Oppdaterer trailing high hvis nåværende pris er høyere."""
    portfolio = load_portfolio()
    ticker = ticker.upper()
    
    if ticker not in portfolio['positions']:
        return False
    
    pos = portfolio['positions'][ticker]
    if current_price > pos.get('trailing_high', 0):
        pos['trailing_high'] = round(current_price, 2)
        save_portfolio(portfolio)
        return True
    return False

def _calculate_holding_days(buy_date_str: str) -> int:
    """Beregner antall dager siden kjøp."""
    try:
        buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d')
        return (datetime.now() - buy_date).days
    except:
        return 0

# =============================================================================
# EXIT SIGNALS - Prioriterte salgssignaler
# =============================================================================

EXIT_PRIORITY = {
    'TRAILING_STOP': 1,      # Høyeste prioritet - beskytter gevinst
    'MAX_DRAWDOWN': 2,       # Stopper tap
    'STOP_LOSS_HIT': 3,      # Initial stop loss trigget
    'DEATH_CROSS': 4,        # Teknisk vendepunkt
    'VOLUME_COLLAPSE': 5,    # Likviditetsproblem
    'RSI_BEARISH_DIV': 6,    # RSI divergens
    'MA_BREAKDOWN': 7,       # Under viktige snitt
    'MACD_CROSS': 8,         # MACD bearish
    'TIME_STOP': 9,          # For lang holdtid uten gevinst
    'PROFIT_TARGET': 10,     # Profittmål nådd (valgfri)
}

def analyze_exit_signals(
    ticker: str,
    df: pd.DataFrame,
    position: Dict,
    current_price: float = None
) -> Dict:
    """
    Analyserer alle exit-signaler for en posisjon.
    
    Args:
        ticker: Aksje-ticker
        df: DataFrame med prisdata og indikatorer
        position: Posisjonsinformasjon fra porteføljen
        current_price: Nåværende pris (default: siste close)
    
    Returns:
        Dict med exit-analyse
    """
    if df.empty or len(df) < 50:
        return {
            'should_exit': False,
            'signals': [],
            'priority': 99,
            'summary': 'Utilstrekkelig data for analyse'
        }
    
    if current_price is None:
        current_price = float(df['Close'].iloc[-1])
    
    avg_price = position['avg_price']
    stop_loss = position.get('stop_loss', avg_price * 0.92)
    trailing_high = position.get('trailing_high', avg_price)
    buy_date = position.get('buy_date', '')
    holding_days = _calculate_holding_days(buy_date)
    
    # Oppdater trailing high
    if current_price > trailing_high:
        trailing_high = current_price
    
    signals = []
    
    # Hent config-verdier
    fallback_trailing_stop_pct = getattr(config, 'EXIT_TRAILING_STOP_PCT', 8.0)
    max_drawdown_pct = getattr(config, 'EXIT_MAX_DRAWDOWN_PCT', 15.0)
    time_stop_days = getattr(config, 'EXIT_TIME_STOP_DAYS', 60)
    profit_target_pct = getattr(config, 'EXIT_PROFIT_TARGET_PCT', 20.0)
    
    # === 1. ATR-ADAPTIV TRAILING STOP (høyeste prioritet) ===
    # Beregn adaptiv stop basert på volatilitet i stedet for fast prosent
    if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]):
        atr = float(df['ATR'].iloc[-1])
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 3.0
        
        # Adaptiv multiplier: volatile aksjer får bredere stop
        if atr_pct < 2.0:
            atr_multiplier = 2.0   # Lav vol → stramt stop
        elif atr_pct < 4.0:
            atr_multiplier = 2.5   # Normal vol
        else:
            atr_multiplier = 3.0   # Høy vol → bredt stop
        
        trailing_stop_pct = atr_pct * atr_multiplier
        trailing_stop_price = trailing_high * (1 - trailing_stop_pct / 100)
    else:
        trailing_stop_pct = fallback_trailing_stop_pct
        trailing_stop_price = trailing_high * (1 - trailing_stop_pct / 100)
        atr_multiplier = None
    
    if current_price <= trailing_stop_price and current_price > avg_price:
        drawdown_from_high = ((trailing_high - current_price) / trailing_high) * 100
        stop_detail = f" (ATR×{atr_multiplier})" if atr_multiplier else " (fast)"
        signals.append({
            'type': 'TRAILING_STOP',
            'priority': EXIT_PRIORITY['TRAILING_STOP'],
            'severity': 'CRITICAL',
            'message': f"🛑 Trailing Stop trigget: -{drawdown_from_high:.1f}% fra topp ({trailing_high:.2f}){stop_detail}",
            'action': 'SELG NÅ for å beskytte gevinst',
            'stop_price': trailing_stop_price
        })
    
    # === 2. MAX DRAWDOWN (taper for mye) ===
    profit_pct = ((current_price - avg_price) / avg_price) * 100
    if profit_pct <= -max_drawdown_pct:
        signals.append({
            'type': 'MAX_DRAWDOWN',
            'priority': EXIT_PRIORITY['MAX_DRAWDOWN'],
            'severity': 'CRITICAL',
            'message': f"🚨 Maks drawdown: {profit_pct:.1f}% tap fra kjøp",
            'action': 'SELG for å stoppe tap'
        })
    
    # === 3. STOP LOSS HIT ===
    if current_price <= stop_loss:
        signals.append({
            'type': 'STOP_LOSS_HIT',
            'priority': EXIT_PRIORITY['STOP_LOSS_HIT'],
            'severity': 'CRITICAL',
            'message': f"⛔ Stop loss trigget: {current_price:.2f} ≤ {stop_loss:.2f}",
            'action': 'SELG umiddelbart'
        })
    
    # === 4. DEATH CROSS ===
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        sma50_now = df['SMA_50'].iloc[-1]
        sma50_prev = df['SMA_50'].iloc[-2]
        sma200_now = df['SMA_200'].iloc[-1]
        sma200_prev = df['SMA_200'].iloc[-2]
        
        if sma50_now < sma200_now and sma50_prev >= sma200_prev:
            signals.append({
                'type': 'DEATH_CROSS',
                'priority': EXIT_PRIORITY['DEATH_CROSS'],
                'severity': 'HIGH',
                'message': "☠️ Death Cross: SMA 50 krysset under SMA 200",
                'action': 'Vurder å selge - langsiktig trendskifte'
            })
    
    # === 5. VOLUME COLLAPSE ===
    if len(df) >= 30:
        vol_avg_20 = df['Volume'].iloc[-21:-1].mean()
        vol_now = df['Volume'].iloc[-1]
        vol_ratio = vol_now / vol_avg_20 if vol_avg_20 > 0 else 1
        
        if vol_ratio < 0.3 and profit_pct < 5:
            signals.append({
                'type': 'VOLUME_COLLAPSE',
                'priority': EXIT_PRIORITY['VOLUME_COLLAPSE'],
                'severity': 'MEDIUM',
                'message': f"📉 Volumkollaps: {vol_ratio:.0%} av normalt volum",
                'action': 'Vurder å selge - lav likviditet'
            })
    
    # === 6. RSI BEARISH DIVERGENCE ===
    if 'RSI' in df.columns:
        rsi_now = df['RSI'].iloc[-1]
        rsi_prev = df['RSI'].iloc[-2]
        
        # RSI faller fra overkjøpt
        if rsi_prev > 70 and rsi_now < rsi_prev and rsi_now < 70:
            signals.append({
                'type': 'RSI_BEARISH_DIV',
                'priority': EXIT_PRIORITY['RSI_BEARISH_DIV'],
                'severity': 'MEDIUM',
                'message': f"📊 RSI snur fra overkjøpt: {rsi_prev:.0f} → {rsi_now:.0f}",
                'action': 'Ta delvis gevinst'
            })
        
        # RSI under 30 og fallende (svak aksje)
        if rsi_now < 30 and rsi_now < rsi_prev and profit_pct < 0:
            signals.append({
                'type': 'RSI_BEARISH_DIV',
                'priority': EXIT_PRIORITY['RSI_BEARISH_DIV'],
                'severity': 'HIGH',
                'message': f"⚠️ RSI ekstremt svak: {rsi_now:.0f} og fallende",
                'action': 'Vurder å kutte tap'
            })
    
    # === 7. MA BREAKDOWN ===
    if 'SMA_50' in df.columns:
        sma50 = df['SMA_50'].iloc[-1]
        vol_avg = df['Volume'].iloc[-20:].mean() if len(df) >= 20 else df['Volume'].mean()
        vol_now = df['Volume'].iloc[-1]
        
        if current_price < sma50 and vol_now > vol_avg * 1.5:
            signals.append({
                'type': 'MA_BREAKDOWN',
                'priority': EXIT_PRIORITY['MA_BREAKDOWN'],
                'severity': 'HIGH',
                'message': f"🔻 Brudd under SMA 50 ({sma50:.2f}) med høyt volum",
                'action': 'Selg eller stram stop loss'
            })
    
    # === 8. MACD BEARISH CROSS ===
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd_now = df['MACD'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2]
        signal_now = df['MACD_Signal'].iloc[-1]
        signal_prev = df['MACD_Signal'].iloc[-2]
        
        if macd_now < signal_now and macd_prev >= signal_prev:
            signals.append({
                'type': 'MACD_CROSS',
                'priority': EXIT_PRIORITY['MACD_CROSS'],
                'severity': 'MEDIUM',
                'message': "📈 MACD krysset under signallinjen",
                'action': 'Momentum avtar - vurder salg'
            })
    
    # === 9. TIME STOP ===
    if holding_days > time_stop_days and profit_pct < 5:
        signals.append({
            'type': 'TIME_STOP',
            'priority': EXIT_PRIORITY['TIME_STOP'],
            'severity': 'LOW',
            'message': f"⏰ Holdt i {holding_days} dager uten vesentlig gevinst",
            'action': 'Vurder å frigjøre kapital'
        })
    
    # === 10. PROFIT TARGET (info) ===
    if profit_pct >= profit_target_pct:
        signals.append({
            'type': 'PROFIT_TARGET',
            'priority': EXIT_PRIORITY['PROFIT_TARGET'],
            'severity': 'INFO',
            'message': f"🎯 Profittmål nådd: +{profit_pct:.1f}%",
            'action': 'Vurder å ta gevinst eller flytte stop loss opp'
        })
    
    # Sorter etter prioritet
    signals = sorted(signals, key=lambda x: x['priority'])
    
    # Bestem om vi bør selge
    critical_signals = [s for s in signals if s['severity'] == 'CRITICAL']
    high_signals = [s for s in signals if s['severity'] == 'HIGH']
    
    should_exit = len(critical_signals) > 0 or len(high_signals) >= 2
    
    # Beregn ny anbefalt trailing stop
    recommended_stop = max(stop_loss, trailing_stop_price) if trailing_high > avg_price else stop_loss
    
    return {
        'should_exit': should_exit,
        'signals': signals,
        'priority': signals[0]['priority'] if signals else 99,
        'critical_count': len(critical_signals),
        'high_count': len(high_signals),
        'current_price': current_price,
        'profit_pct': profit_pct,
        'trailing_high': trailing_high,
        'trailing_stop_price': trailing_stop_price,
        'recommended_stop': round(recommended_stop, 2),
        'holding_days': holding_days,
        'summary': signals[0]['message'] if signals else 'Ingen exit-signaler'
    }

# =============================================================================
# PORTFOLIO ANALYSIS - Samlet oversikt
# =============================================================================

def analyze_portfolio(df_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyserer hele porteføljen.
    
    Args:
        df_dict: Dict med ticker -> DataFrame
    
    Returns:
        Komplett portefølje-analyse
    """
    portfolio = load_portfolio()
    positions = portfolio.get('positions', {})
    
    if not positions:
        return {
            'total_value': 0,
            'total_cost': 0,
            'total_profit': 0,
            'total_profit_pct': 0,
            'positions': [],
            'alerts': [],
            'summary': 'Ingen posisjoner i porteføljen'
        }
    
    analyzed_positions = []
    total_value = 0
    total_cost = 0
    all_alerts = []
    
    for ticker, pos in positions.items():
        df = df_dict.get(ticker)
        
        if df is None or df.empty:
            current_price = pos['avg_price']  # Bruk kjøpskurs hvis ingen data
        else:
            current_price = float(df['Close'].iloc[-1])
            # Oppdater trailing high
            update_trailing_high(ticker, current_price)
        
        quantity = pos['quantity']
        avg_price = pos['avg_price']
        
        position_value = quantity * current_price
        position_cost = quantity * avg_price
        position_profit = position_value - position_cost
        position_profit_pct = ((current_price - avg_price) / avg_price) * 100
        
        # Exit-analyse
        exit_analysis = {}
        if df is not None and not df.empty:
            exit_analysis = analyze_exit_signals(ticker, df, pos, current_price)
            
            # Legg til alerts
            if exit_analysis.get('should_exit'):
                for signal in exit_analysis.get('signals', [])[:3]:  # Topp 3 signaler
                    all_alerts.append({
                        'ticker': ticker,
                        'priority': signal['priority'],
                        'severity': signal['severity'],
                        'message': signal['message'],
                        'action': signal['action']
                    })
        
        analyzed_positions.append({
            'ticker': ticker,
            'quantity': quantity,
            'avg_price': avg_price,
            'current_price': round(current_price, 2),
            'value': round(position_value, 2),
            'cost': round(position_cost, 2),
            'profit': round(position_profit, 2),
            'profit_pct': round(position_profit_pct, 2),
            'buy_date': pos.get('buy_date', ''),
            'holding_days': _calculate_holding_days(pos.get('buy_date', '')),
            'stop_loss': pos.get('stop_loss', 0),
            'trailing_high': pos.get('trailing_high', avg_price),
            'exit_analysis': exit_analysis,
            'notes': pos.get('notes', ''),
            'strategy': pos.get('strategy', '')
        })
        
        total_value += position_value
        total_cost += position_cost
    
    total_profit = total_value - total_cost
    total_profit_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # Sorter posisjoner etter prioritet (mest kritiske først)
    analyzed_positions = sorted(
        analyzed_positions,
        key=lambda x: (
            x.get('exit_analysis', {}).get('priority', 99),
            -x.get('profit_pct', 0)
        )
    )
    
    # Sorter alerts etter prioritet
    all_alerts = sorted(all_alerts, key=lambda x: x['priority'])
    
    return {
        'total_value': round(total_value, 2),
        'total_cost': round(total_cost, 2),
        'total_profit': round(total_profit, 2),
        'total_profit_pct': round(total_profit_pct, 2),
        'position_count': len(analyzed_positions),
        'positions': analyzed_positions,
        'alerts': all_alerts[:10],  # Maks 10 alerts
        'alert_count': len(all_alerts),
        'summary': f"{len(analyzed_positions)} posisjoner, {len(all_alerts)} varsler"
    }

# =============================================================================
# DAILY REPORT - For e-post og daglig oppsummering
# =============================================================================

def generate_daily_report(
    df_dict: Dict[str, pd.DataFrame],
    all_tickers: List[str],
    min_volume_nok: float = 3_000_000,
    min_quality: str = 'A'
) -> Dict:
    """
    Genererer daglig rapport med:
    1. Portefølje-status og alerts
    2. Nye A-kvalitet kjøpssignaler
    3. Exit-signaler som krever handling
    
    Args:
        df_dict: Dict med ticker -> DataFrame
        all_tickers: Liste over alle tilgjengelige tickers
        min_volume_nok: Minimum dagsomsetning for nye signaler
        min_quality: Minimum signalkvalitet (A, B, C, D)
    
    Returns:
        Rapport-dict egnet for visning eller e-post
    """
    report = {
        'generated': datetime.now().isoformat(),
        'portfolio_summary': {},
        'exit_alerts': [],
        'new_signals': [],
        'market_overview': {}
    }
    
    # 1. Portefølje-analyse
    portfolio_analysis = analyze_portfolio(df_dict)
    report['portfolio_summary'] = {
        'total_value': portfolio_analysis['total_value'],
        'total_profit': portfolio_analysis['total_profit'],
        'total_profit_pct': portfolio_analysis['total_profit_pct'],
        'position_count': portfolio_analysis['position_count']
    }
    
    # 2. Exit-alerts (kritiske og høy prioritet)
    for pos in portfolio_analysis['positions']:
        exit_info = pos.get('exit_analysis', {})
        if exit_info.get('should_exit'):
            report['exit_alerts'].append({
                'ticker': pos['ticker'],
                'current_price': pos['current_price'],
                'profit_pct': pos['profit_pct'],
                'signals': [s['message'] for s in exit_info.get('signals', [])[:3]],
                'action': exit_info.get('signals', [{}])[0].get('action', '')
            })
    
    # 3. Nye kjøpssignaler (implementeres med scanner-integrasjon)
    # Dette kan kobles til Scanner-modulen for A-kvalitet signaler
    
    return report

def format_report_for_display(report: Dict) -> str:
    """Formaterer rapport for Streamlit-visning."""
    lines = []
    lines.append(f"# 📊 Daglig Rapport - {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    
    # Portefølje
    ps = report['portfolio_summary']
    lines.append("## 💼 Portefølje")
    lines.append(f"- **Verdi:** {ps['total_value']:,.0f} NOK")
    lines.append(f"- **Resultat:** {ps['total_profit']:+,.0f} NOK ({ps['total_profit_pct']:+.1f}%)")
    lines.append(f"- **Antall posisjoner:** {ps['position_count']}")
    lines.append("")
    
    # Exit-alerts
    if report['exit_alerts']:
        lines.append("## 🚨 Salgssignaler som krever handling")
        for alert in report['exit_alerts']:
            lines.append(f"### {alert['ticker']}")
            lines.append(f"- Kurs: {alert['current_price']:.2f} ({alert['profit_pct']:+.1f}%)")
            for sig in alert['signals']:
                lines.append(f"  - {sig}")
            lines.append(f"- **Handling:** {alert['action']}")
            lines.append("")
    else:
        lines.append("## ✅ Ingen kritiske salgssignaler")
        lines.append("")
    
    return "\n".join(lines)

# =============================================================================
# EMAIL PREPARATION (for fremtidig bruk)
# =============================================================================

# =============================================================================
# MONTE CARLO SIMULERING - Portefølje-risikoanalyse
# =============================================================================

def monte_carlo_portefolje(
    df_dict: Dict[str, pd.DataFrame],
    n_simuleringer: int = 10_000,
    n_dager: int = 252,
    seed: int = 42
) -> Optional[Dict]:
    """
    Kjører Monte Carlo-simulering av porteføljeutvikling.
    
    Bruker Cholesky-dekomponering for å bevare korrelasjoner mellom posisjoner.
    Basert på historisk avkastning og volatilitet (geometrisk brownsk bevegelse).
    
    Args:
        df_dict: Dict med ticker -> DataFrame (må ha 'Close'-kolonne)
        n_simuleringer: Antall simuleringssti (default: 10 000)
        n_dager: Simuleringslengde i handelsdager (default: 252 = 1 år)
        seed: Random seed for reproduserbarhet
    
    Returns:
        Dict med simuleringsresultater, eller None ved feil
    """
    portfolio_data = load_portfolio()
    positions = portfolio_data.get('positions', {})
    
    if not positions:
        return None
    
    # --- 1. Samle daglige avkastninger for alle posisjoner ---
    tickers = []
    vekter = []
    avkastninger = {}
    
    total_value = 0.0
    ticker_verdier = {}
    
    for ticker, pos in positions.items():
        df = df_dict.get(ticker)
        if df is None or df.empty or len(df) < 60:
            continue
        
        close = df['Close'].dropna()
        if len(close) < 60:
            continue
        
        current_price = float(close.iloc[-1])
        pos_value = pos['quantity'] * current_price
        
        # Daglig log-avkastning (siste 252 dager for stabilitet)
        daily_returns = np.log(close / close.shift(1)).dropna()
        if len(daily_returns) > 252:
            daily_returns = daily_returns.iloc[-252:]
        
        tickers.append(ticker)
        ticker_verdier[ticker] = pos_value
        total_value += pos_value
        avkastninger[ticker] = daily_returns
    
    if len(tickers) == 0 or total_value == 0:
        return None
    
    # --- 2. Beregn vekter og porteføljens startverdi ---
    vekter = np.array([ticker_verdier[t] / total_value for t in tickers])
    
    # --- 3. Bygg samkjørt avkastningsmatrise (felles datoer) ---
    # Align alle serier til felles datoindeks
    returns_df = pd.DataFrame({t: avkastninger[t] for t in tickers})
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 30:
        return None
    
    n_assets = len(tickers)
    
    # Annualiserte parametere fra historisk data
    mu_daily = returns_df.mean().values        # Forventet daglig avkastning
    cov_daily = returns_df.cov().values         # Daglig kovariansmatrise
    
    # --- 4. Cholesky-dekomponering for korrelerte simuleringer ---
    try:
        # Sørg for positiv-definitt matrise
        eigvals = np.linalg.eigvalsh(cov_daily)
        if np.any(eigvals <= 0):
            # Korriger til nærmeste PD-matrise
            cov_daily = _nearest_positive_definite(cov_daily)
        L = np.linalg.cholesky(cov_daily)
    except np.linalg.LinAlgError:
        # Fallback: bruk diagonal kovarians (ingen korrelasjon)
        std_daily = returns_df.std().values
        L = np.diag(std_daily)
    
    # --- 5. Kjør simuleringer ---
    rng = np.random.default_rng(seed)
    
    # Matrise: (n_simuleringer, n_dager+1) — porteføljeverdi over tid
    portfolio_paths = np.zeros((n_simuleringer, n_dager + 1))
    portfolio_paths[:, 0] = total_value
    
    for day in range(n_dager):
        # Generer korrelerte normalfordelte tilfeldige tall
        Z = rng.standard_normal((n_simuleringer, n_assets))  # (sims, assets)
        correlated_Z = Z @ L.T  # Korrelerte sjokk
        
        # Geometrisk brownsk bevegelse: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # For daglig: dt = 1
        daily_asset_returns = np.exp(mu_daily - 0.5 * np.diag(cov_daily) + correlated_Z)
        
        # Porteføljeavkastning = vektet sum av asset-avkastninger
        portfolio_return = (daily_asset_returns * vekter).sum(axis=1)
        
        portfolio_paths[:, day + 1] = portfolio_paths[:, day] * portfolio_return
    
    # --- 6. Beregn statistikk ---
    sluttverdier = portfolio_paths[:, -1]
    sluttavkastning = (sluttverdier / total_value - 1) * 100  # I prosent
    
    # Percentiler over tid (for fan-chart)
    pct_5 = np.percentile(portfolio_paths, 5, axis=0)
    pct_10 = np.percentile(portfolio_paths, 10, axis=0)
    pct_25 = np.percentile(portfolio_paths, 25, axis=0)
    pct_50 = np.percentile(portfolio_paths, 50, axis=0)  # Median
    pct_75 = np.percentile(portfolio_paths, 75, axis=0)
    pct_90 = np.percentile(portfolio_paths, 90, axis=0)
    pct_95 = np.percentile(portfolio_paths, 95, axis=0)
    
    # Value-at-Risk (VaR) — 95% konfidens
    # VaR = hva er maks tap med 95% sannsynlighet over perioden?
    var_95 = total_value - np.percentile(sluttverdier, 5)
    var_95_pct = (var_95 / total_value) * 100
    
    # Conditional VaR (CVaR / Expected Shortfall)
    worst_5pct = sluttverdier[sluttverdier <= np.percentile(sluttverdier, 5)]
    cvar_95 = total_value - worst_5pct.mean() if len(worst_5pct) > 0 else var_95
    cvar_95_pct = (cvar_95 / total_value) * 100
    
    # Max drawdown over alle stier (median sti)
    cummax = np.maximum.accumulate(pct_50)
    drawdowns = (cummax - pct_50) / cummax * 100
    max_drawdown_median = drawdowns.max()
    
    # Sannsynlighet for tap
    prob_tap = (sluttverdier < total_value).mean() * 100
    prob_gevinst_10 = (sluttavkastning > 10).mean() * 100
    prob_tap_20 = (sluttavkastning < -20).mean() * 100
    
    # Sharpe-estimat (annualisert fra median sti)
    median_annual_ret = (pct_50[-1] / total_value - 1)
    port_daily_returns = np.diff(pct_50) / pct_50[:-1]
    annual_vol = np.std(port_daily_returns) * np.sqrt(252)
    sharpe_estimate = median_annual_ret / annual_vol if annual_vol > 0 else 0
    
    # Korrelasjon info
    corr_matrix = returns_df.corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean() if n_assets > 1 else 0.0
    
    return {
        'startverdi': round(total_value, 2),
        'n_simuleringer': n_simuleringer,
        'n_dager': n_dager,
        'n_posisjoner': len(tickers),
        'tickers': tickers,
        'vekter': {t: round(w * 100, 1) for t, w in zip(tickers, vekter)},
        
        # Percentil-kurver for fan-chart (som lister for JSON-kompatibilitet)
        'pct_5': pct_5.tolist(),
        'pct_10': pct_10.tolist(),
        'pct_25': pct_25.tolist(),
        'pct_50': pct_50.tolist(),
        'pct_75': pct_75.tolist(),
        'pct_90': pct_90.tolist(),
        'pct_95': pct_95.tolist(),
        
        # Sluttverdi-statistikk
        'median_sluttverdi': round(float(np.median(sluttverdier)), 2),
        'worst_case_5': round(float(np.percentile(sluttverdier, 5)), 2),
        'best_case_95': round(float(np.percentile(sluttverdier, 95)), 2),
        'median_avkastning_pct': round(float(np.median(sluttavkastning)), 1),
        'worst_avkastning_pct': round(float(np.percentile(sluttavkastning, 5)), 1),
        'best_avkastning_pct': round(float(np.percentile(sluttavkastning, 95)), 1),
        
        # Risikomål
        'var_95': round(float(var_95), 2),
        'var_95_pct': round(float(var_95_pct), 1),
        'cvar_95': round(float(cvar_95), 2),
        'cvar_95_pct': round(float(cvar_95_pct), 1),
        'max_drawdown_median_pct': round(float(max_drawdown_median), 1),
        
        # Sannsynligheter
        'prob_tap_pct': round(float(prob_tap), 1),
        'prob_gevinst_10_pct': round(float(prob_gevinst_10), 1),
        'prob_tap_20_pct': round(float(prob_tap_20), 1),
        
        # Porteføljeegenskaper
        'sharpe_estimate': round(float(sharpe_estimate), 2),
        'avg_korrelasjon': round(float(avg_corr), 2),
        'korrelasjon_matrise': {
            'tickers': tickers,
            'matrise': corr_matrix.values.round(2).tolist()
        }
    }


def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    Finn nærmeste positiv-definitte matrise (Higham, 2002).
    Brukes som fallback hvis kovariansmatrisen har numeriske problemer.
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(np.maximum(s, 1e-10)) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    
    # Verifiser PD
    if _is_positive_definite(A3):
        return A3
    
    # Legg til liten diagonal jitter
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not _is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvalsh(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
        if k > 100:
            break
    return A3


def _is_positive_definite(A: np.ndarray) -> bool:
    """Sjekk om matrise er positiv-definitt via Cholesky."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def prepare_email_content(report: Dict) -> Dict:
    """
    Forbereder e-post-innhold fra daglig rapport.
    Returnerer dict med subject og body.
    """
    ps = report['portfolio_summary']
    alert_count = len(report['exit_alerts'])
    
    # Subject
    if alert_count > 0:
        subject = f"🚨 InveStock: {alert_count} salgssignal(er) - Handling kreves!"
    elif ps['total_profit_pct'] > 5:
        subject = f"📈 InveStock: Portefølje +{ps['total_profit_pct']:.1f}%"
    else:
        subject = f"📊 InveStock: Daglig oppdatering"
    
    # Body (HTML-format for e-post)
    body = format_report_for_display(report)
    
    return {
        'subject': subject,
        'body': body,
        'has_critical_alerts': alert_count > 0
    }
