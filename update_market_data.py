#!/usr/bin/env python3
"""
Automatisk oppdatering av markedsdata for InveStock
Kjøres av GitHub Actions etter børsslutt
"""

import sys
import os
import logging
from datetime import datetime

# Legg til prosjektmappe til Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import av våre moduler
import data
import fundamental_data
import insider_monitor
from log_config import get_logger

logger = get_logger(__name__)

def main():
    """Hovedfunksjon for automatisk dataoppdatering"""
    start_tid = datetime.now()
    logger.info("=" * 60)
    logger.info("🤖 AUTOMATISK MARKEDSDATA-OPPDATERING STARTET")
    logger.info(f"Starttid: {start_tid.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    feil = []
    suksess = []
    
    try:
        # 1. Oppdater hovedmarkedsdata (aksjekurser)
        logger.info("📈 Oppdaterer hovedmarkedsdata...")
        try:
            df = data.hent_markedsdata_df(force_refresh=True)
            if not df.empty:
                suksess.append("Hovedmarkedsdata")
                logger.info(f"   ✅ Hentet data for {len(df)} aksjer")
            else:
                feil.append("Hovedmarkedsdata (tom DataFrame)")
                logger.warning("   ⚠️  Tom DataFrame returnert")
        except Exception as e:
            feil.append(f"Hovedmarkedsdata: {str(e)}")
            logger.error(f"   ❌ Feil: {e}")
            
        # 2. Oppdater Brent oljepris
        logger.info("🛢️  Oppdaterer Brent oljepris...")
        try:
            brent_data = data.hent_brent_data(force_refresh=True)
            if not brent_data.empty:
                suksess.append("Brent oljepris")
                logger.info(f"   ✅ Hentet {len(brent_data)} datapunkter")
            else:
                feil.append("Brent oljepris (tom DataFrame)")
        except Exception as e:
            feil.append(f"Brent oljepris: {str(e)}")
            logger.error(f"   ❌ Feil: {e}")
            
        # 3. Oppdater USD/NOK kurs
        logger.info("💱 Oppdaterer USD/NOK kurs...")
        try:
            usdnok_data = data.hent_usdnok_data(force_refresh=True)
            if not usdnok_data.empty:
                suksess.append("USD/NOK kurs")
                logger.info(f"   ✅ Hentet {len(usdnok_data)} datapunkter")
            else:
                feil.append("USD/NOK kurs (tom DataFrame)")
        except Exception as e:
            feil.append(f"USD/NOK kurs: {str(e)}")
            logger.error(f"   ❌ Feil: {e}")
            
        # 4. Oppdater fundamental data cache
        logger.info("📊 Oppdaterer fundamental data...")
        try:
            # Hent alle tickers
            tickers = data.hent_oppdaterte_tickers()
            oppdatert_count = 0
            
            for ticker in tickers[:10]:  # Begrens til første 10 for å ikke overbelaste
                try:
                    fund_data = fundamental_data.get_fundamental_data(ticker)
                    if fund_data:
                        oppdatert_count += 1
                except:
                    continue
                    
            if oppdatert_count > 0:
                suksess.append(f"Fundamental data ({oppdatert_count} aksjer)")
                logger.info(f"   ✅ Oppdatert fundamental data for {oppdatert_count} aksjer")
            else:
                feil.append("Fundamental data (ingen aksjer oppdatert)")
        except Exception as e:
            feil.append(f"Fundamental data: {str(e)}")
            logger.error(f"   ❌ Feil: {e}")
            
        # 5. Oppdater insider data
        logger.info("👥 Oppdaterer insider data...")
        try:
            insider_handler = insider_monitor.hent_innsidehandler(dager=30)
            if insider_handler:
                suksess.append("Insider data")
                logger.info(f"   ✅ Hentet {len(insider_handler)} insider-handler")
            else:
                logger.warning("   ⚠️  Ingen insider-handler funnet")
                suksess.append("Insider data (tom)")
        except Exception as e:
            feil.append(f"Insider data: {str(e)}")
            logger.error(f"   ❌ Feil: {e}")
            
    except Exception as e:
        logger.error(f"💥 KRITISK FEIL i hovedloop: {e}")
        feil.append(f"Kritisk feil: {str(e)}")
        
    # Oppsummering
    slutt_tid = datetime.now()
    varighet = slutt_tid - start_tid
    
    logger.info("=" * 60)
    logger.info("📋 OPPSUMMERING AV OPPDATERING")
    logger.info("=" * 60)
    logger.info(f"Starttid: {start_tid.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Slutttid: {slutt_tid.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Varighet: {varighet}")
    logger.info("")
    
    if suksess:
        logger.info("✅ VELLYKKET:")
        for item in suksess:
            logger.info(f"   • {item}")
        logger.info("")
            
    if feil:
        logger.info("❌ FEIL:")
        for item in feil:
            logger.info(f"   • {item}")
        logger.info("")
        
    # Exit code
    if feil and not suksess:
        logger.error("💥 Alle oppdateringer feilet - avslutter med feilkode")
        sys.exit(1)
    elif feil:
        logger.warning("⚠️  Noen oppdateringer feilet, men fortsetter")
        sys.exit(0)
    else:
        logger.info("🎉 Alle oppdateringer vellykket!")
        sys.exit(0)

if __name__ == "__main__":
    main()