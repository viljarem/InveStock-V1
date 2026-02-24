# log_config.py
"""
Sentralisert logging-oppsett for InveStock.

Bruk:
    from log_config import get_logger
    logger = get_logger(__name__)
    
    logger.debug("Detaljer for feilsøking")
    logger.info("Datahenting startet")
    logger.warning("Manglende data for ticker X")
    logger.error("Kunne ikke koble til API")
"""

import logging
import sys

# Standard format: [MODUL] NIVÅ — melding
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Konfigurer root-logger kun én gang
_configured = False


def setup_logging(level=logging.INFO):
    """Konfigurerer logging for hele applikasjonen."""
    global _configured
    if _configured:
        return
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    
    # Demp støy fra tredjepartsbiblioteker
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("hmmlearn").setLevel(logging.WARNING)
    
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returnerer en logger for gitt modul. Konfigurerer logging ved første kall."""
    setup_logging()
    return logging.getLogger(name)
