# data.py
import yfinance as yf
import pandas as pd
import os
import datetime
from config import DATA_FILE, MARKET_DATA_FILE, HISTORICAL_YEARS, TICKER_LIST_FILE

def hent_oppdaterte_tickers():
    """
    Henter selskaper fra Oslo Børs. 
    Prioriterer 'mine_tickere.txt' i rotmappen hvis den eksisterer.
    Returnerer en alfabetisk sortert liste.
    """
    filsti_bruker = "mine_tickere.txt"
    tickers = []
    
    if os.path.exists(filsti_bruker):
        try:
            with open(filsti_bruker, "r") as f:
                # Les rader, fjern uønsket mellomrom og ignorer tomme linjer
                tickers = [line.strip().upper() for line in f if line.strip()]
            
            # Sikrer .OL suffiks for Oslo Børs hvis det mangler
            tickers = [t if t.endswith(".OL") else f"{t}.OL" for t in tickers]
        except Exception:
            tickers = []
    
    # Hvis filen ikke finnes eller er tom, benyttes standardlisten
    if not tickers:
        tickers = [
            "EQNR.OL", "DNB.OL", "NHY.OL", "ORK.OL", "MOWI.OL", "YAR.OL", 
            "TEL.OL", "GJF.OL", "STB.OL", "AKRBP.OL", "TOM.OL", "SALM.OL",
            "KOG.OL", "SUBC.OL", "NAS.OL", "FRO.OL", "MPCC.OL", "VAR.OL",
            "HAFNI.OL", "LSG.OL", "BAKKA.OL", "NOD.OL", "PGS.OL", "RECSI.OL",
            "SCATC.OL", "ADE.OL", "BOUV.OL", "KIT.OL", "BORR.OL", "ELK.OL",
            "ENTRA.OL", "EPR.OL", "FLNG.OL", "GOGL.OL", "HSHP.OL", "IDEX.OL",
            "KID.OL", "KOA.OL", "LER.OL", "LINK.OL", "MPC.OL", "MULTI.OL",
            "NEL.OL", "NORBIT.OL", "NANOV.OL", "OKEA.OL", "OTOVO.OL", "PEXIP.OL",
            "PHOTO.OL", "PROTECTOR.OL", "SRBANK.OL", "SSG.OL", "TGS.OL", "VEI.OL",
            "VOLUE.OL", "WWI.OL", "XXL.OL", "BELCO.OL", "BWLPG.OL", "CADELER.OL"
        ]
    
    # Sorterer listen alfabetisk for konsistent visning i menyer
    tickers = sorted(list(set(tickers)))
    
    # Lagrer den aktive listen som brukes til en lokal referansefil
    try:
        with open(TICKER_LIST_FILE, "w") as f:
            f.write("\n".join(tickers))
    except Exception:
        pass
        
    return tickers

def last_ned_data():
    """Laster ned 10 aars historikk for markedet og indekser."""
    tickers = hent_oppdaterte_tickers()
    start_date = (datetime.datetime.now() - datetime.timedelta(days=HISTORICAL_YEARS*365)).strftime('%Y-%m-%d')
    
    # 1. Last ned enkeltaksjer
    try:
        # Data hentes i bulk for optimal hastighet på MacBook Air
        data = yf.download(tickers=tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True)
        df_list = []
        
        for ticker in tickers:
            try:
                # Sjekker om tickeren faktisk eksisterer i de nedlastede dataene
                if ticker not in data.columns.get_level_values(0):
                    continue
                    
                temp_df = data[ticker].dropna(how='all').copy()
                if temp_df.empty:
                    continue
                
                temp_df['Ticker'] = ticker
                df_list.append(temp_df)
            except (KeyError, Exception):
                # Hopper over "døde" tickers uten å avbryte prosessen
                continue
                
        if df_list:
            full_df = pd.concat(df_list)
            full_df.reset_index(inplace=True)
            full_df.to_parquet(DATA_FILE, engine='pyarrow', compression='snappy')
    except Exception:
        pass

    # 2. Last ned Benchmark (OSEBX)
    try:
        benchmark = yf.download("^OSEBX", start=start_date, auto_adjust=True)
        if not benchmark.empty:
            benchmark.to_parquet(MARKET_DATA_FILE, engine='pyarrow')
    except Exception:
        pass
        
    return True

def hent_data():
    """Henter aksjedata fra disk."""
    if not os.path.exists(DATA_FILE):
        last_ned_data()
    
    try:
        df = pd.read_parquet(DATA_FILE)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def hent_markedsdata_df():
    """Henter benchmark-data."""
    if not os.path.exists(MARKET_DATA_FILE):
        last_ned_data()
    try:
        return pd.read_parquet(MARKET_DATA_FILE)
    except Exception:
        return pd.DataFrame()

def filtrer_likvide_aksjer(df, min_dagsomsetning=500000):
    """Ekskluderer illikvide aksjer basert på 20-dagers snittomsetning."""
    if df.empty:
        return df
        
    df_c = df.copy()
    df_c['Turnover'] = df_c['Close'] * df_c['Volume']
    
    # Beregner snittomsetning for de siste 20 dagene per ticker
    avg_turnover = df_c.groupby('Ticker')['Turnover'].apply(lambda x: x.tail(20).mean())
    likvide = avg_turnover[avg_turnover >= min_dagsomsetning].index.tolist()
    
    return df[df['Ticker'].isin(likvide)]