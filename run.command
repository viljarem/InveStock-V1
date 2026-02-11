#!/bin/bash

# Gå til prosjektmappen
cd "$(dirname "$0")"

# Last inn shell profil for å få tilgang til npm
source ~/.zshrc 2>/dev/null || source ~/.bash_profile 2>/dev/null

# Sjekk om python er tilgjengelig
if ! command -v python3 &> /dev/null; then
    echo "❌ Python er ikke installert!"
    echo "Last ned fra: https://www.python.org/"
    read -p "Trykk Enter for å avslutte..."
    exit 1
fi

# Sjekk om streamlit er installert
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Installerer avhengigheter..."
    pip3 install streamlit pandas plotly yfinance
fi

echo "🚀 Starter InveStock Pro..."
echo "Åpner nettleseren..."

python3 -m streamlit run app.py
