#!/bin/bash

# InveStock - Stock Analysis App Launcher
# Dobbelklikk denne filen for å starte appen

# Absolutt sti til prosjektet
APP_DIR="/Users/viljar/Programmering/Aksje app/InveStock DEV"
VENV_STREAMLIT="$APP_DIR/.venv/bin/streamlit"

cd "$APP_DIR"

# Sjekk at streamlit finnes i venv
if [ ! -f "$VENV_STREAMLIT" ]; then
    echo "❌ Finner ikke streamlit i .venv. Kjør: pip install streamlit"
    echo "Trykk Enter for å lukke..."
    read
    exit 1
fi

# Start Streamlit appen med venv-streamlit direkte (trenger ikke activate)
echo "🚀 Starter InveStock..."
"$VENV_STREAMLIT" run app.py --logger.level=error
