#!/usr/bin/env bash
# ==============================================================================
#  Projet Gaia - Lancement de Gaia AI Lab & Sandbox (Pop!_OS / Linux)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SO_PATH="$SCRIPT_DIR/projet_gaiapi/target/release/libgaiapi.so"
if [ ! -f "$SO_PATH" ]; then
    echo "[!] libgaiapi.so introuvable. Lancement de la compilation..."
    bash "$SCRIPT_DIR/compile.sh"
fi

# Selection de Python
PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    else
        echo "[!] Python 3 introuvable ! Installez python3."
        exit 1
    fi
fi

echo "[✓] Lancement de Gaia AI Lab..."
cd "$SCRIPT_DIR/gaia_gui_lab"
$PYTHON_BIN main.py
