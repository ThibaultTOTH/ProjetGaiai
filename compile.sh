#!/usr/bin/env bash
# ==============================================================================
#  Projet Gaia - Compilation du Moteur Natif Rust (Pop!_OS / Linux)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Projet Gaia - Compilation du Moteur Natif Rust (Linux)"
echo "============================================================"

# Verification de Cargo
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo "[-] Rust / Cargo non trouve."
        echo "[+] Installation recommandee via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
fi

echo "[✓] Cargo detecte : $(cargo --version)"

# Compilation de projet_gaiapi
cd "$SCRIPT_DIR/projet_gaiapi"
echo "[+] Compilation en mode release..."
cargo build --release

SO_PATH="$SCRIPT_DIR/projet_gaiapi/target/release/libgaiapi.so"
if [ -f "$SO_PATH" ]; then
    echo "[✓] Bibliotheque C-ABI compilee avec succes : $SO_PATH"
    cp -f "$SO_PATH" "$SCRIPT_DIR/training the model/libgaiapi.so" 2>/dev/null || true
else
    echo "[!] Erreur : $SO_PATH introuvable apres compilation !"
    exit 1
fi

cd "$SCRIPT_DIR"
echo "============================================================"
echo "  Compilation terminee avec succes !"
echo "============================================================"
