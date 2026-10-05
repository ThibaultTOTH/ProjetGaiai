#!/usr/bin/env bash
# ==============================================================================
#  🚀 Projet Gaia — Compilation du Moteur Natif Rust (Linux: Ubuntu, Pop!_OS, Debian)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  🚀 Projet Gaia — Compilation du Moteur Natif Rust (Linux)"
echo "============================================================"

# Auto-détection de Cargo / Rustup
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
elif [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v cargo &> /dev/null; then
    echo "[-] Rust / Cargo non trouvé."
    echo "[+] Installation recommandée via :"
    echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "    OU :"
    echo "    sudo apt update && sudo apt install -y cargo rustc"
    exit 1
fi

echo "[✓] Cargo détecté : $(cargo --version)"

# Compilation de projet_gaiapi
cd "$SCRIPT_DIR/projet_gaiapi"
echo "[*] Compilation en mode release (C-ABI cdylib)..."
cargo build --release

SO_PATH="$SCRIPT_DIR/projet_gaiapi/target/release/libgaiapi.so"
if [ ! -f "$SO_PATH" ]; then
    SO_PATH="$SCRIPT_DIR/target/release/libgaiapi.so"
fi

if [ -f "$SO_PATH" ]; then
    echo "[✓] Bibliothèque C-ABI compilée avec succès : $SO_PATH"
    cp -f "$SO_PATH" "$SCRIPT_DIR/training the model/libgaiapi.so" 2>/dev/null || true
    if [ -d "$SCRIPT_DIR/gaia_gui_lab" ]; then
        cp -f "$SO_PATH" "$SCRIPT_DIR/gaia_gui_lab/libgaiapi.so" 2>/dev/null || true
    fi
    echo "[✓] Bibliothèque synchronisée dans 'training the model/' et 'gaia_gui_lab/'."
else
    echo "[!] Erreur : libgaiapi.so introuvable après compilation !"
    exit 1
fi

cd "$SCRIPT_DIR"
echo "============================================================"
echo "  ✅ Compilation terminée avec succès !"
echo "============================================================"

