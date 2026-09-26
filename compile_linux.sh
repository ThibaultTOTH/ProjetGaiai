#!/usr/bin/env bash
# ==============================================================================
# Script de compilation officiel du moteur Rust gaiapi pour Linux (Ubuntu, Debian, Pop!_OS)
# Produit le fichier 'libgaiapi.so' pour le bridge Python C-ABI.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJET_DIR="$SCRIPT_DIR/projet_gaiapi"
MODEL_DIR="$SCRIPT_DIR/training the model"
GUI_DIR="$SCRIPT_DIR/gaia_gui_lab"

echo "=================================================================="
echo "  🚀 COMPILATION DU MOTEUR RUST GAIAPI POUR LINUX (.so)"
echo "=================================================================="

# S'assurer que Cargo / Rustup est dans le PATH
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
elif [ -d "$HOME/.cargo/bin" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 1. Vérification de Cargo / Rust
if ! command -v cargo &> /dev/null; then
    echo "❌ Erreur : 'cargo' (Rust) n'est pas installé sur ce système Linux."
    echo "💡 Pour installer Rust rapidement sur Pop!_OS / Ubuntu, exécutez :"
    echo "   sudo apt update && sudo apt install -y cargo rustc"
    echo "   OU :"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source \$HOME/.cargo/env"
    exit 1
fi

echo "📦 Version de Rust détectée :"
rustc --version
cargo --version

# 2. Compilation release
echo ""
echo "⚙️  Compilation en cours (mode release haute performance)..."
cd "$PROJET_DIR"
cargo build --release

SO_PATH="$PROJET_DIR/target/release/libgaiapi.so"

if [ -f "$SO_PATH" ]; then
    echo ""
    echo "✅ Compilation réussie : $SO_PATH"
    
    # 3. Déploiement automatique vers les dossiers Python
    echo "📋 Copie de libgaiapi.so vers 'training the model' et 'gaia_gui_lab'..."
    cp "$SO_PATH" "$MODEL_DIR/libgaiapi.so"
    cp "$SO_PATH" "$GUI_DIR/libgaiapi.so" 2>/dev/null || true
    
    echo "🎉 Moteur natif Linux prêt à l'emploi !"
    echo "   Vous pouvez maintenant lancer l'entraînement :"
    echo "   cd 'training the model' && python train.py --preset grandmaster --device cuda"
else
    echo "❌ Erreur : le fichier '$SO_PATH' n'a pas été trouvé après la compilation."
    exit 1
fi

echo "=================================================================="
