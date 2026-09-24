#!/usr/bin/env bash
# ==============================================================================
#  Gaia Project Deep RL — Linux One-Click Build & Setup Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  🚀 Gaia Project Deep RL — Setup & Build Linux"
echo "=================================================================="

# 1. Check for Rust & Cargo
if ! command -v cargo &> /dev/null; then
    echo "[-] Rust compiler (cargo) not found."
    echo "[+] Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[✓] Rust compiler found: $(cargo --version)"
fi

# 2. Build Native C-ABI Engine (.so library)
echo ""
echo "=== [1/3] Compiling Native Rust Game Engine (cdylib) ==="
cd "$SCRIPT_DIR/projet_gaiapi"
cargo build --release

SO_PATH="$SCRIPT_DIR/projet_gaiapi/target/release/libgaiapi.so"
if [ -f "$SO_PATH" ]; then
    echo "[✓] Native library built successfully: $SO_PATH"
else
    echo "[!] Error: $SO_PATH not found after build!"
    exit 1
fi
cd "$SCRIPT_DIR"

# 3. Python Requirements
echo ""
echo "=== [2/3] Checking Python & Dependencies ==="
PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    else
        echo "[!] Python 3 not found! Please install python3 and pip."
        exit 1
    fi
fi
echo "[✓] Using Python: $($PYTHON_BIN --version)"

echo "[+] Installing requirements (torch, numpy, matplotlib)..."
$PYTHON_BIN -m pip install -q -r requirements.txt || true

# 4. Run Test Suite
echo ""
echo "=== [3/3] Running Verification Suite (24 Tests) ==="
$PYTHON_BIN "training the model/test_pipeline.py"

echo ""
echo "=================================================================="
echo "  🎉 SETUP SUCCESSFUL! Your Linux environment is 100% ready."
echo "=================================================================="
echo ""
echo "To use the project on Pop!_OS / Linux:"
echo "  1. Launch Gaia AI Lab & Sandbox GUI:"
echo "     ./run_lab.sh"
echo ""
echo "  2. Headless / Server / Overnight Training (Recommended):"
echo "     python3 \"training the model/train.py\" --preset grandmaster"
echo ""
echo "  3. Quick Verification Pipeline:"
echo "     python3 \"training the model/test_pipeline.py\""
echo ""
