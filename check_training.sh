#!/bin/bash
# ==============================================================================
# Gaia Project Deep RL — Production Training Monitor & Dashboard
# ==============================================================================

PROJECT_DIR="/home/thibaulttoth/ProjetGaiai"
LOG_FILE="$PROJECT_DIR/az_run.log"
CKPT_DIR="$PROJECT_DIR/checkpoints"
LEAGUE_DIR="$PROJECT_DIR/league_checkpoints"

# If -w or --watch, loop and clear screen every 4 seconds
if [ "$1" = "-w" ] || [ "$1" = "--watch" ]; then
    while true; do
        clear
        $0 --no-loop
        echo ""
        echo "  [Mode Surveillance Live - Rafraîchissement auto 4s | Ctrl+C pour quitter]"
        sleep 4
    done
    exit 0
fi

echo "==================================================================================================="
echo "  🌌 GAIA PROJECT DEEP RL — TABLEAU DE BORD DE SURVEILLANCE SOTA"
echo "  📅 $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================================================="

# 1. GPU Telemetry
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
    GPU_PWR=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)
    echo "  🚀 GPU          : $GPU_NAME | Util: $GPU_UTIL | VRAM: $GPU_MEM | $GPU_TEMP°C | $GPU_PWR"
else
    echo "  🚀 GPU          : nvidia-smi non disponible"
fi

# 2. RL Training Master Process
TRAIN_PID=$(pgrep -f "training the model/train.py" | head -n 1)
if [ -n "$TRAIN_PID" ]; then
    CPU_MEM=$(ps -p "$TRAIN_PID" -o %cpu,%mem,etime --no-headers 2>/dev/null | xargs)
    echo "  🧠 Processus RL : EN COURS (PID $TRAIN_PID)"
    echo "  ⚙️  Ressources   : CPU/MEM/Durée: $CPU_MEM"
else
    echo "  ⚠️  Processus RL : ARRÊTÉ (aucun train.py actif)"
fi

# 3. Multiprocessing MCTS Workers
NUM_WORKERS=$(pgrep -f "multiprocessing.spawn" | wc -l)
echo "  🌲 Ouvriers MCTS: $NUM_WORKERS / 8 processus CPU actifs en parallèle (Gumbel 128 sims, k=12)"
if [ -f "$PROJECT_DIR/runs/current_status.txt" ]; then
    LIVE_STAT=$(cat "$PROJECT_DIR/runs/current_status.txt" 2>/dev/null)
    if [ -n "$LIVE_STAT" ]; then
        echo "  $LIVE_STAT"
    fi
fi

# 4. Checkpoints & League Status
if [ -f "$CKPT_DIR/gaia_latest.pt" ]; then
    CKPT_TIME=$(stat -c %y "$CKPT_DIR/gaia_latest.pt" 2>/dev/null | cut -d'.' -f1)
    CKPT_SIZE=$(ls -lh "$CKPT_DIR/gaia_latest.pt" 2>/dev/null | awk '{print $5}')
    echo "  💾 Dernier ckpt : $CKPT_DIR/gaia_latest.pt ($CKPT_SIZE, $CKPT_TIME)"
fi

NUM_SNAPS=$(ls -1 "$LEAGUE_DIR"/Snapshot_Epoch_*.pt 2>/dev/null | wc -l)
if [ "$NUM_SNAPS" -gt 0 ]; then
    LATEST_SNAP=$(ls -1t "$LEAGUE_DIR"/Snapshot_Epoch_*.pt 2>/dev/null | head -n 1 | xargs basename)
    echo "  🏆 Ligue        : $NUM_SNAPS snapshot(s) actif(s) (Dernier: $LATEST_SNAP @ 128 sims)"
else
    echo "  🏆 Ligue        : Initialisée (1er snapshot à l'époque 10)"
fi

echo "---------------------------------------------------------------------------------------------------"
echo "  📋 JOURNAL D'ENTRAÎNEMENT & MÉTRIQUES PAR ÉPOQUE :"
echo "---------------------------------------------------------------------------------------------------"

# Show header and recent epochs
if [ -f "$LOG_FILE" ]; then
    grep -E "Époque|Cadence|---" "$LOG_FILE" | tail -n 2
    sed 's/\r/\n/g' "$LOG_FILE" | grep -E '^\[[0-9]{4}/' | tail -n 15
else
    echo "  Fichier log introuvable: $LOG_FILE"
fi

echo "==================================================================================================="

# Follow mode if requested
if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
    echo "  📡 Mode direct activé (Ctrl+C pour quitter)..."
    echo "---------------------------------------------------------------------------------------------------"
    tail -n 0 -f "$LOG_FILE"
elif [ "$1" != "--no-loop" ]; then
    echo "  💡 Commandes utiles pour le suivi :"
    echo "     • Suivre en direct (streaming)      : ./check_training.sh -f"
    echo "     • Tableau de bord rafraîchi (live)  : ./check_training.sh -w"
    echo "     • Voir les 25 dernières lignes      : tail -n 25 az_run.log"
fi
