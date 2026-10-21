#!/bin/bash
cd /home/thibaulttoth/ProjetGaiai || exit 1
export PYTHONUNBUFFERED=1
export PYTHONPATH="training the model":scraper:$PYTHONPATH

# Clean restart from supervised tournament bootstrap requested by user
RESUME_CKPT="checkpoints/gaia_supervised_tournament_ultra.pt"

# Archive previous run log if present and non-empty
if [ -s "az_run.log" ]; then
    ARCHIVE_NAME="az_run_archive_$(date +%Y%m%d_%H%M%S).log"
    cp az_run.log "$ARCHIVE_NAME"
    > az_run.log
    echo "[Launcher] Archived previous log to $ARCHIVE_NAME"
fi

# Reset current_status.txt and archive games.log for clean audit
mkdir -p runs
echo "Initialisation de l'entraînement SOTA (128 sims, k=12, 8 parties/époque, Ligue bootstrap)..." > runs/current_status.txt
if [ -f "runs/games.log" ]; then
    mv runs/games.log "runs/games_archive_$(date +%Y%m%d_%H%M%S).log"
fi

# Ensure league has ONLY the bootstrap snapshot at start
mkdir -p league_checkpoints
rm -f league_checkpoints/Snapshot_Epoch_*.pt
cp "$RESUME_CKPT" "league_checkpoints/Snapshot_Epoch_0000_Bootstrap.pt"

echo "[Launcher] Resuming clean from bootstrap: $RESUME_CKPT"
echo "[Launcher] Hyperparameters: N=128 simulations, k=12 candidates, 8 games/epoch, 8 workers"
echo "[Launcher] League Pool: 1 initial snapshot (Snapshot_Epoch_0000_Bootstrap @ 128 sims)"

/home/thibaulttoth/anaconda3/bin/python -u "training the model/train.py" \
    --algo alphazero \
    --hyperparams runs/best_hyperparams_calibrated.json \
    --resume "$RESUME_CKPT" \
    --reset-epoch \
    --actors 8 \
    --sims 128 \
    --epochs 10000 2>&1 | tee -a az_run.log
