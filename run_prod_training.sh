#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd ~/ProjetGaiai
export PYTHONPATH="training the model":$PYTHONPATH
echo "=== Demarrage de l'entrainement AlphaZero [$(date)] ===" >> training.log
nohup python -u "training the model/train.py" --preset grandmaster --use-best-hyperparams --actors 8 --resume checkpoints/gaia_latest.pt --epochs 10000 >> training.log 2>&1 &
echo $! > training.pid
echo "Entrainement lance en arriere-plan avec PID: $(cat training.pid)"
