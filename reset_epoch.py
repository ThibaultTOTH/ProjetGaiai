import os
import sys
import glob

# Ensure "training the model" is on sys.path so config unpickles cleanly
repo_root = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(repo_root, "training the model")
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch

def main():
    target_epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    ckpt_path = os.path.join(repo_root, "checkpoints", "gaia_latest.pt")

    if not os.path.exists(ckpt_path):
        print(f"[!] Fichier introuvable : {ckpt_path}")
        return

    print(f"Chargement de {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    old_epoch = ckpt.get("epoch", "?")
    ckpt["epoch"] = target_epoch
    if "meta" in ckpt and isinstance(ckpt["meta"], dict):
        ckpt["meta"]["epoch"] = target_epoch
    else:
        ckpt["meta"] = {"epoch": target_epoch, "algorithm": "AlphaZero"}

    torch.save(ckpt, ckpt_path)
    print(f"[OK] gaia_latest.pt recale avec succes : epoque {old_epoch} -> {target_epoch}")

    # Nettoyage des snapshots dummy au-dela de target_epoch
    cleaned = 0
    patterns = [
        os.path.join(repo_root, "league_checkpoints", "Snapshot_Epoch_*.pt"),
        os.path.join(repo_root, "checkpoints", "az_checkpoint_epoch_*.pt"),
        os.path.join(repo_root, "checkpoints", "az_checkpoint_*.pt"),
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            basename = os.path.splitext(os.path.basename(f))[0]
            for part in basename.split("_"):
                if part.isdigit():
                    ep = int(part)
                    if ep > target_epoch:
                        try:
                            os.remove(f)
                            print(f"[OK] Supprime snapshot dummy : {os.path.basename(f)}")
                            cleaned += 1
                        except Exception:
                            pass
                    break

    print(f"[TERMINE] Pret pour relancer l'entrainement a partir de l'epoque {target_epoch} !")

if __name__ == "__main__":
    main()
