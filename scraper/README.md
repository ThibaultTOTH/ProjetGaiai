# 🚀 Gaia Project — Expert Game Scraper & Behavioral Cloning

Ce sous-projet extrait automatiquement des parties de championnat et de haut niveau sur [Boardgamers.space](https://www.boardgamers.space) (BGS) afin de pré-entraîner l'agent **DualGaiaAgent** par **Behavioral Cloning (Imitation)** avant de basculer sur l'optimisation AlphaZero MCTS.

---

## 🎯 Stratégie en 2 Phases

1. **Phase 1 : Imitation Expert (>170 VP)**
   - Collecte de parties de tournoi 4 joueurs où le vainqueur dépasse **170 VP** (moyenne constatée : ~188 VP, max 239 VP, joueurs Elo 400+).
   - Équilibrage des 14 factions de base du jeu de plateau.
   - Entraînement supervisé : en quelques époques, la politique passe de **0.03%** (hasard) à **>30% Top-1** et **>60% Top-5**, évitant l'écueil du blocage à 75 VP en Tabula Rasa.

2. **Phase 2 : Fine-Tuning AlphaZero Grand Maître (220+ VP)**
   - Initialisation directe à partir des poids pré-entraînés `checkpoints/gaia_supervised_pretrained.pt`.
   - Exploration MCTS (Gumbel 16-32 simulations) et Ligue PBT pour surpasser le jeu humain.

---

## 💻 Commandes Rapides

### 1. Télécharger des parties expertes (>170 VP)
```bash
# Télécharger les parties de haut niveau de manière équilibrée
python -m scraper.scraper --target-per-faction 500 --min-vp 171
```

### 2. Consulter les statistiques du pool
```bash
python -m scraper.stats
```

### 3. Compiler le dataset PyTorch
```bash
# Génère scraper/data/dataset/gaia_expert_dataset.pt
python -m scraper.convert_to_dataset --min-vp 171
```

### 4. Lancer le pré-entraînement supervisé
```bash
# Entraîne le réseau et produit checkpoints/gaia_supervised_pretrained.pt
python -m scraper.train_supervised --epochs 25 --batch-size 128
```

---

## 🖥️ Utilisation via le GUI

L'interface graphique `python gui.py` intègre directement ces deux modes :
- Bouton **"🚀 1. Imitation Expert (>170 VP)"** : configure l'algorithme sur **Imitation (Supervisé)** et entraîne le réseau sur le dataset compilé avec retour visuel des courbes en temps réel.
- Bouton **"👑 2. AlphaZero Fine-Tuning (220+ VP)"** : charge automatiquement les poids pré-entraînés pour débuter le self-play à un haut niveau stratégique.
