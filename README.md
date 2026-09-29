# 🌌 Projet Gaia AI — Deep Reinforcement Learning Studio

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B%20(Edition%202021)-orange.svg)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-23%2F23%20Passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20(Ubuntu%2FPop!__OS)-lightgrey.svg)]()

> Système d'Intelligence Artificielle de niveau Grand Maître pour le jeu de stratégie **Gaia Project** (*Projet Gaia* + extension *Lost Fleet*).
> Combine un **moteur de simulation natif Rust ultra-rapide**, une **phase d'imitation supervisée** sur 849 parties de tournoi de très haut niveau, et un **moteur AlphaZero MCTS multi-agents** renforcé par une **Ligue de population (FSP)** et un **curriculum de crise tactique (puzzles RGSC)**.

---

## 📑 Sommaire
1. [Architecture Globale](#-architecture-globale)
2. [Installation & Compilation](#-installation--compilation)
3. [Stratégie d'Entraînement en Deux Temps](#-stratégie-dentraînement-en-deux-temps)
4. [Interfaces & Outils](#-interfaces--outils)
5. [Technologies Deep RL & SOTA 2026](#-technologies-deep-rl--sota-2026)
6. [Structure du Dépôt](#-structure-du-dépôt)
7. [Validation & Tests](#-validation--tests)

---

## 🏛️ Architecture Globale

Le projet est articulé autour d'une architecture hybride **Rust / PyTorch** garantissant une cadence de simulation maximale et une grande expressivité neuronale :

```
                        ┌────────────────────────────────────────────────────────┐
                        │              Projet Gaia AI Ecosystem                  │
                        └──────────────────────────┬─────────────────────────────┘
                                                   │
        ┌──────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                          ▼                                        ▼
┌─────────────────────────┐            ┌─────────────────────────┐              ┌────────────────────────┐
│  projet_gaiapi (Rust)   │            │ training the model (Py) │              │    scraper (Python)    │
│  Moteur natif C-ABI     │            │ Deep RL & AlphaZero     │              │ Données Grands Maîtres │
├─────────────────────────┤            ├─────────────────────────┤              ├────────────────────────┤
│ • 3130 actions discrètes│ ◄──C-ABI──┤ • DualStreamBackbone    │              │ • 849 parties BGS      │
│ • 2476 features d'obs   │            │   (HexGNN + SwiGLU)     │ ◄--Dataset-- │ • 276 806 coups GM     │
│ • Règles complètes base │            │ • AlphaZero Gumbel MCTS │              │ • Gagnant ≥ 150 VP     │
│   + extension Lost Fleet│            │ • League Training (Elo) │              │ • Équilibré 14 factions│
│ • Clonage en micro-sec  │            │ • Puzzles de crise RGSC │              │ • Behavioral Cloning   │
└─────────────────────────┘            └───────────┬─────────────┘              └────────────────────────┘
                                                   │
                                       ┌───────────┴─────────────┐
                                       ▼                         ▼
                           ┌───────────────────────┐ ┌───────────────────────┐
                           │   gui.py (Tkinter)    │ │ play.py (PyQt Lab)    │
                           │ Studio d'entraînement │ │ Station tactique &    │
                           │ & Courbes en direct   │ │ Sandbox interactive   │
                           └───────────────────────┘ └───────────────────────┘
```

---

## ⚙️ Installation & Compilation

### 1. Prérequis
- **Python** : 3.10 ou supérieur (recommandé : Anaconda / venv).
- **Rust & Cargo** : Version 1.75+ ([rustup.rs](https://rustup.rs/)).
- **GPU (Optionnel mais recommandé)** : Carte NVIDIA avec pilotes récents (testé et calibré sur RTX 5070 / CUDA 12.8, avec fallback CPU automatique).

### 2. Compilation du Moteur Natif Rust
Le moteur de simulation doit être compilé en bibliothèque partagée (`.dll` sur Windows, `.so` sur Linux). Deux scripts automatisés à la racine s'occupent de la détection du compilateur, du build release et de la copie des binaires :

* **Sur Windows :**
  ```cmd
  compile.bat
  ```

* **Sur Linux (Ubuntu, Pop!_OS, Debian) :**
  ```bash
  chmod +x compile.sh
  ./compile.sh
  ```

### 3. Dépendances Python
Installez les bibliothèques requises :
```bash
pip install -r requirements.txt
```
*(Si vous disposez d'un GPU NVIDIA, assurez-vous d'installer PyTorch avec support CUDA : `pip install torch --index-url https://download.pytorch.org/whl/cu128`)*

---

## 🚀 Stratégie d'Entraînement en Deux Temps

Pour résoudre la complexité combinatoire extrême de Gaia Project sans rester bloqué sur le plateau des ~75 VP (actions aléatoires myopes), le système utilise une approche en deux phases :

```
[849 Parties BGS ≥ 150 VP]
          │
          ▼
┌────────────────────────────────────────┐
│  Phase 1 : Imitation Supervisée        │  ──► Baseline solide (~140 - 160 VP)
│  (Clonage de Comportement / 277k coups)│      Backbone & Têtes initialisés
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Phase 2 : AlphaZero Fine-Tuning       │  ──► Visée niveau Grand Maître (220+ VP)
│  • MCTS Gumbel Sequential Halving      │      Dépassement du niveau humain
│  • Ligue de Population & Matchmaking   │      sans oubli catastrophique
│  • Puzzles de crise RGSC (Manches 2-5) │
└────────────────────────────────────────┘
```

### Étape 1 : Pré-entraînement Supervisé (Imitation Expert)
Entraîne le réseau à prédire les coups joués par les meilleurs joueurs du monde sur 276 806 situations de jeu :

* **En ligne de commande :**
  ```bash
  python scraper/train_supervised.py --epochs 25 --batch-size 256 --lr 1e-3 --device cuda
  ```
* **Via le GUI :** Sélectionnez le preset `🚀 1. Imitation Expert (>150 VP)` et cliquez sur **Démarrer l'entraînement**.
* **Résultat :** En 20 époques (~15 min sur GPU), le modèle atteint ~35-40% de précision Top-1 (face à 0.03% au hasard) et sauvegarde `checkpoints/gaia_supervised_pretrained.pt`.

### Étape 2 : AlphaZero MCTS Fine-Tuning
L'agent utilise les poids de l'imitation comme point de départ et s'affronte lui-même via recherche arborescente MCTS pour perfectionner ses plans à long terme :

* **En ligne de commande :**
  ```bash
  python "training the model/alphazero_trainer.py"
  ```
* **Via le GUI :** Sélectionnez le preset `👑 2. AlphaZero Fine-Tuning (220+ VP)` et cliquez sur **Démarrer l'entraînement**. Le modèle charge automatiquement les poids de l'imitation et commence le self-play MCTS.

---

## 🖥️ Interfaces & Outils

Le projet fournit trois interfaces complémentaires :

### 1. Studio d'Entraînement (`python gui.py`)
Interface graphique Tkinter complète :
* **Tableau de bord :** Courbes d'apprentissage en direct (perte de politique, perte de valeur, scores réels vs prédits), vitesse en steps/s, indicateurs KPI.
* **Paramétrage SOTA 2026 :** Contrôle granulaire des 14 briques d'algorithmes (MCTS, Ligue, Puzzles RGSC, GNN, R-NaD, RND, etc.).
* **Auto-Tuning Hyperparamètres :** Module d'optimisation bayésienne TPE pour explorer les hyperparamètres sans surapprentissage.
* **Évaluation & Ligue :** Classement Elo en direct de toutes les versions historiques de l'IA.

### 2. Laboratoire Tactique & Sandbox (`python play.py`)
Interface PyQt interactive permettant de :
* Visualiser le plateau spatial hexagonal avec ses 200 cases et ses secteurs stellaires.
* Disposer des mines, stations, laboratoires et vaisseaux.
* Interroger le réseau de neurones en temps réel pour inspecter les probabilités de chaque coup et l'espérance de score prédite.
* Jouer manuellement contre l'agent ou simuler des parties coup par coup.

### 3. Outils du Scraper (`scraper/`)
* `python scraper/scraper.py` : Scrape les parties terminées sur Boardgamers.space via pagination intelligente et filtres de score.
* `python scraper/stats.py` : Analyse statistique détaillée du dataset (répartition des 14 factions, classements Elo des joueurs, scores moyens).
* `python scraper/convert_to_dataset.py` : Compile les fichiers JSON bruts en tenseurs PyTorch optimisés.

---

## 🧬 Technologies Deep RL & SOTA 2026

| Brique Technologique | Module | Description & Rôle |
|---|---|---|
| **HexGNN Spatial** | [`models.py`](training the model/models.py) | Réseau de neurones sur graphes dédié au plateau hexagonal. Propage l'information topologique entre cases adjacentes (portée des vaisseaux, blocages, clusters de fédération). |
| **SwiGLU ResNet** | [`models.py`](training the model/models.py) | Tronc résiduel profond à 4 couches [1024, 1024, 512, 256] avec portes d'activation SwiGLU, calibré pour exploiter le phénomène de *Deep Double Descent*. |
| **MCTS Gumbel (GAZ)** | [`mcts.py`](training the model/mcts.py) | Algorithme *Sequential Halving* 2026 : alloue un budget de simulations (16-32) ciblé sur les 8 actions les plus prometteuses, avec guidage de score optimiste. |
| **Ligue de Population** | [`league.py`](training the model/league.py) | Système inspiré d'AlphaStar : maintient un pool d'instantanés historiques, matchmaking Gaussien (zone proximale de développement) et calcul Elo multi-joueurs Bradley-Terry. |
| **Puzzles RGSC** | [`buffer.py`](training the model/buffer.py) | *Regret-Guided Search Control* (Go-Exploit) : mémorise les états où l'IA a commis une faute en milieu de partie (Manches 2-5) et réinjecte ces états en puzzle tactique lors du self-play. |
| **R-NaD & RND** | [`trainer.py`](training the model/trainer.py), [`rnd.py`](training the model/rnd.py) | Dynamique de Nash régularisée (anti-cyclage 4 joueurs) et distillation de curiosité intrinsèque pour l'algorithme PPO alternatif. |

---

## 📂 Structure du Dépôt

```text
├── compile.bat              # Script de compilation Rust pour Windows
├── compile.sh               # Script de compilation Rust pour Linux (Ubuntu, Pop!_OS)
├── gui.py                   # Lanceur racine du Studio Graphique Tkinter
├── play.py                  # Lanceur racine de la Station Tactique PyQt
├── train.py                 # Lanceur racine d'entraînement CLI
├── requirements.txt         # Dépendances Python du projet
│
├── projet_gaiapi/           # 🦀 Moteur de règles natif en Rust
│   ├── Cargo.toml           # Configuration de compilation Rust (cdylib C-ABI)
│   ├── src/                 # Code source Rust (règles, plateau, actions, Lost Fleet)
│   └── tests/               # Tests unitaires du moteur Rust
│
├── scraper/                 # 🌐 Scraper et dataset d'experts BGS
│   ├── scraper.py           # Téléchargement automatique des parties
│   ├── stats.py             # Statistiques et métadonnées du dataset
│   ├── convert_to_dataset.py# Générateur de dataset PyTorch
│   ├── train_supervised.py  # Entraîneur de clonage de comportement (Phase 1)
│   └── data/                # Fichiers bruts JSON et dataset binaire .pt
│
├── training the model/      # 🧠 Cœur Deep Reinforcement Learning
│   ├── alphazero_trainer.py # Boucle d'entraînement AlphaZero MCTS
│   ├── trainer.py           # Boucle d'entraînement PPO / Acteur-Critique
│   ├── models.py            # Architectures HexGNN, SwiGLU, DualGaiaAgent
│   ├── mcts.py              # Moteur MCTS multi-joueurs PUCT & Gumbel
│   ├── league.py            # Gestionnaire de Ligue de Population & Elo
│   ├── buffer.py            # Tampons de transition et puzzles de crise RGSC
│   ├── config.py            # Presets et hyperparamètres centralisés
│   ├── gui.py               # Implémentation complète de l'interface Tkinter
│   └── test_pipeline.py     # Suite de tests d'intégration (23 tests)
│
└── gaia_gui_lab/            # 🎨 Station tactique & Sandbox interactive (PyQt)
    └── main.py              # Interface graphique avec plateau interactif
```

---

## 🧪 Validation & Tests

La suite complète de tests vérifie l'intégrité de la chaîne de calcul, de la C-ABI Rust jusqu'aux calculs tensoriels PyTorch :

```bash
pytest "training the model/test_pipeline.py" -v
```

**23 tests unitaires et d'intégration validés (100% de succès) :**
- Passes avant et arrière des réseaux (`test_models_forward_and_backward`)
- Bridge C-ABI natif et masquage d'actions 3130D (`test_native_environment`, `test_flat_action_space_and_native_bridge`)
- Recherche MCTS Gumbel & PUCT (`test_mcts_engine`, `test_gumbel_alphazero`)
- Système de ligue, matchmaking Gaussien et Elo (`test_league_training`, `test_gaussian_elo_matchmaking`)
- Puzzles tactiques de crise RGSC (`test_rgsc_state_curriculum`)
- Équilibre de Nash multi-joueurs R-NaD (`test_rnad_equilibrium`)
- Encodeur spatial HexGNN (`test_hex_gnn_encoder`)
- Éléments du jeu et extension Lost Fleet (`test_game_elements_and_lost_fleet_rules`)

---

## 📜 Licence
Projet sous licence MIT — Développé pour la recherche en Deep Reinforcement Learning multi-agents appliqué aux jeux de plateau modernes.
