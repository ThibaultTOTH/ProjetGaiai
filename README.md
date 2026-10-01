# ðŸŒŒ Projet Gaia AI â€” Deep Reinforcement Learning Studio

> **NOTE DE MISE À JOUR : ÉTAT DES LIEUX & CORRECTION ARCHITECTURALE PROFONDE (v2.0)**
> *Suite à une critique exigeante de la stack IA, 5 angles morts critiques ont été identifiés et rigoureusement corrigés pour garantir que l'implémentation reflète réellement les ambitions théoriques de l'AlphaZero multi-agents.*
> 
> ### Angles Morts Identifiés & Résolus :
> 1. **Fuite de Données en Apprentissage Supervisé (Data Leakage) :** 
>    * **Angle mort :** Le script de clonage comportemental (	rain_supervised.py) injectait directement la valeur de victoire finale et les actions futures dans le vecteur d'observation avant la prédiction du réseau, détruisant la généralisation du modèle. L'IA trichait littéralement sur le set d'entraînement.
>    * **Correction :** Nettoyage complet de _build_batch_obs pour s'assurer que l'observation reflète strictement l'état du jeu avant l'action, forçant le modèle à réellement induire les règles.
> 2. **Asymétrie du Self-Play AlphaZero :**
>    * **Angle mort :** La boucle de collecte (lphazero_trainer.py) ne sauvegardait les transitions que pour le siège 0, biaisant l'expérience, jetant 75% de l'expérience et violant les principes fondamentaux du self-play. De plus, la faction Terran était codée en dur.
>    * **Correction :** Refonte de la boucle de self-play. Les factions sont aléatoirement distribuées (les 14 factions), et les transitions de *tous* les joueurs utilisant la politique courante sont collectées en parallèle (tracking CurrentPolicy multi-seats).
> 3. **Hallucination des Valeurs MCTS (Double-Comptage VP) :**
>    * **Angle mort :** L'arbre MCTS ajoutait artificiellement le delta des Points de Victoire (VP) aux prédictions absolues du réseau, menant à des évaluations de nœuds absurdes.
>    * **Correction :** Le réseau est désormais entraîné à prédire le VP final absolu. Le MCTS (mcts.py) projette directement ces valeurs via Gumbel et PUCT sans double-comptage et calcule les vecteurs Max^n correctement.
> 4. **Tenseur de Valeur Multi-Joueurs (4D) au lieu de Scalaire :**
>    * **Angle mort :** Le réseau tentait de prédire le score d'un jeu à 4 joueurs asymétriques via un unique scalaire (1D), écrasant la dynamique de somme non-nulle Max^n.
>    * **Correction :** Le ScorePredictorNet (models.py) génère désormais un tenseur (B, 4) prédisant le score simultané des 4 joueurs, utilisé pour le calcul de regret et la valorisation PUCT.
> 5. **Tête Spatiale Déconnectée (ActionOptimizerNet) :**
>    * **Angle mort :** Le GNN calculait de superbes représentations nodales (HexGNN), mais le réseau d'actions moyennait l'espace (Global Pooling) avant de prédire des actions spatiales à l'aide d'un simple perceptron, détruisant l'information géométrique.
>    * **Correction :** Restructuration en spatial_head et scalar_head. Les actions topologiques (Construire Mine, Projet Gaia) utilisent directement les tenseurs des 200 cases non compressés, restituant l'espace exact (3130 actions discrètes).
>
> *(L'audit du moteur Rust (ctions.rs) a révélé qu'il était en revanche extrêmement robuste : il calcule les portées QIC dynamiques, les adjacences et les graphes BFS de fédération avec une grande précision, sans raccourcis de règles).*

---


[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B%20(Edition%202021)-orange.svg)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-23%2F23%20Passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20(Ubuntu%2FPop!__OS)-lightgrey.svg)]()

> SystÃ¨me d'Intelligence Artificielle de niveau Grand MaÃ®tre pour le jeu de stratÃ©gie **Gaia Project** (*Projet Gaia* + extension *Lost Fleet*).
> Combine un **moteur de simulation natif Rust ultra-rapide**, une **phase d'imitation supervisÃ©e** sur 849 parties de tournoi de trÃ¨s haut niveau, et un **moteur AlphaZero MCTS multi-agents** renforcÃ© par une **Ligue de population (FSP)** et un **curriculum de crise tactique (puzzles RGSC)**.

---

## ðŸ“‘ Sommaire
1. [Architecture Globale](#-architecture-globale)
2. [Installation & Compilation](#-installation--compilation)
3. [StratÃ©gie d'EntraÃ®nement en Deux Temps](#-stratÃ©gie-dentraÃ®nement-en-deux-temps)
4. [Interfaces & Outils](#-interfaces--outils)
5. [Technologies Deep RL & SOTA 2026](#-technologies-deep-rl--sota-2026)
6. [Structure du DÃ©pÃ´t](#-structure-du-dÃ©pÃ´t)
7. [Validation & Tests](#-validation--tests)

---

## ðŸ›ï¸ Architecture Globale

Le projet est articulÃ© autour d'une architecture hybride **Rust / PyTorch** garantissant une cadence de simulation maximale et une grande expressivitÃ© neuronale :

```
                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                        â”‚              Projet Gaia AI Ecosystem                  â”‚
                        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                   â”‚
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â–¼                                          â–¼                                        â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  projet_gaiapi (Rust)   â”‚            â”‚ training the model (Py) â”‚              â”‚    scraper (Python)    â”‚
â”‚  Moteur natif C-ABI     â”‚            â”‚ Deep RL & AlphaZero     â”‚              â”‚ DonnÃ©es Grands MaÃ®tres â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤            â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤              â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ â€¢ 3130 actions discrÃ¨tesâ”‚ â—„â”€â”€C-ABIâ”€â”€â”¤ â€¢ DualStreamBackbone    â”‚              â”‚ â€¢ 849 parties BGS      â”‚
â”‚ â€¢ 2476 features d'obs   â”‚            â”‚   (HexGNN + SwiGLU)     â”‚ â—„--Dataset-- â”‚ â€¢ 276 806 coups GM     â”‚
â”‚ â€¢ RÃ¨gles complÃ¨tes base â”‚            â”‚ â€¢ AlphaZero Gumbel MCTS â”‚              â”‚ â€¢ Gagnant â‰¥ 150 VP     â”‚
â”‚   + extension Lost Fleetâ”‚            â”‚ â€¢ League Training (Elo) â”‚              â”‚ â€¢ Ã‰quilibrÃ© 14 factionsâ”‚
â”‚ â€¢ Clonage en micro-sec  â”‚            â”‚ â€¢ Puzzles de crise RGSC â”‚              â”‚ â€¢ Behavioral Cloning   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                   â”‚
                                       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                       â–¼                         â–¼
                           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                           â”‚   gui.py (Tkinter)    â”‚ â”‚ play.py (PyQt Lab)    â”‚
                           â”‚ Studio d'entraÃ®nement â”‚ â”‚ Station tactique &    â”‚
                           â”‚ & Courbes en direct   â”‚ â”‚ Sandbox interactive   â”‚
                           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## âš™ï¸ Installation & Compilation

### 1. PrÃ©requis
- **Python** : 3.10 ou supÃ©rieur (recommandÃ© : Anaconda / venv).
- **Rust & Cargo** : Version 1.75+ ([rustup.rs](https://rustup.rs/)).
- **GPU (Optionnel mais recommandÃ©)** : Carte NVIDIA avec pilotes rÃ©cents (testÃ© et calibrÃ© sur RTX 5070 / CUDA 12.8, avec fallback CPU automatique).

### 2. Compilation du Moteur Natif Rust
Le moteur de simulation doit Ãªtre compilÃ© en bibliothÃ¨que partagÃ©e (`.dll` sur Windows, `.so` sur Linux). Deux scripts automatisÃ©s Ã  la racine s'occupent de la dÃ©tection du compilateur, du build release et de la copie des binaires :

* **Sur Windows :**
  ```cmd
  compile.bat
  ```

* **Sur Linux (Ubuntu, Pop!_OS, Debian) :**
  ```bash
  chmod +x compile.sh
  ./compile.sh
  ```

### 3. DÃ©pendances Python
Installez les bibliothÃ¨ques requises :
```bash
pip install -r requirements.txt
```
*(Si vous disposez d'un GPU NVIDIA, assurez-vous d'installer PyTorch avec support CUDA : `pip install torch --index-url https://download.pytorch.org/whl/cu128`)*

---

## ðŸš€ StratÃ©gie d'EntraÃ®nement en Deux Temps

Pour rÃ©soudre la complexitÃ© combinatoire extrÃªme de Gaia Project sans rester bloquÃ© sur le plateau des ~75 VP (actions alÃ©atoires myopes), le systÃ¨me utilise une approche en deux phases :

```
[849 Parties BGS â‰¥ 150 VP]
          â”‚
          â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Phase 1 : Imitation SupervisÃ©e        â”‚  â”€â”€â–º Baseline solide (~140 - 160 VP)
â”‚  (Clonage de Comportement / 277k coups)â”‚      Backbone & TÃªtes initialisÃ©s
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Phase 2 : AlphaZero Fine-Tuning       â”‚  â”€â”€â–º VisÃ©e niveau Grand MaÃ®tre (220+ VP)
â”‚  â€¢ MCTS Gumbel Sequential Halving      â”‚      DÃ©passement du niveau humain
â”‚  â€¢ Ligue de Population & Matchmaking   â”‚      sans oubli catastrophique
â”‚  â€¢ Puzzles de crise RGSC (Manches 2-5) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Ã‰tape 1 : PrÃ©-entraÃ®nement SupervisÃ© (Imitation Expert)
EntraÃ®ne le rÃ©seau Ã  prÃ©dire les coups jouÃ©s par les meilleurs joueurs du monde sur 276 806 situations de jeu :

* **En ligne de commande :**
  ```bash
  python scraper/train_supervised.py --epochs 25 --batch-size 256 --lr 1e-3 --device cuda
  ```
* **Via le GUI :** SÃ©lectionnez le preset `ðŸš€ 1. Imitation Expert (>150 VP)` et cliquez sur **DÃ©marrer l'entraÃ®nement**.
* **RÃ©sultat :** En 20 Ã©poques (~15 min sur GPU), le modÃ¨le atteint ~35-40% de prÃ©cision Top-1 (face Ã  0.03% au hasard) et sauvegarde `checkpoints/gaia_supervised_pretrained.pt`.

### Ã‰tape 2 : AlphaZero MCTS Fine-Tuning
L'agent utilise les poids de l'imitation comme point de dÃ©part et s'affronte lui-mÃªme via recherche arborescente MCTS pour perfectionner ses plans Ã  long terme :

* **En ligne de commande :**
  ```bash
  python "training the model/alphazero_trainer.py"
  ```
* **Via le GUI :** SÃ©lectionnez le preset `ðŸ‘‘ 2. AlphaZero Fine-Tuning (220+ VP)` et cliquez sur **DÃ©marrer l'entraÃ®nement**. Le modÃ¨le charge automatiquement les poids de l'imitation et commence le self-play MCTS.

---

## ðŸ–¥ï¸ Interfaces & Outils

Le projet fournit trois interfaces complÃ©mentaires :

### 1. Studio d'EntraÃ®nement (`python gui.py`)
Interface graphique Tkinter complÃ¨te :
* **Tableau de bord :** Courbes d'apprentissage en direct (perte de politique, perte de valeur, scores rÃ©els vs prÃ©dits), vitesse en steps/s, indicateurs KPI.
* **ParamÃ©trage SOTA 2026 :** ContrÃ´le granulaire des 14 briques d'algorithmes (MCTS, Ligue, Puzzles RGSC, GNN, R-NaD, RND, etc.).
* **Auto-Tuning HyperparamÃ¨tres :** Module d'optimisation bayÃ©sienne TPE pour explorer les hyperparamÃ¨tres sans surapprentissage.
* **Ã‰valuation & Ligue :** Classement Elo en direct de toutes les versions historiques de l'IA.

### 2. Laboratoire Tactique & Sandbox (`python play.py`)
Interface PyQt interactive permettant de :
* Visualiser le plateau spatial hexagonal avec ses 200 cases et ses secteurs stellaires.
* Disposer des mines, stations, laboratoires et vaisseaux.
* Interroger le rÃ©seau de neurones en temps rÃ©el pour inspecter les probabilitÃ©s de chaque coup et l'espÃ©rance de score prÃ©dite.
* Jouer manuellement contre l'agent ou simuler des parties coup par coup.

### 3. Outils du Scraper (`scraper/`)
* `python scraper/scraper.py` : Scrape les parties terminÃ©es sur Boardgamers.space via pagination intelligente et filtres de score.
* `python scraper/stats.py` : Analyse statistique dÃ©taillÃ©e du dataset (rÃ©partition des 14 factions, classements Elo des joueurs, scores moyens).
* `python scraper/convert_to_dataset.py` : Compile les fichiers JSON bruts en tenseurs PyTorch optimisÃ©s.

---

## ðŸ§¬ Technologies Deep RL & SOTA 2026

| Brique Technologique | Module | Description & RÃ´le |
|---|---|---|
| **HexGNN Spatial** | [`models.py`](training the model/models.py) | RÃ©seau de neurones sur graphes dÃ©diÃ© au plateau hexagonal. Propage l'information topologique entre cases adjacentes (portÃ©e des vaisseaux, blocages, clusters de fÃ©dÃ©ration). |
| **SwiGLU ResNet** | [`models.py`](training the model/models.py) | Tronc rÃ©siduel profond Ã  4 couches [1024, 1024, 512, 256] avec portes d'activation SwiGLU, calibrÃ© pour exploiter le phÃ©nomÃ¨ne de *Deep Double Descent*. |
| **MCTS Gumbel (GAZ)** | [`mcts.py`](training the model/mcts.py) | Algorithme *Sequential Halving* 2026 : alloue un budget de simulations (16-32) ciblÃ© sur les 8 actions les plus prometteuses, avec guidage de score optimiste. |
| **Ligue de Population** | [`league.py`](training the model/league.py) | SystÃ¨me inspirÃ© d'AlphaStar : maintient un pool d'instantanÃ©s historiques, matchmaking Gaussien (zone proximale de dÃ©veloppement) et calcul Elo multi-joueurs Bradley-Terry. |
| **Puzzles RGSC** | [`buffer.py`](training the model/buffer.py) | *Regret-Guided Search Control* (Go-Exploit) : mÃ©morise les Ã©tats oÃ¹ l'IA a commis une faute en milieu de partie (Manches 2-5) et rÃ©injecte ces Ã©tats en puzzle tactique lors du self-play. |
| **R-NaD & RND** | [`trainer.py`](training the model/trainer.py), [`rnd.py`](training the model/rnd.py) | Dynamique de Nash rÃ©gularisÃ©e (anti-cyclage 4 joueurs) et distillation de curiositÃ© intrinsÃ¨que pour l'algorithme PPO alternatif. |

---

## ðŸ“‚ Structure du DÃ©pÃ´t

```text
â”œâ”€â”€ compile.bat              # Script de compilation Rust pour Windows
â”œâ”€â”€ compile.sh               # Script de compilation Rust pour Linux (Ubuntu, Pop!_OS)
â”œâ”€â”€ gui.py                   # Lanceur racine du Studio Graphique Tkinter
â”œâ”€â”€ play.py                  # Lanceur racine de la Station Tactique PyQt
â”œâ”€â”€ train.py                 # Lanceur racine d'entraÃ®nement CLI
â”œâ”€â”€ requirements.txt         # DÃ©pendances Python du projet
â”‚
â”œâ”€â”€ projet_gaiapi/           # ðŸ¦€ Moteur de rÃ¨gles natif en Rust
â”‚   â”œâ”€â”€ Cargo.toml           # Configuration de compilation Rust (cdylib C-ABI)
â”‚   â”œâ”€â”€ src/                 # Code source Rust (rÃ¨gles, plateau, actions, Lost Fleet)
â”‚   â””â”€â”€ tests/               # Tests unitaires du moteur Rust
â”‚
â”œâ”€â”€ scraper/                 # ðŸŒ Scraper et dataset d'experts BGS
â”‚   â”œâ”€â”€ scraper.py           # TÃ©lÃ©chargement automatique des parties
â”‚   â”œâ”€â”€ stats.py             # Statistiques et mÃ©tadonnÃ©es du dataset
â”‚   â”œâ”€â”€ convert_to_dataset.py# GÃ©nÃ©rateur de dataset PyTorch
â”‚   â”œâ”€â”€ train_supervised.py  # EntraÃ®neur de clonage de comportement (Phase 1)
â”‚   â””â”€â”€ data/                # Fichiers bruts JSON et dataset binaire .pt
â”‚
â”œâ”€â”€ training the model/      # ðŸ§  CÅ“ur Deep Reinforcement Learning
â”‚   â”œâ”€â”€ alphazero_trainer.py # Boucle d'entraÃ®nement AlphaZero MCTS
â”‚   â”œâ”€â”€ trainer.py           # Boucle d'entraÃ®nement PPO / Acteur-Critique
â”‚   â”œâ”€â”€ models.py            # Architectures HexGNN, SwiGLU, DualGaiaAgent
â”‚   â”œâ”€â”€ mcts.py              # Moteur MCTS multi-joueurs PUCT & Gumbel
â”‚   â”œâ”€â”€ league.py            # Gestionnaire de Ligue de Population & Elo
â”‚   â”œâ”€â”€ buffer.py            # Tampons de transition et puzzles de crise RGSC
â”‚   â”œâ”€â”€ config.py            # Presets et hyperparamÃ¨tres centralisÃ©s
â”‚   â”œâ”€â”€ gui.py               # ImplÃ©mentation complÃ¨te de l'interface Tkinter
â”‚   â””â”€â”€ test_pipeline.py     # Suite de tests d'intÃ©gration (23 tests)
â”‚
â””â”€â”€ gaia_gui_lab/            # ðŸŽ¨ Station tactique & Sandbox interactive (PyQt)
    â””â”€â”€ main.py              # Interface graphique avec plateau interactif
```

---

## ðŸ§ª Validation & Tests

La suite complÃ¨te de tests vÃ©rifie l'intÃ©gritÃ© de la chaÃ®ne de calcul, de la C-ABI Rust jusqu'aux calculs tensoriels PyTorch :

```bash
pytest "training the model/test_pipeline.py" -v
```

**23 tests unitaires et d'intÃ©gration validÃ©s (100% de succÃ¨s) :**
- Passes avant et arriÃ¨re des rÃ©seaux (`test_models_forward_and_backward`)
- Bridge C-ABI natif et masquage d'actions 3130D (`test_native_environment`, `test_flat_action_space_and_native_bridge`)
- Recherche MCTS Gumbel & PUCT (`test_mcts_engine`, `test_gumbel_alphazero`)
- SystÃ¨me de ligue, matchmaking Gaussien et Elo (`test_league_training`, `test_gaussian_elo_matchmaking`)
- Puzzles tactiques de crise RGSC (`test_rgsc_state_curriculum`)
- Ã‰quilibre de Nash multi-joueurs R-NaD (`test_rnad_equilibrium`)
- Encodeur spatial HexGNN (`test_hex_gnn_encoder`)
- Ã‰lÃ©ments du jeu et extension Lost Fleet (`test_game_elements_and_lost_fleet_rules`)

---

## ðŸ“œ Licence
Projet sous licence MIT â€” DÃ©veloppÃ© pour la recherche en Deep Reinforcement Learning multi-agents appliquÃ© aux jeux de plateau modernes.

