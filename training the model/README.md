# 🌌 Gaia Project — Deep Reinforcement Learning Studio

Studio d'entraînement et d'optimisation d'agents de Deep Reinforcement Learning pour **Gaia Project**, conçu pour tirer parti d'un GPU **NVIDIA RTX 5070 (12 Go, Blackwell sm_120)** avec repli automatique sur CPU (*fallback mode*). Requiert **PyTorch ≥ 2.6** avec **CUDA ≥ 12.8**.

---

## 🧠 Architecture des Modèles de Deep Learning

Le système entraîne **deux réseaux de neurones distincts** :

1. **Modèle 1 : Prédicteur de Scores (`ScorePredictorNet`)**
   - **Rôle** : Réseau de Valeur (Value Network). Prédit les points de victoire finaux (VP) attendus à partir d'un état de jeu.
   - **Architecture** : Blocs résiduels profonds avec `LayerNorm`, activations `GELU` et régression scalaire.
   - **Objectif** : Minimiser la Huber Loss / Smooth L1 Loss par rapport aux rendements Monte-Carlo et avantages GAE.

2. **Modèle 2 : Optimiseur de Coups (`ActionOptimizerNet`)**
   - **Rôle** : Réseau de Politique (Policy Network / Actor). Choisit et pondère les actions optimales à chaque tour.
   - **Architecture** : Blocs résiduels avec couche de **masquage strict des actions illégales** (`masked_fill(~action_mask, -1e9)`).
   - **Objectif** : Maximisation de la politique par PPO (Proximal Policy Optimization) avec clipping $\epsilon$ et bonus d'entropie.

---

## ⚡ Optimisations NVIDIA RTX 5070 (12 Go) & Fallback CPU

Le code intègre les optimisations modernes pour cartes NVIDIA (architecture Blackwell / Ada) :
- **Détection automatique** : utilise `cuda:0` dès qu'un GPU NVIDIA est présent, ou bascule de façon transparente sur `cpu` sur les PC de développement sans carte graphique dédiée.
- **Précision Mixte Automatique (AMP FP16 / BF16)** : accélère les calculs matriciels sur les Tensor Cores et réduit l'empreinte mémoire.
- **TF32 (TensorFloat-32)** : activé par défaut (`torch.backends.cuda.matmul.allow_tf32 = True`).
- **Gestion VRAM** : configuration par défaut adaptée aux 12 Go de VRAM (batch size 64 à 512, rollout de 512 à 2048 transitions).

---

## 📦 Installation des dépendances

Dans un environnement Python 3.10+ :

```bash
# Installation de PyTorch avec support CUDA (pour la machine équipée de la RTX 5070) :
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Installation des dépendances du projet :
pip install -r requirements.txt
```

---

## 🖥️ Utilisation

### 1. Interface Graphique Tkinter (Par défaut)

Pour lancer le studio graphique complet :

```bash
python main.py
```

L'interface propose 4 onglets :
- **📊 Tableau de bord & Courbes** : démarrage / pause / arrêt de l'apprentissage en arrière-plan, cartes KPI en direct (vitesse en steps/s, pertes PPO et Value, score prédit vs réel) et courbes Matplotlib réactives.
- **⚙️ Paramétrage & Réseaux** : réglage fin des taux d'apprentissage (LR Policy et LR Score séparés), taille de batch, gamma ($\gamma$), lambda GAE ($\lambda$), clipping PPO ($\epsilon$), coefficient d'entropie et architecture des réseaux.
- **🔍 Auto-Tuning Hyperparamètres** : moteur de recherche aléatoire (Random Search) et grille pour tester plusieurs configurations lors de sprints courts, avec tableau comparatif des victoires et scores objectifs.
- **🎮 Simulateur de Matchs** : permet de tester le modèle en direct contre un adversaire légal aléatoire, avec affichage tour par tour du coup choisi, des probabilités d'action et du score final prédit.

---

### 2. Mode Ligne de Commande (Headless CLI)

Pour lancer un entraînement automatisé sans interface graphique (idéal sur serveur ou scripts batch) :

```bash
python main.py --cli --epochs 200 --device cuda:0
```

---

### 3. Génération du Rapport PDF Stratégique (Théorie des Jeux)

Pour extraire la Tier List des factions, les prix ombre des ressources et la matrice de matchups sous forme de livret PDF de 4 pages à partager :

```bash
python main.py --report "Mon_Rapport_Gaia.pdf"
```

Le document PDF généré contient :
1. **Tier List & Enchères Équitables (Fair Bids)** (Classement Elo Bradley-Terry, points de compensation).
2. **Matrice de Matchups & Équilibre de Nash** (Heatmap couleur des confrontations directes).
3. **Prix Ombre des Ressources (Shadow Prices)** (Évolution de la valeur marginale en VP de chaque ressource de la manche 1 à 6).
4. **Guide des Ouvertures (Round 1 Meta)** (Meilleurs coups initiaux recommandés par faction et règles clés).

---

## 🧪 Vérification et Tests du Pipeline

Un script de test unitaire et de validation est fourni pour vérifier l'ensemble des modules (réseaux, masquage, calcul des gradients, buffer GAE, environnement simulé et pas d'entraînement) :

```bash
python test_pipeline.py
```
