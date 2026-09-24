# Documentation Technique et Architecturale — gaiapi

`gaiapi` est un moteur de simulation complet, déterministe et ultra-haute performance du jeu de société **Gaia Project** et de son extension officielle **The Lost Fleet**, écrit en Rust. 

Ce moteur a été conçu dès l'origine pour servir de socle à des algorithmes d'apprentissage par renforcement (RL), de recherche arborescente (MCTS / AlphaZero / Stockfish) et de self-play nécessitant des dizaines de milliers de simulations par seconde.

---

## Sommaire

1. [Objectifs et Principes de Conception](#1-objectifs-et-principes-de-conception)
2. [Structure du Projet](#2-structure-du-projet)
3. [Modélisation du Plateau et Géométrie Spatiale](#3-modélisation-du-plateau-et-géométrie-spatiale)
4. [État des Joueurs et Économie](#4-état-des-joueurs-et-économie)
5. [Moteur d'Actions de Jeu (GameCommand)](#5-moteur-dactions-de-jeu-gamecommand)
6. [Extension The Lost Fleet](#6-extension-the-lost-fleet)
7. [Phase de Revenus et Système de Décompte](#7-phase-de-revenus-et-système-de-décompte)
8. [Énumération des Coups Légaux et Interface RL](#8-énumération-des-coups-légaux-et-interface-rl)
9. [Serveur API REST Local](#9-serveur-api-rest-local)
10. [Tests, Validation et Parité](#10-tests-validation-et-parité)

---

## 1. Objectifs et Principes de Conception

### A. Zéro-Allocation sur le Chemin Chaud
Dans les frameworks de RL (MCTS, AlphaZero), l'allocation mémoire sur le tas (`malloc` / `heap`) constitue le premier goulot d'étranglement de performance. Dans `gaiapi` :
- Tous les types d'état principaux (`HexCoord`, `Hex`, `PowerBowls`, `PlayerData`, `GameCommand`) implémentent le trait `Copy`.
- Le plateau `Map` stocke ses données dans des tableaux de taille fixe `[Hex; MAP_SIZE]` et `[HexCoord; MAP_SIZE]`, sans aucun vecteur alloué dynamiquement lors des transitions.
- L'évaluation des distances, la recherche de connectivité (BFS) pour les fédérations et les charges de puissance s'exécutent entièrement sur la pile (*stack*).

### B. Déterminisme et Reproductibilité
Toutes les générations procédurales (disposition des tuiles de secteur, attribution des tuiles de recherche, décomptes de manche) sont entièrement déterminées par un générateur pseudo-aléatoire à graine fixé (`seed: u64`), garantissant des épisodes 100% reproductibles.

### C. Parité Stricte avec le Moteur de Référence TypeScript
Les formules de coût, la roue des 7 couleurs de planètes, les capacités asymétriques des 14 factions de base et des 4 factions Lost Fleet, ainsi que les règles de tournoi allemandes ont été vérifiées et alignées sur le moteur officiel TypeScript (`gaia-project/engine`).

---

## 2. Structure du Projet

```text
projet_gaiapi/
├── Cargo.toml                # Configuration Cargo, dépendances (axum, serde, tokio)
├── README.md                 # Vue d'ensemble et guide de démarrage rapide
├── MIGRATION.md              # Matrice de parité TypeScript -> Rust (100% complétée)
├── docs/
│   └── ARCHITECTURE.md       # Ce document d'architecture et de référence
└── src/
    ├── lib.rs                # Environnement d'entraînement GaiaEnv, API RL & Axum
    ├── main.rs               # Point d'entrée exécutable (lancement du serveur HTTP)
    ├── board.rs              # Coordonnées cubiques axiales (q, r, s), rotations
    ├── sector.rs             # 10 tuiles de secteurs, configurations 2p et 3-4p
    ├── hex.rs                # Case de plateau Hex (bâtiments, planètes, vaisseaux)
    ├── map.rs                # Graphe spatial Map, BFS sur pile, adjacence, distances
    ├── player.rs             # PowerBowls (3 bols, Brainstone), PlayerData, ressources
    ├── rules.rs              # Factions, coûts, tables de conversion, tuiles tech/scoring
    ├── income.rs             # Phase de revenus (mines, labos, économie, science, boosters)
    └── actions.rs            # Moteur complet des 10 actions, legal_commands, scoring final
```

---

## 3. Modélisation du Plateau et Géométrie Spatiale

### A. Coordonnées Cubiques Axiales (`board.rs`)
Le plateau hexagonal utilise le système de coordonnées cubiques axiales standard :
$$\{ (q, r, s) \in \mathbb{Z}^3 \mid q + r + s = 0 \}$$
- **Distance de Manhattan hexagonale** :
  $$\text{dist}(A, B) = \frac{|q_A - q_B| + |r_A - r_B| + |s_A - s_B|}{2}$$
- **Rotations** : rotation horaire d'un pas par permutation cyclique :
  $$(q', r', s') = (-r, -s, -q)$$
  Support de la rotation autour de l'origine ou d'un centre arbitraire.

### B. Tuiles de Secteur (`sector.rs`)
- Chaque secteur contient 19 cases hexagonales numérotées selon un anneau externe A (12 cases), un anneau intermédiaire B (6 cases) et le centre C (1 case).
- 10 secteurs uniques avec faces recto/verso (`S1` à `S10`, variantes `5A/5B`, `6A/6B`, `7A/7B`).
- Agencements officiels :
  - **Petite configuration (2 joueurs)** : 7 secteurs avec centres prédéfinis `SMALL_CENTERS`.
  - **Grande configuration (3-4 joueurs)** : 10 secteurs avec centres prédéfinis `BIG_CENTERS`.
- Validation de conformité selon les règles de tournoi allemandes : deux planètes mères de même couleur ne peuvent jamais être adjacentes.

### C. Graphe Spatial (`map.rs` & `hex.rs`)
- Capacité fixe `MAP_SIZE = 200` couvrant jusqu'à 10 secteurs complets ($10 \times 19 = 190$ cases).
- Table de connectivité `adjacency: [[u8; 6]; MAP_SIZE]` calculée en $O(N)$ sur la pile.
- Recherche de plus court chemin (BFS) sur la pile pour calculer la distance entre une case et la structure la plus proche d'un joueur, en tenant compte des cubes Q.I.C. et des portées temporaires.

---

## 4. État des Joueurs et Économie

### A. Les 3 Bols de Puissance (`player.rs`)
Le système d'énergie (*Power*) obéit aux règles strictes du jeu physique :
- **Trois zones** : Zone 1 (vide/inactif), Zone 2 (chargé intermédiaire), Zone 3 (prêt à dépenser).
- **Cascade de charge** :
  1. Si la zone 1 contient des jetons, ils montent en zone 2.
  2. Une fois la zone 1 vide, les jetons de la zone 2 montent en zone 3.
  3. Tout excédent au-delà de la zone 3 est perdu.
- **Brûlage d'énergie (*Power Burning*)** : pour chaque jeton sacrifié (défaussé définitivement) depuis la zone 2, un autre jeton passe immédiatement de la zone 2 à la zone 3.
- **Brainstone des Taklons** :
  - Le Brainstone est stocké dans `PowerBowls::brainstone: Option<PowerArea>`.
  - Il charge **en priorité absolue** avant les jetons ordinaires.
  - Lorsqu'il est dépensé depuis la zone 3, il fournit **3 points de puissance** à lui seul et redescend en zone 1.

### B. Plafonds de Ressources
Les ressources du joueur sont plafonnées aux valeurs officielles du jeu :
- Minerai : $0 \le \text{ore} \le 15$
- Crédits : $0 \le \text{credits} \le 30$
- Connaissances : $0 \le \text{knowledge} \le 15$
- Q.I.C. : pas de plafond strict (représenté par `i16`).

### C. Recherche Technologique
6 domaines de recherche (`Terraforming`, `Navigation`, `Intelligence`, `GaiaProject`, `Economy`, `Science`) gradués de 0 à 5 :
- Coût de passage : 4 connaissances.
- Niveau 3 : déclenche immédiatement une charge de 3 énergies.
- Niveau 5 : accessible à un seul joueur par domaine, nécessite de retourner un jeton de fédération vert en gris, et octroie des récompenses majeures (ex: 9 PV, gain de Gaiaformer, etc.).
- Bal T'aks : restriction bloquant l'avancée en Navigation au-delà du niveau 1 tant que l'Institut Planétaire n'est pas érigé.

---

## 5. Moteur d'Actions de Jeu (`GameCommand`)

Toutes les actions de jeu sont regroupées dans l'énumération `GameCommand` (`src/actions.rs`) :

| Commande | Paramètres | Description & Règles Métier |
| :--- | :--- | :--- |
| `BuildMine` | `coord: HexCoord` | Construction de mine. Portée vérifiée avec cubes Q.I.C. Terraformation calculée selon la roue des couleurs (3, 2 ou 1 minerai par étape). Gratuit sur astéroïde (consomme 1 Gaiaformer). 3 étapes et +6 PV sur protoplanète. Mine supplémentaire Lantids supportée. |
| `StartGaiaProject` | `coord: HexCoord` | Lancement de projet Gaia sur planète Transdim. Déplacement de 6, 4 ou 3 énergies vers la zone Gaia selon le niveau de recherche. |
| `Upgrade` | `coord: HexCoord, to: Building` | Amélioration de structure : Mine $\rightarrow$ Station commerciale (3c, 2o si voisin $\le 2$ cases, sinon 6c, 2o), Station $\rightarrow$ IP ou Labo, Labo $\rightarrow$ Académie 1 ou 2. Arbre inversé des Bescods respecté. |
| `FormFederation` | `coords: &[HexCoord], token: FederationToken` | Fondation de fédération. Vérifie la valeur de puissance $\ge 7$ (ou 6 pour Xénos avec IP), place des satellites sur les cases vides, valide la connectivité BFS, défausse 1 jeton d'énergie par satellite, et attribue le jeton. |
| `AdvanceResearch` | `field: ResearchField` | Avancement sur l'une des 6 pistes pour 4 connaissances. Vérification d'exclusivité et coût de jeton vert pour le niveau 5. |
| `BoardAction` | `action: BoardAction, ...` | 7 actions de puissance (`Power1`..`Power7`) et 3 actions Q.I.C. (`Qic1`..`Qic3`). Verrouillage strict par manche. |
| `SpecialAction` | `action: SpecialAction, ...` | Capacités uniques de faction : échange d'IP Ambas, rétrogradation Labo Firaks avec tech gratuite, avancée de la piste la plus basse Bescods, station spatiale Ivits, action 1x/manche Tech9. |
| `Pass` | `new_booster: Option<u8>` | Fin du tour de manche. Décompte immédiat des PV du booster actif et choix du booster pour la manche suivante. L'ordre de passage détermine l'ordre du premier joueur au tour suivant. |
| `ExploreSpaceship` | `ship: Spaceship, coord: HexCoord` | Exploration d'un vaisseau The Lost Fleet (voir section dédiée). |
| `SpaceshipBoardAction`| `ship: Spaceship, action_type: SpaceshipActionType, ...` | Action de plateau exclusive sur un vaisseau exploré (voir section dédiée). |

---

## 6. Extension The Lost Fleet

L'extension *The Lost Fleet* apporte des vaisseaux spatiaux abandonnés, de nouveaux types de planètes et de nouvelles dynamiques :

### A. Vaisseaux Spatiaux Disponibles
- **Twilight** : Toujours en jeu (2 à 4 joueurs).
- **Rebellion** : En jeu à 3 et 4 joueurs uniquement (exclu à 2 joueurs selon la règle §C2/§H3).
- **TFMars** : Toujours en jeu.
- **Eclipse** : Toujours en jeu.

### B. Exploration des Vaisseaux (`ExploreSpaceship`)
- **Limite de navettes** : 2 navettes maximum par joueur à 2 joueurs, 3 navettes à 3-4 joueurs.
- **Coût de déploiement** : 5 PV pour toutes les factions standards, **7 PV pour les Bal T'aks** (§D2/§D5).
- **Piste de charge partagée** : `[0, 2, 2, 3]` power tokens selon la place d'arrivée de la navette sur les 4 emplacements du vaisseau.
- Chaque joueur ne peut explorer un vaisseau donné qu'une seule fois.

### C. Actions de Plateau des Vaisseaux (`SpaceshipBoardAction`)
Chaque vaisseau offre des actions exclusives utilisables **une seule fois par manche pour l'ensemble des joueurs** :

1. **Twilight** :
   - *Q.I.C.* (3q) : Re-score immédiat d'un jeton de fédération déjà possédé.
   - *Power* (3pw, 2o) : Amélioration directe d'une station commerciale en laboratoire de recherche.
   - *Knowledge* (1k) : +3 de portée temporaire pour le tour en cours.
2. **Rebellion** (3-4 joueurs) :
   - *Q.I.C.* (3q) : Acquisition immédiate d'une tuile technologique.
   - *Power* (3pw, 1o) : Amélioration d'une mine en station commerciale sans contrainte d'adjacence adverse.
   - *Knowledge* (2k) : Gain direct de 2 crédits et 1 Q.I.C.
3. **TFMars** :
   - *Q.I.C.* (2q) : Gain de 2 PV + 1 PV par tuile technologique possédée.
   - *Power* (2pw) : Projet Gaia instantané transformant une planète Transdim à portée en planète Gaia.
   - *Credit* (3c + 2o) : 1 étape de terraformation offerte et construction d'une mine.
4. **Eclipse** :
   - *Q.I.C.* (2q) : Gain de 2 PV + 1 PV par type de planète distinct colonisé.
   - *Power* (3pw, 2k) : Progression gratuite d'un niveau sur n'importe quel domaine de recherche.
   - *Credit* (6c) : Construction d'une mine sur un astéroïde à portée sans coût en minerai ni consommation de Gaiaformer.

---

## 7. Phase de Revenus et Système de Décompte

### A. Phase de Revenus (`income.rs`)
La fonction `compute_player_income` cumule en une seule passe :
- Revenus de structures (mines, stations, laboratoires, académies).
- Revenus spécifiques des Instituts Planétaires de faction (ex: Terrans 4pw, Ivits 4c, Geodens 1o, etc.).
- Revenus de pistes Économie (crédits, minerai, power) et Science (connaissances).
- Revenus des boosters de manche et des tuiles technologiques.
- Résolution optimisée de la charge de pouvoir à deux passes (priorité Brainstone et cascade 1 $\rightarrow$ 2 $\rightarrow$ 3).

### B. Décompte de Manche (`RoundScoringTile`)
10 tuiles de base (mines sur Gaia, améliorations en station, pose d'IP, fondation de fédération, étapes de terraformation, etc.) + 3 tuiles Lost Fleet.

### C. Décompte Final (`FinalTile`)
Calculé à la fin de la 6e manche par `compute_final_scores` :
1. **Tuiles d'objectifs finaux** : 18 PV pour le 1er, 12 PV pour le 2e, 6 PV pour le 3e, 0 PV pour le 4e (avec partage équitable et arrondi inférieur en cas d'égalité).
2. **Recherche de pointe** : 4 PV par niveau pour tout domaine au niveau $\ge 3$ ($\text{PV} = (\text{niveau} - 2) \times 4$).
3. **Conversion de fin de partie** : 1 PV pour chaque lot de 3 ressources restantes (crédits + minerai + connaissances).

---

## 8. Énumération des Coups Légaux et Interface RL

### A. Algorithme `legal_commands`
Pour alimenter les réseaux de neurones de politique (Policy Networks) et les algorithmes MCTS :
```rust
pub fn legal_commands(
    player_seat: usize,
    player: &PlayerData,
    map: &Map,
    claimed_l5: &[Option<u8>; 6],
    claimed_board_actions: &[Option<u8>; 10],
    claimed_spaceship_actions: &[[bool; 4]; 4],
    available_boosters: &[u8],
    is_final_round: bool,
) -> Vec<GameCommand>
```
Cette fonction explore l'ensemble complet des coups admissibles selon l'état actuel :
- Vérification des coûts et de l'inventaire des bâtiments disponibles.
- Évaluation de la portée et des cubes Q.I.C. nécessaires.
- Filtrage des actions de plateau et d'exploration déjà revendiquées.
- Filtrage des passes et sélection de boosters libres.

### B. Trait Gym-like `GaiaEnv` (`src/lib.rs`)
- `env.reset() -> Observation` : Réinitialise la partie avec la graine courante, instancie le plateau et attribue les factions.
- `env.legal_commands(player) -> Vec<GameCommand>` : Renvoie la liste des coups légaux pour le joueur actif.
- `env.execute_command(player, cmd) -> Result<(), ActionError>` : Exécute une commande complète de jeu de plateau.
- `env.step(action) -> Result<StepResult, EnvError>` : Avance l'environnement via l'espace discret d'actions RL.
- `env.observe() -> Observation` : Renvoie le vecteur d'observation normalisé et le masque booléen d'actions légales.

---

## 9. Serveur API REST Local

Un serveur HTTP local asynchrone bâti sur **Axum** et **Tokio** est inclus (`src/main.rs`) pour permettre le pilotage de l'environnement depuis Python ou tout autre langage sans binding natif obligatoire :

```bash
cargo run --release
# Écoute sur http://127.0.0.1:3000
```

### Routes Principales

| Méthode | Route | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/environments` | Créer une instance d'environnement (`{"config": {"players": 2, "seed": 42}}`). |
| `GET` | `/v1/environments/{id}/observation` | Lire le vecteur d'observation et le masque d'action actuel. |
| `POST` | `/v1/environments/{id}/reset` | Réinitialiser l'environnement de manière déterministe. |
| `POST` | `/v1/environments/{id}/step` | Appliquer une action discrète (`{"action": "build_mine"}`). |
| `GET` | `/v1/rules/factions/{faction}/free-actions` | Obtenir le vocabulaire d'actions gratuites de la faction. |
| `GET` | `/health` | Vérifier l'état de santé du serveur (`{"status": "ok"}`). |

---

## 10. Tests, Validation et Parité

### A. Exécution de la Suite de Tests
```bash
cargo test --release
```
**Résultat actuel** : **50 tests unitaires et d'intégration réussis en 0.01 seconde**.

### B. Contrôle Qualité Clippy
```bash
cargo clippy --all-targets -- -D warnings
```
**Résultat actuel** : **0 avertissement, 0 erreur**.

### C. Génération de la Documentation Rustdoc
```bash
cargo doc --no-deps --open
```
Génère la documentation HTML native des modules et fonctions de la bibliothèque dans `target/doc/gaiapi/index.html`.
