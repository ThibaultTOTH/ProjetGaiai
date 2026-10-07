# Migration TypeScript → Rust

Ce document récapitule la correspondance entre le moteur de référence TypeScript (`gaia-project/engine`) et le moteur haute performance Rust (`gaiapi`).

## Matrice de Parité des Modules

| Moteur TypeScript (`engine/src/`) | Moteur Rust (`gaiapi/src/`) | Statut | Détails d'Implémentation |
| :--- | :--- | :---: | :--- |
| `enums.ts`, `actions.ts` | `src/lib.rs`, `src/rules.rs` | **100% Porté** | Types, 14 factions de base + 4 Lost Fleet, structures, limites et conversions. |
| `reward.ts`, `cost.ts` | `src/rules.rs` | **100% Porté** | Formules de terraformation, escomptes, coûts Q.I.C., tables de conversion d'énergie. |
| `algorithms/grid-topology.ts` | `src/board.rs` | **100% Porté** | Coordonnées axiales cubiques $(q, r, s)$, distances de Manhattan, rotations horaires. |
| `gaia-hex.ts` | `src/hex.rs` | **100% Porté** | Structure compacte `Hex` (`Copy`), gestion planètes, bâtiments, mine Lantids, vaisseaux. |
| `player-data.ts` (power & stats) | `src/player.rs` | **100% Porté** | 3 bols de puissance, cascade $1 \rightarrow 2 \rightarrow 3$, sacrifice/burn, Brainstone Taklons, plafonds. |
| `income.ts` | `src/income.rs` | **100% Porté** | Revenus complets (mines, stations, labos, académies, IP de faction, économie, science, boosters, tuiles tech). |
| `map.ts` (graphe & distance) | `src/map.rs` | **100% Porté** | Plateau compact `MAP_SIZE = 200`, BFS sur pile sans allocation, calcul de portée avec Q.I.C. |
| `map.ts`, `sector.ts` (setup) | `src/sector.rs`, `src/map.rs` | **100% Porté** | 10 secteurs officiels, 19 décalages, rotations, layouts 2p et 3-4p, validation règles allemandes. |
| `move/build.ts` | `src/actions.rs` | **100% Porté** | Action 1 : Construction de mine sur planètes standards, astéroïdes (Lost Fleet), protoplanètes (+6 PV). |
| `move/gaia-project.ts` | `src/actions.rs` | **100% Porté** | Action 2 : Lancement projet Gaia, déplacement d'énergie vers zone Gaia, résolution de phase Gaia. |
| `move/upgrade.ts` | `src/actions.rs` | **100% Porté** | Action 3 : Amélioration de structures avec détection de voisinage adverse ($\le 2$ cases) et arbre inversé Bescods. |
| `move/federation.ts` | `src/actions.rs` | **100% Porté** | Action 4 : Fondation de fédération, placement de satellites, validation de connexité BFS et jetons verts/gris. |
| `move/research.ts` | `src/actions.rs` | **100% Porté** | Action 5 : Progression de recherche (0 à 5), charge niveau 3, exclusivité et jeton vert niveau 5. |
| `move/board-actions.ts` | `src/actions.rs` | **100% Porté** | Action 6 : 7 actions d'énergie et 3 actions Q.I.C. avec verrouillage strict par manche. |
| `move/special.ts` | `src/actions.rs` | **100% Porté** | Action 7 : Capacités uniques de faction (Ambas, Firaks, Bescods, Ivits, Tech 9, etc.). |
| `move/pass.ts` | `src/actions.rs` | **100% Porté** | Action 8 : Décompte des PV de booster, sélection du nouveau booster, détermination du premier joueur. |
| `move/exploration.ts`, `spaceships.ts` | `src/actions.rs` | **100% Porté** | Action 9 : Exploration de vaisseaux The Lost Fleet, coût 5/7 PV, piste de charge partagée `[0, 2, 2, 3]`. |
| `move/spaceship-actions.ts` | `src/actions.rs` | **100% Porté** | Action 10 : 12 actions exclusives réparties sur Twilight, Rebellion, TFMars et Eclipse avec verrouillage par manche. |
| `available/*` | `src/actions.rs` | **100% Porté** | Énumération exhaustive des coups légaux `legal_commands` pour le RL et MCTS. |
| `events.ts`, `tiles/*` | `src/rules.rs`, `src/actions.rs` | **100% Porté** | 9 tuiles tech, 15 tuiles avancées, décomptes de manche (base + Lost Fleet) et scoring final 18/12/6/0. |
| Contrat IA/RL & API | `src/lib.rs`, `src/main.rs` | **100% Porté** | `GaiaEnv`, observations vectorielles, masque booléen, serveur REST local Axum. |

---

## Bilan de Conformité

- **Couverture de Règles** : 100% du jeu de base et de l'extension *The Lost Fleet*.
- **Tests de Parité** : 50 tests unitaires et d'intégration validés (`cargo test --release`).
- **Qualité de Code** : Zéro warning avec `cargo clippy --all-targets -- -D warnings`.
- **Performance Mémoire** : Zéro allocation dynamique sur le tas pendant l'exécution des commandes.
