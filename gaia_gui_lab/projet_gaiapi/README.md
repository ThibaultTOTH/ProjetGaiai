# gaiapi

[![Rust](https://img.shields.io/badge/rust-1.80%2B-blue.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-50%2F50%20passing-brightgreen.svg)]()
[![Clippy](https://img.shields.io/badge/clippy-zero%20warnings-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()

**gaiapi** est un moteur de simulation complet, déterministe et ultra-haute performance du jeu de société **Gaia Project** et de son extension officielle **The Lost Fleet**, implémenté en Rust.

Conçu spécialement pour l'apprentissage par renforcement (RL), la recherche arborescente (MCTS / AlphaZero) et la construction d'un « Stockfish » pour Gaia Project, le moteur garantit **zéro allocation dynamique sur le tas** sur le chemin critique de simulation.

---

## 🚀 Points Clés

- **100% Fidélité aux Règles** : Moteur complet du jeu de base et de l'extension *The Lost Fleet*, conforme aux règles officielles et aligné sur les oracles de test du moteur de référence TypeScript (`gaia-project/engine`).
- **Zéro Allocation sur le Chemin Chaud** : Tous les états (`PlayerData`, `Hex`, `Map`, `PowerBowls`, `GameCommand`) sont `Copy` et stockés sur la pile (*stack*) avec des tableaux à taille fixe. (Note : la méthode `observe_into()` a été ajoutée pour permettre la réutilisation de buffers côté FFI).
- **Énumération Exhaustive des Coups Légaux (`legal_commands`)** : Permet le masquage strict des actions de politique et l'exploration MCTS à plusieurs dizaines de milliers de nœuds par seconde.
- **Extension The Lost Fleet Complète** :
  - Vaisseaux spatiaux (`Twilight`, `Rebellion`, `TFMars`, `Eclipse`) avec pistes de charge `[0, 2, 2, 3]`.
  - Actions de plateau exclusives verrouillées par manche.
  - Planètes Astéroïdes (mines gratuites via Gaiaformers) et Protoplanètes (3 étapes de terraformation, +6 PV).
- **Double Interface d'Utilisation** :
  - **Bibliothèque Rust native** pour un débit maximal en self-play / MCTS.
  - **Serveur HTTP REST asynchrone (Axum)** pour pilotage facile depuis Python, PyTorch ou Ray RLlib.

---

## 📦 Démarrage Rapide

### Utilisation en Bibliothèque Rust

```rust
use gaiapi::{GaiaEnv, GameConfig};
use gaiapi::actions::GameCommand;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise une partie déterministe à 2 joueurs
    let mut env = GaiaEnv::new(GameConfig {
        players: 2,
        seed: 42,
        ..Default::default()
    })?;

    // Récupère la liste exhaustive des commandes légales pour le joueur actif
    let current_seat = env.current_player;
    let legal_cmds = env.legal_commands(current_seat);
    println!("Nombre d'actions légales disponibles : {}", legal_cmds.len());

    // Exécute une commande de jeu complète (ex: passer avec un nouveau booster)
    if let Some(&cmd) = legal_cmds.first() {
        env.execute_command(current_seat, cmd)?;
    }

    Ok(())
}
```

### Lancement du Serveur HTTP / REST Local

```bash
cargo run --release
# Le serveur écoute sur http://127.0.0.1:3000
```

Exemple d'interaction via cURL / JSON :

```bash
# Vérifier la santé du serveur
curl http://127.0.0.1:3000/health

# Créer un environnement
curl -X POST http://127.0.0.1:3000/v1/environments \
     -H "Content-Type: application/json" \
     -d '{"config": {"players": 2, "seed": 42}}'

# Lire l'observation et le masque d'action
curl http://127.0.0.1:3000/v1/environments/1/observation

# Effectuer une action discrète
curl -X POST http://127.0.0.1:3000/v1/environments/1/step \
     -H "Content-Type: application/json" \
     -d '{"action": "build_mine"}'
```

---

## 🏛️ Architecture et Composants

Le moteur est découpé en modules spécialisés sans dépendances circulaires :

| Module | Rôle |
| :--- | :--- |
| [`src/board.rs`](docs/ARCHITECTURE.md#3-modélisation-du-plateau-et-géométrie-spatiale) | Coordonnées axiales cubiques $(q, r, s)$, distances de Manhattan et rotations. |
| [`src/sector.rs`](docs/ARCHITECTURE.md#3-modélisation-du-plateau-et-géométrie-spatiale) | 10 tuiles de secteurs officielles, agencements 2p (`SMALL_CENTERS`) et 3-4p (`BIG_CENTERS`). |
| [`src/hex.rs`](docs/ARCHITECTURE.md#3-modélisation-du-plateau-et-géométrie-spatiale) | Modèle compact `Hex` (`Copy`), gestion des planètes, bâtiments, satellites et vaisseaux. |
| [`src/map.rs`](docs/ARCHITECTURE.md#3-modélisation-du-plateau-et-géométrie-spatiale) | Graphe spatial de 200 cases, adjacence en $O(N)$, BFS sur la pile pour portée et Q.I.C. |
| [`src/player.rs`](docs/ARCHITECTURE.md#4-état-des-joueurs-et-économie) | Gestion des 3 bols de puissance, Brainstone des Taklons, ressources plafonnées et recherche. |
| [`src/rules.rs`](docs/ARCHITECTURE.md#4-état-des-joueurs-et-économie) | Factions, tables de conversion, tuiles technologiques standard et avancées, scoring. |
| [`src/income.rs`](docs/ARCHITECTURE.md#7-phase-de-revenus-et-système-de-décompte) | Calcul exhaustif des revenus (structures, IP, économie, science, boosters) et charge d'énergie. |
| [`src/actions.rs`](docs/ARCHITECTURE.md#5-moteur-dactions-de-jeu-gamecommand) | Les 10 actions de jeu, exploration de vaisseaux, `legal_commands` et décompte final. |
| [`src/lib.rs`](docs/ARCHITECTURE.md#8-énumération-des-coups-légaux-et-interface-rl) | Environnement d'entraînement `GaiaEnv`, cycle des manches, ordre du tour et routes API. |

Pour une description technique détaillée de chaque structure de données et règle métier, consultez le document complet : **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## 🕹️ Les 10 Commandes de Jeu Supportées

1. **`BuildMine`** : Construction d'une mine sur planète mère, autres couleurs (avec terraformation), astéroïde (gratuit avec Gaiaformer) ou protoplanète (+6 PV).
2. **`StartGaiaProject`** : Lancement d'un projet Gaia sur planète Transdim avec déplacement d'énergie vers la zone Gaia.
3. **`Upgrade`** : Amélioration de structure (Mine $\rightarrow$ Station, Station $\rightarrow$ Labo ou IP, Labo $\rightarrow$ Académies) avec escompte de voisinage adverse.
4. **`FormFederation`** : Fondation d'une fédération (valeur $\ge 7$), placement de satellites, validation de connexité BFS et attribution du jeton.
5. **`AdvanceResearch`** : Progression sur l'une des 6 pistes pour 4 connaissances (charge de 3 énergies au niveau 3, exclusivité et jeton vert au niveau 5).
6. **`BoardAction`** : Les 7 actions de puissance (`Power1`..`Power7`) et 3 actions Q.I.C. (`Qic1`..`Qic3`) exclusives par manche.
7. **`SpecialAction`** : Capacités uniques de faction (Ambas, Firaks, Bescods, Ivits, Tech 9, etc.).
8. **`Pass`** : Fin de tour de manche, décompte du booster et sélection du prochain booster.
9. **`ExploreSpaceship`** : Exploration d'un vaisseau *The Lost Fleet* avec attribution de navette, coût en PV et charge sur la piste `[0, 2, 2, 3]`.
10. **`SpaceshipBoardAction`** : 12 actions de vaisseaux exclusives verrouillées par manche réparties sur Twilight, Rebellion, TFMars et Eclipse.

---

## 🧪 Tests et Qualité

La bibliothèque est accompagnée d'une suite de 50 tests unitaires et de parité vérifiant l'ensemble des règles du jeu et des oracles de test TypeScript :

```bash
# Exécuter les tests en mode release (environ 0.01 seconde)
cargo test --release

# Linter strict sans aucun warning toléré
cargo clippy --all-targets -- -D warnings

# Générer la documentation HTML native
cargo doc --no-deps --open
```

---

## 📖 Documentation Complémentaire

- **[Architecture & Référence Développeur](docs/ARCHITECTURE.md)** : Conception détaillée, règles de simulation, mémoire et RL.
- **[Matrice de Migration TypeScript $\rightarrow$ Rust](MIGRATION.md)** : Correspondance exacte entre le code TypeScript d'origine et le moteur Rust.
