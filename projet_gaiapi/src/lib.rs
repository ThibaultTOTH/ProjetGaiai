//! A deterministic, turn-based reinforcement-learning API inspired by Gaia Project.
//!
//! `GaiaEnv` deliberately exposes the familiar RL lifecycle: [`GaiaEnv::reset`],
//! [`GaiaEnv::observe`], [`GaiaEnv::legal_actions`], and [`GaiaEnv::step`]. The
//! rules implemented here are a compact, fully deterministic training kernel.

pub mod map_serialize;
use std::fmt;

use serde::{Deserialize, Serialize};

pub mod actions;
pub mod action_space;
pub mod board;
pub mod ffi;
pub mod hex;
pub mod income;
pub mod map;
pub mod player;
pub mod rules;
pub mod sector;

pub use actions::{ActionError, GameCommand};
pub use hex::Hex;
pub use map::Map;
pub use player::{PlayerData, PowerBowls};
pub use rules::{
    AdvTechTile, BoardAction, FederationToken, FinalTile, ScoringTile, SpecialAction, TechTile,
};
pub use sector::SectorId;

/// Maximum number of seats supported by the environment.
pub const MAX_PLAYERS: usize = 4;
/// Number of discrete actions in the stable action space.
pub const ACTION_SPACE: usize = crate::action_space::FLAT_ACTION_SPACE;
/// Total exhaustive observation vector dimension (Global 88 + 4 Players 388 + Map 2000).
pub const OBS_SPACE: usize = 2476;

/// Factions currently represented by the training kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum Faction {
    Terrans,
    Lantids,
    HadschHallas,
    Ivits,
    Geodens,
    BalTaks,
    Xenos,
    Gleens,
    Taklons,
    Ambas,
    Firaks,
    Bescods,
    Nevlas,
    Itars,
    Tinkeroids,
    Darkanians,
    Moweyds,
    SpaceGiants,
}

impl Faction {
    /// Factions available without the Lost Fleet expansion.
    pub const BASE: [Self; 14] = [
        Self::Terrans,
        Self::Lantids,
        Self::Xenos,
        Self::HadschHallas,
        Self::Ivits,
        Self::Geodens,
        Self::BalTaks,
        Self::Gleens,
        Self::Taklons,
        Self::Ambas,
        Self::Firaks,
        Self::Bescods,
        Self::Nevlas,
        Self::Itars,
    ];
    /// Factions introduced by Lost Fleet.
    pub const LOST_FLEET: [Self; 4] = [
        Self::Tinkeroids,
        Self::Darkanians,
        Self::Moweyds,
        Self::SpaceGiants,
    ];
    pub const ALL: [Self; 18] = [
        Self::Terrans,
        Self::Lantids,
        Self::HadschHallas,
        Self::Ivits,
        Self::Geodens,
        Self::BalTaks,
        Self::Xenos,
        Self::Gleens,
        Self::Taklons,
        Self::Ambas,
        Self::Firaks,
        Self::Bescods,
        Self::Nevlas,
        Self::Itars,
        Self::Tinkeroids,
        Self::Darkanians,
        Self::Moweyds,
        Self::SpaceGiants,
    ];
}

/// Resources exposed in observations, in this exact order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    Credits = 0,
    Ore = 1,
    Knowledge = 2,
    Qic = 3,
    Power = 4,
}

/// A compact resource wallet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resources {
    values: [i16; 5],
}

impl Resources {
    pub const fn new(credits: i16, ore: i16, knowledge: i16, qic: i16, power: i16) -> Self {
        Self {
            values: [credits, ore, knowledge, qic, power],
        }
    }
    pub const fn get(self, resource: Resource) -> i16 {
        self.values[resource as usize]
    }
    pub fn add(&mut self, resource: Resource, amount: i16) {
        self.values[resource as usize] += amount;
    }
    pub fn can_afford(self, cost: Self) -> bool {
        self.values
            .iter()
            .zip(cost.values)
            .all(|(have, need)| *have >= need)
    }
    pub fn pay(&mut self, cost: Self) {
        for (have, need) in self.values.iter_mut().zip(cost.values) {
            *have -= need;
        }
    }
    pub const fn as_array(self) -> [i16; 5] {
        self.values
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameConfig {
    pub players: usize,
    pub max_rounds: u8,
    pub seed: u64,
    pub factions: Option<Vec<Faction>>,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            players: 2,
            max_rounds: 6,
            seed: 0,
            factions: None,
        }
    }
}

/// Information made available to an agent.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Observation {
    pub current_player: usize,
    pub round: u8,
    pub values: Vec<f32>,
    pub action_mask: Vec<bool>,
    pub free_actions: Vec<crate::rules::FreeAction>,
}

/// Result of a state transition. Reward is per seat, enabling self-play.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepResult {
    pub observation: Observation,
    pub rewards: Vec<f32>,
    pub terminated: bool,
    pub truncated: bool,
    pub round: u8,
    pub current_player: usize,
}

/// Validation failures are returned instead of panicking so a trainer can recover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    InvalidPlayerCount(usize),
    InvalidFactionCount { expected: usize, actual: usize },
    EpisodeFinished,
    IllegalAction { player: usize, action: usize },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlayerCount(value) => write!(
                f,
                "players must be between 2 and {MAX_PLAYERS}, got {value}"
            ),
            Self::InvalidFactionCount { expected, actual } => {
                write!(f, "expected {expected} factions, got {actual}")
            }
            Self::EpisodeFinished => f.write_str("episode is finished; call reset before stepping"),
            Self::IllegalAction { player, action } => {
                write!(f, "action {action:?} is illegal for player {player}")
            }
        }
    }
}
impl std::error::Error for EnvError {}

/// Deterministic environment suitable for synchronous single-process self-play.
#[derive(Debug, Clone)]
pub struct GaiaEnv {
    pub config: GameConfig,
    pub map: Map,
    pub players: Vec<PlayerData>,
    pub research_level_5_claimed: [Option<u8>; 6],
    pub claimed_board_actions: [Option<u8>; 10],
    pub claimed_adv_techs: [bool; 15],
    pub claimed_spaceship_actions: [[bool; 4]; 4],
    pub claimed_artefacts: [bool; 13],
    pub round_scoring_tiles: [ScoringTile; 6],
    pub final_scoring_tiles: [FinalTile; 2],
    pub pass_order: Vec<u8>,
    pub current_player: usize,
    pub round: u8,
    pub terminated: bool,
    pub rng_state: u64,
}

impl GaiaEnv {
    pub fn new(config: GameConfig) -> Result<Self, EnvError> {
        if !(2..=MAX_PLAYERS).contains(&config.players) {
            return Err(EnvError::InvalidPlayerCount(config.players));
        }
        if let Some(factions) = &config.factions
            && factions.len() != config.players
        {
            return Err(EnvError::InvalidFactionCount {
                expected: config.players,
                actual: factions.len(),
            });
        }
        let mut environment = Self {
            rng_state: config.seed,
            config,
            map: Map::empty(),
            players: Vec::new(),
            research_level_5_claimed: [None; 6],
            claimed_board_actions: [None; 10],
            claimed_adv_techs: [false; 15],
            claimed_spaceship_actions: [[false; 4]; 4],
            claimed_artefacts: [false; 13],
            round_scoring_tiles: [
                ScoringTile::Score3, // Round 1: mine 2vp
                ScoringTile::Score8, // Round 2: ts 3vp
                ScoringTile::Score5, // Round 3: ts 4vp
                ScoringTile::Score1, // Round 4: step 2vp
                ScoringTile::Score2, // Round 5: a 2vp
                ScoringTile::Score4, // Round 6: fed 5vp
            ],
            final_scoring_tiles: [FinalTile::Structure, FinalTile::PlanetType],
            pass_order: Vec::new(),
            current_player: 0,
            round: 1,
            terminated: false,
        };
        environment.reset();
        Ok(environment)
    }

    /// Start a fresh episode with the configured seed and return its first observation.
    pub fn reset(&mut self) -> Observation {
        self.map = crate::map::Map::generate_standard(self.config.players, self.config.seed);
        self.round = 1;
        self.terminated = false;
        self.claimed_artefacts = [false; 13];
        if self.players.len() != self.config.players {
            self.players = (0..self.config.players)
                .map(|seat| {
                    let faction = self
                        .config
                        .factions
                        .as_ref()
                        .and_then(|f| f.get(seat).copied())
                        .unwrap_or_else(|| match seat {
                            0 => Faction::Terrans,
                            1 => Faction::Lantids,
                            2 => Faction::HadschHallas,
                            _ => Faction::Ivits,
                        });
                    crate::player::PlayerData::new(seat as u8, faction)
                })
                .collect();
        } else {
            for (seat, p) in self.players.iter_mut().enumerate() {
                let faction = p.faction;
                *p = crate::player::PlayerData::new(seat as u8, faction);
            }
        }

        // Place starting structures — faction-specific rules apply.
        // Ivits: 1 Planetary Institute on Oxide planet, 0 mines.
        // Tinkeroids / Darkanians: 1 Planetary Institute (Tinkeroids) or 1 Mine (Darkanians) on Asteroid.
        // Moweyds / Space Giants: 1 Mine on Protoplanet.
        // All others: 2 Mines on their home planet type.
        for seat in 0..self.config.players {
            let faction = self.players[seat].faction;
            let home_planet = crate::rules::faction_planet(faction);
            match faction {
                Faction::Ivits => {
                    // Ivits start with a Planetary Institute on Oxide, no mines
                    for hex in &mut self.map.hexes[..self.map.count] {
                        if hex.building.is_none() && hex.planet == crate::rules::Planet::Oxide {
                            hex.building = Some(crate::rules::Building::PlanetaryInstitute);
                            hex.player = Some(seat as u8);
                            self.players[seat].buildings[crate::rules::Building::PlanetaryInstitute as usize] = 1;
                            break;
                        }
                    }
                }
                Faction::Tinkeroids => {
                    // Tinkeroids: Planetary Institute on an Asteroid
                    let mut placed = false;
                    for hex in &mut self.map.hexes[..self.map.count] {
                        if hex.building.is_none() && hex.planet == crate::rules::Planet::Asteroid {
                            hex.building = Some(crate::rules::Building::PlanetaryInstitute);
                            hex.player = Some(seat as u8);
                            self.players[seat].buildings[crate::rules::Building::PlanetaryInstitute as usize] = 1;
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        for hex in &mut self.map.hexes[..self.map.count] {
                            if hex.building.is_none() && (hex.planet == crate::rules::Planet::Empty || hex.planet == crate::rules::Planet::Transdim) {
                                hex.planet = crate::rules::Planet::Asteroid;
                                hex.building = Some(crate::rules::Building::PlanetaryInstitute);
                                hex.player = Some(seat as u8);
                                self.players[seat].buildings[crate::rules::Building::PlanetaryInstitute as usize] = 1;
                                break;
                            }
                        }
                    }
                }
                Faction::Darkanians => {
                    // Darkanians: 1 Mine on Asteroid
                    let mut placed = false;
                    for hex in &mut self.map.hexes[..self.map.count] {
                        if hex.building.is_none() && hex.planet == crate::rules::Planet::Asteroid {
                            hex.building = Some(crate::rules::Building::Mine);
                            hex.player = Some(seat as u8);
                            self.players[seat].buildings[crate::rules::Building::Mine as usize] = 1;
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        for hex in &mut self.map.hexes[..self.map.count] {
                            if hex.building.is_none() && (hex.planet == crate::rules::Planet::Empty || hex.planet == crate::rules::Planet::Transdim) {
                                hex.planet = crate::rules::Planet::Asteroid;
                                hex.building = Some(crate::rules::Building::Mine);
                                hex.player = Some(seat as u8);
                                self.players[seat].buildings[crate::rules::Building::Mine as usize] = 1;
                                break;
                            }
                        }
                    }
                }
                _ => {
                    // Standard factions: 2 Mines on home-colored planets
                    // Moweyds/SpaceGiants home is Protoplanet — place 1 mine there
                    let target_planet = home_planet;
                    let max_starting = if matches!(faction, Faction::Moweyds | Faction::SpaceGiants) { 1 } else { 2 };
                    let mut placed = 0;
                    for hex in &mut self.map.hexes[..self.map.count] {
                        if placed >= max_starting { break; }
                        if hex.building.is_none()
                            && hex.planet == target_planet
                        {
                            hex.building = Some(crate::rules::Building::Mine);
                            hex.player = Some(seat as u8);
                            self.players[seat].buildings[crate::rules::Building::Mine as usize] += 1;
                            placed += 1;
                        }
                    }
                    if placed < max_starting && matches!(faction, Faction::Moweyds | Faction::SpaceGiants) {
                        for hex in &mut self.map.hexes[..self.map.count] {
                            if hex.building.is_none() && (hex.planet == crate::rules::Planet::Empty || hex.planet == crate::rules::Planet::Transdim) {
                                hex.planet = crate::rules::Planet::Protoplanet;
                                hex.building = Some(crate::rules::Building::Mine);
                                hex.player = Some(seat as u8);
                                self.players[seat].buildings[crate::rules::Building::Mine as usize] += 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
        self.observe()
    }

    pub fn current_player(&self) -> usize {
        self.current_player
    }
    pub fn round(&self) -> u8 {
        self.round
    }
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }
    pub fn map(&self) -> &Map {
        &self.map
    }
    pub fn map_mut(&mut self) -> &mut Map {
        &mut self.map
    }
    pub fn players(&self) -> &[PlayerData] {
        &self.players
    }
    pub fn players_mut(&mut self) -> &mut [PlayerData] {
        &mut self.players
    }

    /// Executes a full structured Gaia Project game command.
    pub fn execute_command(&mut self, player: usize, cmd: GameCommand) -> Result<(), ActionError> {
        if self.terminated {
            return Err(ActionError::GameTerminated);
        }
        if player >= self.config.players {
            return Err(ActionError::GameTerminated);
        }
        match cmd {
            GameCommand::BuildMine { coord } => {
                let hex_before = self.map.get_hex(coord).copied();
                actions::execute_build_mine(&mut self.players[player], &mut self.map, coord)?;
                if (self.round as usize) <= self.round_scoring_tiles.len() {
                    let tile = self.round_scoring_tiles[(self.round - 1) as usize];
                    match tile {
                        rules::ScoringTile::Score3 => self.players[player].add_victory_points(2),
                        rules::ScoringTile::Score6
                            if hex_before.map(|h| h.planet) == Some(rules::Planet::Gaia) =>
                        {
                            self.players[player].add_victory_points(4);
                        }
                        rules::ScoringTile::Score9
                            if hex_before.map(|h| h.planet) == Some(rules::Planet::Gaia) =>
                        {
                            self.players[player].add_victory_points(3);
                        }
                        _ => {}
                    }
                }
                // Passive leeching: opponents within range 2 may charge power (automatic max leech)
                {
                    let opps = actions::find_leech_opportunities(&self.map, coord, player as u8, &self.players);
                    for opp in &opps {
                        if let Some(opp) = opp {
                            // Auto-accept maximum leech for the opponent (simplified: always leech max)
                            let _ = actions::execute_leech(&mut self.players[opp.seat as usize], opp.power_value);
                        }
                    }
                }
            }
            GameCommand::StartGaiaProject { coord } => {
                actions::execute_start_gaia_project(&mut self.players[player], &mut self.map, coord)?;
            }
            GameCommand::Upgrade { coord, to } => {
                actions::execute_upgrade(&mut self.players[player], &mut self.map, coord, to)?;
                if (self.round as usize) <= self.round_scoring_tiles.len() {
                    let tile = self.round_scoring_tiles[(self.round - 1) as usize];
                    match tile {
                        rules::ScoringTile::Score5 if to == rules::Building::TradingStation => {
                            self.players[player].add_victory_points(4);
                        }
                        rules::ScoringTile::Score8 if to == rules::Building::TradingStation => {
                            self.players[player].add_victory_points(3);
                        }
                        rules::ScoringTile::Score7 | rules::ScoringTile::Score10
                            if matches!(
                                to,
                                rules::Building::PlanetaryInstitute
                                    | rules::Building::Academy1
                                    | rules::Building::Academy2
                            ) =>
                        {
                            self.players[player].add_victory_points(5);
                        }
                        rules::ScoringTile::LfLab4 if to == rules::Building::ResearchLab => {
                            self.players[player].add_victory_points(4);
                        }
                        _ => {}
                    }
                }
                // Passive leeching: opponents within range 2 may charge power (automatic max leech)
                {
                    let opps = actions::find_leech_opportunities(&self.map, coord, player as u8, &self.players);
                    for opp in &opps {
                        if let Some(opp) = opp {
                            let _ = actions::execute_leech(&mut self.players[opp.seat as usize], opp.power_value);
                        }
                    }
                }
                // Mandatory tech tile claim trigger on Lab/Academy upgrade
                // (The agent MUST follow up with a ClaimTechTile command; this is enforced by legal_commands)
                // Note: actual claiming is done via ClaimTechTile command in the next action.
            }
            GameCommand::FormFederation {
                planets,
                planet_count,
                satellites,
                satellite_count,
                token,
            } => {
                actions::execute_form_federation(
                    &mut self.players[player],
                    &mut self.map,
                    &planets[..planet_count as usize],
                    &satellites[..satellite_count as usize],
                    token,
                )?;
                if (self.round as usize) <= self.round_scoring_tiles.len()
                    && self.round_scoring_tiles[(self.round - 1) as usize] == rules::ScoringTile::Score4
                {
                    self.players[player].add_victory_points(5);
                }
            }
            GameCommand::FormFederationAuto { token } => {
                actions::execute_form_federation_auto(&mut self.players[player], &mut self.map, token)?;
                if (self.round as usize) <= self.round_scoring_tiles.len()
                    && self.round_scoring_tiles[(self.round - 1) as usize] == rules::ScoringTile::Score4
                {
                    self.players[player].add_victory_points(5);
                }
            }
            GameCommand::AdvanceResearch { field } => {
                actions::execute_advance_research(
                    &mut self.players[player],
                    field,
                    &mut self.research_level_5_claimed,
                )?;
                if (self.round as usize) <= self.round_scoring_tiles.len()
                    && self.round_scoring_tiles[(self.round - 1) as usize] == rules::ScoringTile::Score2
                {
                    self.players[player].add_victory_points(2);
                }
            }
            GameCommand::Pass { new_booster } => {
                actions::execute_pass(&mut self.players[player], new_booster)?;
                self.pass_order.push(player as u8);
            }
            GameCommand::ChargePower { charge_amount } => {
                actions::execute_leech(&mut self.players[player], charge_amount)?;
            }
            GameCommand::BoardAction {
                action,
                target_mine,
                rescore_token,
            } => {
                actions::execute_board_action(
                    &mut self.players[player],
                    &mut self.map,
                    &mut self.claimed_board_actions,
                    action,
                    target_mine,
                    rescore_token,
                )?;
            }
            GameCommand::SpecialAction {
                action,
                target_coord,
                target_field,
            } => {
                actions::execute_special_action(
                    &mut self.players[player],
                    &mut self.map,
                    action,
                    target_coord,
                    target_field,
                    &mut self.research_level_5_claimed,
                )?;
            }
            GameCommand::ClaimTechTile {
                tech,
                advance_field,
            } => {
                actions::execute_claim_tech_tile(
                    &mut self.players[player],
                    tech,
                    advance_field,
                    &mut self.research_level_5_claimed,
                )?;
                // Clear pending flag — player has claimed their mandatory tech tile
                self.players[player].pending_tech_claim = false;
            }
            GameCommand::ClaimAdvTechTile {
                adv_tech,
                cover_tech,
                field,
            } => {
                actions::execute_claim_adv_tech_tile(
                    &mut self.players[player],
                    &self.map,
                    adv_tech,
                    cover_tech,
                    field,
                    &mut self.claimed_adv_techs,
                )?;
                // Clear pending flag — player claimed their mandatory tech tile (adv variant)
                self.players[player].pending_tech_claim = false;
            }
            GameCommand::ExploreSpaceship { ship, coord } => {
                let max_shuttles = if self.config.players == 2 { 2 } else { 3 };
                let mut occupied = [false; 5];
                for p in &self.players {
                    if let Some(slot) = p.exploration_ships[ship as usize].filter(|&s| (s as usize) < occupied.len()) {
                        occupied[slot as usize] = true;
                    }
                }
                actions::execute_explore_spaceship(
                    &mut self.players[player],
                    &occupied,
                    &self.map,
                    ship,
                    coord,
                    max_shuttles,
                )?;
            }
            GameCommand::SpaceshipBoardAction {
                ship,
                action_type,
                target_coord,
                target_field,
                rescore_token,
            } => {
                if ship == rules::Spaceship::Twilight && action_type == rules::SpaceshipActionType::Credit {
                    if let Some(art_idx) = self.claimed_artefacts.iter().enumerate().position(|(i, &c)| !c && !self.players[player].claimed_artefacts[i]) {
                        let art = match art_idx {
                            0 => rules::Artefact::IncomeKnowledgeOre,
                            1 => rules::Artefact::Credits3Ore3,
                            2 => rules::Artefact::Knowledge3Qic1,
                            3 => rules::Artefact::Credits5Ore2,
                            4 => rules::Artefact::ChargePower2,
                            5 => rules::Artefact::AsteroidVp,
                            6 => rules::Artefact::ProtoplanetVp,
                            7 => rules::Artefact::ResearchAreaLevel,
                            8 => rules::Artefact::ResearchTracksCount,
                            9 => rules::Artefact::RescoreFederation,
                            10 => rules::Artefact::GaiaformingTrack,
                            11 => rules::Artefact::PlanetTypes,
                            _ => rules::Artefact::DeepSpace,
                        };
                        actions::execute_examine_artefact(
                            &mut self.players[player],
                            &self.map,
                            art,
                            rescore_token,
                            &mut self.claimed_artefacts,
                        )?;
                        self.claimed_spaceship_actions[ship as usize][action_type as usize] = true;
                    } else {
                        return Err(ActionError::HexNotColonizable);
                    }
                } else {
                    actions::execute_spaceship_action(
                        &mut self.players[player],
                        &mut self.map,
                        &mut self.claimed_spaceship_actions,
                        &mut self.research_level_5_claimed,
                        ship,
                        action_type,
                        target_coord,
                        target_field,
                        rescore_token,
                    )?;
                }
            }
            GameCommand::FreeAction { action } => {
                actions::execute_free_action(&mut self.players[player], action)?;
            }
            GameCommand::ExamineArtefact { artefact, rescore_token } => {
                actions::execute_examine_artefact(
                    &mut self.players[player],
                    &self.map,
                    artefact,
                    rescore_token,
                    &mut self.claimed_artefacts,
                )?;
            }
        }
        Ok(())
    }

    /// Enumerates all currently legal game commands for an active player.
    pub fn legal_commands(&self, player: usize) -> Vec<GameCommand> {
        if self.terminated || player >= self.config.players {
            return Vec::new();
        }
        let available_boosters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let is_final = self.round >= self.config.max_rounds;
        actions::legal_commands(
            player,
            &self.players[player],
            &self.map,
            &self.research_level_5_claimed,
            &self.claimed_board_actions,
            &self.claimed_spaceship_actions,
            &available_boosters,
            is_final,
        )
    }


    /// Returns a fixed-size legality mask aligned with [`Action::ALL`].
    pub fn action_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SPACE];
        if self.terminated {
            return mask;
        }
        for i in 0..ACTION_SPACE {
            if let Some(cmd) = crate::action_space::decode_action(i, &self.map) {
                let mut cloned = self.clone();
                if cloned.execute_command(self.current_player, cmd).is_ok() {
                    mask[i] = true;
                }
            }
        }
        mask
    }

    pub fn observe(&self) -> Observation {
        let mut v = vec![0.0f32; OBS_SPACE];

        // =========================================================================
        // PART A: GLOBAL STATE & COMMON MARKET (88 floats, indices 0..87)
        // =========================================================================
        let num_players = self.config.players as f32;
        v[0] = (self.current_player as f32) / (num_players - 1.0).max(1.0);
        v[1] = (self.round as f32) / (self.config.max_rounds as f32);

        // Pass order ranking for 4 seats (indices 2..6)
        for s in 0..4 {
            if let Some(pos) = self.pass_order.iter().position(|&seat| seat == s as u8) {
                v[2 + s] = (pos + 1) as f32 / 4.0;
            } else {
                v[2 + s] = 0.0;
            }
        }

        // Round scoring tiles (indices 6..12)
        for i in 0..6 {
            v[6 + i] = (self.round_scoring_tiles[i] as u8 as f32) / 13.0;
        }

        // Final scoring tiles (indices 12..14)
        for i in 0..2 {
            v[12 + i] = (self.final_scoring_tiles[i] as u8 as f32) / 9.0;
        }

        // Research level 5 claimed (indices 14..20)
        for i in 0..6 {
            if let Some(c) = self.research_level_5_claimed[i] {
                v[14 + i] = (c as f32 + 1.0) / 4.0;
            } else {
                v[14 + i] = 0.0;
            }
        }

        // Standard tech tiles remaining (indices 20..29)
        for tech_idx in 0..9 {
            let claimed_count = self.players.iter().filter(|p| tech_idx < p.tech_tiles.len() && p.tech_tiles[tech_idx]).count();
            v[20 + tech_idx] = (4.0 - claimed_count as f32).max(0.0) / 4.0;
        }

        // Advanced tech tiles available (indices 29..44)
        for adv_idx in 0..15 {
            v[29 + adv_idx] = if self.claimed_adv_techs[adv_idx] { 0.0 } else { 1.0 };
        }

        // Board power actions claimed (indices 44..54)
        for b_idx in 0..10 {
            v[44 + b_idx] = if self.claimed_board_actions[b_idx].is_some() { 1.0 } else { 0.0 };
        }

        // Federation tokens stock remaining (indices 54..62)
        // Fed1..Fed6: 3 copies each in base game; Gleens token: 1 copy.
        {
            let fed_variants = [
                rules::FederationToken::Fed1,
                rules::FederationToken::Fed2,
                rules::FederationToken::Fed3,
                rules::FederationToken::Fed4,
                rules::FederationToken::Fed5,
                rules::FederationToken::Fed6,
                rules::FederationToken::Gleens,
            ];
            let fed_stock: [u8; 7] = [3, 3, 3, 3, 3, 3, 1];
            for (f_idx, (&variant, &stock)) in fed_variants.iter().zip(fed_stock.iter()).enumerate() {
                let claimed: u8 = self.players.iter().map(|p| {
                    p.claimed_federations.iter().filter(|&&f| f == Some(variant)).count() as u8
                }).sum();
                v[54 + f_idx] = (stock.saturating_sub(claimed) as f32) / (stock.max(1) as f32);
            }
            v[61] = 0.0; // 8th slot unused
        }

        // Boosters held by player seat (indices 62..72)
        for b_id in 0..10 {
            let booster_num = (b_id + 1) as u8;
            let held_by = self.players.iter().position(|p| p.current_booster == Some(booster_num));
            v[62 + b_id] = match held_by {
                Some(p_idx) => (p_idx as f32 + 2.0) / 6.0,
                None => 1.0,
            };
        }

        // Spaceship actions claimed (indices 72..88, 4x4)
        let mut idx_s = 72;
        for s_idx in 0..4 {
            for act_idx in 0..4 {
                v[idx_s] = if self.claimed_spaceship_actions[s_idx][act_idx] { 1.0 } else { 0.0 };
                idx_s += 1;
            }
        }

        // =========================================================================
        // PART B: 4 PLAYERS DETAILED STATE (4 x 97 = 388 floats, indices 88..475)
        // =========================================================================
        let mut base_idx = 88;
        for s in 0..4 {
            if s < self.players.len() {
                let p = &self.players[s];
                // 1. Identity & Score (3)
                v[base_idx + 0] = (p.faction as u8 as f32) / 17.0;
                v[base_idx + 1] = (s as f32) / 3.0;
                v[base_idx + 2] = (p.victory_points as f32) / 300.0;

                // 2. Wallet Resources (4)
                v[base_idx + 3] = (p.credits as f32) / 30.0;
                v[base_idx + 4] = (p.ore as f32) / 15.0;
                v[base_idx + 5] = (p.knowledge as f32) / 15.0;
                v[base_idx + 6] = (p.qic as f32) / 15.0;

                // 3. Power Bowls & Energy (7)
                v[base_idx + 7] = (p.power.area1 as f32) / 15.0;
                v[base_idx + 8] = (p.power.area2 as f32) / 15.0;
                v[base_idx + 9] = (p.power.area3 as f32) / 15.0;
                v[base_idx + 10] = (p.power.gaia as f32) / 15.0;
                v[base_idx + 11] = match p.power.brainstone {
                    Some(crate::rules::PowerArea::Area1) => 1.0 / 4.0,
                    Some(crate::rules::PowerArea::Area2) => 2.0 / 4.0,
                    Some(crate::rules::PowerArea::Area3) => 3.0 / 4.0,
                    Some(crate::rules::PowerArea::Gaia) => 4.0 / 4.0,
                    None => 0.0,
                };
                v[base_idx + 12] = (p.power.spendable_power() as f32) / 15.0;
                v[base_idx + 13] = ((p.power.area1 + p.power.area2 + p.power.area3) as f32) / 15.0;

                // 4. Research Tracks (6)
                for t_idx in 0..6 {
                    v[base_idx + 14 + t_idx] = (p.research[t_idx] as f32) / 5.0;
                }

                // 5. Buildings Stocks & Deployed (9)
                v[base_idx + 20] = (p.buildings[crate::rules::Building::Mine as usize] as f32) / 8.0;
                v[base_idx + 21] = (p.buildings[crate::rules::Building::TradingStation as usize] as f32) / 4.0;
                v[base_idx + 22] = (p.buildings[crate::rules::Building::ResearchLab as usize] as f32) / 3.0;
                v[base_idx + 23] = (p.buildings[crate::rules::Building::PlanetaryInstitute as usize] as f32) / 1.0;
                v[base_idx + 24] = (p.buildings[crate::rules::Building::Academy1 as usize] as f32) / 1.0;
                v[base_idx + 25] = (p.buildings[crate::rules::Building::Academy2 as usize] as f32) / 1.0;
                v[base_idx + 26] = (p.gaiaformers_unlocked as f32) / 3.0;
                v[base_idx + 27] = (p.gaiaformers_in_gaia as f32) / 3.0;
                v[base_idx + 28] = (p.satellites as f32) / 15.0;

                // 6. Federations (12)
                v[base_idx + 29] = (p.green_federation_tokens as f32) / 5.0;
                v[base_idx + 30] = (p.gray_federation_tokens as f32) / 5.0;
                v[base_idx + 31] = (p.total_federation_tokens() as f32) / 6.0;
                v[base_idx + 32] = (p.buildings[crate::rules::Building::Mine as usize] as f32
                    + (p.buildings[crate::rules::Building::TradingStation as usize] as f32) * 2.0
                    + (p.buildings[crate::rules::Building::ResearchLab as usize] as f32) * 2.0
                    + (p.buildings[crate::rules::Building::PlanetaryInstitute as usize] as f32) * 3.0
                    + ((p.buildings[crate::rules::Building::Academy1 as usize] + p.buildings[crate::rules::Building::Academy2 as usize]) as f32) * 3.0) / 30.0;
                let fed_tokens = [
                    crate::rules::FederationToken::Fed1,
                    crate::rules::FederationToken::Fed2,
                    crate::rules::FederationToken::Fed3,
                    crate::rules::FederationToken::Fed4,
                    crate::rules::FederationToken::Fed5,
                    crate::rules::FederationToken::Fed6,
                    crate::rules::FederationToken::Gleens,
                ];
                for f_idx in 0..7 {
                    v[base_idx + 33 + f_idx] = if p.claimed_federations.iter().any(|f| *f == Some(fed_tokens[f_idx])) { 1.0 } else { 0.0 };
                }
                v[base_idx + 40] = 0.0;

                // 7. Tech Tiles (33)
                for i in 0..9 {
                    v[base_idx + 41 + i] = if p.tech_tiles[i] { 1.0 } else { 0.0 };
                }
                for i in 0..9 {
                    v[base_idx + 50 + i] = if p.covered_tech_tiles[i] { 1.0 } else { 0.0 };
                }
                for i in 0..15 {
                    v[base_idx + 59 + i] = if p.adv_tech_tiles[i] { 1.0 } else { 0.0 };
                }

                // 8. Turn, Booster & Special Actions (12)
                v[base_idx + 74] = if p.passed { 1.0 } else { 0.0 };
                v[base_idx + 75] = p.current_booster.map(|b| b as f32).unwrap_or(0.0) / 10.0;
                for i in 0..10 {
                    v[base_idx + 76 + i] = if p.special_actions_used[i] { 1.0 } else { 0.0 };
                }

                // 9. Lost Fleet Exploration (5)
                v[base_idx + 86] = (p.deployed_shuttles() as f32) / 3.0;
                for i in 0..4 {
                    v[base_idx + 87 + i] = if p.exploration_ships[i].is_some() { 1.0 } else { 0.0 };
                }

                // 10. Final Scoring Metrics (6)
                v[base_idx + 91] = (10.0f32).min((p.buildings[crate::rules::Building::Mine as usize] + p.buildings[crate::rules::Building::TradingStation as usize]) as f32) / 10.0;
                v[base_idx + 92] = (10.0f32).min(p.buildings[crate::rules::Building::Mine as usize] as f32 + 1.0) / 10.0;
                v[base_idx + 93] = (p.gaiaformers_unlocked as f32) / 6.0;
                v[base_idx + 94] = (p.total_federation_tokens() as f32) / 15.0;
                v[base_idx + 95] = (p.satellites as f32) / 15.0;
                v[base_idx + 96] = (6.0f32).min(p.deployed_shuttles() as f32) / 6.0;
            }
            base_idx += 97;
        }

        // =========================================================================
        // PART C: GALAXY MAP BOARD (200 Hexes x 10 features = 2000 floats, indices 476..2475)
        //   feat 0 = planet type, feat 1 = building, feat 2 = player, feat 3 = additional_mine,
        //   feat 4..7 = in_federation[0..3], feat 8 = spaceship, feat 9 = sector_id
        // =========================================================================
        let mut map_idx = 476;
        for k in 0..200 {
            if k < self.map.count {
                let h = &self.map.hexes[k];
                v[map_idx + 0] = (h.planet as u8 as f32) / 13.0;
                v[map_idx + 1] = match h.building {
                    Some(b) => (b as usize + 1) as f32,
                    None => 0.0,
                } / 9.0;
                v[map_idx + 2] = match h.player {
                    Some(p) => (p as f32 + 1.0) / 4.0,
                    None => 0.0,
                };
                v[map_idx + 3] = match h.additional_mine {
                    Some(p) => (p as f32 + 1.0) / 4.0,
                    None => 0.0,
                };
                for seat in 0..4 {
                    v[map_idx + 4 + seat] = if h.in_federation[seat] { 1.0 } else { 0.0 };
                }
                v[map_idx + 8] = match h.spaceship {
                    Some(s) => (s as usize as f32 + 1.0) / 4.0,
                    None => 0.0,
                };
                // Feature 9: sector_id (0-9 for base sectors, normalized to [0,1])
                v[map_idx + 9] = (h.sector_id as f32) / 10.0;
            }
            map_idx += 10;
        }

        Observation {
            current_player: self.current_player,
            round: self.round,
            values: v,
            action_mask: self.action_mask(),
            free_actions: vec![],
        }
    }
    pub fn execute_command_from_rl(&mut self, actor: usize, cmd: crate::actions::GameCommand) -> Result<StepResult, EnvError> {
        if self.terminated {
            return Err(EnvError::EpisodeFinished);
        }
        if actor != self.current_player {
            return Err(EnvError::IllegalAction { player: actor, action: 0 }); // dummy action ID
        }
        
        let is_free_action = false;

        let old_vps: Vec<i16> = self.players.iter().map(|p| p.victory_points).collect();

        if self.execute_command(actor, cmd).is_err() {
            return Err(EnvError::IllegalAction { player: actor, action: 0 });
        }
        
        if !is_free_action {
            self.advance_turn();
        }

        let mut rewards = vec![0.0; self.config.players];
        for i in 0..self.config.players {
            rewards[i] = (self.players[i].victory_points - old_vps[i]) as f32;
        }

        Ok(StepResult {
            observation: self.observe(),
            
            rewards,
            terminated: self.terminated,
            truncated: self.terminated && self.round > self.config.max_rounds,
            round: self.round,
            current_player: self.current_player,
        })
    }
    pub fn advance_turn(&mut self) {
        if self.players.iter().all(|player| player.passed) {
            self.round += 1;
            if self.round > self.config.max_rounds {
                actions::compute_final_scores(&mut self.players, &self.map, self.final_scoring_tiles);
                self.terminated = true;
                return;
            }
            // Phase II: Gaia Phase
            actions::resolve_gaia_phase(&mut self.map, &mut self.players);

            // Phase IV: Clean-Up & Phase I: Income for next round
            self.claimed_board_actions = [None; 10];
            self.claimed_spaceship_actions = [[false; 4]; 4];
            for player in &mut self.players {
                player.passed = false;
                player.temporary_step = 0;
                player.temporary_range = 0;
                player.special_actions_used = [false; 10];
                income::collect_round_income(player);
            }
            self.current_player = self.pass_order.first().copied().unwrap_or(0) as usize;
            self.pass_order.clear();
            return;
        }
        for _ in 0..self.config.players {
            self.current_player = (self.current_player + 1) % self.config.players;
            if !self.players[self.current_player].passed {
                return;
            }
        }
    }

    pub fn random_faction(&mut self, seat: usize) -> Faction {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        Faction::ALL[((self.rng_state >> 32) as usize + seat) % Faction::ALL.len()]
    }
}







