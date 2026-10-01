//! Game action engine for Gaia Project: Build Mine, Start Gaia Project, Upgrade, Form Federation, Research, Pass.
//!
//! Fully compliant with the official Base Game rules and The Lost Fleet expansion.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::Faction;
use crate::board::HexCoord;
use crate::map::Map;
use crate::player::PlayerData;
use crate::rules::{
    AdvTechTile, BoardAction, Building, FederationToken, FinalTile, Planet, ResearchField,
    SpecialAction, Spaceship, SpaceshipActionType, TS_COST_ISOLATED, TS_COST_NEIGHBOR, TechTile,
    building_power_value, faction_planet, qic_for_distance, terraforming_ore_cost,
    terraforming_steps, upgraded_buildings, EXPLORATION_CHARGE_TRACK,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionError {
    GameTerminated,
    PlayerAlreadyPassed,
    HexNotFound,
    HexNotColonizable,
    HexAlreadyOccupied,
    CannotBuildOnTransdimDirectly,
    UnreachableHex,
    OutOfRange { needed: u8, max_available: u8 },
    InsufficientCredits { needed: i16, have: i16 },
    InsufficientOre { needed: i16, have: i16 },
    InsufficientKnowledge { needed: i16, have: i16 },
    InsufficientQic { needed: i16, have: i16 },
    NoMinesAvailable,
    NoGaiaformerAvailable,
    BuildingMaxReached(Building),
    InvalidUpgradePath { from: Option<Building>, to: Building },
    NotYourStructure,
    LantidsCannotUpgradeAdditionalMine,
    GaiaProjectResearchRequired,
    NotTransdimPlanet,
    InsufficientPowerForGaiaProject { needed: u8, have: u8 },
    InsufficientPowerForSatellites { needed: u8, have: u8 },
    InsufficientSatellites { needed: u8, available: u8 },
    InsufficientFederationPower { needed: u8, have: u8 },
    PlanetsNotConnected,
    PlanetAlreadyFederated,
    InvalidSatellitePlacement,
    ResearchMaxReached(ResearchField),
    ResearchLevel5AlreadyClaimed(ResearchField),
    FederationTokenRequiredForLevel5,
    BalTaksNavigationRestricted,
    BoardActionAlreadyClaimed(BoardAction),
    InsufficientPowerForBoardAction { needed: u8, have: u8 },
    SpecialActionAlreadyUsed(SpecialAction),
    SpecialActionUnavailable(SpecialAction),
    TechTileAlreadyOwned(TechTile),
    AdvTechTileAlreadyClaimed(AdvTechTile),
    AdvTechTileRequirementsNotMet,
    NoFederationToRescore,
    NoExplorationShuttleAvailable,
    SpaceshipAlreadyExplored(Spaceship),
    SpaceshipNoFreeSlot(Spaceship),
    SpaceshipNotExplored(Spaceship),
    SpaceshipActionAlreadyUsed(Spaceship, SpaceshipActionType),
    NotSpaceshipHex,
    TwilightNotExplored,
    ArtefactAlreadyClaimed(crate::rules::Artefact),
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GameTerminated => f.write_str("game is already finished"),
            Self::PlayerAlreadyPassed => f.write_str("player has already passed this round"),
            Self::HexNotFound => f.write_str("target hex coordinate does not exist on map"),
            Self::HexNotColonizable => f.write_str("space does not contain a colonizable planet"),
            Self::HexAlreadyOccupied => f.write_str("planet is already occupied by a structure"),
            Self::CannotBuildOnTransdimDirectly => {
                f.write_str("cannot build directly on a Transdim planet without a Gaia project")
            }
            Self::UnreachableHex => f.write_str("target hex cannot be reached from any colonized planet"),
            Self::OutOfRange { needed, max_available } => write!(
                f,
                "out of range: needed {needed}, but max available with QIC is {max_available}"
            ),
            Self::InsufficientCredits { needed, have } => {
                write!(f, "insufficient credits: needed {needed}, have {have}")
            }
            Self::InsufficientOre { needed, have } => {
                write!(f, "insufficient ore: needed {needed}, have {have}")
            }
            Self::InsufficientKnowledge { needed, have } => {
                write!(f, "insufficient knowledge: needed {needed}, have {have}")
            }
            Self::InsufficientQic { needed, have } => {
                write!(f, "insufficient QIC: needed {needed}, have {have}")
            }
            Self::NoMinesAvailable => f.write_str("no mines remaining on faction board"),
            Self::NoGaiaformerAvailable => f.write_str("no Gaiaformer available"),
            Self::BuildingMaxReached(b) => write!(f, "maximum limit reached for building {b:?}"),
            Self::InvalidUpgradePath { from, to } => {
                write!(f, "cannot upgrade from {from:?} to {to:?}")
            }
            Self::NotYourStructure => f.write_str("player does not own a structure on this hex"),
            Self::LantidsCannotUpgradeAdditionalMine => {
                f.write_str("Lantids cannot upgrade an additional mine built on an opponent planet")
            }
            Self::GaiaProjectResearchRequired => {
                f.write_str("Gaia Project research level 1 or higher is required to start a Gaia project")
            }
            Self::NotTransdimPlanet => f.write_str("target hex is not a Transdim planet"),
            Self::InsufficientPowerForGaiaProject { needed, have } => write!(
                f,
                "insufficient power for Gaia project: needed {needed}, have {have}"
            ),
            Self::InsufficientPowerForSatellites { needed, have } => write!(
                f,
                "insufficient power to discard for satellites: needed {needed}, have {have}"
            ),
            Self::InsufficientSatellites { needed, available } => write!(
                f,
                "insufficient satellites in supply: needed {needed}, available {available}"
            ),
            Self::InsufficientFederationPower { needed, have } => write!(
                f,
                "insufficient structure power value for federation: needed {needed}, have {have}"
            ),
            Self::PlanetsNotConnected => {
                f.write_str("federation structures and satellites do not form a single connected group")
            }
            Self::PlanetAlreadyFederated => {
                f.write_str("one or more of the selected planets already belongs to a federation")
            }
            Self::InvalidSatellitePlacement => {
                f.write_str("satellites must be placed on empty space hexes that do not already have your satellite")
            }
            Self::ResearchMaxReached(field) => {
                write!(f, "research field {field:?} is already at maximum level 5")
            }
            Self::ResearchLevel5AlreadyClaimed(field) => {
                write!(f, "level 5 in {field:?} has already been claimed by another player")
            }
            Self::FederationTokenRequiredForLevel5 => {
                f.write_str("advancing to level 5 requires flipping a green federation token")
            }
            Self::BalTaksNavigationRestricted => {
                f.write_str("Bal T'aks cannot advance in Navigation before building their Planetary Institute")
            }
            Self::BoardActionAlreadyClaimed(action) => {
                write!(f, "board action {action:?} has already been claimed this round")
            }
            Self::InsufficientPowerForBoardAction { needed, have } => {
                write!(f, "insufficient power for board action: needed {needed}, have {have}")
            }
            Self::SpecialActionAlreadyUsed(action) => {
                write!(f, "special action {action:?} has already been used this round")
            }
            Self::SpecialActionUnavailable(action) => {
                write!(f, "special action {action:?} is not available for this faction or building state")
            }
            Self::TechTileAlreadyOwned(tech) => {
                write!(f, "standard tech tile {tech:?} is already owned")
            }
            Self::AdvTechTileAlreadyClaimed(adv) => {
                write!(f, "advanced tech tile {adv:?} has already been claimed")
            }
            Self::AdvTechTileRequirementsNotMet => {
                f.write_str("requirements for advanced tech tile not met (research level 4/5, green federation token, uncovered standard tech tile)")
            }
            Self::NoFederationToRescore => {
                f.write_str("no federation token owned to rescore")
            }
            Self::NoExplorationShuttleAvailable => {
                f.write_str("all exploration shuttles have been deployed")
            }
            Self::SpaceshipAlreadyExplored(ship) => {
                write!(f, "player has already explored spaceship {ship:?}")
            }
            Self::SpaceshipNoFreeSlot(ship) => {
                write!(f, "no free exploration slots remaining on spaceship {ship:?}")
            }
            Self::SpaceshipNotExplored(ship) => {
                write!(f, "player must explore spaceship {ship:?} first")
            }
            Self::SpaceshipActionAlreadyUsed(ship, action) => {
                write!(f, "spaceship action {action:?} on {ship:?} has already been used this round")
            }
            Self::NotSpaceshipHex => {
                f.write_str("hex does not contain the requested spaceship")
            }
            Self::TwilightNotExplored => {
                f.write_str("player must explore the Twilight spaceship before examining artifacts")
            }
            Self::ArtefactAlreadyClaimed(art) => {
                write!(f, "artifact {art:?} has already been claimed")
            }
        }
    }
}
impl std::error::Error for ActionError {}

/// High-level structured command for player actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameCommand {
    BuildMine {
        coord: HexCoord,
    },
    StartGaiaProject {
        coord: HexCoord,
    },
    Upgrade {
        coord: HexCoord,
        to: Building,
    },
    FormFederation {
        planets: [HexCoord; 6],
        planet_count: u8,
        satellites: [HexCoord; 12],
        satellite_count: u8,
        token: FederationToken,
    },
    FormFederationAuto {
        token: FederationToken,
    },
    AdvanceResearch {
        field: ResearchField,
    },
    Pass {
        new_booster: Option<u8>,
    },
    ChargePower {
        charge_amount: u8,
    },
    DeclineLeech,
    BoardAction {
        action: BoardAction,
        target_mine: Option<HexCoord>,
        rescore_token: Option<FederationToken>,
    },
    SpecialAction {
        action: SpecialAction,
        target_coord: Option<HexCoord>,
        target_field: Option<ResearchField>,
    },
    ClaimTechTile {
        tech: TechTile,
        advance_field: Option<ResearchField>,
    },
    ClaimAdvTechTile {
        adv_tech: AdvTechTile,
        cover_tech: TechTile,
        field: ResearchField,
    },
    ExploreSpaceship {
        ship: Spaceship,
        coord: HexCoord,
    },
    SpaceshipBoardAction {
        ship: Spaceship,
        action_type: SpaceshipActionType,
        target_coord: Option<HexCoord>,
        target_field: Option<ResearchField>,
        rescore_token: Option<FederationToken>,
    },
    FreeAction {
        action: crate::rules::FreeAction,
    },
    ExamineArtefact {
        artefact: crate::rules::Artefact,
        rescore_token: Option<FederationToken>,
    },
}

impl GameCommand {
    pub fn form_federation(
        planets: &[HexCoord],
        satellites: &[HexCoord],
        token: FederationToken,
    ) -> Self {
        let mut p = [HexCoord::origin(); 6];
        let p_count = planets.len().min(6) as u8;
        for (i, coord) in planets.iter().take(6).enumerate() {
            p[i] = *coord;
        }

        let mut s = [HexCoord::origin(); 12];
        let s_count = satellites.len().min(12) as u8;
        for (i, coord) in satellites.iter().take(12).enumerate() {
            s[i] = *coord;
        }

        Self::FormFederation {
            planets: p,
            planet_count: p_count,
            satellites: s,
            satellite_count: s_count,
            token,
        }
    }
}

/// Computes the exact cost to build a mine on `target_coord`.
pub struct BuildMineCost {
    pub credits: i16,
    pub ore: i16,
    pub qic: i16,
    pub consumes_gaiaformer: bool,
    pub immediate_vp: i16,
}

pub fn calculate_build_mine_cost(
    player: &PlayerData,
    map: &Map,
    coord: HexCoord,
) -> Result<(usize, BuildMineCost), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if player.buildings_available(Building::Mine) == 0 {
        return Err(ActionError::NoMinesAvailable);
    }

    let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
    let hex = map.hexes[hex_idx];

    if !hex.has_planet() {
        return Err(ActionError::HexNotColonizable);
    }
    if hex.planet == Planet::Transdim {
        return Err(ActionError::CannotBuildOnTransdimDirectly);
    }

    // Occupancy check (Lantids special exception & Gaiaformer on Gaia planet exception)
    let has_own_gaiaformer = hex.planet == Planet::Gaia
        && hex.building == Some(Building::GaiaFormer)
        && hex.player == Some(player.seat);

    if hex.occupied() && !has_own_gaiaformer {
        if player.faction == Faction::Lantids {
            if hex.additional_mine.is_some() || hex.player == Some(player.seat) || hex.building == Some(Building::GaiaFormer) {
                return Err(ActionError::HexAlreadyOccupied);
            }
        } else {
            return Err(ActionError::HexAlreadyOccupied);
        }
    }

    // Distance and range check (no range needed if Gaiaformer already on site)
    let qic_for_dist = if has_own_gaiaformer {
        0
    } else {
        let dist = match map.min_distance_from_player(player.seat, hex_idx) {
            Some(d) => d,
            None => {
                // First mine in game setup can be placed without range starting point
                if player.buildings[Building::Mine as usize] == 0 {
                    0
                } else {
                    return Err(ActionError::UnreachableHex);
                }
            }
        };

        let base_range = player.effective_range();
        qic_for_distance(dist, base_range, 0) as i16
    };

    // Special planet type costs:
    let mut credits = 2;
    let mut ore = 1;
    let mut qic = qic_for_dist;
    let mut consumes_gaiaformer = false;
    let mut immediate_vp = 0;

    if hex.planet == Planet::Asteroid {
        // Lost Fleet Asteroid: requires available Gaiaformer, permanently consumes it!
        // 0 ore and 0 credits build cost.
        if player.available_gaiaformers() == 0 {
            return Err(ActionError::NoGaiaformerAvailable);
        }
        consumes_gaiaformer = true;
        credits = 0;
        ore = 0;
    } else if hex.planet == Planet::Protoplanet {
        // Lost Fleet Protoplanet: 3 terraforming steps for all factions, +6 VP reward!
        let discount = player.terraform_cost_discount();
        let terra_ore = terraforming_ore_cost(player.temporary_step as i16, discount, 3);
        ore += terra_ore;
        immediate_vp = 6;
    } else if hex.planet == Planet::Gaia {
        // Gaia planet:
        // If a Gaiaformer is already on it, no QIC cost to make habitable!
        let has_gaiaformer = hex.building == Some(Building::GaiaFormer) && hex.player == Some(player.seat);
        if !has_gaiaformer {
            if player.faction == Faction::Gleens {
                // Gleens pay 1 ore instead of 1 QIC and get +2 VP
                ore += 1;
                immediate_vp += 2;
            } else if player.faction == Faction::Darkanians || player.faction == Faction::SpaceGiants {
                qic += 2;
            } else {
                qic += 1;
            }
        }
        if player.tech_tiles[TechTile::Tech7 as usize] {
            immediate_vp += 3;
        }
    } else {
        // Standard 7-color planet terraforming
        if hex.occupied() && player.faction == Faction::Lantids {
            // Lantids do not pay terraforming on occupied planets
        } else {
            let steps = terraforming_steps(player.faction, hex.planet, &[]) as i16;
            let discount = player.terraform_cost_discount();
            ore += terraforming_ore_cost(player.temporary_step as i16, discount, steps);
        }
    }

    if player.adv_tech_tiles[AdvTechTile::AdvTech14 as usize] {
        immediate_vp += 3;
    }

    if player.credits < credits {
        return Err(ActionError::InsufficientCredits {
            needed: credits,
            have: player.credits,
        });
    }
    if player.ore < ore {
        return Err(ActionError::InsufficientOre {
            needed: ore,
            have: player.ore,
        });
    }
    if player.qic < qic {
        return Err(ActionError::InsufficientQic {
            needed: qic,
            have: player.qic,
        });
    }

    Ok((
        hex_idx,
        BuildMineCost {
            credits,
            ore,
            qic,
            consumes_gaiaformer,
            immediate_vp,
        },
    ))
}

/// Executes building a mine on `target_coord`.
pub fn execute_build_mine(
    player: &mut PlayerData,
    map: &mut Map,
    coord: HexCoord,
) -> Result<(), ActionError> {
    let (hex_idx, cost) = calculate_build_mine_cost(player, map, coord)?;

    // Deduct resources
    player.credits -= cost.credits;
    player.ore -= cost.ore;
    player.qic -= cost.qic;

    if cost.consumes_gaiaformer {
        player.used_gaiaformers_asteroid += 1;
    }

    // If building on a Gaia planet that already holds player's Gaiaformer, return it
    if map.hexes[hex_idx].building == Some(Building::GaiaFormer)
        && map.hexes[hex_idx].player == Some(player.seat)
    {
        player.gaiaformers_in_gaia = player.gaiaformers_in_gaia.saturating_sub(1);
    }

    player.buildings[Building::Mine as usize] += 1;
    player.victory_points += cost.immediate_vp;

    let hex = &mut map.hexes[hex_idx];
    if hex.occupied() && player.faction == Faction::Lantids {
        hex.additional_mine = Some(player.seat);
        // Lantids PI ability: each time building a mine on an opponent's planet, gain 2 knowledge
        if player.buildings[Building::PlanetaryInstitute as usize] > 0 {
            player.add_knowledge(2);
        }
    } else {
        let target_planet = hex.planet;
        hex.building = Some(Building::Mine);
        hex.player = Some(player.seat);

        // Geodens PI ability: first time colonizing each planet type, gain 3 knowledge
        if player.faction == Faction::Geodens
            && player.buildings[Building::PlanetaryInstitute as usize] > 0
        {
            let is_new_type = !map.hexes[..map.count].iter().enumerate().any(|(i, h)| {
                i != hex_idx && h.planet == target_planet && h.colonized_by(player.seat)
            });
            if is_new_type {
                player.add_knowledge(3);
            }
        }
    }

    Ok(())
}

/// Checks if an upgrade from the current building on `coord` to `to` is valid and computes cost.
pub fn calculate_upgrade_cost(
    player: &PlayerData,
    map: &Map,
    coord: HexCoord,
    to: Building,
) -> Result<(usize, i16, i16), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
    let hex = map.hexes[hex_idx];

    // Lantids additional mines cannot be upgraded (Base Rulebook p. 20)
    if hex.additional_mine == Some(player.seat) {
        return Err(ActionError::LantidsCannotUpgradeAdditionalMine);
    }

    let current = hex.building_of(player.seat).ok_or(ActionError::NotYourStructure)?;
    if !upgraded_buildings(current, player.faction).contains(&to) {
        return Err(ActionError::InvalidUpgradePath {
            from: Some(current),
            to,
        });
    }

    if player.buildings_available(to) == 0 {
        return Err(ActionError::BuildingMaxReached(to));
    }

    let (credits, ore) = match (current, to) {
        (Building::Mine, Building::TradingStation) => {
            // Check for opponent structures within distance <= 2
            let mut has_neighbor_opponent = false;
            for j in 0..map.count {
                if j != hex_idx
                    && map.hexes[j].has_structure()
                    && map.hexes[j].player != Some(player.seat)
                    && map.distance(hex_idx, j) <= 2
                {
                    has_neighbor_opponent = true;
                    break;
                }
            }
            if has_neighbor_opponent {
                (TS_COST_NEIGHBOR.credits, TS_COST_NEIGHBOR.ore)
            } else {
                (TS_COST_ISOLATED.credits, TS_COST_ISOLATED.ore)
            }
        }
        (Building::TradingStation, Building::PlanetaryInstitute) => (6, 4),
        (Building::TradingStation, Building::ResearchLab) => (5, 3),
        (Building::ResearchLab, Building::Academy1)
        | (Building::ResearchLab, Building::Academy2) => (6, 6),
        // Bescods inversion exceptions:
        (Building::TradingStation, Building::Academy1)
        | (Building::TradingStation, Building::Academy2) => (6, 6),
        (Building::ResearchLab, Building::PlanetaryInstitute) => (6, 4),
        _ => {
            return Err(ActionError::InvalidUpgradePath {
                from: Some(current),
                to,
            });
        }
    };

    if player.credits < credits {
        return Err(ActionError::InsufficientCredits {
            needed: credits,
            have: player.credits,
        });
    }
    if player.ore < ore {
        return Err(ActionError::InsufficientOre {
            needed: ore,
            have: player.ore,
        });
    }

    Ok((hex_idx, credits, ore))
}

/// Executes upgrading the structure on `coord` to `to`.
pub fn execute_upgrade(
    player: &mut PlayerData,
    map: &mut Map,
    coord: HexCoord,
    to: Building,
) -> Result<(), ActionError> {
    let (hex_idx, credits, ore) = calculate_upgrade_cost(player, map, coord, to)?;
    let current = map.hexes[hex_idx].building_of(player.seat).unwrap();

    player.credits -= credits;
    player.ore -= ore;

    player.buildings[current as usize] = player.buildings[current as usize].saturating_sub(1);
    player.buildings[to as usize] = player.buildings[to as usize].saturating_add(1);
    map.hexes[hex_idx].building = Some(to);

    // Gleens PI ability: immediately gain Gleens federation token upon upgrading to PI
    if to == Building::PlanetaryInstitute && player.faction == Faction::Gleens {
        player.claim_federation_token(FederationToken::Gleens);
    }

    // Mandatory tech tile claim: upgrading to ResearchLab, Academy1, or Academy2 grants one tech tile.
    // Set a flag so the game loop knows the player must pick a tech tile next.
    if matches!(to, Building::ResearchLab | Building::Academy1 | Building::Academy2) {
        player.pending_tech_claim = true;
    }

    Ok(())
}

/// Advances one level on a research track without charging knowledge cost.
/// Handles level 5 federation token requirement, level 5 exclusivity, level 3 power charging,
/// and immediate track rewards. Returns Ok(true) if advanced, Ok(false) if blocked.
pub fn advance_research_free(
    player: &mut PlayerData,
    field: ResearchField,
    level_5_claimed: &mut [Option<u8>; 6],
) -> Result<bool, ActionError> {
    let cur = player.research_level(field);
    if cur >= 5 {
        return Ok(false);
    }

    // Bal T'aks restriction: cannot advance in Navigation before PI
    if player.faction == Faction::BalTaks
        && field == ResearchField::Navigation
        && player.buildings[Building::PlanetaryInstitute as usize] == 0
    {
        return Err(ActionError::BalTaksNavigationRestricted);
    }

    // Level 5 requires green federation token and exclusivity
    if cur == 4 {
        if level_5_claimed[field as usize].is_some() {
            return Ok(false);
        }
        if !player.flip_federation_token() {
            return Ok(false);
        }
    }

    // Crossing level 2 to level 3 charges 3 power
    if cur == 2 {
        player.power.charge(3);
    }

    let new_level = cur + 1;
    player.research[field as usize] = new_level;

    if new_level == 5 {
        level_5_claimed[field as usize] = Some(player.seat);
    }

    // Immediate rewards per field & level
    match field {
        ResearchField::Terraforming => match new_level {
            1 | 4 => player.add_ore(2),
            5 => player.green_federation_tokens += 1, // Gain federation token on L5
            _ => {}
        },
        ResearchField::Navigation => match new_level {
            1 | 3 => player.add_qic(1),
            _ => {}
        },
        ResearchField::Intelligence => match new_level {
            1 | 2 => player.add_qic(1),
            3 | 4 => player.add_qic(2),
            5 => player.add_qic(4),
            _ => {}
        },
        ResearchField::GaiaProject => match new_level {
            1 | 3 | 4 => player.gaiaformers_unlocked += 1,
            2 => player.power.area1 += 3,
            5 => player.add_victory_points(4),
            _ => {}
        },
        ResearchField::Economy => {
            if new_level == 5 {
                player.add_ore(3);
                player.add_credits(6);
                player.power.charge(6);
            }
        }
        ResearchField::Science => {
            if new_level == 5 {
                player.add_knowledge(9);
            }
        }
    }

    Ok(true)
}

/// Validates and executes advancing one level in `field`.
pub fn execute_advance_research(
    player: &mut PlayerData,
    field: ResearchField,
    level_5_claimed: &mut [Option<u8>; 6],
) -> Result<(), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    let cur = player.research_level(field);
    if cur >= 5 {
        return Err(ActionError::ResearchMaxReached(field));
    }

    // Bal T'aks restriction: cannot advance in Navigation before PI
    if player.faction == Faction::BalTaks
        && field == ResearchField::Navigation
        && player.buildings[Building::PlanetaryInstitute as usize] == 0
    {
        return Err(ActionError::BalTaksNavigationRestricted);
    }

    // Level 5 requires green federation token and exclusivity
    if cur == 4 {
        if level_5_claimed[field as usize].is_some() {
            return Err(ActionError::ResearchLevel5AlreadyClaimed(field));
        }
        if player.green_federation_tokens == 0 {
            return Err(ActionError::FederationTokenRequiredForLevel5);
        }
    }

    if player.knowledge < 4 {
        return Err(ActionError::InsufficientKnowledge {
            needed: 4,
            have: player.knowledge,
        });
    }
    player.knowledge -= 4;

    advance_research_free(player, field, level_5_claimed)?;

    Ok(())
}

/// Executes the Pass action: computes booster VP, updates booster, and marks player as passed.
pub fn execute_pass(
    player: &mut PlayerData,
    new_booster: Option<u8>,
) -> Result<i16, ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }

    let mut pass_vp = 0i16;

    // Compute VP from returning round booster
    if let Some(booster) = player.current_booster {
        match booster {
            6 => {
                // 1 VP per mine
                pass_vp += player.buildings[Building::Mine as usize] as i16;
            }
            7 => {
                // 3 VP per research lab
                pass_vp += (player.buildings[Building::ResearchLab as usize] as i16) * 3;
            }
            8 => {
                // 2 VP per trading station
                pass_vp += (player.buildings[Building::TradingStation as usize] as i16) * 2;
            }
            9 => {
                // 4 VP per PI and Academy
                let count = player.buildings[Building::PlanetaryInstitute as usize]
                    + player.buildings[Building::Academy1 as usize]
                    + player.buildings[Building::Academy2 as usize];
                pass_vp += (count as i16) * 4;
            }
            _ => {}
        }
    }

    player.add_victory_points(pass_vp);
    player.current_booster = new_booster;
    player.passed = true;

    Ok(pass_vp)
}

/// Computes the cost to start a Gaia project on `coord`.
pub fn calculate_start_gaia_project_cost(
    player: &PlayerData,
    map: &Map,
    coord: HexCoord,
) -> Result<(usize, u8, i16), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if player.available_gaiaformers() == 0 {
        return Err(ActionError::NoGaiaformerAvailable);
    }
    let power_cost = player
        .gaia_power_cost()
        .ok_or(ActionError::GaiaProjectResearchRequired)?;

    let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
    let hex = map.hexes[hex_idx];

    if hex.planet != Planet::Transdim {
        return Err(ActionError::NotTransdimPlanet);
    }
    if hex.building.is_some() {
        return Err(ActionError::HexAlreadyOccupied);
    }

    let dist = match map.min_distance_from_player(player.seat, hex_idx) {
        Some(d) => d,
        None => return Err(ActionError::UnreachableHex),
    };

    let base_range = player.effective_range();
    let qic_for_dist = qic_for_distance(dist, base_range, 0) as i16;

    if player.qic < qic_for_dist {
        return Err(ActionError::InsufficientQic {
            needed: qic_for_dist,
            have: player.qic,
        });
    }

    let total_power = player.power.area1 + player.power.area2 + player.power.area3;
    if total_power < power_cost {
        return Err(ActionError::InsufficientPowerForGaiaProject {
            needed: power_cost,
            have: total_power,
        });
    }

    Ok((hex_idx, power_cost, qic_for_dist))
}

/// Executes starting a Gaia project on a Transdim planet.
pub fn execute_start_gaia_project(
    player: &mut PlayerData,
    map: &mut Map,
    coord: HexCoord,
) -> Result<(), ActionError> {
    let (hex_idx, power_cost, qic_cost) = calculate_start_gaia_project_cost(player, map, coord)?;

    player.qic -= qic_cost;
    let moved = player.move_power_to_gaia(power_cost);
    debug_assert!(moved);
    player.gaiaformers_in_gaia += 1;

    let hex = &mut map.hexes[hex_idx];
    hex.building = Some(Building::GaiaFormer);
    hex.player = Some(player.seat);

    Ok(())
}

/// Resolves Phase II (Gaia Phase) of the round:
/// 1. All Transdim planets with a Gaiaformer transform into Gaia planets.
/// 2. Power tokens in each player's Gaia area return to Area 1 (or Area 2 for Terrans).
pub fn resolve_gaia_phase(map: &mut Map, players: &mut [PlayerData]) {
    for hex in &mut map.hexes[..map.count] {
        if hex.planet == Planet::Transdim && hex.building == Some(Building::GaiaFormer) {
            hex.planet = Planet::Gaia;
        }
    }
    for player in players {
        let gaia_power = player.power.gaia;
        if gaia_power > 0 {
            player.power.gaia = 0;
            if player.faction == Faction::Terrans {
                player.power.area2 += gaia_power;
            } else {
                player.power.area1 += gaia_power;
            }
        }
    }
}

/// Validates forming a federation with `planets` and connecting `satellites`.
pub fn calculate_form_federation(
    player: &PlayerData,
    map: &Map,
    planets: &[HexCoord],
    satellites: &[HexCoord],
) -> Result<(u8, u8), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if planets.is_empty() {
        return Err(ActionError::PlanetsNotConnected);
    }

    let has_pi = player.buildings[Building::PlanetaryInstitute as usize] > 0;
    let home_planet = faction_planet(player.faction);

    let mut total_power = 0u8;
    let mut planet_indices = [0usize; 8];
    if planets.len() > planet_indices.len() {
        return Err(ActionError::PlanetsNotConnected);
    }

    for (i, &coord) in planets.iter().enumerate() {
        let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
        let hex = map.hexes[hex_idx];
        if hex.belongs_to_federation_of(player.seat) {
            return Err(ActionError::PlanetAlreadyFederated);
        }
        let building = hex.building_of(player.seat).ok_or(ActionError::NotYourStructure)?;
        total_power += building_power_value(
            building,
            player.faction,
            has_pi,
            hex.planet == home_planet,
        );
        planet_indices[i] = hex_idx;
    }

    let required_power = if player.faction == Faction::Xenos && has_pi {
        6
    } else if player.faction == Faction::Ivits {
        7 * (player.total_federation_tokens() + 1)
    } else {
        7
    };

    if total_power < required_power {
        return Err(ActionError::InsufficientFederationPower {
            needed: required_power,
            have: total_power,
        });
    }

    let sat_count = satellites.len() as u8;
    if sat_count > player.available_satellites() {
        return Err(ActionError::InsufficientSatellites {
            needed: sat_count,
            available: player.available_satellites(),
        });
    }

    let mut sat_indices = [0usize; 16];
    if satellites.len() > sat_indices.len() {
        return Err(ActionError::InsufficientSatellites {
            needed: sat_count,
            available: player.available_satellites(),
        });
    }
    for (i, &coord) in satellites.iter().enumerate() {
        let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
        let hex = map.hexes[hex_idx];
        if hex.has_planet() {
            return Err(ActionError::InvalidSatellitePlacement);
        }
        if hex.belongs_to_federation_of(player.seat) {
            return Err(ActionError::InvalidSatellitePlacement);
        }
        sat_indices[i] = hex_idx;
    }

    if player.faction == Faction::Ivits {
        if player.qic < sat_count as i16 {
            return Err(ActionError::InsufficientQic {
                needed: sat_count as i16,
                have: player.qic,
            });
        }
    } else {
        let total_bowl = player.power.area1 + player.power.area2 + player.power.area3;
        if total_bowl < sat_count {
            return Err(ActionError::InsufficientPowerForSatellites {
                needed: sat_count,
                have: total_bowl,
            });
        }
    }

    // Graph connectivity verification via BFS
    let total_nodes = planets.len() + satellites.len();
    let mut all_indices = [0usize; 24];
    all_indices[..planets.len()].copy_from_slice(&planet_indices[..planets.len()]);
    all_indices[planets.len()..total_nodes].copy_from_slice(&sat_indices[..satellites.len()]);

    let mut visited = [false; 24];
    let mut queue = [0usize; 24];
    let mut head = 0;
    let mut tail = 0;

    visited[0] = true;
    queue[tail] = 0;
    tail += 1;

    while head < tail {
        let cur = queue[head];
        head += 1;
        let cur_hex = all_indices[cur];

        for next in 0..total_nodes {
            if !visited[next] {
                let next_hex = all_indices[next];
                if map.distance(cur_hex, next_hex) == 1 {
                    visited[next] = true;
                    queue[tail] = next;
                    tail += 1;
                }
            }
        }
    }

    if visited[..total_nodes].iter().any(|&v| !v) {
        return Err(ActionError::PlanetsNotConnected);
    }

    Ok((total_power, sat_count))
}

/// Executes forming a federation: discards power for satellites, marks hexes, and awards federation token.
pub fn execute_form_federation(
    player: &mut PlayerData,
    map: &mut Map,
    planets: &[HexCoord],
    satellites: &[HexCoord],
    token: FederationToken,
) -> Result<(), ActionError> {
    let (_, sat_count) = calculate_form_federation(player, map, planets, satellites)?;

    if player.faction == Faction::Ivits {
        player.qic -= sat_count as i16;
    } else {
        let discarded = player.discard_power(sat_count);
        debug_assert!(discarded);
    }
    player.satellites += sat_count;

    for &coord in planets {
        let hex_idx = map.index_of(coord).unwrap();
        map.hexes[hex_idx].add_to_federation(player.seat);
    }
    for &coord in satellites {
        let hex_idx = map.index_of(coord).unwrap();
        map.hexes[hex_idx].add_to_federation(player.seat);
    }

    player.claim_federation_token(token);
    Ok(())
}

/// Finds the minimal set of candidate planets and satellites needed to form a valid federation.
/// Returns Ok((planets, satellites)) or an ActionError explaining why forming a federation is impossible.
pub fn find_minimal_federation(
    player: &PlayerData,
    map: &Map,
) -> Result<(Vec<HexCoord>, Vec<HexCoord>), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }

    let has_pi = player.buildings[Building::PlanetaryInstitute as usize] > 0;
    let home_planet = faction_planet(player.faction);
    let required_power = if player.faction == Faction::Xenos && has_pi {
        6
    } else if player.faction == Faction::Ivits {
        7 * (player.total_federation_tokens() + 1)
    } else {
        7
    };

    let mut candidate_planets: Vec<(usize, u8)> = Vec::new();
    let mut total_cand_power = 0u8;

    for i in 0..map.count {
        let hex = map.hexes[i];
        if hex.belongs_to_federation_of(player.seat) {
            continue;
        }
        if let Some(building) = hex.building_of(player.seat) {
            let p_val = building_power_value(
                building,
                player.faction,
                has_pi,
                hex.planet == home_planet,
            );
            candidate_planets.push((i, p_val));
            total_cand_power += p_val;
        }
    }

    if total_cand_power < required_power {
        return Err(ActionError::InsufficientFederationPower {
            needed: required_power,
            have: total_cand_power,
        });
    }

    // 1. Check if any connected component of candidate planets has power >= required_power with 0 satellites
    let n = candidate_planets.len();
    let mut visited = vec![false; n];
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut comp_power = 0u8;
        let mut q = std::collections::VecDeque::new();
        q.push_back(start);
        visited[start] = true;

        while let Some(u) = q.pop_front() {
            comp.push(candidate_planets[u].0);
            comp_power += candidate_planets[u].1;
            for v in 0..n {
                if !visited[v] && map.distance(candidate_planets[u].0, candidate_planets[v].0) == 1 {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        if comp_power >= required_power {
            let mut sub_planets: Vec<(usize, u8)> = comp.iter().map(|&idx| {
                let p_val = candidate_planets.iter().find(|&&(c, _)| c == idx).map(|&(_, v)| v).unwrap_or(1);
                (idx, p_val)
            }).collect();
            let mut cur_p = comp_power;

            let mut pruned = true;
            while pruned && cur_p > required_power {
                pruned = false;
                for i in 0..sub_planets.len() {
                    let cand_val = sub_planets[i].1;
                    if cur_p.saturating_sub(cand_val) >= required_power {
                        let remaining: Vec<usize> = sub_planets.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, p)| p.0).collect();
                        if !remaining.is_empty() {
                            let mut test_vis = vec![false; remaining.len()];
                            let mut test_q = std::collections::VecDeque::new();
                            test_q.push_back(0);
                            test_vis[0] = true;
                            while let Some(tu) = test_q.pop_front() {
                                for tv in 0..remaining.len() {
                                    if !test_vis[tv] && map.distance(remaining[tu], remaining[tv]) == 1 {
                                        test_vis[tv] = true;
                                        test_q.push_back(tv);
                                    }
                                }
                            }
                            if test_vis.iter().all(|&v| v) {
                                cur_p -= cand_val;
                                sub_planets.remove(i);
                                pruned = true;
                                break;
                            }
                        }
                    }
                }
            }

            let planets = sub_planets.into_iter().map(|(idx, _)| map.coords[idx]).collect();
            return Ok((planets, Vec::new()));
        }
    }

    // 2. We need satellites to bridge separated candidate planets.
    let max_sats = if player.faction == Faction::Ivits {
        (player.available_satellites() as usize).min(player.qic.max(0) as usize)
    } else {
        let total_bowl = (player.power.area1 + player.power.area2 + player.power.area3) as usize;
        (player.available_satellites() as usize).min(total_bowl)
    };

    if max_sats == 0 {
        return Err(ActionError::InsufficientSatellites {
            needed: 1,
            available: 0,
        });
    }

    // Helper: is a hex valid as a satellite?
    let is_valid_satellite_hex = |hex_idx: usize| -> bool {
        let hex = map.hexes[hex_idx];
        !hex.has_planet() && !hex.belongs_to_federation_of(player.seat)
    };

    // Helper: check if hex is candidate planet
    let is_candidate_planet = |hex_idx: usize| -> Option<u8> {
        candidate_planets.iter().find(|&&(idx, _)| idx == hex_idx).map(|&(_, p)| p)
    };

    let mut best_solution: Option<(Vec<HexCoord>, Vec<HexCoord>)> = None;
    let mut min_sat_count = usize::MAX;

    // Try growing a tree from each candidate planet
    for root_idx in 0..n {
        let root_hex = candidate_planets[root_idx].0;

        let mut tree_planets = vec![root_hex];
        let mut tree_satellites: Vec<usize> = Vec::new();
        let mut tree_hexes = std::collections::HashSet::new();
        tree_hexes.insert(root_hex);
        let mut cur_power = candidate_planets[root_idx].1;

        while cur_power < required_power {
            // Multi-source BFS from all hexes currently in the tree
            // To find the closest unadded candidate planet
            let mut dist = vec![usize::MAX; map.count];
            let mut parent = vec![None; map.count];
            let mut q = std::collections::VecDeque::new();

            for &th in &tree_hexes {
                dist[th] = 0;
                q.push_back(th);
            }

            let mut target_planet = None;

            while let Some(u) = q.pop_front() {
                // If u is a candidate planet not yet in tree, we reached it!
                if !tree_hexes.contains(&u) && is_candidate_planet(u).is_some() {
                    target_planet = Some(u);
                    break;
                }

                // Expand to adjacent hexes
                for v in 0..map.count {
                    if map.distance(u, v) == 1 {
                        let is_cand = is_candidate_planet(v).is_some();
                        let is_sat = is_valid_satellite_hex(v);

                        if is_cand || is_sat {
                            let new_cost = dist[u] + if is_cand { 0 } else { 1 };
                            if new_cost < dist[v] {
                                dist[v] = new_cost;
                                parent[v] = Some(u);
                                q.push_back(v);
                            }
                        }
                    }
                }
            }

            let Some(tp) = target_planet else {
                break; // Could not reach another candidate planet
            };

            // Trace path from target_planet back to tree_hexes
            let mut curr = tp;
            let mut new_sats = Vec::new();
            while !tree_hexes.contains(&curr) {
                if is_candidate_planet(curr).is_none() {
                    new_sats.push(curr);
                }
                if let Some(p) = parent[curr] {
                    curr = p;
                } else {
                    break;
                }
            }

            // Add new satellites and the target planet to the tree
            for &s in &new_sats {
                if !tree_hexes.contains(&s) {
                    tree_hexes.insert(s);
                    tree_satellites.push(s);
                }
            }
            tree_hexes.insert(tp);
            tree_planets.push(tp);
            cur_power += is_candidate_planet(tp).unwrap_or(0);
        }

        if cur_power >= required_power
            && tree_satellites.len() <= max_sats
            && tree_satellites.len() <= 16
            && tree_planets.len() <= 8
        {
            if tree_satellites.len() < min_sat_count {
                min_sat_count = tree_satellites.len();
                let planets = tree_planets.into_iter().map(|idx| map.coords[idx]).collect();
                let sats = tree_satellites.into_iter().map(|idx| map.coords[idx]).collect();
                best_solution = Some((planets, sats));
            }
        }
    }

    if let Some((planets, satellites)) = best_solution {
        Ok((planets, satellites))
    } else {
        Err(ActionError::PlanetsNotConnected)
    }
}

pub fn execute_form_federation_auto(
    player: &mut PlayerData,
    map: &mut Map,
    token: FederationToken,
) -> Result<(), ActionError> {
    let (planets, satellites) = find_minimal_federation(player, map)?;
    execute_form_federation(player, map, &planets, &satellites, token)
}

/// Executes a free conversion action for a player (no turn consumed).
/// Applies the resource cost and grants the income defined by the conversion rule.
pub fn execute_free_action(
    player: &mut PlayerData,
    action: crate::rules::FreeAction,
) -> Result<(), ActionError> {
    use crate::rules::{free_action_conversion, GameResource};
    let rule = free_action_conversion(action);

    // Verify and spend costs
    for cost_item in rule.cost {
        match cost_item.resource {
            GameResource::ChargePower => {
                if !player.power.spend(cost_item.amount as u8) {
                    return Err(ActionError::InsufficientPowerForBoardAction {
                        needed: cost_item.amount as u8,
                        have: player.power.spendable_power(),
                    });
                }
            }
            GameResource::Ore => {
                if player.ore < cost_item.amount {
                    return Err(ActionError::InsufficientOre {
                        needed: cost_item.amount,
                        have: player.ore,
                    });
                }
                player.ore -= cost_item.amount;
            }
            GameResource::Credit => {
                if player.credits < cost_item.amount {
                    return Err(ActionError::InsufficientCredits {
                        needed: cost_item.amount,
                        have: player.credits,
                    });
                }
                player.credits -= cost_item.amount;
            }
            GameResource::Knowledge => {
                if player.knowledge < cost_item.amount {
                    return Err(ActionError::InsufficientKnowledge {
                        needed: cost_item.amount,
                        have: player.knowledge,
                    });
                }
                player.knowledge -= cost_item.amount;
            }
            GameResource::Qic => {
                if player.qic < cost_item.amount {
                    return Err(ActionError::InsufficientQic {
                        needed: cost_item.amount,
                        have: player.qic,
                    });
                }
                player.qic -= cost_item.amount;
            }
            GameResource::GaiaToken => {
                // Gaia tokens are tracked in power.gaia
                if (player.power.gaia as i16) < cost_item.amount {
                    return Err(ActionError::InsufficientPowerForBoardAction {
                        needed: cost_item.amount as u8,
                        have: player.power.gaia,
                    });
                }
                player.power.gaia -= cost_item.amount as u8;
            }
            GameResource::GaiaFormer => {
                if player.available_gaiaformers() < cost_item.amount as u8 {
                    return Err(ActionError::NoGaiaformerAvailable);
                }
                // Consume one gaiaformer (move to gaia project area)
                player.gaiaformers_in_gaia += cost_item.amount as u8;
            }
            GameResource::PowerToken => {
                // Power token cost (area 1 discard)
                if player.power.area1 < cost_item.amount as u8 {
                    return Err(ActionError::InsufficientPowerForBoardAction {
                        needed: cost_item.amount as u8,
                        have: player.power.area1,
                    });
                }
                player.power.area1 -= cost_item.amount as u8;
            }
            _ => {}
        }
    }

    // Apply income
    for income_item in rule.income {
        match income_item.resource {
            GameResource::Ore => player.add_ore(income_item.amount),
            GameResource::Credit => player.add_credits(income_item.amount),
            GameResource::Knowledge => player.add_knowledge(income_item.amount),
            GameResource::Qic => player.add_qic(income_item.amount),
            GameResource::VictoryPoint => player.add_victory_points(income_item.amount),
            GameResource::ChargePower => {
                player.power.charge(income_item.amount as u8);
            }
            GameResource::PowerToken => {
                player.power.area1 += income_item.amount as u8;
            }
            GameResource::TechTile => {
                // Mark that a tech tile must be claimed
                player.pending_tech_claim = true;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Opportunity for an opponent to charge power (passive action) when an actor builds or upgrades.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeechOpportunity {
    pub seat: u8,
    pub power_value: u8,
    pub vp_cost: i16,
}

/// Finds all opponents within distance <= 2 that have eligible structures to charge power.
pub fn find_leech_opportunities(
    map: &Map,
    coord: HexCoord,
    actor_seat: u8,
    players: &[PlayerData],
) -> [Option<LeechOpportunity>; 4] {
    let mut opps = [None; 4];
    let target_idx = match map.index_of(coord) {
        Some(idx) => idx,
        None => return opps,
    };

    for seat in 0..players.len().min(4) as u8 {
        if seat == actor_seat {
            continue;
        }
        let player = &players[seat as usize];
        let has_pi = player.buildings[Building::PlanetaryInstitute as usize] > 0;
        let home = faction_planet(player.faction);

        let mut max_val = 0u8;
        for j in 0..map.count {
            if map.distance(target_idx, j) <= 2 {
                let hex = map.hexes[j];
                if let Some(building) = hex.building_of(seat) {
                    let val = building_power_value(
                        building,
                        player.faction,
                        has_pi,
                        hex.planet == home,
                    );
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
        }
        if max_val > 0 {
            let vp_cost = (max_val as i16 - 1).max(0);
            opps[seat as usize] = Some(LeechOpportunity {
                seat,
                power_value: max_val,
                vp_cost,
            });
        }
    }
    opps
}

/// Executes a leeching decision for a player: spends VP (if > 1 power gained) and charges power.
/// Follows Gaia Project official rule: VP is only deducted for power actually gained,
/// and never if the player cannot charge power (or gains 0 power).
pub fn execute_leech(
    player: &mut PlayerData,
    charge_amount: u8,
) -> Result<(), ActionError> {
    if charge_amount == 0 || !player.power.can_charge() {
        return Ok(());
    }

    // Taklons PI: each time charging from passive leech, gain 1 power token
    if player.faction == Faction::Taklons && player.buildings[Building::PlanetaryInstitute as usize] > 0 {
        player.power.area1 += 1;
    }

    // Maximum power player can afford based on VP (0 VP allows charging 1 power for 0 VP cost)
    let max_affordable = if player.victory_points <= 0 {
        1u8
    } else {
        (player.victory_points as u8).saturating_add(1)
    };

    let target = charge_amount.min(max_affordable);
    let (charged, _wasted) = player.power.charge(target);

    if charged > 0 {
        let vp_cost = (charged as i16 - 1).max(0);
        player.victory_points = player.victory_points.saturating_sub(vp_cost);
    }

    Ok(())
}

/// Executes a board power or Q.I.C. action (Action 6).
pub fn execute_board_action(
    player: &mut PlayerData,
    map: &mut Map,
    claimed_actions: &mut [Option<u8>; 10],
    action: BoardAction,
    target_mine: Option<HexCoord>,
    rescore_token: Option<FederationToken>,
) -> Result<(), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if claimed_actions[action as usize].is_some() {
        return Err(ActionError::BoardActionAlreadyClaimed(action));
    }

    if BoardAction::POWER_ACTIONS.contains(&action) {
        let cost = action.power_cost();
        if player.power.spendable_power() < cost {
            return Err(ActionError::InsufficientPowerForBoardAction {
                needed: cost,
                have: player.power.spendable_power(),
            });
        }
        player.power.spend(cost);
    } else if BoardAction::QIC_ACTIONS.contains(&action) {
        let cost = action.qic_cost() as i16;
        if player.qic < cost {
            return Err(ActionError::InsufficientQic {
                needed: cost,
                have: player.qic,
            });
        }
        player.qic -= cost;
    }

    claimed_actions[action as usize] = Some(player.seat);

    match action {
        BoardAction::Power1 => {
            player.add_knowledge(3);
        }
        BoardAction::Power2 => {
            if let Some(coord) = target_mine {
                player.temporary_step = 2;
                let res = execute_build_mine(player, map, coord);
                player.temporary_step = 0;
                res?;
            } else {
                player.temporary_step = 2;
            }
        }
        BoardAction::Power3 => {
            player.add_ore(2);
        }
        BoardAction::Power4 => {
            player.add_credits(7);
        }
        BoardAction::Power5 => {
            player.add_knowledge(2);
        }
        BoardAction::Power6 => {
            if let Some(coord) = target_mine {
                player.temporary_step = 1;
                let res = execute_build_mine(player, map, coord);
                player.temporary_step = 0;
                res?;
            } else {
                player.temporary_step = 1;
            }
        }
        BoardAction::Power7 => {
            player.power.area1 += 2;
        }
        BoardAction::Qic1 => {}
        BoardAction::Qic2 => {
            let token = rescore_token
                .or_else(|| player.claimed_federations.iter().filter_map(|&t| t).next())
                .ok_or(ActionError::NoFederationToRescore)?;
            let rew = token.reward();
            player.add_victory_points(rew.vp);
            player.add_credits(rew.credits);
            player.add_ore(rew.ore);
            player.add_knowledge(rew.knowledge);
            player.add_qic(rew.qic);
            player.power.area1 += rew.power_tokens as u8;
        }
        BoardAction::Qic3 => {
            let distinct = map.distinct_planets_colonized_by(player.seat);
            player.add_victory_points(3 + distinct as i16);
        }
    }

    Ok(())
}

/// Executes once-per-round special faction or tech tile actions (Action 7).
pub fn execute_special_action(
    player: &mut PlayerData,
    map: &mut Map,
    action: SpecialAction,
    target_coord: Option<HexCoord>,
    target_field: Option<ResearchField>,
    claimed_l5: &mut [Option<u8>; 6],
) -> Result<(), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if player.special_actions_used[action as usize] {
        return Err(ActionError::SpecialActionAlreadyUsed(action));
    }

    match action {
        SpecialAction::AmbasPiSwap => {
            if player.faction != Faction::Ambas
                || player.buildings[Building::PlanetaryInstitute as usize] == 0
            {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            let hex_idx = match target_coord {
                Some(coord) => map.index_of(coord).ok_or(ActionError::HexNotFound)?,
                None => {
                    let mut cand = None;
                    for (idx, hex) in map.hexes[..map.count].iter().enumerate() {
                        if hex.player == Some(player.seat) && hex.building == Some(Building::Mine) {
                            cand = Some(idx);
                            break;
                        }
                    }
                    cand.ok_or(ActionError::NotYourStructure)?
                }
            };
            if map.hexes[hex_idx].player != Some(player.seat)
                || map.hexes[hex_idx].building != Some(Building::Mine)
            {
                return Err(ActionError::NotYourStructure);
            }
            let mut pi_idx = None;
            for (idx, hex) in map.hexes[..map.count].iter().enumerate() {
                if hex.player == Some(player.seat)
                    && hex.building == Some(Building::PlanetaryInstitute)
                {
                    pi_idx = Some(idx);
                    break;
                }
            }
            let pi_idx = pi_idx.ok_or(ActionError::NotYourStructure)?;
            map.hexes[pi_idx].building = Some(Building::Mine);
            map.hexes[hex_idx].building = Some(Building::PlanetaryInstitute);
        }
        SpecialAction::FiraksDowngradeLab => {
            if player.faction != Faction::Firaks
                || player.buildings[Building::PlanetaryInstitute as usize] == 0
            {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            let hex_idx = match target_coord {
                Some(coord) => map.index_of(coord).ok_or(ActionError::HexNotFound)?,
                None => {
                    let mut cand = None;
                    for (idx, hex) in map.hexes[..map.count].iter().enumerate() {
                        if hex.player == Some(player.seat) && hex.building == Some(Building::ResearchLab) {
                            cand = Some(idx);
                            break;
                        }
                    }
                    cand.ok_or(ActionError::NotYourStructure)?
                }
            };
            if map.hexes[hex_idx].player != Some(player.seat)
                || map.hexes[hex_idx].building != Some(Building::ResearchLab)
            {
                return Err(ActionError::NotYourStructure);
            }
            if player.buildings_available(Building::TradingStation) == 0 {
                return Err(ActionError::BuildingMaxReached(Building::TradingStation));
            }
            map.hexes[hex_idx].building = Some(Building::TradingStation);
            player.buildings[Building::ResearchLab as usize] -= 1;
            player.buildings[Building::TradingStation as usize] += 1;
            let field = target_field.unwrap_or_else(|| {
                let mut best_f = ResearchField::Terraforming;
                let mut min_l = 255;
                for &f in &[
                    ResearchField::Terraforming,
                    ResearchField::Navigation,
                    ResearchField::Intelligence,
                    ResearchField::GaiaProject,
                    ResearchField::Economy,
                    ResearchField::Science,
                ] {
                    let l = player.research_level(f);
                    if l < 5 && l < min_l {
                        min_l = l;
                        best_f = f;
                    }
                }
                best_f
            });
            let _ = advance_research_free(player, field, claimed_l5);
        }
        SpecialAction::BescodsAdvanceLowest => {
            if player.faction != Faction::Bescods {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            let min_lvl = *player.research.iter().min().unwrap_or(&0);
            if min_lvl >= 5 {
                return Err(ActionError::ResearchMaxReached(ResearchField::Terraforming));
            }
            let chosen_field = target_field.unwrap_or_else(|| {
                [
                    ResearchField::Terraforming,
                    ResearchField::Navigation,
                    ResearchField::Intelligence,
                    ResearchField::GaiaProject,
                    ResearchField::Economy,
                    ResearchField::Science,
                ]
                .into_iter()
                .find(|&f| player.research_level(f) == min_lvl)
                .unwrap_or(ResearchField::Terraforming)
            });
            if player.research_level(chosen_field) == min_lvl {
                let _ = advance_research_free(player, chosen_field, claimed_l5);
            }
        }
        SpecialAction::IvitsSpaceStation => {
            if player.faction != Faction::Ivits
                || player.buildings[Building::PlanetaryInstitute as usize] == 0
            {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            if player.buildings_available(Building::SpaceStation) == 0 {
                return Err(ActionError::BuildingMaxReached(Building::SpaceStation));
            }
            let hex_idx = match target_coord {
                Some(coord) => map.index_of(coord).ok_or(ActionError::HexNotFound)?,
                None => {
                    let mut cand = None;
                    let mut best_dist = 255;
                    for (idx, hex) in map.hexes[..map.count].iter().enumerate() {
                        if hex.planet == Planet::Empty && !hex.has_structure() && hex.building.is_none() {
                            if let Some(d) = map.min_distance_from_player(player.seat, idx) {
                                if d < best_dist {
                                    best_dist = d;
                                    cand = Some(idx);
                                    if d == 1 { break; }
                                }
                            }
                        }
                    }
                    cand.ok_or(ActionError::HexNotColonizable)?
                }
            };
            if map.hexes[hex_idx].planet != Planet::Empty || map.hexes[hex_idx].has_structure() {
                return Err(ActionError::HexNotColonizable);
            }
            map.hexes[hex_idx].building = Some(Building::SpaceStation);
            map.hexes[hex_idx].player = Some(player.seat);
            player.buildings[Building::SpaceStation as usize] += 1;
        }
        SpecialAction::SpaceGiantsTerraform => {
            if player.faction != Faction::SpaceGiants {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            let coord = match target_coord {
                Some(c) => c,
                None => {
                    let mut best_c = None;
                    for i in 0..map.count {
                        let c = map.coords[i];
                        player.temporary_step = 2;
                        let ok = calculate_build_mine_cost(player, map, c).is_ok();
                        player.temporary_step = 0;
                        if ok {
                            best_c = Some(c);
                            break;
                        }
                    }
                    best_c.ok_or(ActionError::HexNotFound)?
                }
            };
            player.temporary_step = 2;
            let res = execute_build_mine(player, map, coord);
            player.temporary_step = 0;
            res?;
        }
        SpecialAction::Tech9Charge4Power => {
            if !player.tech_tiles[TechTile::Tech9 as usize]
                || player.covered_tech_tiles[TechTile::Tech9 as usize]
            {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            player.power.charge(4);
        }
        SpecialAction::AdvTech3QicCredit => {
            if !player.adv_tech_tiles[AdvTechTile::AdvTech3 as usize] {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            player.add_qic(1);
            player.add_credits(5);
        }
        SpecialAction::AdvTech11Gain3Ore => {
            if !player.adv_tech_tiles[AdvTechTile::AdvTech11 as usize] {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            player.add_ore(3);
        }
        SpecialAction::AdvTech13Gain3Knowledge => {
            if !player.adv_tech_tiles[AdvTechTile::AdvTech13 as usize] {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
            player.add_knowledge(3);
        }
        SpecialAction::Booster5TemporaryRange => {
            if player.current_booster != Some(5) {
                return Err(ActionError::SpecialActionUnavailable(action));
            }
        }
    }

    player.special_actions_used[action as usize] = true;
    Ok(())
}

/// Claims a standard tech tile for a player.
pub fn execute_claim_tech_tile(
    player: &mut PlayerData,
    tech: TechTile,
    advance_field: Option<ResearchField>,
    claimed_l5: &mut [Option<u8>; 6],
) -> Result<(), ActionError> {
    if player.tech_tiles[tech as usize] {
        return Err(ActionError::TechTileAlreadyOwned(tech));
    }
    player.tech_tiles[tech as usize] = true;
    player.pending_tech_claim = false;

    match tech {
        TechTile::Tech1 => {
            player.add_ore(1);
            player.add_qic(1);
        }
        TechTile::Tech4 => {
            player.add_victory_points(7);
        }
        _ => {}
    }

    if let Some(field) = advance_field {
        let _ = advance_research_free(player, field, claimed_l5);
    }

    Ok(())
}

/// Claims an advanced tech tile, covering a standard tech tile.
pub fn execute_claim_adv_tech_tile(
    player: &mut PlayerData,
    map: &Map,
    adv_tech: AdvTechTile,
    cover_tech: TechTile,
    field: ResearchField,
    claimed_adv: &mut [bool; 15],
) -> Result<(), ActionError> {
    if claimed_adv[adv_tech as usize] {
        return Err(ActionError::AdvTechTileAlreadyClaimed(adv_tech));
    }
    if !player.tech_tiles[cover_tech as usize] || player.covered_tech_tiles[cover_tech as usize] {
        return Err(ActionError::AdvTechTileRequirementsNotMet);
    }
    if player.research_level(field) < 4 {
        return Err(ActionError::AdvTechTileRequirementsNotMet);
    }
    if player.green_federation_tokens == 0 {
        return Err(ActionError::FederationTokenRequiredForLevel5);
    }

    player.flip_federation_token();
    player.covered_tech_tiles[cover_tech as usize] = true;
    player.adv_tech_tiles[adv_tech as usize] = true;
    claimed_adv[adv_tech as usize] = true;

    match adv_tech {
        AdvTechTile::AdvTech4 => {
            let mines = player.buildings[Building::Mine as usize];
            player.add_victory_points(mines as i16 * 2);
        }
        AdvTechTile::AdvTech6 => {
            let sectors = map.sectors_with_player(player.seat);
            player.add_ore(sectors as i16);
        }
        AdvTechTile::AdvTech8 => {
            let gaia = map.gaia_planets_colonized_by(player.seat);
            player.add_victory_points(gaia as i16 * 2);
        }
        AdvTechTile::AdvTech9 => {
            let ts = player.buildings[Building::TradingStation as usize];
            player.add_victory_points(ts as i16 * 4);
        }
        AdvTechTile::AdvTech10 => {
            let sectors = map.sectors_with_player(player.seat);
            player.add_victory_points(sectors as i16 * 2);
        }
        AdvTechTile::AdvTech12 => {
            let feds = player.total_federation_tokens();
            player.add_victory_points(feds as i16 * 5);
        }
        _ => {}
    }

    Ok(())
}

/// Validates and executes deploying an exploration shuttle to a spaceship board.
pub fn execute_explore_spaceship(
    player: &mut PlayerData,
    occupied_slots: &[bool; 5],
    map: &Map,
    ship: Spaceship,
    coord: HexCoord,
    max_shuttles: u8,
) -> Result<u8, ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if player.deployed_shuttles() >= max_shuttles {
        return Err(ActionError::NoExplorationShuttleAvailable);
    }
    if player.has_explored(ship) {
        return Err(ActionError::SpaceshipAlreadyExplored(ship));
    }
    let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
    let hex = map.hexes[hex_idx];
    if hex.spaceship != Some(ship) {
        return Err(ActionError::NotSpaceshipHex);
    }

    // Distance calculation from colonized structures
    let dist = match map.min_distance_from_player(player.seat, hex_idx) {
        Some(d) => d,
        None => return Err(ActionError::UnreachableHex),
    };
    let base_range = player.effective_range();
    let qic_needed = qic_for_distance(dist, base_range, player.temporary_range) as i16;
    if player.qic < qic_needed {
        return Err(ActionError::InsufficientQic {
            needed: qic_needed,
            have: player.qic,
        });
    }

    // Base cost: 7 VP for Bal T'aks, 5 VP for other factions
    let vp_cost = if player.faction == Faction::BalTaks { 7 } else { 5 };
    if player.victory_points < vp_cost {
        return Err(ActionError::InsufficientCredits {
            needed: vp_cost,
            have: player.victory_points,
        });
    }

    // Find next free slot (1..=4) on this ship among all players
    let slot = occupied_slots
        .iter()
        .enumerate()
        .skip(1)
        .find(|&(_, &occupied)| !occupied)
        .map(|(s, _)| s as u8)
        .ok_or(ActionError::SpaceshipNoFreeSlot(ship))?;

    // Deduct costs
    player.qic -= qic_needed;
    player.victory_points -= vp_cost;

    // Record shuttle placement
    player.exploration_ships[ship as usize] = Some(slot);

    // Charge power from exploration track (spaces 1..4 -> index 0..3)
    let charge = EXPLORATION_CHARGE_TRACK[(slot - 1) as usize];
    if charge > 0 {
        player.power.charge(charge);
    }

    Ok(slot)
}

/// Executes a spaceship board action (Action 6 on a spaceship).
#[allow(clippy::too_many_arguments)]
pub fn execute_spaceship_action(
    player: &mut PlayerData,
    map: &mut Map,
    claimed_actions: &mut [[bool; 4]; 4],
    claimed_l5: &mut [Option<u8>; 6],
    ship: Spaceship,
    action_type: SpaceshipActionType,
    target_coord: Option<HexCoord>,
    target_field: Option<ResearchField>,
    rescore_token: Option<FederationToken>,
) -> Result<(), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if !player.has_explored(ship) {
        return Err(ActionError::SpaceshipNotExplored(ship));
    }
    if claimed_actions[ship as usize][action_type as usize] {
        return Err(ActionError::SpaceshipActionAlreadyUsed(ship, action_type));
    }

    match (ship, action_type) {
        (Spaceship::Twilight, SpaceshipActionType::Qic) => {
            if player.qic < 3 {
                return Err(ActionError::InsufficientQic {
                    needed: 3,
                    have: player.qic,
                });
            }
            player.qic -= 3;
            let token = rescore_token
                .or_else(|| player.claimed_federations.iter().filter_map(|&t| t).next())
                .ok_or(ActionError::NoFederationToRescore)?;
            let rew = token.reward();
            player.add_victory_points(rew.vp);
            player.add_credits(rew.credits);
            player.add_ore(rew.ore);
            player.add_knowledge(rew.knowledge);
            player.add_qic(rew.qic);
            player.power.area1 += rew.power_tokens as u8;
        }
        (Spaceship::Twilight, SpaceshipActionType::Power) => {
            if player.power.spendable_power() < 3 {
                return Err(ActionError::InsufficientPowerForBoardAction {
                    needed: 3,
                    have: player.power.spendable_power(),
                });
            }
            if player.ore < 2 {
                return Err(ActionError::InsufficientOre {
                    needed: 2,
                    have: player.ore,
                });
            }
            let coord = target_coord.ok_or(ActionError::HexNotFound)?;
            let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
            if map.hexes[hex_idx].player != Some(player.seat)
                || map.hexes[hex_idx].building != Some(Building::TradingStation)
            {
                return Err(ActionError::NotYourStructure);
            }
            if player.buildings_available(Building::ResearchLab) == 0 {
                return Err(ActionError::BuildingMaxReached(Building::ResearchLab));
            }
            player.power.spend(3);
            player.ore -= 2;
            map.hexes[hex_idx].building = Some(Building::ResearchLab);
            player.buildings[Building::TradingStation as usize] -= 1;
            player.buildings[Building::ResearchLab as usize] += 1;
        }
        (Spaceship::Twilight, SpaceshipActionType::Knowledge) => {
            if player.knowledge < 1 {
                return Err(ActionError::InsufficientKnowledge {
                    needed: 1,
                    have: player.knowledge,
                });
            }
            player.knowledge -= 1;
            player.temporary_range += 3;
        }
        (Spaceship::Rebellion, SpaceshipActionType::Qic) => {
            if player.qic < 3 {
                return Err(ActionError::InsufficientQic {
                    needed: 3,
                    have: player.qic,
                });
            }
            player.qic -= 3;
        }
        (Spaceship::Rebellion, SpaceshipActionType::Power) => {
            if player.power.spendable_power() < 3 {
                return Err(ActionError::InsufficientPowerForBoardAction {
                    needed: 3,
                    have: player.power.spendable_power(),
                });
            }
            if player.ore < 1 {
                return Err(ActionError::InsufficientOre {
                    needed: 1,
                    have: player.ore,
                });
            }
            let coord = target_coord.ok_or(ActionError::HexNotFound)?;
            let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
            if map.hexes[hex_idx].player != Some(player.seat)
                || map.hexes[hex_idx].building != Some(Building::Mine)
            {
                return Err(ActionError::NotYourStructure);
            }
            if player.buildings_available(Building::TradingStation) == 0 {
                return Err(ActionError::BuildingMaxReached(Building::TradingStation));
            }
            player.power.spend(3);
            player.ore -= 1;
            map.hexes[hex_idx].building = Some(Building::TradingStation);
            player.buildings[Building::Mine as usize] -= 1;
            player.buildings[Building::TradingStation as usize] += 1;
        }
        (Spaceship::Rebellion, SpaceshipActionType::Knowledge) => {
            if player.knowledge < 2 {
                return Err(ActionError::InsufficientKnowledge {
                    needed: 2,
                    have: player.knowledge,
                });
            }
            player.knowledge -= 2;
            player.add_credits(2);
            player.add_qic(1);
        }
        (Spaceship::TFMars, SpaceshipActionType::Qic) => {
            if player.qic < 2 {
                return Err(ActionError::InsufficientQic {
                    needed: 2,
                    have: player.qic,
                });
            }
            player.qic -= 2;
            let tech_count = player.tech_tiles.iter().filter(|&&t| t).count() as i16;
            player.add_victory_points(2 + tech_count);
        }
        (Spaceship::TFMars, SpaceshipActionType::Power) => {
            if player.power.spendable_power() < 2 {
                return Err(ActionError::InsufficientPowerForBoardAction {
                    needed: 2,
                    have: player.power.spendable_power(),
                });
            }
            let coord = target_coord.ok_or(ActionError::HexNotFound)?;
            let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
            if map.hexes[hex_idx].planet != Planet::Transdim {
                return Err(ActionError::NotTransdimPlanet);
            }
            player.power.spend(2);
            map.hexes[hex_idx].planet = Planet::Gaia;
        }
        (Spaceship::TFMars, SpaceshipActionType::Credit) => {
            if player.credits < 3 {
                return Err(ActionError::InsufficientCredits {
                    needed: 3,
                    have: player.credits,
                });
            }
            let coord = target_coord.ok_or(ActionError::HexNotFound)?;
            player.credits -= 3;
            player.temporary_step = 1;
            let res = execute_build_mine(player, map, coord);
            player.temporary_step = 0;
            res?;
        }
        (Spaceship::Eclipse, SpaceshipActionType::Qic) => {
            if player.qic < 2 {
                return Err(ActionError::InsufficientQic {
                    needed: 2,
                    have: player.qic,
                });
            }
            player.qic -= 2;
            let types = map.distinct_planets_colonized_by(player.seat);
            player.add_victory_points(2 + types as i16);
        }
        (Spaceship::Eclipse, SpaceshipActionType::Power) => {
            if player.power.spendable_power() < 3 {
                return Err(ActionError::InsufficientPowerForBoardAction {
                    needed: 3,
                    have: player.power.spendable_power(),
                });
            }
            if player.knowledge < 2 {
                return Err(ActionError::InsufficientKnowledge {
                    needed: 2,
                    have: player.knowledge,
                });
            }
            let field = target_field
                .ok_or(ActionError::ResearchMaxReached(ResearchField::Terraforming))?;
            player.power.spend(3);
            player.knowledge -= 2;
            advance_research_free(player, field, claimed_l5)?;
        }
        (Spaceship::Eclipse, SpaceshipActionType::Credit) => {
            if player.credits < 6 {
                return Err(ActionError::InsufficientCredits {
                    needed: 6,
                    have: player.credits,
                });
            }
            let coord = target_coord.ok_or(ActionError::HexNotFound)?;
            let hex_idx = map.index_of(coord).ok_or(ActionError::HexNotFound)?;
            if map.hexes[hex_idx].planet != Planet::Asteroid || map.hexes[hex_idx].occupied() {
                return Err(ActionError::HexNotColonizable);
            }
            if player.buildings_available(Building::Mine) == 0 {
                return Err(ActionError::NoMinesAvailable);
            }
            let dist = match map.min_distance_from_player(player.seat, hex_idx) {
                Some(d) => d,
                None => return Err(ActionError::UnreachableHex),
            };
            let base_range = player.effective_range();
            let qic_needed = qic_for_distance(dist, base_range, player.temporary_range) as i16;
            if player.qic < qic_needed {
                return Err(ActionError::InsufficientQic {
                    needed: qic_needed,
                    have: player.qic,
                });
            }
            player.credits -= 6;
            player.qic -= qic_needed;
            map.hexes[hex_idx].building = Some(Building::Mine);
            map.hexes[hex_idx].player = Some(player.seat);
            player.buildings[Building::Mine as usize] += 1;
        }
        _ => return Err(ActionError::SpecialActionUnavailable(SpecialAction::AmbasPiSwap)),
    }

    claimed_actions[ship as usize][action_type as usize] = true;
    Ok(())
}

/// Computes final game scoring after round 6.
pub fn compute_final_scores(players: &mut [PlayerData], map: &Map, final_tiles: [FinalTile; 2]) {
    for &tile in &final_tiles {
        let n = players.len();
        let mut scores = [0i32; 4];
        for (i, p) in players.iter().enumerate() {
            scores[i] = match tile {
                FinalTile::Structure => {
                    let b = p.buildings;
                    (b[Building::Mine as usize]
                        + b[Building::TradingStation as usize]
                        + b[Building::ResearchLab as usize]
                        + b[Building::PlanetaryInstitute as usize]
                        + b[Building::Academy1 as usize]
                        + b[Building::Academy2 as usize]) as i32
                }
                FinalTile::StructureFed => {
                    let mut count = 0;
                    for hex in &map.hexes[..map.count] {
                        if hex.colonized_by(p.seat) && hex.belongs_to_federation_of(p.seat) {
                            count += 1;
                        }
                    }
                    count
                }
                FinalTile::PlanetType => map.distinct_planets_colonized_by(p.seat) as i32,
                FinalTile::Gaia => map.gaia_planets_colonized_by(p.seat) as i32,
                FinalTile::Sector => map.sectors_with_player(p.seat) as i32,
                FinalTile::Satellite => p.satellites as i32,
                FinalTile::Asteroid => {
                    map.hexes[..map.count]
                        .iter()
                        .filter(|h| h.planet == Planet::Asteroid && h.colonized_by(p.seat))
                        .count() as i32
                }
                FinalTile::PlanetaryInstituteAcademyDistance => {
                    let mut pi_idx = None;
                    let mut academy_indices = Vec::new();
                    for (idx, hex) in map.hexes[..map.count].iter().enumerate() {
                        if hex.player == Some(p.seat) {
                            if hex.building == Some(Building::PlanetaryInstitute) {
                                pi_idx = Some(idx);
                            } else if matches!(
                                hex.building,
                                Some(Building::Academy1) | Some(Building::Academy2)
                            ) {
                                academy_indices.push(idx);
                            }
                        }
                    }
                    if let Some(pi) = pi_idx {
                        academy_indices
                            .into_iter()
                            .map(|ac| map.distance(pi, ac) as i32)
                            .max()
                            .unwrap_or(0)
                    } else {
                        0
                    }
                }
                FinalTile::DeepSpaceSector => 0,
            };
        }

        let base_awards = [18, 12, 6, 0];
        let mut ranked: Vec<usize> = (0..n).collect();
        ranked.sort_by(|&a, &b| scores[b].cmp(&scores[a]));

        let mut i = 0;
        while i < n {
            if scores[ranked[i]] == 0 {
                break;
            }
            let mut j = i;
            while j < n && scores[ranked[j]] == scores[ranked[i]] {
                j += 1;
            }
            let total_vp: i16 = base_awards[i..j.min(4)].iter().copied().sum();
            let share = total_vp / ((j - i) as i16);
            for k in i..j {
                players[ranked[k]].add_victory_points(share);
            }
            i = j;
        }
    }

    for p in players.iter_mut() {
        let mut research_vp = 0i16;
        for &lvl in &p.research {
            if lvl >= 3 {
                research_vp += ((lvl - 2) * 4) as i16;
            }
        }
        p.add_victory_points(research_vp);
        let total_res = p.credits + p.ore + p.knowledge;
        p.add_victory_points(total_res / 3);
    }
}

/// Enumerates all currently legal game commands for an active player.
/// Designed for high-throughput RL action masking and tree search.
#[allow(clippy::too_many_arguments)]
pub fn legal_commands(
    _player_seat: usize,
    player: &PlayerData,
    map: &Map,
    claimed_l5: &[Option<u8>; 6],
    claimed_board_actions: &[Option<u8>; 10],
    claimed_spaceship_actions: &[[bool; 4]; 4],
    available_boosters: &[u8],
    is_final_round: bool,
) -> Vec<GameCommand> {
    let mut cmds = Vec::with_capacity(32);
    if player.passed {
        return cmds;
    }

    // If player has upgraded to ResearchLab or Academy, they MUST claim a tech tile first
    if player.pending_tech_claim {
        let all_techs = [
            TechTile::Tech1, TechTile::Tech2, TechTile::Tech3,
            TechTile::Tech4, TechTile::Tech5, TechTile::Tech6,
            TechTile::Tech7, TechTile::Tech8, TechTile::Tech9,
        ];
        let all_fields = [
            ResearchField::Terraforming, ResearchField::Navigation, ResearchField::Intelligence,
            ResearchField::GaiaProject, ResearchField::Economy, ResearchField::Science,
        ];
        for &tech in &all_techs {
            if !player.tech_tiles[tech as usize] {
                // Option without research advancement
                cmds.push(GameCommand::ClaimTechTile { tech, advance_field: None });
                // Options with research advancement
                for &field in &all_fields {
                    let cur = player.research_level(field);
                    if cur < 5 {
                        if cur == 4 {
                            if claimed_l5[field as usize].is_none() && player.green_federation_tokens > 0 {
                                cmds.push(GameCommand::ClaimTechTile { tech, advance_field: Some(field) });
                            }
                        } else if !(player.faction == Faction::BalTaks
                            && field == ResearchField::Navigation
                            && player.buildings[Building::PlanetaryInstitute as usize] == 0)
                        {
                            cmds.push(GameCommand::ClaimTechTile { tech, advance_field: Some(field) });
                        }
                    }
                }
            }
        }
        return cmds;
    }

    // 1. Build Mine
    if player.buildings_available(Building::Mine) > 0 {
        for i in 0..map.count {
            let coord = map.coords[i];
            if calculate_build_mine_cost(player, map, coord).is_ok() {
                cmds.push(GameCommand::BuildMine { coord });
            }
        }
    }

    // 2. Start Gaia Project
    if player.available_gaiaformers() > 0 && player.gaia_power_cost().is_some() {
        for i in 0..map.count {
            let coord = map.coords[i];
            if calculate_start_gaia_project_cost(player, map, coord).is_ok() {
                cmds.push(GameCommand::StartGaiaProject { coord });
            }
        }
    }

    // 3. Upgrade Structure
    for i in 0..map.count {
        let hex = map.hexes[i];
        if let Some(building) = hex.building_of(player.seat) {
            let coord = map.coords[i];
            for &to in upgraded_buildings(building, player.faction) {
                if calculate_upgrade_cost(player, map, coord, to).is_ok() {
                    cmds.push(GameCommand::Upgrade { coord, to });
                }
            }
        }
    }

    // 4. Advance Research
    if player.knowledge >= 4 {
        for &field in &[
            ResearchField::Terraforming,
            ResearchField::Navigation,
            ResearchField::Intelligence,
            ResearchField::GaiaProject,
            ResearchField::Economy,
            ResearchField::Science,
        ] {
            let cur = player.research_level(field);
            if cur < 5 {
                if cur == 4 {
                    if claimed_l5[field as usize].is_none() && player.green_federation_tokens > 0 {
                        cmds.push(GameCommand::AdvanceResearch { field });
                    }
                } else if !(player.faction == Faction::BalTaks
                    && field == ResearchField::Navigation
                    && player.buildings[Building::PlanetaryInstitute as usize] == 0)
                {
                    cmds.push(GameCommand::AdvanceResearch { field });
                }
            }
        }
    }

    // 5. Board Actions (Power & QIC)
    for &action in &BoardAction::ALL {
        if claimed_board_actions[action as usize].is_none() {
            if BoardAction::POWER_ACTIONS.contains(&action) {
                if player.power.spendable_power() >= action.power_cost() {
                    cmds.push(GameCommand::BoardAction {
                        action,
                        target_mine: None,
                        rescore_token: None,
                    });
                }
            } else if BoardAction::QIC_ACTIONS.contains(&action)
                && player.qic >= action.qic_cost() as i16
            {
                if action == BoardAction::Qic2 {
                    if player.total_federation_tokens() > 0 {
                        cmds.push(GameCommand::BoardAction {
                            action,
                            target_mine: None,
                            rescore_token: None,
                        });
                    }
                } else {
                    cmds.push(GameCommand::BoardAction {
                        action,
                        target_mine: None,
                        rescore_token: None,
                    });
                }
            }
        }
    }

    // 6. Special Actions
    if player.faction == Faction::Bescods
        && !player.special_actions_used[SpecialAction::BescodsAdvanceLowest as usize]
    {
        cmds.push(GameCommand::SpecialAction {
            action: SpecialAction::BescodsAdvanceLowest,
            target_coord: None,
            target_field: None,
        });
    }
    if player.tech_tiles[TechTile::Tech9 as usize]
        && !player.covered_tech_tiles[TechTile::Tech9 as usize]
        && !player.special_actions_used[SpecialAction::Tech9Charge4Power as usize]
    {
        cmds.push(GameCommand::SpecialAction {
            action: SpecialAction::Tech9Charge4Power,
            target_coord: None,
            target_field: None,
        });
    }

    // 7. Spaceship Exploration
    let vp_cost = if player.faction == Faction::BalTaks { 7 } else { 5 };
    if player.deployed_shuttles() < 3 && player.victory_points >= vp_cost {
        for i in 0..map.count {
            let Some(ship) = map.hexes[i].spaceship else { continue };
            if player.has_explored(ship) {
                continue;
            }
            let Some(dist) = map.min_distance_from_player(player.seat, i) else { continue };
            let qic_needed = qic_for_distance(dist, player.effective_range(), player.temporary_range) as i16;
            if player.qic >= qic_needed {
                cmds.push(GameCommand::ExploreSpaceship {
                    ship,
                    coord: map.coords[i],
                });
            }
        }
    }

    // 8. Spaceship Board Actions
    for &ship in &Spaceship::ALL {
        if player.has_explored(ship) {
            match ship {
                Spaceship::Twilight => {
                    // Qic: rescore federation token for 3 QIC
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Qic as usize]
                        && player.qic >= 3
                        && player.total_federation_tokens() > 0
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Qic,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                    // Power: TS -> Lab for 3pw, 2o
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Power as usize]
                        && player.power.spendable_power() >= 3
                        && player.ore >= 2
                        && player.buildings_available(Building::ResearchLab) > 0
                    {
                        for i in 0..map.count {
                            if map.hexes[i].building_of(player.seat) == Some(Building::TradingStation) {
                                cmds.push(GameCommand::SpaceshipBoardAction {
                                    ship,
                                    action_type: SpaceshipActionType::Power,
                                    target_coord: Some(map.coords[i]),
                                    target_field: None,
                                    rescore_token: None,
                                });
                            }
                        }
                    }
                    // Knowledge: +3 temporary range for 1 knowledge
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Knowledge as usize]
                        && player.knowledge >= 1
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Knowledge,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                    // Credit: Examine Artefact (costs 6 power tokens discarded)
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Credit as usize]
                        && player.power.total_tokens() >= 6
                        && player.claimed_artefacts.iter().any(|&c| !c)
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Credit,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                }
                Spaceship::Rebellion => {
                    // Qic: claim tech tile for 3 QIC
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Qic as usize]
                        && player.qic >= 3
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Qic,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                    // Power: Mine -> TS for 3pw, 1o
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Power as usize]
                        && player.power.spendable_power() >= 3
                        && player.ore >= 1
                        && player.buildings_available(Building::TradingStation) > 0
                    {
                        for i in 0..map.count {
                            if map.hexes[i].building_of(player.seat) == Some(Building::Mine) {
                                cmds.push(GameCommand::SpaceshipBoardAction {
                                    ship,
                                    action_type: SpaceshipActionType::Power,
                                    target_coord: Some(map.coords[i]),
                                    target_field: None,
                                    rescore_token: None,
                                });
                            }
                        }
                    }
                    // Knowledge: 2 credits, 1 QIC for 2 knowledge
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Knowledge as usize]
                        && player.knowledge >= 2
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Knowledge,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                }
                Spaceship::TFMars => {
                    // Qic: 2 VP + 1 VP per tech tile for 2 QIC
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Qic as usize]
                        && player.qic >= 2
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Qic,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                    // Power: instant Transdim -> Gaia for 2pw
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Power as usize]
                        && player.power.spendable_power() >= 2
                    {
                        for i in 0..map.count {
                            let hex = map.hexes[i];
                            if hex.planet != Planet::Transdim || hex.occupied() {
                                continue;
                            }
                            let Some(dist) = map.min_distance_from_player(player.seat, i) else { continue };
                            let qic_needed = qic_for_distance(dist, player.effective_range(), player.temporary_range) as i16;
                            if player.qic >= qic_needed {
                                cmds.push(GameCommand::SpaceshipBoardAction {
                                    ship,
                                    action_type: SpaceshipActionType::Power,
                                    target_coord: Some(map.coords[i]),
                                    target_field: None,
                                    rescore_token: None,
                                });
                            }
                        }
                    }
                    // Credit: 1 terraforming step + mine for 3 credits + mine ore cost
                    let ore_cost = if player.faction == Faction::Ivits { 1 } else { 2 };
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Credit as usize]
                        && player.credits >= 3
                        && player.ore >= ore_cost
                        && player.buildings_available(Building::Mine) > 0
                    {
                        for i in 0..map.count {
                            let hex = map.hexes[i];
                            if !hex.has_planet()
                                || hex.occupied()
                                || hex.planet == Planet::Transdim
                                || hex.planet == Planet::Asteroid
                                || terraforming_steps(player.faction, hex.planet, &[]) != 1
                            {
                                continue;
                            }
                            let Some(dist) = map.min_distance_from_player(player.seat, i) else { continue };
                            let qic_needed = qic_for_distance(dist, player.effective_range(), player.temporary_range) as i16;
                            if player.qic >= qic_needed {
                                cmds.push(GameCommand::SpaceshipBoardAction {
                                    ship,
                                    action_type: SpaceshipActionType::Credit,
                                    target_coord: Some(map.coords[i]),
                                    target_field: None,
                                    rescore_token: None,
                                });
                            }
                        }
                    }
                }
                Spaceship::Eclipse => {
                    // Qic: 2 VP + 1 VP per planet type for 2 QIC
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Qic as usize]
                        && player.qic >= 2
                    {
                        cmds.push(GameCommand::SpaceshipBoardAction {
                            ship,
                            action_type: SpaceshipActionType::Qic,
                            target_coord: None,
                            target_field: None,
                            rescore_token: None,
                        });
                    }
                    // Power: advance research field for 3pw, 2k
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Power as usize]
                        && player.power.spendable_power() >= 3
                        && player.knowledge >= 2
                    {
                        for &field in &[
                            ResearchField::Terraforming,
                            ResearchField::Navigation,
                            ResearchField::Intelligence,
                            ResearchField::GaiaProject,
                            ResearchField::Economy,
                            ResearchField::Science,
                        ] {
                            let cur = player.research_level(field);
                            if cur < 5 {
                                if cur == 4 {
                                    if claimed_l5[field as usize].is_none() && player.green_federation_tokens > 0 {
                                        cmds.push(GameCommand::SpaceshipBoardAction {
                                            ship,
                                            action_type: SpaceshipActionType::Power,
                                            target_coord: None,
                                            target_field: Some(field),
                                            rescore_token: None,
                                        });
                                    }
                                } else {
                                    cmds.push(GameCommand::SpaceshipBoardAction {
                                        ship,
                                        action_type: SpaceshipActionType::Power,
                                        target_coord: None,
                                        target_field: Some(field),
                                        rescore_token: None,
                                    });
                                }
                            }
                        }
                    }
                    // Credit: mine on Asteroid for 6 credits
                    if !claimed_spaceship_actions[ship as usize][SpaceshipActionType::Credit as usize]
                        && player.credits >= 6
                        && player.buildings_available(Building::Mine) > 0
                    {
                        for i in 0..map.count {
                            let hex = map.hexes[i];
                            if hex.planet != Planet::Asteroid || hex.occupied() {
                                continue;
                            }
                            let Some(dist) = map.min_distance_from_player(player.seat, i) else { continue };
                            let qic_needed = qic_for_distance(dist, player.effective_range(), player.temporary_range) as i16;
                            if player.qic >= qic_needed {
                                cmds.push(GameCommand::SpaceshipBoardAction {
                                    ship,
                                    action_type: SpaceshipActionType::Credit,
                                    target_coord: Some(map.coords[i]),
                                    target_field: None,
                                    rescore_token: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // 9. Form Federation
    if find_minimal_federation(player, map).is_ok() {
        for &tok in &[
            FederationToken::Fed1,
            FederationToken::Fed2,
            FederationToken::Fed3,
            FederationToken::Fed4,
            FederationToken::Fed5,
            FederationToken::Fed6,
        ] {
            cmds.push(GameCommand::FormFederationAuto { token: tok });
        }
    }

    // 10. Free Actions (Universal conversions)
    if player.power.spendable_power() >= 4 {
        cmds.push(GameCommand::FreeAction { action: crate::rules::FreeAction::PowerToQic });
        cmds.push(GameCommand::FreeAction { action: crate::rules::FreeAction::PowerToKnowledge });
    }
    if player.power.spendable_power() >= 3 {
        cmds.push(GameCommand::FreeAction { action: crate::rules::FreeAction::PowerToOre });
    }
    if player.power.spendable_power() >= 1 {
        cmds.push(GameCommand::FreeAction { action: crate::rules::FreeAction::PowerToCredit });
    }
    if player.qic >= 1 {
        cmds.push(GameCommand::FreeAction { action: crate::rules::FreeAction::QicToOre });
    }
    if player.ore >= 1 {
        cmds.push(GameCommand::FreeAction { action: crate::rules::FreeAction::OreToToken });
    }

    // 11. Pass
    if is_final_round {
        cmds.push(GameCommand::Pass { new_booster: None });
    } else {
        for &booster in available_boosters {
            if player.current_booster != Some(booster) {
                cmds.push(GameCommand::Pass {
                    new_booster: Some(booster),
                });
            }
        }
    }

    cmds
}


/// Validates and executes examining an Artifact on the Twilight spaceship (Action 12).
/// Costs 6 power tokens discarded from Bowls I/II/III.
pub fn execute_examine_artefact(
    player: &mut PlayerData,
    map: &Map,
    artefact: crate::rules::Artefact,
    rescore_token: Option<FederationToken>,
    claimed_artefacts: &mut [bool; 13],
) -> Result<(), ActionError> {
    if player.passed {
        return Err(ActionError::PlayerAlreadyPassed);
    }
    if !player.has_explored(Spaceship::Twilight) {
        return Err(ActionError::SpaceshipNotExplored(Spaceship::Twilight));
    }
    if claimed_artefacts[artefact as usize] || player.claimed_artefacts[artefact as usize] {
        return Err(ActionError::ArtefactAlreadyClaimed(artefact));
    }
    let total_power = player.power.area1 + player.power.area2 + player.power.area3;
    if total_power < 6 {
        return Err(ActionError::InsufficientPowerForSatellites {
            needed: 6,
            have: total_power,
        });
    }
    player.discard_power(6);
    claimed_artefacts[artefact as usize] = true;
    player.claimed_artefacts[artefact as usize] = true;

    match artefact {
        crate::rules::Artefact::IncomeKnowledgeOre => {
            // Ongoing: +1 knowledge, +1 ore per income phase
            player.add_knowledge(1);
            player.add_ore(1);
        }
        crate::rules::Artefact::Credits3Ore3 => {
            player.add_credits(3);
            player.add_ore(3);
        }
        crate::rules::Artefact::Knowledge3Qic1 => {
            player.add_knowledge(3);
            player.add_qic(1);
        }
        crate::rules::Artefact::Credits5Ore2 => {
            player.add_credits(5);
            player.add_ore(2);
        }
        crate::rules::Artefact::ChargePower2 => {
            player.power.charge(2);
        }
        crate::rules::Artefact::AsteroidVp => {
            player.add_victory_points(7);
        }
        crate::rules::Artefact::ProtoplanetVp => {
            player.add_victory_points(7);
        }
        crate::rules::Artefact::ResearchAreaLevel => {
            let max_lvl = player.research.iter().copied().max().unwrap_or(0);
            player.add_victory_points(max_lvl as i16 * 3);
        }
        crate::rules::Artefact::ResearchTracksCount => {
            let count_3 = player.research.iter().filter(|&&lvl| lvl >= 3).count();
            player.add_victory_points(count_3 as i16 * 3);
        }
        crate::rules::Artefact::RescoreFederation => {
            let token = rescore_token
                .or_else(|| player.claimed_federations.iter().filter_map(|&t| t).next())
                .ok_or(ActionError::NoFederationToRescore)?;
            let rew = token.reward();
            player.add_victory_points(rew.vp);
            player.add_credits(rew.credits);
            player.add_ore(rew.ore);
            player.add_knowledge(rew.knowledge);
            player.add_qic(rew.qic);
            player.power.area1 += rew.power_tokens as u8;
        }
        crate::rules::Artefact::GaiaformingTrack => {
            let gaia_lvl = player.research[ResearchField::GaiaProject as usize];
            player.add_victory_points(gaia_lvl as i16 * 3);
        }
        crate::rules::Artefact::PlanetTypes => {
            let types = map.distinct_planets_colonized_by(player.seat);
            player.add_victory_points(3 + types as i16);
        }
        crate::rules::Artefact::DeepSpace => {
            let sectors = map.sectors_with_player(player.seat);
            player.add_victory_points(sectors as i16 * 3);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;
    use crate::player::PowerBowls;

    #[test]
    fn build_mine_on_home_planet_costs_standard() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;

        // Existing mine on 0
        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        // Target Terra planet on 1
        map.hexes[1].planet = Planet::Terra;

        let res = execute_build_mine(&mut player, &mut map, HexCoord::new(1, -1, 0).unwrap());
        assert!(res.is_ok());
        assert_eq!(player.credits, 15 - 2);
        assert_eq!(player.ore, 4 - 1);
        assert_eq!(player.buildings[Building::Mine as usize], 2);
        assert_eq!(map.hexes[1].building, Some(Building::Mine));
    }

    #[test]
    fn build_mine_on_asteroid_consumes_gaiaformer_and_is_free() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.gaiaformers_unlocked = 1;
        player.credits = 0;
        player.ore = 0;

        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        map.hexes[1].planet = Planet::Asteroid;

        let res = execute_build_mine(&mut player, &mut map, HexCoord::new(1, -1, 0).unwrap());
        assert!(res.is_ok());
        assert_eq!(player.used_gaiaformers_asteroid, 1);
        assert_eq!(player.credits, 0);
        assert_eq!(player.ore, 0);
    }

    #[test]
    fn build_mine_on_protoplanet_gives_six_vp() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.ore = 10;
        player.credits = 10;
        let start_vp = player.victory_points;

        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        map.hexes[1].planet = Planet::Protoplanet;

        let res = execute_build_mine(&mut player, &mut map, HexCoord::new(1, -1, 0).unwrap());
        assert!(res.is_ok());
        assert_eq!(player.victory_points, start_vp + 6);
    }

    #[test]
    fn upgrade_mine_to_trading_station_checks_neighbors() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        // Without opponent: costs 6c, 2o
        let (_, c1, o1) = calculate_upgrade_cost(&player, &map, HexCoord::origin(), Building::TradingStation).unwrap();
        assert_eq!((c1, o1), (6, 2));

        // With opponent on hex 1: costs 3c, 2o
        map.hexes[1].planet = Planet::Desert;
        map.hexes[1].building = Some(Building::Mine);
        map.hexes[1].player = Some(1);

        let (_, c2, o2) = calculate_upgrade_cost(&player, &map, HexCoord::origin(), Building::TradingStation).unwrap();
        assert_eq!((c2, o2), (3, 2));
    }

    #[test]
    fn research_advancement_level_5_and_charge_power() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.knowledge = 10;
        player.research[ResearchField::Science as usize] = 2;
        player.power = crate::player::PowerBowls::new(2, 4, 0, 0);
        let mut claimed = [None; 6];

        // 2 -> 3 costs 4 knowledge, charges 3 power
        assert!(execute_advance_research(&mut player, ResearchField::Science, &mut claimed).is_ok());
        assert_eq!(player.research[ResearchField::Science as usize], 3);
        assert_eq!(player.knowledge, 6);
        assert_eq!(player.power.area1, 0);
        assert_eq!(player.power.area2, 5);
        assert_eq!(player.power.area3, 1);

        // Advance 3 -> 4
        assert!(execute_advance_research(&mut player, ResearchField::Science, &mut claimed).is_ok());

        // 4 -> 5 requires green federation token
        player.knowledge = 10;
        assert_eq!(
            execute_advance_research(&mut player, ResearchField::Science, &mut claimed),
            Err(ActionError::FederationTokenRequiredForLevel5)
        );

        player.green_federation_tokens = 1;
        assert!(execute_advance_research(&mut player, ResearchField::Science, &mut claimed).is_ok());
        assert_eq!(player.research[ResearchField::Science as usize], 5);
        assert_eq!(player.green_federation_tokens, 0);
        assert_eq!(player.gray_federation_tokens, 1);
        assert_eq!(claimed[ResearchField::Science as usize], Some(0));
        // L5 Science immediately grants 9 knowledge
        assert_eq!(player.knowledge, 15); // Capped at MAX_KNOWLEDGE 15
    }

    #[test]
    fn pass_computes_booster_vp() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.current_booster = Some(6); // 1 VP per mine
        player.buildings[Building::Mine as usize] = 4;
        let start_vp = player.victory_points;

        let vp = execute_pass(&mut player, Some(7)).unwrap();
        assert_eq!(vp, 4);
        assert_eq!(player.victory_points, start_vp + 4);
        assert_eq!(player.current_booster, Some(7));
        assert!(player.passed);
    }

    #[test]
    fn start_gaia_project_and_resolve_phase_flow() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.research[ResearchField::GaiaProject as usize] = 1; // Unlocks Gaia project, 6 power cost
        player.gaiaformers_unlocked = 1;
        player.power = crate::player::PowerBowls::new(4, 4, 0, 0);

        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);

        map.hexes[1].planet = Planet::Transdim;

        // Execute Start Gaia Project on hex 1
        let res = execute_start_gaia_project(&mut player, &mut map, HexCoord::new(1, -1, 0).unwrap());
        assert!(res.is_ok());
        assert_eq!(player.gaiaformers_in_gaia, 1);
        assert_eq!(player.power.gaia, 6);
        assert_eq!(map.hexes[1].building, Some(Building::GaiaFormer));
        assert_eq!(map.hexes[1].player, Some(0));

        // Resolve Gaia Phase
        let mut players = [player];
        resolve_gaia_phase(&mut map, &mut players);
        // Transdim transforms into Gaia planet
        assert_eq!(map.hexes[1].planet, Planet::Gaia);
        // Terrans move Gaia power directly to Area 2!
        // (Started with Area 2 = 4, spent 2 to Gaia, now receives 6 from Gaia -> 4 - 2 + 6 = 8)
        assert_eq!(players[0].power.gaia, 0);
        assert_eq!(players[0].power.area2, 8);
    }

    #[test]
    fn form_federation_with_satellites_and_rewards() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.power = crate::player::PowerBowls::new(2, 4, 0, 0);

        let mut map = Map::empty();
        map.count = 4;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap(); // Empty space
        map.coords[2] = HexCoord::new(2, -2, 0).unwrap();
        map.coords[3] = HexCoord::new(0, 1, -1).unwrap();
        map.recompute_adjacency();

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::PlanetaryInstitute);
        map.hexes[0].player = Some(0);

        map.hexes[3].planet = Planet::Terra;
        map.hexes[3].building = Some(Building::Mine);
        map.hexes[3].player = Some(0);

        map.hexes[2].planet = Planet::Terra;
        map.hexes[2].building = Some(Building::Academy1);
        map.hexes[2].player = Some(0);

        // Hex 1 is empty space for a satellite connecting Hex 0 and Hex 2
        map.hexes[1].planet = Planet::Empty;

        let planets = [map.coords[0], map.coords[3], map.coords[2]];
        let satellites = [map.coords[1]];

        let res = execute_form_federation(
            &mut player,
            &mut map,
            &planets,
            &satellites,
            FederationToken::Fed4, // 7 VP, 2 ore
        );
        assert!(res.is_ok());
        assert_eq!(player.satellites, 1);
        assert_eq!(player.victory_points, 10 + 7);
        assert_eq!(player.ore, 4 + 2);
        assert_eq!(player.green_federation_tokens, 1);
        // 1 power discarded for satellite
        assert_eq!(player.power.area1, 1);
        // Hexes marked in federation
        assert!(map.hexes[0].belongs_to_federation_of(0));
        assert!(map.hexes[1].belongs_to_federation_of(0));
        assert!(map.hexes[2].belongs_to_federation_of(0));
        assert!(map.hexes[3].belongs_to_federation_of(0));
    }

    #[test]
    fn find_leech_opportunities_and_charge_power() {
        let p0 = PlayerData::new(0, Faction::Terrans);
        let mut p1 = PlayerData::new(1, Faction::Xenos);
        p1.power = crate::player::PowerBowls::new(4, 0, 0, 0);

        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.recompute_adjacency();

        // Player 1 has a Trading Station on Hex 1 (power value 2)
        map.hexes[1].planet = Planet::Desert;
        map.hexes[1].building = Some(Building::TradingStation);
        map.hexes[1].player = Some(1);

        let players = [p0, p1];
        // Player 0 builds or upgrades on Hex 0 (distance 1 <= 2)
        let opps = find_leech_opportunities(&map, HexCoord::origin(), 0, &players);
        let p1_opp = opps[1].unwrap();
        assert_eq!(p1_opp.power_value, 2);
        assert_eq!(p1_opp.vp_cost, 1); // 2 power costs 1 VP

        // Player 1 decides to charge 2 power
        let mut p1_mut = players[1].clone();
        assert!(execute_leech(&mut p1_mut, 2).is_ok());
        assert_eq!(p1_mut.victory_points, 10 - 1);
        assert_eq!(p1_mut.power.area1, 2);
        assert_eq!(p1_mut.power.area2, 2);
    }

    #[test]
    fn legal_commands_enumerates_valid_actions() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.recompute_adjacency();

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        map.hexes[1].planet = Planet::Terra;

        let claimed_l5 = [None; 6];
        let boosters = [1, 2, 3];
        let cmds = legal_commands(0, &player, &map, &claimed_l5, &[None; 10], &[[false; 4]; 4], &boosters, false);

        // Should contain BuildMine on Hex 1, Upgrade on Hex 0, Pass with boosters
        assert!(cmds.iter().any(|c| matches!(c, GameCommand::BuildMine { coord } if *coord == map.coords[1])));
        assert!(cmds.iter().any(|c| matches!(c, GameCommand::Upgrade { coord, to } if *coord == map.coords[0] && *to == Building::TradingStation)));
        assert!(cmds.iter().any(|c| matches!(c, GameCommand::Pass { new_booster: Some(1) })));
    }

    #[test]
    fn legal_commands_enumerates_spaceship_exploration_and_board_actions() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.victory_points = 15;
        player.knowledge = 3;
        player.qic = 2;

        let mut map = Map::empty();
        map.count = 3;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.coords[2] = HexCoord::new(2, -2, 0).unwrap();
        map.recompute_adjacency();

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        // Place TFMars on hex 2 (dist 2 from origin, base range 1 -> needs 1 QIC)
        map.hexes[2].spaceship = Some(Spaceship::TFMars);

        // Player has explored Twilight
        player.exploration_ships[Spaceship::Twilight as usize] = Some(1);

        let claimed_l5 = [None; 6];
        let claimed_board = [None; 10];
        let mut claimed_ships = [[false; 4]; 4];
        let boosters = [1];

        // 1. Check commands: should include ExploreSpaceship(TFMars) and SpaceshipBoardAction(Twilight, Knowledge)
        let cmds = legal_commands(0, &player, &map, &claimed_l5, &claimed_board, &claimed_ships, &boosters, false);
        assert!(cmds.iter().any(|c| matches!(c, GameCommand::ExploreSpaceship { ship, coord } if *ship == Spaceship::TFMars && *coord == map.coords[2])));
        assert!(cmds.iter().any(|c| matches!(c, GameCommand::SpaceshipBoardAction { ship, action_type, .. } if *ship == Spaceship::Twilight && *action_type == SpaceshipActionType::Knowledge)));

        // 2. If Twilight Knowledge is claimed this round, it should NOT be enumerated
        claimed_ships[Spaceship::Twilight as usize][SpaceshipActionType::Knowledge as usize] = true;
        let cmds2 = legal_commands(0, &player, &map, &claimed_l5, &claimed_board, &claimed_ships, &boosters, false);
        assert!(!cmds2.iter().any(|c| matches!(c, GameCommand::SpaceshipBoardAction { ship, action_type, .. } if *ship == Spaceship::Twilight && *action_type == SpaceshipActionType::Knowledge)));
    }

    #[test]
    fn board_actions_cost_deduction_and_exclusivity() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        map.hexes[1].planet = Planet::Desert;
        map.hexes[1].building = Some(Building::Mine);
        map.hexes[1].player = Some(0);

        let mut claimed = [None; 10];
        player.power.area3 = 7;
        player.knowledge = 0;

        // Power1: costs 7 power, awards 3 knowledge
        assert!(execute_board_action(&mut player, &mut map, &mut claimed, BoardAction::Power1, None, None).is_ok());
        assert_eq!(player.power.area3, 0);
        assert_eq!(player.power.area1, 4 + 7);
        assert_eq!(player.knowledge, 3);
        assert_eq!(claimed[BoardAction::Power1 as usize], Some(0));

        // Attempting to claim Power1 again in same round fails
        player.power.area3 = 7;
        assert_eq!(
            execute_board_action(&mut player, &mut map, &mut claimed, BoardAction::Power1, None, None),
            Err(ActionError::BoardActionAlreadyClaimed(BoardAction::Power1))
        );

        // Qic2: rescore a claimed federation
        player.qic = 3;
        player.claim_federation_token(FederationToken::Fed1); // 12 VP
        let initial_vp = player.victory_points;
        assert!(execute_board_action(&mut player, &mut map, &mut claimed, BoardAction::Qic2, None, Some(FederationToken::Fed1)).is_ok());
        assert_eq!(player.victory_points, initial_vp + 12);
        assert_eq!(player.qic, 0);

        // Qic3: 3 VP + 1 VP per planet type (Terra + Desert = 2 types -> 5 VP)
        player.qic = 2;
        let pre_qic3_vp = player.victory_points;
        assert!(execute_board_action(&mut player, &mut map, &mut claimed, BoardAction::Qic3, None, None).is_ok());
        assert_eq!(player.victory_points, pre_qic3_vp + 3 + 2);
    }

    #[test]
    fn special_actions_faction_abilities() {
        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();

        // Ambas PI swap
        let mut ambas = PlayerData::new(0, Faction::Ambas);
        ambas.buildings[Building::PlanetaryInstitute as usize] = 1;
        ambas.buildings[Building::Mine as usize] = 1;
        map.hexes[0].player = Some(0);
        map.hexes[0].building = Some(Building::PlanetaryInstitute);
        map.hexes[1].player = Some(0);
        map.hexes[1].building = Some(Building::Mine);
        let mut claimed_l5 = [None; 6];
        let target_ambas = map.coords[1];

        assert!(execute_special_action(&mut ambas, &mut map, SpecialAction::AmbasPiSwap, Some(target_ambas), None, &mut claimed_l5).is_ok());
        assert_eq!(map.hexes[0].building, Some(Building::Mine));
        assert_eq!(map.hexes[1].building, Some(Building::PlanetaryInstitute));
        assert!(ambas.special_actions_used[SpecialAction::AmbasPiSwap as usize]);

        // Firaks downgrade Lab to TS and advance research
        let mut firaks = PlayerData::new(1, Faction::Firaks);
        firaks.buildings[Building::PlanetaryInstitute as usize] = 1;
        firaks.buildings[Building::ResearchLab as usize] = 1;
        firaks.buildings[Building::TradingStation as usize] = 0;
        map.hexes[0].player = Some(1);
        map.hexes[0].building = Some(Building::ResearchLab);
        let target_firaks = map.coords[0];

        assert!(execute_special_action(&mut firaks, &mut map, SpecialAction::FiraksDowngradeLab, Some(target_firaks), Some(ResearchField::Science), &mut claimed_l5).is_ok());
        assert_eq!(map.hexes[0].building, Some(Building::TradingStation));
        assert_eq!(firaks.buildings[Building::ResearchLab as usize], 0);
        assert_eq!(firaks.buildings[Building::TradingStation as usize], 1);
        assert_eq!(firaks.research_level(ResearchField::Science), 1);

        // Bescods advance lowest
        let mut bescods = PlayerData::new(2, Faction::Bescods);
        bescods.research = [0, 1, 1, 1, 1, 1];
        bescods.ore = 0;
        assert!(execute_special_action(&mut bescods, &mut map, SpecialAction::BescodsAdvanceLowest, None, None, &mut claimed_l5).is_ok());
        assert_eq!(bescods.research_level(ResearchField::Terraforming), 1);
        assert_eq!(bescods.ore, 2); // Terraforming level 1 awards 2 ore
    }

    #[test]
    fn tech_tiles_claiming_and_advanced_tech() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        let mut map = Map::empty();
        map.count = 2;
        map.coords[0] = HexCoord::origin();
        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        let mut claimed_l5 = [None; 6];
        let mut claimed_adv = [false; 15];

        // Claim Tech1: gives 1 ore and 1 QIC, advance Navigation from 0 to 1 (awards 1 QIC)
        assert!(execute_claim_tech_tile(&mut player, TechTile::Tech1, Some(ResearchField::Navigation), &mut claimed_l5).is_ok());
        assert!(player.tech_tiles[TechTile::Tech1 as usize]);
        assert_eq!(player.research_level(ResearchField::Navigation), 1);
        assert_eq!(player.ore, 5); // 4 initial + 1 from Tech1
        assert_eq!(player.qic, 3); // 1 initial + 1 from Tech1 + 1 from Nav1

        // Cannot claim Tech1 again
        assert_eq!(
            execute_claim_tech_tile(&mut player, TechTile::Tech1, None, &mut claimed_l5),
            Err(ActionError::TechTileAlreadyOwned(TechTile::Tech1))
        );

        // Prepare player for advanced tech: research level 4 and green federation token
        player.research[ResearchField::Terraforming as usize] = 4;
        player.claim_federation_token(FederationToken::Fed2); // green token (8 VP, 1 QIC)
        assert_eq!(player.green_federation_tokens, 1);

        // Claim AdvTech4 (2 VP per mine): covers Tech1
        let pre_vp = player.victory_points;
        assert!(execute_claim_adv_tech_tile(&mut player, &map, AdvTechTile::AdvTech4, TechTile::Tech1, ResearchField::Terraforming, &mut claimed_adv).is_ok());
        assert!(player.adv_tech_tiles[AdvTechTile::AdvTech4 as usize]);
        assert!(player.covered_tech_tiles[TechTile::Tech1 as usize]);
        assert_eq!(player.green_federation_tokens, 0);
        assert_eq!(player.gray_federation_tokens, 1);
        assert_eq!(player.victory_points, pre_vp + 2); // 1 mine * 2 VP
    }

    #[test]
    fn final_scoring_ranking_and_tie_splits() {
        let mut p0 = PlayerData::new(0, Faction::Terrans);
        let mut p1 = PlayerData::new(1, Faction::HadschHallas);
        let map = Map::empty();

        // Final tiles: Structure (18, 12, 6, 0) and PlanetType
        p0.buildings[Building::Mine as usize] = 4; // 4 structures
        p1.buildings[Building::Mine as usize] = 2; // 2 structures

        // Research VP: p0 has level 4 (+8 VP), p1 has level 3 (+4 VP)
        p0.research[0] = 4;
        p1.research[0] = 3;

        // Resource conversion: p0 has 6 resources (+2 VP), p1 has 3 resources (+1 VP)
        p0.credits = 3;
        p0.ore = 3;
        p0.knowledge = 0;
        p1.credits = 3;
        p1.ore = 0;
        p1.knowledge = 0;

        let final_tiles = [FinalTile::Structure, FinalTile::PlanetType];
        let mut players = [p0, p1];
        compute_final_scores(&mut players, &map, final_tiles);

        // Player 0: 1st in Structure (18 VP) + research ((4-2)*4 = 8 VP) + resources (6/3 = 2 VP) = +28 VP
        // Player 1: 2nd in Structure (12 VP) + research ((3-2)*4 = 4 VP) + resources (3/3 = 1 VP) = +17 VP
        assert_eq!(players[0].victory_points, 10 + 28);
        assert_eq!(players[1].victory_points, 10 + 17);
    }

    #[test]
    fn spaceship_exploration_and_charge_track() {
        let mut p0 = PlayerData::new(0, Faction::Terrans);
        p0.victory_points = 15;
        p0.qic = 0;
        p0.power = crate::player::PowerBowls::new(4, 0, 0, 0);

        let mut map = Map::empty();
        map.count = 3;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.coords[2] = HexCoord::new(2, -2, 0).unwrap();
        map.recompute_adjacency();

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        p0.buildings[Building::Mine as usize] = 1;

        // Place Twilight on hex 2 (dist 2 from origin, base range 1 -> needs 1 QIC)
        map.hexes[2].spaceship = Some(Spaceship::Twilight);

        let mut occupied = [false; 5];
        // 0 QIC -> error
        assert!(matches!(
            execute_explore_spaceship(&mut p0, &occupied, &map, Spaceship::Twilight, map.coords[2], 2),
            Err(ActionError::InsufficientQic { needed: 1, have: 0 })
        ));

        // Give 1 QIC
        p0.qic = 1;
        let slot = execute_explore_spaceship(&mut p0, &occupied, &map, Spaceship::Twilight, map.coords[2], 2).unwrap();
        assert_eq!(slot, 1);
        assert_eq!(p0.victory_points, 10); // 15 - 5 VP
        assert_eq!(p0.qic, 0); // 1 - 1 QIC
        assert_eq!(p0.exploration_ships[Spaceship::Twilight as usize], Some(1));
        assert!(p0.has_explored(Spaceship::Twilight));
        // Slot 1 charge is 0
        assert_eq!(p0.power.area3, 0);

        // Cannot explore again
        assert!(matches!(
            execute_explore_spaceship(&mut p0, &occupied, &map, Spaceship::Twilight, map.coords[2], 2),
            Err(ActionError::SpaceshipAlreadyExplored(Spaceship::Twilight))
        ));

        // Player 1 explores slot 2: slot 1 is occupied
        occupied[1] = true;
        let mut p1 = PlayerData::new(1, Faction::BalTaks); // Bal T'aks pays 7 VP
        p1.victory_points = 20;
        p1.qic = 2;
        p1.power = crate::player::PowerBowls::new(4, 0, 0, 0);
        map.hexes[1].planet = Planet::Gaia;
        map.hexes[1].building = Some(Building::Mine);
        map.hexes[1].player = Some(1);
        p1.buildings[Building::Mine as usize] = 1;

        let slot2 = execute_explore_spaceship(&mut p1, &occupied, &map, Spaceship::Twilight, map.coords[2], 2).unwrap();
        assert_eq!(slot2, 2);
        assert_eq!(p1.victory_points, 13); // 20 - 7 VP for Bal T'aks
        assert_eq!(p1.exploration_ships[Spaceship::Twilight as usize], Some(2));
        // Slot 2 charges 2 power: bowl 1 -> bowl 2
        assert_eq!(p1.power.area2, 2);
        assert_eq!(p1.power.area1, 2);
    }

    #[test]
    fn spaceship_board_actions_all_effects_and_lock() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.exploration_ships[Spaceship::Twilight as usize] = Some(1);
        player.exploration_ships[Spaceship::Rebellion as usize] = Some(1);
        player.exploration_ships[Spaceship::TFMars as usize] = Some(1);
        player.exploration_ships[Spaceship::Eclipse as usize] = Some(1);

        let mut map = Map::empty();
        map.count = 4;
        map.coords[0] = HexCoord::origin();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.coords[2] = HexCoord::new(2, -2, 0).unwrap();
        map.coords[3] = HexCoord::new(0, 1, -1).unwrap();
        map.recompute_adjacency();

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);
        player.buildings[Building::Mine as usize] = 1;

        let mut claimed_actions = [[false; 4]; 4];
        let mut claimed_l5 = [None; 6];

        // 1. Twilight Knowledge: 2k -> +3 temporary range
        player.knowledge = 3;
        assert!(execute_spaceship_action(
            &mut player,
            &mut map,
            &mut claimed_actions,
            &mut claimed_l5,
            Spaceship::Twilight,
            SpaceshipActionType::Knowledge,
            None,
            None,
            None,
        ).is_ok());
        assert_eq!(player.knowledge, 2);
        assert_eq!(player.temporary_range, 3);
        assert!(claimed_actions[Spaceship::Twilight as usize][SpaceshipActionType::Knowledge as usize]);

        // Trying again in same round fails
        assert!(matches!(
            execute_spaceship_action(
                &mut player,
                &mut map,
                &mut claimed_actions,
                &mut claimed_l5,
                Spaceship::Twilight,
                SpaceshipActionType::Knowledge,
                None,
                None,
                None,
            ),
            Err(ActionError::SpaceshipActionAlreadyUsed(Spaceship::Twilight, SpaceshipActionType::Knowledge))
        ));

        // 2. Rebellion Knowledge: 2k -> 2c, 1q
        player.knowledge = 2;
        player.credits = 5;
        player.qic = 0;
        assert!(execute_spaceship_action(
            &mut player,
            &mut map,
            &mut claimed_actions,
            &mut claimed_l5,
            Spaceship::Rebellion,
            SpaceshipActionType::Knowledge,
            None,
            None,
            None,
        ).is_ok());
        assert_eq!(player.knowledge, 0);
        assert_eq!(player.credits, 7);
        assert_eq!(player.qic, 1);

        // 3. TFMars Qic: 2q -> 2 VP + 1/tech
        player.qic = 2;
        player.victory_points = 10;
        player.tech_tiles[TechTile::Tech1 as usize] = true;
        player.tech_tiles[TechTile::Tech2 as usize] = true;
        assert!(execute_spaceship_action(
            &mut player,
            &mut map,
            &mut claimed_actions,
            &mut claimed_l5,
            Spaceship::TFMars,
            SpaceshipActionType::Qic,
            None,
            None,
            None,
        ).is_ok());
        assert_eq!(player.qic, 0);
        assert_eq!(player.victory_points, 10 + 2 + 2); // 10 + 2 base + 2 techs = 14

        // 4. TFMars Power: 2pw -> instant Transdim to Gaia + free mine
        player.power = crate::player::PowerBowls::new(0, 0, 4, 0);
        map.hexes[1].planet = Planet::Transdim;
        let target_transdim = map.coords[1];
        assert!(execute_spaceship_action(
            &mut player,
            &mut map,
            &mut claimed_actions,
            &mut claimed_l5,
            Spaceship::TFMars,
            SpaceshipActionType::Power,
            Some(target_transdim),
            None,
            None,
        ).is_ok());
        assert_eq!(player.power.area3, 2);
        assert_eq!(player.power.area1, 2);
        assert_eq!(map.hexes[1].planet, Planet::Gaia);
        assert_eq!(map.hexes[1].building, None);

        // 5. Eclipse Power: 3pw, 2k -> advance research
        player.power = crate::player::PowerBowls::new(0, 0, 3, 0);
        player.knowledge = 2;
        assert_eq!(player.research[ResearchField::Science as usize], 0);
        assert!(execute_spaceship_action(
            &mut player,
            &mut map,
            &mut claimed_actions,
            &mut claimed_l5,
            Spaceship::Eclipse,
            SpaceshipActionType::Power,
            None,
            Some(ResearchField::Science),
            None,
        ).is_ok());
        assert_eq!(player.knowledge, 0);
        assert_eq!(player.research[ResearchField::Science as usize], 1);

        // 6. Eclipse Credit: 6c -> mine on Asteroid
        player.credits = 10;
        map.hexes[3].planet = Planet::Asteroid;
        let target_asteroid = map.coords[3];
        assert!(execute_spaceship_action(
            &mut player,
            &mut map,
            &mut claimed_actions,
            &mut claimed_l5,
            Spaceship::Eclipse,
            SpaceshipActionType::Credit,
            Some(target_asteroid),
            None,
            None,
        ).is_ok());
        assert_eq!(player.credits, 4); // 10 - 6
        assert_eq!(map.hexes[3].building, Some(Building::Mine));
        assert_eq!(map.hexes[3].player, Some(0));
    }

    #[test]
    fn test_examine_artefact_mechanics() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        let map = Map::empty();
        let mut claimed = [false; 13];

        // 1. Should fail if Twilight not explored
        assert!(execute_examine_artefact(
            &mut player,
            &map,
            crate::rules::Artefact::Credits3Ore3,
            None,
            &mut claimed,
        ).is_err());

        // Explore Twilight
        player.exploration_ships[Spaceship::Twilight as usize] = Some(0);

        // 2. Should fail if insufficient power (less than 6 tokens total)
        player.power = PowerBowls::new(2, 2, 0, 0); // only 4 tokens
        assert!(execute_examine_artefact(
            &mut player,
            &map,
            crate::rules::Artefact::Credits3Ore3,
            None,
            &mut claimed,
        ).is_err());

        // 3. Give 6 power tokens: 2 in area1, 2 in area2, 2 in area3
        player.power = PowerBowls::new(2, 2, 2, 0);
        player.credits = 10;
        player.ore = 2;
        assert!(execute_examine_artefact(
            &mut player,
            &map,
            crate::rules::Artefact::Credits3Ore3,
            None,
            &mut claimed,
        ).is_ok());

        assert_eq!(player.credits, 13);
        assert_eq!(player.ore, 5);
        assert_eq!(player.power.area1 + player.power.area2 + player.power.area3, 0); // all 6 discarded
        assert!(claimed[crate::rules::Artefact::Credits3Ore3 as usize]);
        assert!(player.claimed_artefacts[crate::rules::Artefact::Credits3Ore3 as usize]);

        // 4. Duplicate claim should fail
        player.power = PowerBowls::new(3, 3, 0, 0);
        assert!(execute_examine_artefact(
            &mut player,
            &map,
            crate::rules::Artefact::Credits3Ore3,
            None,
            &mut claimed,
        ).is_err());
    }

    #[test]
    fn test_faction_startup_setups() {
        let tinkeroids = PlayerData::new(0, Faction::Tinkeroids);
        assert_eq!(tinkeroids.knowledge, 2);
        assert_eq!(tinkeroids.ore, 4);
        assert_eq!(tinkeroids.credits, 15);
        assert_eq!(tinkeroids.qic, 1);
        assert_eq!(tinkeroids.power.area1, 4);
        assert_eq!(tinkeroids.power.area2, 2);
        assert_eq!(tinkeroids.research[ResearchField::Science as usize], 1);

        let darkanians = PlayerData::new(1, Faction::Darkanians);
        assert_eq!(darkanians.knowledge, 3);
        assert_eq!(darkanians.ore, 7);
        assert_eq!(darkanians.credits, 15);
        assert_eq!(darkanians.qic, 1);
        assert_eq!(darkanians.power.area1, 4);
        assert_eq!(darkanians.power.area2, 2);
        assert_eq!(darkanians.research[ResearchField::Navigation as usize], 1);
        assert_eq!(darkanians.research[ResearchField::Economy as usize], 1);

        let moweyds = PlayerData::new(2, Faction::Moweyds);
        assert_eq!(moweyds.knowledge, 5);
        assert_eq!(moweyds.ore, 6);
        assert_eq!(moweyds.credits, 15);
        assert_eq!(moweyds.qic, 2);
        assert_eq!(moweyds.power.area1, 4);
        assert_eq!(moweyds.power.area2, 4);
        assert_eq!(moweyds.research[ResearchField::GaiaProject as usize], 1);
        assert_eq!(moweyds.exploration_ships[Spaceship::TFMars as usize], Some(0));
        assert_eq!(moweyds.power_rings, 6);

        let space_giants = PlayerData::new(3, Faction::SpaceGiants);
        assert_eq!(space_giants.knowledge, 3);
        assert_eq!(space_giants.ore, 6);
        assert_eq!(space_giants.credits, 15);
        assert_eq!(space_giants.qic, 1);
        assert_eq!(space_giants.power.area1, 4);
        assert_eq!(space_giants.power.area2, 4);
        assert_eq!(space_giants.research[ResearchField::Navigation as usize], 1);
    }

    #[test]
    fn test_free_action_conversions() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.power = PowerBowls::new(0, 0, 10, 0); // 10 power in bowl 3
        player.ore = 0;
        player.credits = 0;
        player.knowledge = 0;
        player.qic = 0;

        // 1. Power to Ore (3 power -> 1 ore)
        assert!(execute_free_action(&mut player, crate::rules::FreeAction::PowerToOre).is_ok());
        assert_eq!(player.ore, 1);
        assert_eq!(player.power.area3, 7);

        // 2. Power to Credit (1 power -> 1 credit)
        assert!(execute_free_action(&mut player, crate::rules::FreeAction::PowerToCredit).is_ok());
        assert_eq!(player.credits, 1);
        assert_eq!(player.power.area3, 6);

        // 3. Power to Knowledge (4 power -> 1 knowledge)
        assert!(execute_free_action(&mut player, crate::rules::FreeAction::PowerToKnowledge).is_ok());
        assert_eq!(player.knowledge, 1);
        assert_eq!(player.power.area3, 2);

        // 4. QIC to Ore (1 QIC -> 1 Ore)
        player.qic = 1;
        assert!(execute_free_action(&mut player, crate::rules::FreeAction::QicToOre).is_ok());
        assert_eq!(player.qic, 0);
        assert_eq!(player.ore, 2);

        // 5. Ore to Token (1 Ore -> 1 Power Token in Area 1)
        let initial_tokens = player.power.total_tokens();
        assert!(execute_free_action(&mut player, crate::rules::FreeAction::OreToToken).is_ok());
        assert_eq!(player.ore, 1);
        assert_eq!(player.power.total_tokens(), initial_tokens + 1);
    }
}




