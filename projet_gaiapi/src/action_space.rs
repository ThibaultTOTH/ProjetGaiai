use crate::actions::GameCommand;
use crate::rules::{
    AdvTechTile, BoardAction, Building, FederationToken, ResearchField, SpecialAction,
    TechTile,
};

pub const A_BUILD_MINE_OFFSET: usize = 0;
pub const A_BUILD_MINE_COUNT: usize = 200;

pub const A_START_GAIA_OFFSET: usize = A_BUILD_MINE_OFFSET + A_BUILD_MINE_COUNT;
pub const A_START_GAIA_COUNT: usize = 200;

pub const A_UPGRADE_OFFSET: usize = A_START_GAIA_OFFSET + A_START_GAIA_COUNT;
pub const A_UPGRADE_COUNT: usize = 200 * 5;

pub const A_FEDERATION_OFFSET: usize = A_UPGRADE_OFFSET + A_UPGRADE_COUNT;
pub const A_FEDERATION_COUNT: usize = 6;

pub const A_ADVANCE_RESEARCH_OFFSET: usize = A_FEDERATION_OFFSET + A_FEDERATION_COUNT;
pub const A_ADVANCE_RESEARCH_COUNT: usize = 6;

pub const A_PASS_OFFSET: usize = A_ADVANCE_RESEARCH_OFFSET + A_ADVANCE_RESEARCH_COUNT;
pub const A_PASS_COUNT: usize = 10;

pub const A_CHARGE_POWER_OFFSET: usize = A_PASS_OFFSET + A_PASS_COUNT;
pub const A_CHARGE_POWER_COUNT: usize = 2;

pub const A_BOARD_ACTION_OFFSET: usize = A_CHARGE_POWER_OFFSET + A_CHARGE_POWER_COUNT;
pub const A_BOARD_ACTION_COUNT: usize = 10;

pub const A_SPECIAL_ACTION_OFFSET: usize = A_BOARD_ACTION_OFFSET + A_BOARD_ACTION_COUNT;
pub const A_SPECIAL_ACTION_COUNT: usize = 10;

pub const A_CLAIM_TECH_OFFSET: usize = A_SPECIAL_ACTION_OFFSET + A_SPECIAL_ACTION_COUNT;
pub const A_CLAIM_TECH_COUNT: usize = 9 * 6;

pub const A_CLAIM_ADV_TECH_OFFSET: usize = A_CLAIM_TECH_OFFSET + A_CLAIM_TECH_COUNT;
pub const A_CLAIM_ADV_TECH_COUNT: usize = 15 * 9 * 6;

pub const A_EXPLORE_SPACESHIP_OFFSET: usize = A_CLAIM_ADV_TECH_OFFSET + A_CLAIM_ADV_TECH_COUNT;
pub const A_EXPLORE_SPACESHIP_COUNT: usize = 4 * 200;

pub const A_SPACESHIP_BOARD_OFFSET: usize = A_EXPLORE_SPACESHIP_OFFSET + A_EXPLORE_SPACESHIP_COUNT;
pub const A_SPACESHIP_BOARD_COUNT: usize = 4 * 4;

pub const A_FREE_ACTION_OFFSET: usize = A_SPACESHIP_BOARD_OFFSET + A_SPACESHIP_BOARD_COUNT;
pub const A_FREE_ACTION_COUNT: usize = 6;

pub const FLAT_ACTION_SPACE: usize = A_FREE_ACTION_OFFSET + A_FREE_ACTION_COUNT;

pub fn decode_action(index: usize, map: &crate::map::Map) -> Option<GameCommand> {
    if index < A_START_GAIA_OFFSET {
        return Some(GameCommand::BuildMine {
            coord: map.coords[index],
        });
    }
    
    if index < A_UPGRADE_OFFSET {
        let i = index - A_START_GAIA_OFFSET;
        return Some(GameCommand::StartGaiaProject {
            coord: map.coords[i],
        });
    }

    if index < A_FEDERATION_OFFSET {
        let i = index - A_UPGRADE_OFFSET;
        let coord = map.coords[i / 5];
        let to = match i % 5 {
            0 => Building::TradingStation,
            1 => Building::ResearchLab,
            2 => Building::PlanetaryInstitute,
            3 => Building::Academy1,
            _ => Building::Academy2,
        };
        return Some(GameCommand::Upgrade { coord, to });
    }

    if index < A_ADVANCE_RESEARCH_OFFSET {
        let i = index - A_FEDERATION_OFFSET;
        // Use FormFederationAuto so the engine auto-computes a valid planet/satellite set
        let token = match i {
            0 => FederationToken::Fed1,
            1 => FederationToken::Fed2,
            2 => FederationToken::Fed3,
            3 => FederationToken::Fed4,
            4 => FederationToken::Fed5,
            _ => FederationToken::Fed6,
        };
        return Some(GameCommand::FormFederationAuto { token });
    }

    if index < A_PASS_OFFSET {
        let i = index - A_ADVANCE_RESEARCH_OFFSET;
        let field = match i {
            0 => ResearchField::Terraforming,
            1 => ResearchField::Navigation,
            2 => ResearchField::Intelligence,
            3 => ResearchField::GaiaProject,
            4 => ResearchField::Economy,
            _ => ResearchField::Science,
        };
        return Some(GameCommand::AdvanceResearch { field });
    }

    if index < A_CHARGE_POWER_OFFSET {
        let i = index - A_PASS_OFFSET;
        return Some(GameCommand::Pass { new_booster: Some(i as u8) });
    }

    if index < A_BOARD_ACTION_OFFSET {
        let i = index - A_CHARGE_POWER_OFFSET;
        return Some(GameCommand::ChargePower { charge_amount: if i == 0 { 0 } else { 1 } });
    }

    if index < A_SPECIAL_ACTION_OFFSET {
        let i = index - A_BOARD_ACTION_OFFSET;
        let action = match i {
            0 => BoardAction::Power1,
            1 => BoardAction::Power2,
            2 => BoardAction::Power3,
            3 => BoardAction::Power4,
            4 => BoardAction::Power5,
            5 => BoardAction::Power6,
            6 => BoardAction::Power7,
            7 => BoardAction::Qic1,
            8 => BoardAction::Qic2,
            _ => BoardAction::Qic3,
        };
        return Some(GameCommand::BoardAction { action, target_mine: None, rescore_token: None });
    }

    if index < A_CLAIM_TECH_OFFSET {
        let i = index - A_SPECIAL_ACTION_OFFSET;
        let action = match i {
            0 => SpecialAction::AmbasPiSwap,
            1 => SpecialAction::FiraksDowngradeLab,
            2 => SpecialAction::BescodsAdvanceLowest,
            3 => SpecialAction::IvitsSpaceStation,
            4 => SpecialAction::SpaceGiantsTerraform,
            5 => SpecialAction::Tech9Charge4Power,
            6 => SpecialAction::AdvTech3QicCredit,
            7 => SpecialAction::AdvTech11Gain3Ore,
            8 => SpecialAction::AdvTech13Gain3Knowledge,
            _ => SpecialAction::Booster5TemporaryRange,
        };
        return Some(GameCommand::SpecialAction { action, target_coord: None, target_field: None });
    }

    if index < A_CLAIM_ADV_TECH_OFFSET {
        let i = index - A_CLAIM_TECH_OFFSET;
        let tech_idx = i / 6;
        let field_idx = i % 6;
        let tech = match tech_idx {
            0 => TechTile::Tech1, 1 => TechTile::Tech2, 2 => TechTile::Tech3,
            3 => TechTile::Tech4, 4 => TechTile::Tech5, 5 => TechTile::Tech6,
            6 => TechTile::Tech7, 7 => TechTile::Tech8, _ => TechTile::Tech9,
        };
        let field = match field_idx {
            0 => ResearchField::Terraforming, 1 => ResearchField::Navigation, 2 => ResearchField::Intelligence,
            3 => ResearchField::GaiaProject, 4 => ResearchField::Economy, _ => ResearchField::Science,
        };
        return Some(GameCommand::ClaimTechTile { tech, advance_field: Some(field) });
    }

    if index < A_EXPLORE_SPACESHIP_OFFSET {
        let i = index - A_CLAIM_ADV_TECH_OFFSET;
        let adv_idx = i / 54;
        let rem = i % 54;
        let cover_idx = rem / 6;
        let field_idx = rem % 6;

        let adv_tech = match adv_idx {
            0 => AdvTechTile::AdvTech1, 1 => AdvTechTile::AdvTech2, 2 => AdvTechTile::AdvTech3,
            3 => AdvTechTile::AdvTech4, 4 => AdvTechTile::AdvTech5, 5 => AdvTechTile::AdvTech6,
            6 => AdvTechTile::AdvTech7, 7 => AdvTechTile::AdvTech8, 8 => AdvTechTile::AdvTech9,
            9 => AdvTechTile::AdvTech10, 10 => AdvTechTile::AdvTech11, 11 => AdvTechTile::AdvTech12,
            12 => AdvTechTile::AdvTech13, 13 => AdvTechTile::AdvTech14, _ => AdvTechTile::AdvTech15,
        };
        let cover_tech = match cover_idx {
            0 => TechTile::Tech1, 1 => TechTile::Tech2, 2 => TechTile::Tech3,
            3 => TechTile::Tech4, 4 => TechTile::Tech5, 5 => TechTile::Tech6,
            6 => TechTile::Tech7, 7 => TechTile::Tech8, _ => TechTile::Tech9,
        };
        let field = match field_idx {
            0 => ResearchField::Terraforming, 1 => ResearchField::Navigation, 2 => ResearchField::Intelligence,
            3 => ResearchField::GaiaProject, 4 => ResearchField::Economy, _ => ResearchField::Science,
        };
        return Some(GameCommand::ClaimAdvTechTile { adv_tech, cover_tech, field });
    }

    // ExploreSpaceship: 4 ships x 200 coords = 800 actions
    if index < A_SPACESHIP_BOARD_OFFSET {
        let i = index - A_EXPLORE_SPACESHIP_OFFSET;
        let ship_idx = i / 200;
        let coord_idx = i % 200;
        let ship = match ship_idx {
            0 => crate::rules::Spaceship::Twilight,
            1 => crate::rules::Spaceship::Rebellion,
            2 => crate::rules::Spaceship::TFMars,
            _ => crate::rules::Spaceship::Eclipse,
        };
        return Some(GameCommand::ExploreSpaceship { ship, coord: map.coords[coord_idx] });
    }

    // SpaceshipBoardAction: 4 ships x 4 action types = 16 actions
    if index < A_FREE_ACTION_OFFSET {
        let i = index - A_SPACESHIP_BOARD_OFFSET;
        let ship_idx = i / 4;
        let action_idx = i % 4;
        let ship = match ship_idx {
            0 => crate::rules::Spaceship::Twilight,
            1 => crate::rules::Spaceship::Rebellion,
            2 => crate::rules::Spaceship::TFMars,
            _ => crate::rules::Spaceship::Eclipse,
        };
        let action_type = match action_idx {
            0 => crate::rules::SpaceshipActionType::Qic,
            1 => crate::rules::SpaceshipActionType::Power,
            2 => crate::rules::SpaceshipActionType::Knowledge,
            _ => crate::rules::SpaceshipActionType::Credit,
        };
        return Some(GameCommand::SpaceshipBoardAction {
            ship,
            action_type,
            target_coord: None,
            target_field: None,
            rescore_token: None,
        });
    }

    // FreeAction: 6 most common universal free actions
    if index < FLAT_ACTION_SPACE {
        let i = index - A_FREE_ACTION_OFFSET;
        let action = match i {
            0 => crate::rules::FreeAction::PowerToQic,
            1 => crate::rules::FreeAction::PowerToKnowledge,
            2 => crate::rules::FreeAction::PowerToOre,
            3 => crate::rules::FreeAction::PowerToCredit,
            4 => crate::rules::FreeAction::QicToOre,
            _ => crate::rules::FreeAction::OreToToken,
        };
        return Some(GameCommand::FreeAction { action });
    }

    None
}

pub fn get_action_name(index: usize, map: &crate::map::Map) -> String {
    if let Some(cmd) = decode_action(index, map) {
        match cmd {
            crate::actions::GameCommand::Pass { new_booster } => {
                match new_booster {
                    Some(b) => format!("Passer (Prendre Booster {})", b),
                    None => "Passer son tour".to_string(),
                }
            }
            crate::actions::GameCommand::BuildMine { coord } => {
                format!("Construire Mine en [q:{}, r:{}, s:{}]", coord.q, coord.r, coord.s)
            }
            crate::actions::GameCommand::StartGaiaProject { coord } => {
                format!("Lancer Projet Gaïa en [q:{}, r:{}, s:{}]", coord.q, coord.r, coord.s)
            }
            crate::actions::GameCommand::Upgrade { coord, to } => {
                format!("Améliorer vers {:?} en [q:{}, r:{}, s:{}]", to, coord.q, coord.r, coord.s)
            }
            crate::actions::GameCommand::FormFederation { token, .. } => {
                format!("Fonder Fédération (Jeton {:?})", token)
            }
            crate::actions::GameCommand::AdvanceResearch { field } => {
                format!("Recherche: Avancer sur {:?}", field)
            }
            crate::actions::GameCommand::ChargePower { charge_amount } => {
                format!("Charger {} Puissance", charge_amount)
            }
            crate::actions::GameCommand::BoardAction { action, .. } => {
                format!("Action Plateau Puissance: {:?}", action)
            }
            crate::actions::GameCommand::SpecialAction { action, .. } => {
                format!("Action Spéciale de Faction: {:?}", action)
            }
            crate::actions::GameCommand::ClaimTechTile { tech, advance_field } => {
                format!("Prendre Tuile Tech {:?} (Avancée: {:?})", tech, advance_field)
            }
            crate::actions::GameCommand::ClaimAdvTechTile { adv_tech, field, .. } => {
                format!("Prendre Tuile Tech Avancée {:?} sur {:?}", adv_tech, field)
            }
            crate::actions::GameCommand::ExploreSpaceship { ship, coord } => {
                format!("Explorer avec Vaisseau {:?} vers [q:{}, r:{}, s:{}]", ship, coord.q, coord.r, coord.s)
            }
            crate::actions::GameCommand::SpaceshipBoardAction { ship, action_type, .. } => {
                format!("Action Vaisseau {:?} ({:?})", ship, action_type)
            }
            crate::actions::GameCommand::FreeAction { action } => {
                format!("Action Libre: {:?}", action)
            }
            crate::actions::GameCommand::ExamineArtefact { artefact, .. } => {
                format!("Examiner Artéfact {:?}", artefact)
            }
            crate::actions::GameCommand::FormFederationAuto { token } => {
                format!("Fonder Fédération Auto (Jeton {:?})", token)
            }
        }
    } else {
        format!("Action Invalide #{}", index)
    }
}




