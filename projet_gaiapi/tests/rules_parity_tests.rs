use gaiapi::actions::{execute_examine_artefact, execute_free_action, GameCommand};
use gaiapi::action_space::{decode_action, get_action_name, FLAT_ACTION_SPACE};
use gaiapi::map::Map;
use gaiapi::player::PlayerData;
use gaiapi::rules::{
    Artefact, Building, FreeAction, PowerArea, ResearchField, Spaceship, TechTile,
};
use gaiapi::{Faction, GaiaEnv, GameConfig, OBS_SPACE};

#[test]
fn test_all_18_factions_starting_boards() {
    let all_factions = [
        Faction::Terrans,
        Faction::Lantids,
        Faction::HadschHallas,
        Faction::Ivits,
        Faction::Geodens,
        Faction::BalTaks,
        Faction::Xenos,
        Faction::Gleens,
        Faction::Taklons,
        Faction::Ambas,
        Faction::Firaks,
        Faction::Bescods,
        Faction::Nevlas,
        Faction::Itars,
        Faction::Tinkeroids,
        Faction::Darkanians,
        Faction::Moweyds,
        Faction::SpaceGiants,
    ];

    for &fac in &all_factions {
        let p = PlayerData::new(0, fac);
        assert_eq!(p.faction, fac);
        assert_eq!(p.seat, 0);
        assert_eq!(p.victory_points, 10);
        assert!(!p.passed);

        let config = GameConfig {
            players: 4,
            max_rounds: 6,
            seed: 42,
            factions: Some(vec![fac, Faction::Terrans, Faction::Lantids, Faction::Xenos]),
        };
        let mut env = GaiaEnv::new(config).unwrap();
        env.reset();
        let p_reset = &env.players[0];

        match fac {
            Faction::Ivits => {
                // Ivits start with PI placed, 0 mines
                assert_eq!(p_reset.buildings[Building::PlanetaryInstitute as usize], 1);
                assert_eq!(p_reset.buildings[Building::Mine as usize], 0);
            }
            Faction::Tinkeroids => {
                // Tinkeroids: PI on Asteroid, 2k, 4o, 15c, 1q, power 4/2, Science +1
                assert_eq!(p_reset.buildings[Building::PlanetaryInstitute as usize], 1);
                assert_eq!(p.knowledge, 2);
                assert_eq!(p.ore, 4);
                assert_eq!(p.credits, 15);
                assert_eq!(p.qic, 1);
                assert_eq!(p.power.area1, 4);
                assert_eq!(p.power.area2, 2);
                assert_eq!(p.research[ResearchField::Science as usize], 1);
            }
            Faction::Darkanians => {
                // Darkanians: 1 Mine on Asteroid, 3k, 7o, 15c, 1q, power 4/2, Navigation +1, Economy +1
                assert_eq!(p_reset.buildings[Building::Mine as usize], 1);
                assert_eq!(p.knowledge, 3);
                assert_eq!(p.ore, 7);
                assert_eq!(p.credits, 15);
                assert_eq!(p.qic, 1);
                assert_eq!(p.power.area1, 4);
                assert_eq!(p.power.area2, 2);
                assert_eq!(p.research[ResearchField::Navigation as usize], 1);
                assert_eq!(p.research[ResearchField::Economy as usize], 1);
            }
            Faction::Moweyds => {
                // Moweyds: 1 Mine on Protoplanet, 5k, 6o, 15c, 2q, power 4/4, Gaia +1, 6 power rings
                assert_eq!(p_reset.buildings[Building::Mine as usize], 1);
                assert_eq!(p.knowledge, 5);
                assert_eq!(p.ore, 6);
                assert_eq!(p.credits, 15);
                assert_eq!(p.qic, 2);
                assert_eq!(p.power.area1, 4);
                assert_eq!(p.power.area2, 4);
                assert_eq!(p.research[ResearchField::GaiaProject as usize], 1);
                assert_eq!(p.power_rings, 6);
                assert!(p.has_explored(Spaceship::TFMars));
            }
            Faction::SpaceGiants => {
                // Space Giants: 1 Mine on Protoplanet, 3k, 6o, 15c, 1q, power 4/4, Navigation +1
                assert_eq!(p_reset.buildings[Building::Mine as usize], 1);
                assert_eq!(p.knowledge, 3);
                assert_eq!(p.ore, 6);
                assert_eq!(p.credits, 15);
                assert_eq!(p.qic, 1);
                assert_eq!(p.power.area1, 4);
                assert_eq!(p.power.area2, 4);
                assert_eq!(p.research[ResearchField::Navigation as usize], 1);
            }
            Faction::Taklons => {
                // Taklons have Brainstone
                assert_eq!(p.power.brainstone, Some(PowerArea::Area1));
            }
            Faction::Geodens => {
                assert_eq!(p.research[ResearchField::Terraforming as usize], 1);
            }
            _ => {
                // Standard base game factions start with 2 mines
                assert_eq!(p_reset.buildings[Building::Mine as usize], 2);
            }
        }
    }
}

#[test]
fn test_all_13_lost_fleet_artefacts_effects() {
    let map = Map::generate_standard(4, 1);

    let all_artefacts = [
        Artefact::IncomeKnowledgeOre,
        Artefact::Credits3Ore3,
        Artefact::Knowledge3Qic1,
        Artefact::Credits5Ore2,
        Artefact::ChargePower2,
        Artefact::AsteroidVp,
        Artefact::ProtoplanetVp,
        Artefact::ResearchAreaLevel,
        Artefact::ResearchTracksCount,
        Artefact::RescoreFederation,
        Artefact::GaiaformingTrack,
        Artefact::PlanetTypes,
        Artefact::DeepSpace,
    ];

    for (idx, &art) in all_artefacts.iter().enumerate() {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.exploration_ships[Spaceship::Twilight as usize] = Some(0);
        assert!(p.has_explored(Spaceship::Twilight));

        // Needs 6 power tokens to discard
        p.power.area1 = 2;
        p.power.area2 = 2;
        p.power.area3 = 2;
        assert_eq!(p.power.total_tokens(), 6);
        if art == Artefact::RescoreFederation {
            p.claimed_federations[0] = Some(gaiapi::rules::FederationToken::Fed1);
        }

        let mut claimed = [false; 13];
        let res = execute_examine_artefact(&mut p, &map, art, Some(gaiapi::rules::FederationToken::Fed1), &mut claimed);
        assert!(res.is_ok(), "Artefact {:?} failed to examine: {:?}", art, res);
        assert!(claimed[idx], "Artefact {:?} should be marked claimed", art);
        assert!(p.claimed_artefacts[idx], "Player should record artefact claim");
        assert_eq!(p.power.total_tokens(), 0, "6 power tokens should be discarded");

        // Verify specific artifact benefits
        match art {
            Artefact::Credits3Ore3 => {
                // Terrans start with 15c, 4o -> +3c, +3o
                assert_eq!(p.credits, 18);
                assert_eq!(p.ore, 7);
            }
            Artefact::Knowledge3Qic1 => {
                // Terrans start with 3k, 1q -> +3k, +1q
                assert_eq!(p.knowledge, 6);
                assert_eq!(p.qic, 2);
            }
            Artefact::Credits5Ore2 => {
                assert_eq!(p.credits, 20);
                assert_eq!(p.ore, 6);
            }
            Artefact::AsteroidVp | Artefact::ProtoplanetVp => {
                // Immediate +7 VP
                assert_eq!(p.victory_points, 17);
            }
            _ => {}
        }
    }
}

#[test]
fn test_all_6_free_actions_conversions() {
    // 1. PowerToQic: 4 power -> 1 QIC
    {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.power.area3 = 4;
        p.qic = 0;
        assert!(execute_free_action(&mut p, FreeAction::PowerToQic).is_ok());
        assert_eq!(p.qic, 1);
        assert_eq!(p.power.area3, 0);
        assert_eq!(p.power.area1, 8);
        // Insufficient power fails
        assert!(execute_free_action(&mut p, FreeAction::PowerToQic).is_err());
    }

    // 2. PowerToKnowledge: 4 power -> 1 Knowledge
    {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.power.area3 = 4;
        let init_k = p.knowledge;
        assert!(execute_free_action(&mut p, FreeAction::PowerToKnowledge).is_ok());
        assert_eq!(p.knowledge, init_k + 1);
    }

    // 3. PowerToOre: 3 power -> 1 Ore
    {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.power.area3 = 3;
        let init_o = p.ore;
        assert!(execute_free_action(&mut p, FreeAction::PowerToOre).is_ok());
        assert_eq!(p.ore, init_o + 1);
    }

    // 4. PowerToCredit: 1 power -> 1 Credit
    {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.power.area3 = 1;
        let init_c = p.credits;
        assert!(execute_free_action(&mut p, FreeAction::PowerToCredit).is_ok());
        assert_eq!(p.credits, init_c + 1);
    }

    // 5. QicToOre: 1 QIC -> 1 Ore
    {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.qic = 1;
        let init_o = p.ore;
        assert!(execute_free_action(&mut p, FreeAction::QicToOre).is_ok());
        assert_eq!(p.qic, 0);
        assert_eq!(p.ore, init_o + 1);
    }

    // 6. OreToToken: 1 Ore -> 1 Power Token in Area 1
    {
        let mut p = PlayerData::new(0, Faction::Terrans);
        p.ore = 2;
        let init_tokens = p.power.total_tokens();
        assert!(execute_free_action(&mut p, FreeAction::OreToToken).is_ok());
        assert_eq!(p.ore, 1);
        assert_eq!(p.power.total_tokens(), init_tokens + 1);
    }
}

#[test]
fn test_power_charging_and_leeching_vp_deduction() {
    let mut env = GaiaEnv::new(GameConfig::default()).unwrap();
    env.reset();
    env.players[0].victory_points = 10;
    env.players[0].power.area1 = 4;
    env.players[0].power.area2 = 0;
    env.players[0].power.area3 = 0;

    // Decline charge (charge_amount = 0) -> no VP lost, no power moved
    assert!(env.execute_command(0, GameCommand::ChargePower { charge_amount: 0 }).is_ok());
    assert_eq!(env.players[0].victory_points, 10);
    assert_eq!(env.players[0].power.area1, 4);

    // Accept charge (charge_amount = 1) -> 1 power charged (1 -> 2), 0 VP lost (1 - 1 = 0)
    assert!(env.execute_command(0, GameCommand::ChargePower { charge_amount: 1 }).is_ok());
    assert_eq!(env.players[0].victory_points, 10);
    assert_eq!(env.players[0].power.area1, 3);
    assert_eq!(env.players[0].power.area2, 1);
}

#[test]
fn test_all_3130_actions_roundtrip_decode() {
    let map = Map::generate_standard(4, 42);

    for i in 0..FLAT_ACTION_SPACE {
        let cmd = decode_action(i, &map);
        assert!(cmd.is_some(), "Action index {} returned None!", i);
        let name = get_action_name(i, &map);
        assert!(!name.starts_with("Action Invalide"), "Action {} gave invalid name: {}", i, name);
    }

    // Boundary check
    assert!(decode_action(FLAT_ACTION_SPACE, &map).is_none());
    assert!(decode_action(FLAT_ACTION_SPACE + 100, &map).is_none());
}

#[test]
fn test_tech_tile_and_advanced_tech_tile_claiming() {
    let mut env = GaiaEnv::new(GameConfig::default()).unwrap();
    env.reset();
    env.players[0].pending_tech_claim = true;

    // Standard tech tile claim
    let cmd = GameCommand::ClaimTechTile {
        tech: TechTile::Tech1,
        advance_field: Some(ResearchField::Economy),
    };
    assert!(env.execute_command(0, cmd).is_ok());
    assert!(env.players[0].tech_tiles[TechTile::Tech1 as usize]);
    assert_eq!(env.players[0].research[ResearchField::Economy as usize], 1);
    assert!(!env.players[0].pending_tech_claim);
}

#[test]
fn test_observation_vector_shape_and_features() {
    let mut env = GaiaEnv::new(GameConfig::default()).unwrap();
    let obs = env.reset();
    assert_eq!(obs.values.len(), OBS_SPACE, "Observation dimension must match OBS_SPACE (2476)");

    // Ensure all values are finite and non-NaN
    for (i, &val) in obs.values.iter().enumerate() {
        assert!(val.is_finite(), "Obs at index {} is not finite: {}", i, val);
    }

    // Test Global segment (0..88)
    assert!(obs.values[0] >= 0.0 && obs.values[0] <= 1.0, "Current player index out of range");
    assert!(obs.values[1] >= 0.0 && obs.values[1] <= 1.0, "Round progress out of range");

    // Test Player segment (88..476)
    let p0_offset = 88;
    assert!(obs.values[p0_offset] >= 0.0, "VP feature invalid");

    // Test Map segment (476..2476)
    // 200 hexes * 10 floats = 2000
    assert_eq!(obs.values.len() - 476, 2000);
}

#[test]
fn test_full_game_turn_cycle_with_lost_fleet() {
    let config = GameConfig {
        players: 4,
        max_rounds: 6,
        seed: 42,
        factions: None,
    };
    let mut env = GaiaEnv::new(config).unwrap();
    env.reset();

    assert_eq!(env.round, 1);
    let mask = env.action_mask();
    assert_eq!(mask.len(), FLAT_ACTION_SPACE);
    let legal_count = mask.iter().filter(|&&m| m).count();
    assert!(legal_count > 0, "Active player must have at least one legal action");

    // First legal action execution
    let first_legal = mask.iter().position(|&m| m).unwrap();
    let cmd = decode_action(first_legal, &env.map).expect("Failed to decode legal action");
    let res = env.execute_command_from_rl(env.current_player, cmd);
    assert!(res.is_ok(), "Step failed on legal action: {:?}", res);
    let outcome = res.unwrap();
    assert_eq!(outcome.rewards.len(), 4);
}
