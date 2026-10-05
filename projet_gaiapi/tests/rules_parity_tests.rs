use gaiapi::actions::{execute_examine_artefact, execute_free_action, GameCommand};
use gaiapi::action_space::{decode_action, encode_action, get_action_name, FLAT_ACTION_SPACE};
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

#[test]
fn test_action_encoding_bijection_roundtrip() {
    let map = Map::generate_standard(4, 42);
    let mut roundtripped = 0;
    for i in 0..FLAT_ACTION_SPACE {
        if let Some(cmd) = decode_action(i, &map) {
            let is_valid_map_hex = match &cmd {
                GameCommand::BuildMine { coord } | GameCommand::StartGaiaProject { coord } => {
                    map.index_of(*coord).is_some() && (i % 200 < map.count)
                }
                GameCommand::Upgrade { coord, .. } => {
                    let hex_idx = (i - 400) / 5;
                    map.index_of(*coord).is_some() && (hex_idx < map.count)
                }
                GameCommand::ExploreSpaceship { coord, .. } => {
                    let hex_idx = (i - 2308) % 200;
                    map.index_of(*coord).is_some() && (hex_idx < map.count)
                }
                _ => true,
            };
            if is_valid_map_hex {
                let encoded = encode_action(&cmd, &map);
                assert_eq!(encoded, Some(i), "Action {} failed roundtrip encoding! Decoded: {:?}", i, cmd);
                roundtripped += 1;
            }
        }
    }
    assert!(roundtripped >= 2800, "Expected >= 2800 valid actions roundtripped, got {}", roundtripped);
}

#[test]
fn test_canonical_coordinates_stability_across_seeds() {
    let mut map1 = Map::generate_standard(4, 100);
    map1.sort_canonical();
    
    // Verify canonical sort order: q ascending, then r ascending
    for i in 1..map1.count {
        let prev = map1.coords[i - 1];
        let curr = map1.coords[i];
        assert!(
            curr.q > prev.q || (curr.q == prev.q && curr.r > prev.r),
            "Hex coordinates not in canonical order at {}: {:?} then {:?}",
            i, prev, curr
        );
        // Verify index_of binary search works
        assert_eq!(map1.index_of(curr), Some(i));
    }
}

#[test]
fn test_form_federation_auto_graph_routing_and_power_discard() {
    let mut map = Map::generate_standard(4, 42);
    let mut player = PlayerData::new(0, Faction::Terrans);
    player.power.area1 = 4;
    player.power.area2 = 4;
    player.power.area3 = 0;
    assert_eq!(player.power.total_tokens(), 8);

    // Place a Planetary Institute (power 3 in Gaia Project) on hex 0
    map.hexes[0].player = Some(0);
    map.hexes[0].building = Some(Building::PlanetaryInstitute);
    player.buildings[Building::PlanetaryInstitute as usize] = 1;

    // Place a Trading Station (power 2) on hex 1
    map.hexes[1].player = Some(0);
    map.hexes[1].building = Some(Building::TradingStation);
    player.buildings[Building::TradingStation as usize] = 1;

    // Total power = 3 + 2 = 5 < 7 -> should fail with InsufficientFederationPower
    let res = gaiapi::actions::execute_form_federation_auto(&mut player, &mut map, gaiapi::rules::FederationToken::Fed1);
    assert!(matches!(res, Err(gaiapi::actions::ActionError::InsufficientFederationPower { .. })));

    // Place another Trading Station (power 2) on hex 2 (adjacent to hex 1)
    map.hexes[2].player = Some(0);
    map.hexes[2].building = Some(Building::TradingStation);
    player.buildings[Building::TradingStation as usize] = 2;

    // Total power = 3 + 2 + 2 = 7. Execute federation auto
    let initial_sats = player.satellites;
    let initial_tokens = player.power.total_tokens();
    let res = gaiapi::actions::execute_form_federation_auto(&mut player, &mut map, gaiapi::rules::FederationToken::Fed1);
    assert!(res.is_ok(), "Auto federation should succeed: {:?}", res);
    assert!(player.claimed_federations[0].is_some(), "Federation token should be claimed");

    // Satellites placed must equal power tokens discarded
    let sats_placed = player.satellites - initial_sats;
    let tokens_lost = initial_tokens - player.power.total_tokens();
    assert_eq!(sats_placed, tokens_lost, "Tokens discarded must match satellites placed");
}

#[test]
fn test_passive_leech_non_punitive_when_power_full() {
    let mut player = PlayerData::new(0, Faction::Terrans);
    player.victory_points = 15;
    // Bowls 1 and 2 are empty, Bowl 3 has all 12 power tokens
    player.power.area1 = 0;
    player.power.area2 = 0;
    player.power.area3 = 12;
    assert!(!player.power.can_charge(), "Player with bowls 1 & 2 empty cannot charge");

    // Attempt to leech 3 power
    let res = gaiapi::actions::execute_leech(&mut player, 3);
    assert!(res.is_ok());
    // Victory points must NOT be penalized because no power was charged!
    assert_eq!(player.victory_points, 15, "VP must not be lost when power cannot be charged");

    // Now test with 1 power in Bowl 2: offered 3 power
    player.power.area2 = 1;
    player.power.area3 = 11;
    assert!(player.power.can_charge());
    let res = gaiapi::actions::execute_leech(&mut player, 3);
    assert!(res.is_ok());
    // Only 1 power could be charged (area2 -> area3), so charged = 1, cost = (1-1) = 0 VP!
    assert_eq!(player.victory_points, 15, "Charging 1 power costs 0 VP");
    assert_eq!(player.power.area2, 0);
    assert_eq!(player.power.area3, 12);
}

#[test]
fn test_observe_egocentric_symmetry() {
    let mut env = GaiaEnv::new(GameConfig {
        players: 4,
        max_rounds: 6,
        seed: 42,
        factions: None,
    }).unwrap();
    env.reset();

    // Verify Seat 0 perspective
    env.current_player = 0;
    let obs0 = env.observe_egocentric();
    assert_eq!(obs0.values.len(), OBS_SPACE);
    let p0_block = 88;
    assert_eq!(obs0.values[p0_block + 1], 0.0);

    // Verify Seat 2 perspective
    env.current_player = 2;
    let obs2 = env.observe_egocentric();
    assert_eq!(obs2.values.len(), OBS_SPACE);
    assert_eq!(obs2.values[p0_block + 1], 0.0);
    let seat2_faction = (env.players[2].faction as u8 as f32) / 17.0;
    assert_eq!(obs2.values[p0_block + 0], seat2_faction);
}

#[test]
fn test_action_mask_fast_and_accurate() {
    let mut env = GaiaEnv::new(GameConfig::default()).unwrap();
    env.reset();

    let start = std::time::Instant::now();
    let mask = env.action_mask();
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 50, "action_mask should take < 50ms, took {:?}", elapsed);

    let legal_cmds = env.legal_commands(env.current_player);
    for cmd in &legal_cmds {
        if let Some(idx) = gaiapi::action_space::encode_action(cmd, &env.map) {
            assert!(mask[idx], "Action index {} encoded from {:?} must be true in mask", idx, cmd);
        }
    }
}

#[test]
fn test_interactive_leeching_queue_flow() {
    let mut env = GaiaEnv::new(GameConfig::default()).unwrap();
    env.reset();

    // Find Player 0's mine on the map
    let p0_hex = (0..env.map.count).find(|&i| env.map.hexes[i].player == Some(0)).unwrap();
    // Find an empty hex adjacent to p0_hex (distance 1)
    let p1_hex = (0..env.map.count).find(|&i| env.map.hexes[i].player.is_none() && env.map.distance(p0_hex, i) == 1).unwrap();
    // Set a TradingStation for Player 1 on p1_hex (power value 2)
    env.map.hexes[p1_hex].player = Some(1);
    env.map.hexes[p1_hex].building = Some(Building::TradingStation);
    env.players[1].buildings[Building::TradingStation as usize] = 1;

    // Find another empty hex adjacent to p1_hex for Player 0 to build a mine (distance 1 <= 2)
    let build_hex = (0..env.map.count).find(|&i| env.map.hexes[i].player.is_none() && env.map.distance(p1_hex, i) == 1 && env.map.distance(p0_hex, i) <= 2).unwrap();
    let build_coord = env.map.coords[build_hex];
    // Make planet colonizable by Player 0 (home planet type)
    let home_planet = gaiapi::rules::faction_planet(env.players[0].faction);
    env.map.hexes[build_hex].planet = home_planet;

    // Give Player 0 ample resources
    env.players[0].credits = 15;
    env.players[0].ore = 10;
    env.players[0].qic = 5;

    // Give Player 1 power to charge and VP
    env.players[1].victory_points = 10;
    env.players[1].power.area1 = 2;
    env.players[1].power.area2 = 2;
    env.players[1].power.area3 = 0;

    env.current_player = 0;
    let res = env.execute_command_from_rl(0, GameCommand::BuildMine { coord: build_coord });
    assert!(res.is_ok(), "BuildMine failed: {:?}", res);

    // Turn should NOT advance to Player 1's normal turn yet; Player 1 is prompted to leech!
    assert_eq!(env.current_player, 1);
    assert_eq!(env.pending_leeches.len(), 1);
    assert_eq!(env.pending_leeches[0].seat, 1);
    assert_eq!(env.pending_leeches[0].power_value, 2);

    // Legal commands for Player 1 during leeching: ONLY Decline and ChargePower
    let legal = env.legal_commands(1);
    assert_eq!(legal.len(), 2);
    assert!(legal.contains(&GameCommand::DeclineLeech));
    assert!(legal.contains(&GameCommand::ChargePower { charge_amount: 2 }));

    let mask = env.action_mask();
    assert!(mask[1422]); // Decline
    assert!(mask[1423]); // Charge
    assert!(!mask[0]);   // Mine is illegal during leech reaction

    // Player 1 accepts leech
    let leech_res = env.execute_command_from_rl(1, GameCommand::ChargePower { charge_amount: 2 });
    assert!(leech_res.is_ok(), "Leech ChargePower failed: {:?}", leech_res);

    // Leech queue is now empty
    assert!(env.pending_leeches.is_empty());
    // Player 1 paid 1 VP (2 pw - 1 = 1 VP) and charged 2 power from area 1 to area 2
    assert_eq!(env.players[1].victory_points, 9);
    assert_eq!(env.players[1].power.area1, 0);
    assert_eq!(env.players[1].power.area2, 4);

    // Turn now advanced to next player (Player 1) for normal play
    assert_eq!(env.current_player, 1);
    let normal_cmds = env.legal_commands(1);
    assert!(normal_cmds.len() > 2);
}

