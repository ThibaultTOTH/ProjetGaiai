use gaiapi::actions::{ActionError, GameCommand};
use gaiapi::board::HexCoord;
use gaiapi::map::Map;
use gaiapi::player::PowerBowls;
use gaiapi::rules::{
    BoardAction, Building, FederationToken, Planet, ResearchField, Spaceship,
    SpaceshipActionType,
};
use gaiapi::{Faction, GaiaEnv, GameConfig};

#[test]
fn test_comprehensive_adversarial_cheat_game() {
    // Initialize a 2-player game: Player 0 = Terrans, Player 1 = BalTaks
    let mut env = GaiaEnv::new(GameConfig {
        players: 2,
        seed: 12345,
        ..Default::default()
    })
    .unwrap();

    env.players[0].faction = Faction::Terrans;
    env.players[1].faction = Faction::BalTaks;

    // Clear board and place controlled test hexes
    let mut map = Map::empty();
    map.count = 8;
    map.coords[0] = HexCoord::origin(); // Player 0 home mine (Terra)
    map.coords[1] = HexCoord::new(1, -1, 0).unwrap(); // Adjacent Terra
    map.coords[2] = HexCoord::new(2, -2, 0).unwrap(); // Player 1 home mine (Volcanic)
    map.coords[3] = HexCoord::new(3, -3, 0).unwrap(); // Transdim planet (dist 3 from p0)
    map.coords[4] = HexCoord::new(4, -4, 0).unwrap(); // Asteroid (dist 4 from p0)
    map.coords[5] = HexCoord::new(5, -5, 0).unwrap(); // Deep space hex with Spaceship Twilight
    map.coords[6] = HexCoord::new(0, 1, -1).unwrap(); // Empty space
    map.coords[7] = HexCoord::new(0, 2, -2).unwrap(); // Protoplanet
    map.recompute_adjacency();

    // Setup hex 0: Player 0 Mine
    map.hexes[0].planet = Planet::Terra;
    map.hexes[0].building = Some(Building::Mine);
    map.hexes[0].player = Some(0);
    env.players[0].buildings[Building::Mine as usize] = 1;

    // Setup hex 2: Player 1 Mine
    map.hexes[2].planet = Planet::Volcanic;
    map.hexes[2].building = Some(Building::Mine);
    map.hexes[2].player = Some(1);
    env.players[1].buildings[Building::Mine as usize] = 1;

    // Setup other hexes
    map.hexes[1].planet = Planet::Terra;
    map.hexes[3].planet = Planet::Transdim;
    map.hexes[4].planet = Planet::Asteroid;
    map.hexes[5].spaceship = Some(Spaceship::Twilight);
    map.hexes[7].planet = Planet::Protoplanet;

    env.map = map;

    println!("=== PHASE 1: TESTING BUILD MINE CHEATS ===");

    // Cheat 1: Build mine on an opponent's colonized planet (hex 2 belongs to Player 1)
    let cheat_occupy = GameCommand::BuildMine {
        coord: env.map.coords[2],
    };
    let res = env.execute_command(0, cheat_occupy);
    assert_eq!(
        res,
        Err(ActionError::HexAlreadyOccupied),
        "Failed to reject building on opponent's occupied hex"
    );

    // Cheat 2: Build mine on empty space without planet (hex 6 is empty space)
    let cheat_no_planet = GameCommand::BuildMine {
        coord: env.map.coords[6],
    };
    let res = env.execute_command(0, cheat_no_planet);
    assert_eq!(
        res,
        Err(ActionError::HexNotColonizable),
        "Failed to reject building on empty space without planet"
    );

    // Cheat 3: Build mine directly on Transdim planet (hex 3 is Transdim)
    let cheat_transdim_mine = GameCommand::BuildMine {
        coord: env.map.coords[3],
    };
    let res = env.execute_command(0, cheat_transdim_mine);
    assert_eq!(
        res,
        Err(ActionError::CannotBuildOnTransdimDirectly),
        "Failed to reject building directly on Transdim planet"
    );

    // Cheat 4: Build mine on a planet too far away without enough QIC
    // Hex 4 is dist 4 from Hex 0. Player 0 has range 1, 1 QIC (reaches dist 1 + 2 = 3). Dist 4 needs 2 QIC!
    // Since Hex 4 is Asteroid, unlock 1 Gaiaformer so the gaiaformer check passes and QIC is checked!
    env.players[0].gaiaformers_unlocked = 1;
    env.players[0].qic = 1;
    let cheat_too_far = GameCommand::BuildMine {
        coord: env.map.coords[4],
    };
    let res = env.execute_command(0, cheat_too_far);
    assert_eq!(
        res,
        Err(ActionError::InsufficientQic { needed: 2, have: 1 }),
        "Failed to reject building out of range without QIC"
    );

    // Cheat 5: Build mine on Asteroid without available Gaiaformer
    // Hex 4 is Asteroid. Reset gaiaformers_unlocked to 0.
    env.players[0].gaiaformers_unlocked = 0;
    env.players[0].qic = 3; // Give enough QIC to reach dist 4
    assert_eq!(env.players[0].available_gaiaformers(), 0);
    let cheat_asteroid_no_gf = GameCommand::BuildMine {
        coord: env.map.coords[4],
    };
    let res = env.execute_command(0, cheat_asteroid_no_gf);
    assert_eq!(
        res,
        Err(ActionError::NoGaiaformerAvailable),
        "Failed to reject asteroid mine without available gaiaformer"
    );

    // Cheat 6: Build mine without enough credits
    env.players[0].credits = 1; // Needs 2 credits
    env.players[0].ore = 5;
    let cheat_no_credits = GameCommand::BuildMine {
        coord: env.map.coords[1],
    };
    let res = env.execute_command(0, cheat_no_credits);
    assert_eq!(
        res,
        Err(ActionError::InsufficientCredits { needed: 2, have: 1 }),
        "Failed to reject mine without enough credits"
    );

    // Cheat 7: Build mine without enough ore
    env.players[0].credits = 5;
    env.players[0].ore = 0; // Needs 1 ore for Terra
    let cheat_no_ore = GameCommand::BuildMine {
        coord: env.map.coords[1],
    };
    let res = env.execute_command(0, cheat_no_ore);
    assert_eq!(
        res,
        Err(ActionError::InsufficientOre { needed: 1, have: 0 }),
        "Failed to reject mine without enough ore"
    );

    // LEGAL MOVE 1: Player 0 legitimately builds a Mine on Hex 1 (Terra)
    env.players[0].credits = 15;
    env.players[0].ore = 4;
    let valid_mine = GameCommand::BuildMine {
        coord: env.map.coords[1],
    };
    assert!(
        env.execute_command(0, valid_mine).is_ok(),
        "Legal mine build failed"
    );
    assert_eq!(env.players[0].credits, 13); // 15 - 2
    assert_eq!(env.players[0].ore, 3); // 4 - 1
    assert_eq!(env.players[0].buildings[Building::Mine as usize], 2);
    assert_eq!(env.map.hexes[1].building, Some(Building::Mine));
    assert_eq!(env.map.hexes[1].player, Some(0));

    println!("=== PHASE 2: TESTING GAIA PROJECT CHEATS ===");

    // Cheat 8: Start Gaia Project on a non-Transdim planet (hex 1 is Terra)
    let cheat_gp_wrong_planet = GameCommand::StartGaiaProject {
        coord: env.map.coords[1],
    };
    let res = env.execute_command(0, cheat_gp_wrong_planet);
    assert_eq!(
        res,
        Err(ActionError::NoGaiaformerAvailable), // Player 0 currently has 0 Gaiaformers
        "Failed to reject starting Gaia project without Gaiaformer"
    );

    // Give 1 Gaiaformer but 0 research level
    env.players[0].gaiaformers_unlocked = 1;
    env.players[0].research[ResearchField::GaiaProject as usize] = 0;
    let cheat_gp_no_research = GameCommand::StartGaiaProject {
        coord: env.map.coords[3],
    };
    let res = env.execute_command(0, cheat_gp_no_research);
    assert_eq!(
        res,
        Err(ActionError::GaiaProjectResearchRequired),
        "Failed to reject Gaia project when Gaia Project research is at level 0"
    );

    // Level 1 research: requires moving 6 power to Gaia area
    env.players[0].research[ResearchField::GaiaProject as usize] = 1;
    env.players[0].power = PowerBowls::new(2, 0, 0, 0); // Only 2 total power tokens!
    let cheat_gp_insufficient_power = GameCommand::StartGaiaProject {
        coord: env.map.coords[3],
    };
    let res = env.execute_command(0, cheat_gp_insufficient_power);
    assert_eq!(
        res,
        Err(ActionError::InsufficientPowerForGaiaProject {
            needed: 6,
            have: 2
        }),
        "Failed to reject Gaia project with insufficient power tokens"
    );

    println!("=== PHASE 3: TESTING UPGRADE CHEATS ===");

    // Cheat 9: Upgrading opponent's structure (hex 2 belongs to Player 1)
    let cheat_upgrade_opponent = GameCommand::Upgrade {
        coord: env.map.coords[2],
        to: Building::TradingStation,
    };
    let res = env.execute_command(0, cheat_upgrade_opponent);
    assert_eq!(
        res,
        Err(ActionError::NotYourStructure),
        "Failed to reject upgrading opponent's structure"
    );

    // Cheat 10: Invalid upgrade path: Mine directly to Research Lab
    let cheat_invalid_path = GameCommand::Upgrade {
        coord: env.map.coords[0],
        to: Building::ResearchLab,
    };
    let res = env.execute_command(0, cheat_invalid_path);
    assert_eq!(
        res,
        Err(ActionError::InvalidUpgradePath {
            from: Some(Building::Mine),
            to: Building::ResearchLab
        }),
        "Failed to reject illegal direct upgrade Mine -> ResearchLab"
    );

    // Cheat 11: Invalid upgrade path: Mine directly to Academy
    let cheat_mine_to_academy = GameCommand::Upgrade {
        coord: env.map.coords[0],
        to: Building::Academy1,
    };
    let res = env.execute_command(0, cheat_mine_to_academy);
    assert_eq!(
        res,
        Err(ActionError::InvalidUpgradePath {
            from: Some(Building::Mine),
            to: Building::Academy1
        }),
        "Failed to reject illegal direct upgrade Mine -> Academy"
    );

    // LEGAL MOVE 2: Upgrade Mine at Hex 0 to Trading Station
    // Hex 0 is dist 2 from Player 1's mine at Hex 2! Neighbour discount applies: 3 credits, 2 ore!
    env.players[0].credits = 10;
    env.players[0].ore = 5;
    let valid_upgrade = GameCommand::Upgrade {
        coord: env.map.coords[0],
        to: Building::TradingStation,
    };
    assert!(
        env.execute_command(0, valid_upgrade).is_ok(),
        "Legal upgrade to Trading Station failed"
    );
    assert_eq!(env.players[0].credits, 7); // 10 - 3 (discounted!)
    assert_eq!(env.players[0].ore, 3); // 5 - 2
    assert_eq!(env.map.hexes[0].building, Some(Building::TradingStation));
    assert_eq!(
        env.players[0].buildings[Building::TradingStation as usize],
        1
    );
    assert_eq!(env.players[0].buildings[Building::Mine as usize], 1); // 1 mine upgraded

    println!("=== PHASE 4: TESTING RESEARCH CHEATS ===");

    // Cheat 12: Advance research with insufficient knowledge
    env.players[0].knowledge = 3; // Needs 4
    let cheat_advance_no_k = GameCommand::AdvanceResearch {
        field: ResearchField::Science,
    };
    let res = env.execute_command(0, cheat_advance_no_k);
    assert_eq!(
        res,
        Err(ActionError::InsufficientKnowledge { needed: 4, have: 3 }),
        "Failed to reject research advancement without 4 knowledge"
    );

    // Cheat 13: Bal T'aks advancing Navigation beyond level 1 without Planetary Institute
    env.players[1].knowledge = 10;
    env.players[1].research[ResearchField::Navigation as usize] = 1;
    assert_eq!(
        env.players[1].buildings[Building::PlanetaryInstitute as usize],
        0
    );
    let cheat_baltaks_nav = GameCommand::AdvanceResearch {
        field: ResearchField::Navigation,
    };
    let res = env.execute_command(1, cheat_baltaks_nav);
    assert_eq!(
        res,
        Err(ActionError::BalTaksNavigationRestricted),
        "Failed to enforce Bal T'aks Navigation research restriction"
    );

    // Cheat 14: Advancing to Level 5 without a Green Federation token
    env.players[0].knowledge = 10;
    env.players[0].research[ResearchField::Science as usize] = 4;
    assert_eq!(env.players[0].green_federation_tokens, 0);
    let cheat_l5_no_green_fed = GameCommand::AdvanceResearch {
        field: ResearchField::Science,
    };
    let res = env.execute_command(0, cheat_l5_no_green_fed);
    assert_eq!(
        res,
        Err(ActionError::FederationTokenRequiredForLevel5),
        "Failed to require green federation token for Level 5 research"
    );

    // LEGAL MOVE 3: Player 0 claims a green federation token and reaches Level 5 Science
    env.players[0].claim_federation_token(FederationToken::Fed2);
    assert_eq!(env.players[0].green_federation_tokens, 1);
    let valid_l5_advance = GameCommand::AdvanceResearch {
        field: ResearchField::Science,
    };
    assert!(
        env.execute_command(0, valid_l5_advance).is_ok(),
        "Legal advancement to Level 5 failed"
    );
    assert_eq!(env.players[0].research[ResearchField::Science as usize], 5);
    assert_eq!(env.players[0].green_federation_tokens, 0);
    assert_eq!(env.players[0].gray_federation_tokens, 1);
    assert_eq!(
        env.research_level_5_claimed[ResearchField::Science as usize],
        Some(0)
    );

    // Cheat 15: Player 1 trying to claim the SAME Level 5 Science already claimed by Player 0 (Exclusivity!)
    env.players[1].claim_federation_token(FederationToken::Fed2);
    env.players[1].knowledge = 10;
    env.players[1].research[ResearchField::Science as usize] = 4;
    let cheat_l5_already_claimed = GameCommand::AdvanceResearch {
        field: ResearchField::Science,
    };
    let res = env.execute_command(1, cheat_l5_already_claimed);
    assert_eq!(
        res,
        Err(ActionError::ResearchLevel5AlreadyClaimed(
            ResearchField::Science
        )),
        "Failed to enforce Level 5 research exclusivity between players"
    );

    println!("=== PHASE 5: TESTING BOARD ACTION CHEATS ===");

    // Cheat 16: Taking Power board action with insufficient power
    // Power1 (3 knowledge) costs 7 power. Player 0 has 0 in Bowl 3!
    env.players[0].power = PowerBowls::new(4, 2, 0, 0);
    assert_eq!(env.players[0].power.spendable_power(), 0);
    let cheat_power1_no_pw = GameCommand::BoardAction {
        action: BoardAction::Power1,
        target_mine: None,
        rescore_token: None,
    };
    let res = env.execute_command(0, cheat_power1_no_pw);
    assert_eq!(
        res,
        Err(ActionError::InsufficientPowerForBoardAction {
            needed: 7,
            have: 0
        }),
        "Failed to reject Power board action with insufficient spendable power"
    );

    // Cheat 17: Taking QIC board action with insufficient QIC
    // Qic1 costs 4 QIC. Player 0 has 0 QIC.
    env.players[0].qic = 0;
    let cheat_qic1_no_qic = GameCommand::BoardAction {
        action: BoardAction::Qic1,
        target_mine: None,
        rescore_token: None,
    };
    let res = env.execute_command(0, cheat_qic1_no_qic);
    assert_eq!(
        res,
        Err(ActionError::InsufficientQic { needed: 4, have: 0 }),
        "Failed to reject QIC board action with insufficient QIC"
    );

    // LEGAL MOVE 4: Player 0 takes Power5 (2 knowledge for 4 power)
    env.players[0].power = PowerBowls::new(0, 0, 4, 0);
    env.players[0].knowledge = 3; // Ensure under 15 MAX_KNOWLEDGE cap
    let valid_power5 = GameCommand::BoardAction {
        action: BoardAction::Power5,
        target_mine: None,
        rescore_token: None,
    };
    let pre_k = env.players[0].knowledge;
    assert!(
        env.execute_command(0, valid_power5).is_ok(),
        "Legal Power5 board action failed"
    );
    assert_eq!(env.players[0].knowledge, pre_k + 2);
    assert_eq!(env.players[0].power.area3, 0);
    assert_eq!(env.players[0].power.area1, 4);
    assert_eq!(
        env.claimed_board_actions[BoardAction::Power5 as usize],
        Some(0)
    );

    // Cheat 18: Player 1 trying to take Power5 in the SAME round (Round Exclusivity!)
    env.players[1].power = PowerBowls::new(0, 0, 4, 0);
    let cheat_power5_duplicate = GameCommand::BoardAction {
        action: BoardAction::Power5,
        target_mine: None,
        rescore_token: None,
    };
    let res = env.execute_command(1, cheat_power5_duplicate);
    assert_eq!(
        res,
        Err(ActionError::BoardActionAlreadyClaimed(BoardAction::Power5)),
        "Failed to enforce board action exclusivity in the same round"
    );

    println!("=== PHASE 6: TESTING LOST FLEET SPACESHIP CHEATS ===");

    // Cheat 19: Taking a spaceship action on a spaceship that has NOT been explored yet
    let cheat_unexplored_ship_action = GameCommand::SpaceshipBoardAction {
        ship: Spaceship::Twilight,
        action_type: SpaceshipActionType::Knowledge,
        target_coord: None,
        target_field: None,
        rescore_token: None,
    };
    let res = env.execute_command(0, cheat_unexplored_ship_action);
    assert_eq!(
        res,
        Err(ActionError::SpaceshipNotExplored(Spaceship::Twilight)),
        "Failed to reject spaceship board action on an unexplored ship"
    );

    // Cheat 20: Exploring spaceship without enough VP (needs 5 VP)
    env.players[0].victory_points = 3;
    env.players[0].qic = 5;
    let cheat_explore_no_vp = GameCommand::ExploreSpaceship {
        ship: Spaceship::Twilight,
        coord: env.map.coords[5],
    };
    let res = env.execute_command(0, cheat_explore_no_vp);
    assert_eq!(
        res,
        Err(ActionError::InsufficientCredits { needed: 5, have: 3 }),
        "Failed to reject exploration without enough VP"
    );

    // Cheat 21: Bal T'aks needs 7 VP instead of 5 VP!
    env.players[1].victory_points = 6;
    env.players[1].qic = 5;
    let cheat_baltaks_explore_6vp = GameCommand::ExploreSpaceship {
        ship: Spaceship::Twilight,
        coord: env.map.coords[5],
    };
    let res = env.execute_command(1, cheat_baltaks_explore_6vp);
    assert_eq!(
        res,
        Err(ActionError::InsufficientCredits { needed: 7, have: 6 }),
        "Failed to require 7 VP for Bal T'aks exploration"
    );

    // LEGAL MOVE 5: Player 0 legitimately explores Spaceship Twilight
    env.players[0].victory_points = 15;
    // Hex 5 is dist 4 from Hex 1. Player 0 has range 1, needs (4 - 1)/2 = 2 QIC
    env.players[0].qic = 2;
    let valid_explore = GameCommand::ExploreSpaceship {
        ship: Spaceship::Twilight,
        coord: env.map.coords[5],
    };
    assert!(
        env.execute_command(0, valid_explore).is_ok(),
        "Legal spaceship exploration failed"
    );
    assert_eq!(env.players[0].victory_points, 10); // 15 - 5 VP
    assert_eq!(env.players[0].qic, 0); // 2 - 2 QIC
    assert_eq!(
        env.players[0].exploration_ships[Spaceship::Twilight as usize],
        Some(1)
    );
    assert!(env.players[0].has_explored(Spaceship::Twilight));

    // Cheat 22: Player 0 trying to explore Spaceship Twilight a SECOND time (Double exploration)
    env.players[0].victory_points = 10;
    env.players[0].qic = 5;
    let cheat_reexplore = GameCommand::ExploreSpaceship {
        ship: Spaceship::Twilight,
        coord: env.map.coords[5],
    };
    let res = env.execute_command(0, cheat_reexplore);
    assert_eq!(
        res,
        Err(ActionError::SpaceshipAlreadyExplored(Spaceship::Twilight)),
        "Failed to reject second exploration of the same spaceship"
    );

    // LEGAL MOVE 6: Player 0 uses Twilight Knowledge action (+3 range for 1k)
    env.players[0].knowledge = 2;
    let valid_twilight_knowledge = GameCommand::SpaceshipBoardAction {
        ship: Spaceship::Twilight,
        action_type: SpaceshipActionType::Knowledge,
        target_coord: None,
        target_field: None,
        rescore_token: None,
    };
    assert!(
        env.execute_command(0, valid_twilight_knowledge).is_ok(),
        "Legal Twilight Knowledge action failed"
    );
    assert_eq!(env.players[0].knowledge, 1); // 2 - 1k
    assert_eq!(env.players[0].temporary_range, 3);
    assert!(env.claimed_spaceship_actions[Spaceship::Twilight as usize]
        [SpaceshipActionType::Knowledge as usize]);

    // Cheat 23: Player 0 trying to reuse Twilight Knowledge in the SAME round (Round Lock)
    let cheat_reuse_ship_action = GameCommand::SpaceshipBoardAction {
        ship: Spaceship::Twilight,
        action_type: SpaceshipActionType::Knowledge,
        target_coord: None,
        target_field: None,
        rescore_token: None,
    };
    let res = env.execute_command(0, cheat_reuse_ship_action);
    assert_eq!(
        res,
        Err(ActionError::SpaceshipActionAlreadyUsed(
            Spaceship::Twilight,
            SpaceshipActionType::Knowledge
        )),
        "Failed to enforce spaceship board action lock for the round"
    );

    println!("=== PHASE 7: TESTING PASSING & POST-PASS CHEATS ===");

    // LEGAL MOVE 7: Player 0 passes with booster 2
    let valid_pass = GameCommand::Pass {
        new_booster: Some(2),
    };
    assert!(
        env.execute_command(0, valid_pass).is_ok(),
        "Legal pass failed"
    );
    assert!(env.players[0].passed);
    assert_eq!(env.players[0].current_booster, Some(2));

    // Cheat 24: Player 0 trying to take another action after having passed!
    let cheat_act_after_pass = GameCommand::AdvanceResearch {
        field: ResearchField::Economy,
    };
    let res = env.execute_command(0, cheat_act_after_pass);
    assert_eq!(
        res,
        Err(ActionError::PlayerAlreadyPassed),
        "Failed to reject action from a player who has already passed"
    );

    // Player 1 passes with booster 3 to finish the round
    let valid_p1_pass = GameCommand::Pass {
        new_booster: Some(3),
    };
    assert!(env.execute_command(1, valid_p1_pass).is_ok());
    assert!(env.players[1].passed);

    // Advance turn: all players have passed -> round clean-up and new round transition!
    env.advance_turn();
    assert_eq!(env.round, 2);
    assert!(!env.players[0].passed);
    assert!(!env.players[1].passed);
    assert_eq!(env.players[0].temporary_range, 0); // Temporary range reset

    // Board actions and spaceship actions must be unlocked for the new round!
    assert!(env.claimed_board_actions[BoardAction::Power5 as usize].is_none());
    assert!(
        !env.claimed_spaceship_actions[Spaceship::Twilight as usize]
            [SpaceshipActionType::Knowledge as usize]
    );

    println!("🎉 ALL 24 ADVERSARIAL CHEATS REJECTED WITH 100% PRECISION!");
}
