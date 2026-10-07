//! Income phase calculation and power charging for Gaia Project.

use crate::Faction;
use crate::player::PlayerData;
use crate::rules::{Building, ResearchField, TechTile};

/// Charges power for a player with cascading bowl transfers.
/// Returns `(charged, wasted)`.
pub fn charge_player_power(player: &mut PlayerData, amount: u8) -> (u8, u8) {
    player.power.charge(amount)
}

/// Applies raw resource income to a player, respecting resource caps.
pub fn apply_income(
    player: &mut PlayerData,
    credits: i16,
    ore: i16,
    knowledge: i16,
    qic: i16,
    power_charge: u8,
) {
    player.add_credits(credits);
    player.add_ore(ore);
    player.add_knowledge(knowledge);
    player.add_qic(qic);
    if power_charge > 0 {
        charge_player_power(player, power_charge);
    }
}

/// Computes raw income totals (credits, ore, knowledge, qic, tokens_to_gain, power_to_charge)
/// for a player across base faction, buildings, research, booster, and active tech tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RawIncome {
    pub credits: i16,
    pub ore: i16,
    pub knowledge: i16,
    pub qic: i16,
    pub power_tokens: u8,
    pub power_charge: u8,
}

impl RawIncome {
    pub fn add(&mut self, c: i16, o: i16, k: i16, q: i16, pt: u8, pw: u8) {
        self.credits += c;
        self.ore += o;
        self.knowledge += k;
        self.qic += q;
        self.power_tokens += pt;
        self.power_charge += pw;
    }
}

/// Computes the complete income for a player for a new round.
pub fn compute_player_income(player: &PlayerData) -> RawIncome {
    let mut inc = RawIncome::default();

    // 1. Base faction income
    match player.faction {
        Faction::Terrans
        | Faction::Lantids
        | Faction::Xenos
        | Faction::Gleens
        | Faction::Taklons
        | Faction::Geodens
        | Faction::BalTaks
        | Faction::Nevlas
        | Faction::Tinkeroids
        | Faction::Darkanians
        | Faction::Moweyds
        | Faction::SpaceGiants => {
            inc.add(0, 1, 1, 0, 0, 0); // +1o, +1k
        }
        Faction::Ambas => {
            inc.add(0, 2, 1, 0, 0, 0); // +2o, +1k
        }
        Faction::HadschHallas => {
            inc.add(3, 1, 1, 0, 0, 0); // +3c, +1o, +1k
        }
        Faction::Ivits => {
            inc.add(0, 1, 1, 1, 0, 0); // +1o, +1k, +1q
        }
        Faction::Firaks => {
            inc.add(0, 1, 2, 0, 0, 0); // +1o, +2k
        }
        Faction::Bescods => {
            inc.add(0, 1, 0, 0, 0, 0); // +1o
        }
        Faction::Itars => {
            inc.add(0, 1, 1, 0, 1, 0); // +1o, +1k, +1t
        }
    }

    // 2. Building income based on uncovered faction board slots
    // Mines:
    let mines = player.buildings[Building::Mine as usize];
    let mine_ore = match mines {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        n => (n - 1) as i16,
    };
    inc.add(0, mine_ore, 0, 0, 0, 0);

    // Trading Stations:
    let ts = player.buildings[Building::TradingStation as usize];
    if player.faction == Faction::Bescods {
        inc.add(0, 0, ts as i16, 0, 0, 0);
    } else {
        let ts_c = match ts {
            0 => 0,
            1 => 3,
            2 => 7,
            3 => 11,
            _ => 16,
        };
        inc.add(ts_c, 0, 0, 0, 0, 0);
    }

    // Research Labs:
    let labs = player.buildings[Building::ResearchLab as usize];
    if player.faction == Faction::Bescods {
        let lab_c = match labs {
            0 => 0,
            1 => 3,
            2 => 7,
            _ => 12,
        };
        inc.add(lab_c, 0, 0, 0, 0, 0);
    } else if player.faction == Faction::Nevlas {
        inc.add(0, 0, 0, 0, 0, labs * 2);
    } else {
        inc.add(0, 0, labs as i16, 0, 0, 0);
    }

    // Planetary Institute:
    if player.buildings[Building::PlanetaryInstitute as usize] > 0 {
        match player.faction {
            Faction::Gleens => inc.add(0, 1, 0, 0, 0, 4),
            Faction::Ambas | Faction::Bescods => inc.add(0, 0, 0, 0, 2, 4),
            Faction::Lantids | Faction::Terrans => inc.add(0, 0, 0, 0, 0, 4),
            Faction::SpaceGiants => inc.add(0, 0, 0, 0, 1, 6),
            Faction::Xenos => inc.add(0, 0, 0, 1, 0, 4),
            _ => inc.add(0, 0, 0, 0, 1, 4),
        }
    }

    // Academies:
    if player.buildings[Building::Academy1 as usize] > 0 {
        let k = if player.faction == Faction::Itars { 3 } else { 2 };
        inc.add(0, 0, k, 0, 0, 0);
    }
    if player.buildings[Building::Academy2 as usize] > 0 && player.faction == Faction::BalTaks {
        inc.add(4, 0, 0, 0, 0, 0);
    }

    // 3. Research track income
    // Economy track:
    match player.research_level(ResearchField::Economy) {
        1 => inc.add(2, 0, 0, 0, 0, 1),
        2 => inc.add(2, 1, 0, 0, 0, 2),
        3 => inc.add(3, 1, 0, 0, 0, 3),
        4 | 5 => inc.add(4, 2, 0, 0, 0, 4),
        _ => {}
    }
    // Science track:
    match player.research_level(ResearchField::Science) {
        1 => inc.add(0, 0, 1, 0, 0, 0),
        2 => inc.add(0, 0, 2, 0, 0, 0),
        3 => inc.add(0, 0, 3, 0, 0, 0),
        4 | 5 => inc.add(0, 0, 4, 0, 0, 0),
        _ => {}
    }

    // 4. Booster income
    if let Some(booster) = player.current_booster {
        match booster {
            1 => inc.add(0, 1, 1, 0, 0, 0),
            2 => inc.add(2, 0, 0, 1, 0, 0),
            3 => inc.add(0, 1, 0, 0, 0, 2),
            4 => inc.add(2, 0, 0, 0, 0, 0),
            5 => inc.add(0, 0, 0, 0, 0, 2),
            6 => inc.add(0, 1, 0, 0, 0, 0),
            7 => inc.add(0, 0, 1, 0, 0, 0),
            8 => inc.add(0, 1, 0, 0, 0, 0),
            9 => inc.add(0, 0, 0, 0, 0, 4),
            10 => inc.add(4, 0, 0, 0, 0, 0),
            _ => {}
        }
    }

    // 5. Active tech tile income (only if not covered by an advanced tech tile)
    if player.tech_tiles[TechTile::Tech2 as usize] && !player.covered_tech_tiles[TechTile::Tech2 as usize] {
        inc.add(1, 0, 1, 0, 0, 0);
    }
    if player.tech_tiles[TechTile::Tech3 as usize] && !player.covered_tech_tiles[TechTile::Tech3 as usize] {
        inc.add(0, 1, 0, 0, 1, 0);
    }
    if player.tech_tiles[TechTile::Tech5 as usize] && !player.covered_tech_tiles[TechTile::Tech5 as usize] {
        inc.add(4, 0, 0, 0, 0, 0);
    }
    if player.tech_tiles[TechTile::Tech6 as usize] && !player.covered_tech_tiles[TechTile::Tech6 as usize] {
        inc.add(0, 1, 0, 0, 0, 1);
    }

    inc
}

/// Collects complete round income for a player with optimal power charge ordering.
pub fn collect_round_income(player: &mut PlayerData) {
    let inc = compute_player_income(player);

    player.add_credits(inc.credits);
    player.add_ore(inc.ore);
    player.add_knowledge(inc.knowledge);
    player.add_qic(inc.qic);

    // Optimize power token addition vs power charging:
    // Evaluate Option A: Add tokens first, then charge
    let mut bowls_a = player.power;
    bowls_a.area1 += inc.power_tokens;
    let (_, waste_a) = bowls_a.charge(inc.power_charge);

    // Evaluate Option B: Charge first, then add tokens
    let mut bowls_b = player.power;
    let (_, waste_b) = bowls_b.charge(inc.power_charge);
    bowls_b.area1 += inc.power_tokens;

    // Pick sequence with minimum waste; if tied, maximum Area 3
    if waste_a < waste_b || (waste_a == waste_b && bowls_a.area3 >= bowls_b.area3) {
        player.power = bowls_a;
    } else {
        player.power = bowls_b;
    }
}

/// Computes and applies income granted by research tracks (Economy and Science).
pub fn apply_research_income(player: &mut PlayerData) {
    match player.research_level(ResearchField::Economy) {
        1 => apply_income(player, 2, 0, 0, 0, 1),
        2 => apply_income(player, 2, 1, 0, 0, 2),
        3 => apply_income(player, 3, 1, 0, 0, 3),
        4 | 5 => apply_income(player, 4, 2, 0, 0, 4),
        _ => {}
    }
    match player.research_level(ResearchField::Science) {
        1 => player.add_knowledge(1),
        2 => player.add_knowledge(2),
        3 => player.add_knowledge(3),
        4 | 5 => player.add_knowledge(4),
        _ => {}
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Faction;

    #[test]
    fn research_income_applies_economy_and_science() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.credits = 0;
        player.ore = 0;
        player.knowledge = 0;
        player.research[ResearchField::Economy as usize] = 2; // +2c, 1o, 2pw
        player.research[ResearchField::Science as usize] = 3; // +3k

        apply_research_income(&mut player);

        assert_eq!(player.credits, 2);
        assert_eq!(player.ore, 1);
        assert_eq!(player.knowledge, 3);
        // Initial power for Terrans is (4, 4, 0, 0). Charging 2 moves 2 from Area 1 -> Area 2.
        assert_eq!(player.power.area1, 2);
        assert_eq!(player.power.area2, 6);
    }

    #[test]
    fn full_round_income_combines_all_sources_and_optimizes_power() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.credits = 5;
        player.ore = 2;
        player.knowledge = 1;
        player.qic = 0;
        player.buildings[Building::Mine as usize] = 3; // 2 ore from mines
        player.buildings[Building::TradingStation as usize] = 2; // 7 credits from TS
        player.buildings[Building::ResearchLab as usize] = 1; // 1 knowledge from lab
        player.buildings[Building::PlanetaryInstitute as usize] = 1; // 4 power charge for Terrans
        player.current_booster = Some(2); // Booster 2: +2c, +1q
        player.tech_tiles[TechTile::Tech2 as usize] = true; // +1c, +1k

        // Faction base for Terrans: +1o, +1k
        // Expected:
        // credits: 5 (init) + 7 (TS) + 2 (booster) + 1 (tech) = 15
        // ore: 2 (init) + 1 (base) + 2 (mines) = 5
        // knowledge: 1 (init) + 1 (base) + 1 (lab) + 1 (tech) = 4
        // qic: 0 (init) + 1 (booster) = 1
        collect_round_income(&mut player);

        assert_eq!(player.credits, 15);
        assert_eq!(player.ore, 5);
        assert_eq!(player.knowledge, 4);
        assert_eq!(player.qic, 1);
        // Initial power for Terrans is (4, 4, 0, 0). 4 charge moves 4 from A1->A2 (leaves 0 in A1, 8 in A2, 0 in A3)
        assert_eq!(player.power.area1, 0);
        assert_eq!(player.power.area2, 8);
        assert_eq!(player.power.area3, 0);
    }
}


