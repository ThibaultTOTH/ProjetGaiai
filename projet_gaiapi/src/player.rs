//! Player state and power management for Gaia Project.
//!
//! Designed for zero-allocation simulation and high-throughput RL.

use serde::{Deserialize, Serialize};

use crate::Faction;
use crate::rules::{
    Building, FederationToken, MAX_CREDIT, MAX_KNOWLEDGE, MAX_ORE, PowerArea, ResearchField,
};

/// 3-bowl power system + Gaia area, with optional Brainstone support (Taklons).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerBowls {
    pub area1: u8,
    pub area2: u8,
    pub area3: u8,
    pub gaia: u8,
    pub brainstone: Option<PowerArea>,
}

impl Default for PowerBowls {
    fn default() -> Self {
        Self::new(2, 4, 0, 0)
    }
}

impl PowerBowls {
    pub const fn new(area1: u8, area2: u8, area3: u8, gaia: u8) -> Self {
        Self {
            area1,
            area2,
            area3,
            gaia,
            brainstone: None,
        }
    }

    pub const fn with_brainstone(area1: u8, area2: u8, area3: u8, gaia: u8, brainstone: PowerArea) -> Self {
        Self {
            area1,
            area2,
            area3,
            gaia,
            brainstone: Some(brainstone),
        }
    }

    /// Total number of standard power tokens.
    #[inline]
    pub const fn total_tokens(&self) -> u8 {
        self.area1 + self.area2 + self.area3 + self.gaia
    }

    /// Total effective spending power in Area 3 (Brainstone counts as 3 power).
    #[inline]
    pub fn spendable_power(&self) -> u8 {
        let bs_value = if self.brainstone == Some(PowerArea::Area3) { 3 } else { 0 };
        self.area3 + bs_value
    }

    /// Cascading power charge: Bowl 1 -> Bowl 2, then Bowl 2 -> Bowl 3.
    /// Returns `(charged_amount, wasted_amount)`.
    pub fn charge(&mut self, mut amount: u8) -> (u8, u8) {
        if amount == 0 {
            return (0, 0);
        }
        let starting = amount;

        // 1. Move Brainstone from Area 1 to Area 2 if possible
        if self.brainstone == Some(PowerArea::Area1) && amount > 0 {
            self.brainstone = Some(PowerArea::Area2);
            amount -= 1;
        }

        // 2. Move standard tokens from Area 1 to Area 2
        let move_1_to_2 = amount.min(self.area1);
        self.area1 -= move_1_to_2;
        self.area2 += move_1_to_2;
        amount -= move_1_to_2;

        // 3. Move Brainstone from Area 2 to Area 3 if possible
        if self.brainstone == Some(PowerArea::Area2) && amount > 0 {
            self.brainstone = Some(PowerArea::Area3);
            amount -= 1;
        }

        // 4. Move standard tokens from Area 2 to Area 3
        let move_2_to_3 = amount.min(self.area2);
        self.area2 -= move_2_to_3;
        self.area3 += move_2_to_3;
        amount -= move_2_to_3;

        let charged = starting - amount;
        let wasted = amount;
        (charged, wasted)
    }

    /// Burns power: for each token moved to Area 3, one token from Area 2 is permanently discarded.
    /// `pairs_to_burn` is the number of tokens to destroy (and therefore gain in Area 3).
    pub fn burn(&mut self, pairs_to_burn: u8) -> bool {
        let needed_in_area2 = pairs_to_burn.saturating_mul(2);
        if self.area2 < needed_in_area2 {
            return false;
        }
        self.area2 -= needed_in_area2;
        self.area3 += pairs_to_burn;
        true
    }

    /// Spends power tokens from Area 3 (moving them to Area 1).
    /// If Brainstone is in Area 3, it counts as 3 power and is moved to Area 1 when used.
    pub fn spend(&mut self, amount: u8) -> bool {
        if self.spendable_power() < amount {
            return false;
        }
        let mut remaining = amount;

        // If Brainstone is in Area 3 and needed or optimal:
        if self.brainstone == Some(PowerArea::Area3) && (remaining >= 3 || self.area3 < remaining) {
            self.brainstone = Some(PowerArea::Area1);
            remaining = remaining.saturating_sub(3);
        }

        let from_area3 = remaining.min(self.area3);
        self.area3 -= from_area3;
        self.area1 += from_area3;
        true
    }
}

/// Complete state of a player seat.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerData {
    pub seat: u8,
    pub faction: Faction,
    pub victory_points: i16,
    pub credits: i16,
    pub ore: i16,
    pub knowledge: i16,
    pub qic: i16,
    pub power: PowerBowls,
    pub research: [u8; 6],
    pub buildings: [u8; 8],
    pub satellites: u8,
    pub gaiaformers_unlocked: u8,
    pub gaiaformers_in_gaia: u8,
    pub used_gaiaformers_asteroid: u8,
    pub green_federation_tokens: u8,
    pub gray_federation_tokens: u8,
    pub current_booster: Option<u8>,
    pub passed: bool,
    pub temporary_step: u8,
    pub tech_tiles: [bool; 9],
    pub covered_tech_tiles: [bool; 9],
    pub adv_tech_tiles: [bool; 15],
    pub claimed_federations: [Option<FederationToken>; 8],
    pub special_actions_used: [bool; 10],
    pub exploration_ships: [Option<u8>; 4],
    pub temporary_range: u8,
    /// Set to true when the player has upgraded to ResearchLab/Academy and must claim a tech tile.
    pub pending_tech_claim: bool,
    /// Claimed Lost Fleet artifacts (13 tokens from Twilight spaceship).
    pub claimed_artefacts: [bool; 13],
    /// Power rings remaining to place (Moweyds start with 6).
    pub power_rings: u8,
}

impl PlayerData {
    pub fn new(seat: u8, faction: Faction) -> Self {
        let mut player = Self {
            seat,
            faction,
            victory_points: 10,
            credits: 15,
            ore: 4,
            knowledge: 3,
            qic: 1,
            power: PowerBowls::new(2, 4, 0, 0),
            research: [0; 6],
            buildings: [0; 8],
            satellites: 0,
            gaiaformers_unlocked: 0,
            gaiaformers_in_gaia: 0,
            used_gaiaformers_asteroid: 0,
            green_federation_tokens: 0,
            gray_federation_tokens: 0,
            current_booster: None,
            passed: false,
            temporary_step: 0,
            tech_tiles: [false; 9],
            covered_tech_tiles: [false; 9],
            adv_tech_tiles: [false; 15],
            claimed_federations: [None; 8],
            special_actions_used: [false; 10],
            exploration_ships: [None; 4],
            temporary_range: 0,
            pending_tech_claim: false,
            claimed_artefacts: [false; 13],
            power_rings: if faction == Faction::Moweyds { 6 } else { 0 },
        };
        // Faction-specific starting bonuses & adjustments
        match faction {
            Faction::Terrans => {
                // 4/4 power, GaiaProject +1
                player.power = PowerBowls::new(4, 4, 0, 0);
                player.research[ResearchField::GaiaProject as usize] = 1;
            }
            Faction::Lantids => {
                // 13 credits, 4/0 power
                player.credits = 13;
                player.power = PowerBowls::new(4, 0, 0, 0);
            }
            Faction::Xenos => {
                // Intelligence +1
                player.research[ResearchField::Intelligence as usize] = 1;
            }
            Faction::Gleens => {
                // Navigation +1, 0 QIC
                player.qic = 0;
                player.research[ResearchField::Navigation as usize] = 1;
            }
            Faction::Taklons => {
                player.power.brainstone = Some(PowerArea::Area1);
            }
            Faction::Ambas => {
                // Navigation +1
                player.research[ResearchField::Navigation as usize] = 1;
            }
            Faction::HadschHallas => {
                // Economy +1
                player.research[ResearchField::Economy as usize] = 1;
            }
            Faction::Ivits => {
                // 2/2 power
                player.power = PowerBowls::new(2, 2, 0, 0);
            }
            Faction::Geodens => {
                // Terraforming +1
                player.research[ResearchField::Terraforming as usize] = 1;
            }
            Faction::BalTaks => {
                // 0 QIC, 2/2 power, GaiaProject +1
                player.qic = 0;
                player.power = PowerBowls::new(2, 2, 0, 0);
                player.research[ResearchField::GaiaProject as usize] = 1;
            }
            Faction::Firaks => {
                // 2 knowledge
                player.knowledge = 2;
            }
            Faction::Bescods => {}
            Faction::Nevlas => {
                // 2 knowledge, Science +1
                player.knowledge = 2;
                player.research[ResearchField::Science as usize] = 1;
            }
            Faction::Itars => {
                // 5 ore, 4/4 power
                player.ore = 5;
                player.power = PowerBowls::new(4, 4, 0, 0);
            }
            Faction::Tinkeroids => {
                // 2k, 4o, 15c, 1q, power 4/2, Science +1
                player.knowledge = 2;
                player.ore = 4;
                player.credits = 15;
                player.qic = 1;
                player.power = PowerBowls::new(4, 2, 0, 0);
                player.research[ResearchField::Science as usize] = 1;
            }
            Faction::Darkanians => {
                // 3k, 7o, 15c, 1q, power 4/2, Navigation +1, Economy +1
                player.knowledge = 3;
                player.ore = 7;
                player.credits = 15;
                player.qic = 1;
                player.power = PowerBowls::new(4, 2, 0, 0);
                player.research[ResearchField::Navigation as usize] = 1;
                player.research[ResearchField::Economy as usize] = 1;
            }
            Faction::Moweyds => {
                // 5k, 6o, 15c, 2q, power 4/4, Gaiaforming +1, starts with shuttle on TFMars
                player.knowledge = 5;
                player.ore = 6;
                player.credits = 15;
                player.qic = 2;
                player.power = PowerBowls::new(4, 4, 0, 0);
                player.research[ResearchField::GaiaProject as usize] = 1;
                player.exploration_ships[crate::rules::Spaceship::TFMars as usize] = Some(0);
            }
            Faction::SpaceGiants => {
                // 3k, 6o, 15c, 1q, power 4/4, Navigation +1
                player.knowledge = 3;
                player.ore = 6;
                player.credits = 15;
                player.qic = 1;
                player.power = PowerBowls::new(4, 4, 0, 0);
                player.research[ResearchField::Navigation as usize] = 1;
            }
        }
        player
    }

    #[inline]
    pub fn deployed_shuttles(&self) -> u8 {
        self.exploration_ships.iter().filter(|s| s.is_some()).count() as u8
    }

    #[inline]
    pub fn has_explored(&self, ship: crate::rules::Spaceship) -> bool {
        self.exploration_ships[ship as usize].is_some()
    }

    #[inline]
    pub fn available_gaiaformers(&self) -> u8 {
        self.gaiaformers_unlocked
            .saturating_sub(self.gaiaformers_in_gaia.saturating_add(self.used_gaiaformers_asteroid))
    }

    #[inline]
    pub fn total_federation_tokens(&self) -> u8 {
        self.green_federation_tokens + self.gray_federation_tokens
    }

    pub fn flip_federation_token(&mut self) -> bool {
        if self.green_federation_tokens > 0 {
            self.green_federation_tokens -= 1;
            self.gray_federation_tokens += 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn available_satellites(&self) -> u8 {
        25u8.saturating_sub(self.satellites)
    }

    /// Power required to start a Gaia project based on Gaia Project research level.
    pub fn gaia_power_cost(&self) -> Option<u8> {
        match self.research_level(ResearchField::GaiaProject) {
            0 => None,
            1 | 2 => Some(6),
            3 => Some(4),
            _ => Some(3),
        }
    }

    /// Moves power from power areas (Area 1, then Area 2, then Area 3) into the Gaia area.
    pub fn move_power_to_gaia(&mut self, mut amount: u8) -> bool {
        let total = self.power.area1 + self.power.area2 + self.power.area3;
        if total < amount {
            return false;
        }
        let from_1 = self.power.area1.min(amount);
        self.power.area1 -= from_1;
        amount -= from_1;

        let from_2 = self.power.area2.min(amount);
        self.power.area2 -= from_2;
        amount -= from_2;

        let from_3 = self.power.area3.min(amount);
        self.power.area3 -= from_3;
        amount -= from_3;

        let moved = from_1 + from_2 + from_3;
        self.power.gaia += moved;
        debug_assert_eq!(amount, 0);
        true
    }

    /// Discards power tokens from Area 1, 2, or 3 (used to build satellites).
    pub fn discard_power(&mut self, mut amount: u8) -> bool {
        let total = self.power.area1 + self.power.area2 + self.power.area3;
        if total < amount {
            return false;
        }
        let from_1 = self.power.area1.min(amount);
        self.power.area1 -= from_1;
        amount -= from_1;

        let from_2 = self.power.area2.min(amount);
        self.power.area2 -= from_2;
        amount -= from_2;

        let from_3 = self.power.area3.min(amount);
        self.power.area3 -= from_3;
        true
    }

    /// Claims a federation token, immediately awarding resources/VP and updating tokens.
    pub fn claim_federation_token(&mut self, token: FederationToken) {
        let rew = token.reward();
        self.add_victory_points(rew.vp);
        self.add_credits(rew.credits);
        self.add_ore(rew.ore);
        self.add_knowledge(rew.knowledge);
        self.add_qic(rew.qic);
        self.power.area1 += rew.power_tokens as u8;
        if rew.is_green {
            self.green_federation_tokens += 1;
        } else {
            self.gray_federation_tokens += 1;
        }
        for slot in &mut self.claimed_federations {
            if slot.is_none() {
                *slot = Some(token);
                break;
            }
        }
    }


    #[inline]
    pub fn resources_array(&self) -> [i16; 5] {
        [
            self.credits,
            self.ore,
            self.knowledge,
            self.qic,
            self.power.spendable_power() as i16,
        ]
    }

    // --- Resource limits and additions ---

    pub fn add_credits(&mut self, amount: i16) {
        self.credits = (self.credits + amount).clamp(0, MAX_CREDIT);
    }

    pub fn add_ore(&mut self, amount: i16) {
        self.ore = (self.ore + amount).clamp(0, MAX_ORE);
    }

    pub fn add_knowledge(&mut self, amount: i16) {
        self.knowledge = (self.knowledge + amount).clamp(0, MAX_KNOWLEDGE);
    }

    pub fn add_qic(&mut self, amount: i16) {
        self.qic = (self.qic + amount).max(0);
    }

    pub fn add_victory_points(&mut self, amount: i16) {
        self.victory_points += amount;
    }

    // --- Affordability and payments ---

    pub fn can_afford(&self, credits: i16, ore: i16, knowledge: i16, qic: i16, power: u8) -> bool {
        self.credits >= credits
            && self.ore >= ore
            && self.knowledge >= knowledge
            && self.qic >= qic
            && self.power.spendable_power() >= power
    }

    pub fn pay(&mut self, credits: i16, ore: i16, knowledge: i16, qic: i16, power: u8) -> bool {
        if !self.can_afford(credits, ore, knowledge, qic, power) {
            return false;
        }
        self.credits -= credits;
        self.ore -= ore;
        self.knowledge -= knowledge;
        self.qic -= qic;
        if power > 0 {
            self.power.spend(power);
        }
        true
    }

    // --- Research and Capabilities ---

    #[inline]
    pub fn research_level(&self, field: ResearchField) -> u8 {
        self.research[field as usize]
    }

    /// Navigation track levels:
    /// Level 0-1: range 1
    /// Level 2-3: range 2
    /// Level 4-5: range 3 (Level 5 has range 4 in Lost Fleet/lost planet, range 3 base + special)
    pub fn effective_range(&self) -> u8 {
        let nav = self.research_level(ResearchField::Navigation);
        match nav {
            0..=1 => 1,
            2..=3 => 2,
            4 => 3,
            _ => 4,
        }
    }

    /// Terraforming track levels:
    /// Level 0-1: 3 ore per terraforming step (discount 0)
    /// Level 2: 2 ore per step (discount 1)
    /// Level 3+: 1 ore per step (discount 2)
    pub fn terraform_cost_discount(&self) -> i16 {
        let terra = self.research_level(ResearchField::Terraforming);
        match terra {
            0..=1 => 0,
            2 => 1,
            _ => 2,
        }
    }

    /// Building limits
    pub fn max_buildings(building: Building) -> u8 {
        match building {
            Building::Mine => 8,
            Building::TradingStation => 4,
            Building::ResearchLab => 3,
            Building::PlanetaryInstitute => 1,
            Building::Academy1 | Building::Academy2 => 1,
            Building::GaiaFormer => 3,
            Building::SpaceStation => 1,
        }
    }

    pub fn buildings_available(&self, building: Building) -> u8 {
        let built = self.buildings[building as usize];
        Self::max_buildings(building).saturating_sub(built)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_charging_cascades_from_bowl_1_to_3() {
        let mut power = PowerBowls::new(2, 2, 0, 0);

        // Charge 3 power:
        // Area 1 (2 tokens) -> Area 2 (now 4 in Area 2)
        // 1 remaining charge: Area 2 -> Area 3 (now 3 in Area 2, 1 in Area 3)
        let (charged, wasted) = power.charge(3);
        assert_eq!(charged, 3);
        assert_eq!(wasted, 0);
        assert_eq!(power.area1, 0);
        assert_eq!(power.area2, 3);
        assert_eq!(power.area3, 1);
    }

    #[test]
    fn power_charging_wastes_overflow() {
        let mut power = PowerBowls::new(1, 1, 2, 0);
        // Total tokens = 4. Max capacity in Bowl 3 = 4.
        // Needs 1 charge to move Area 1 -> Area 2.
        // Needs 2 charges to move both Area 2 tokens -> Area 3.
        // Extra charges beyond 3 should be wasted.
        let (charged, wasted) = power.charge(5);
        assert_eq!(charged, 3);
        assert_eq!(wasted, 2);
        assert_eq!(power.area1, 0);
        assert_eq!(power.area2, 0);
        assert_eq!(power.area3, 4);
    }

    #[test]
    fn power_burning_sacrifices_from_area_2() {
        let mut power = PowerBowls::new(0, 4, 0, 0);
        // Burn 1 pair: 2 tokens destroyed from Area 2, 1 token gained in Area 3.
        assert!(power.burn(1));
        assert_eq!(power.area2, 2);
        assert_eq!(power.area3, 1);

        // Burning 2 pairs requires 4 tokens in Area 2, we only have 2: fails
        assert!(!power.burn(2));
    }

    #[test]
    fn taklons_brainstone_charges_first_and_spends_as_three() {
        let mut power = PowerBowls::with_brainstone(1, 0, 0, 0, PowerArea::Area1);
        // Charge 1: Brainstone moves Area 1 -> Area 2
        let (charged, _) = power.charge(1);
        assert_eq!(charged, 1);
        assert_eq!(power.brainstone, Some(PowerArea::Area2));
        assert_eq!(power.area1, 1);

        // Charge 2: Brainstone moves to Area 3, then 1 token from Area 1 moves to Area 2
        let (charged, _) = power.charge(2);
        assert_eq!(charged, 2);
        assert_eq!(power.brainstone, Some(PowerArea::Area3));
        assert_eq!(power.area1, 0);
        assert_eq!(power.area2, 1);

        // Spendable power is 3 from brainstone + 0 normal tokens = 3
        assert_eq!(power.spendable_power(), 3);
        assert!(power.spend(3));
        assert_eq!(power.brainstone, Some(PowerArea::Area1));
    }

    #[test]
    fn player_resource_caps_and_affordability() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.add_credits(50);
        assert_eq!(player.credits, MAX_CREDIT);

        player.add_ore(30);
        assert_eq!(player.ore, MAX_ORE);

        player.add_knowledge(20);
        assert_eq!(player.knowledge, MAX_KNOWLEDGE);

        assert!(player.can_afford(30, 15, 15, 0, 0));
        assert!(player.pay(10, 5, 5, 0, 0));
        assert_eq!(player.credits, 20);
        assert_eq!(player.ore, 10);
        assert_eq!(player.knowledge, 10);
    }

    #[test]
    fn player_power_movement_to_gaia_and_satellites() {
        let mut player = PlayerData::new(0, Faction::Terrans);
        player.power = PowerBowls::new(2, 4, 0, 0);

        // Move 5 power to Gaia: 2 from Area 1, 3 from Area 2
        assert!(player.move_power_to_gaia(5));
        assert_eq!(player.power.area1, 0);
        assert_eq!(player.power.area2, 1);
        assert_eq!(player.power.gaia, 5);

        // Discard 1 power for a satellite: takes from Area 2
        assert!(player.discard_power(1));
        assert_eq!(player.power.area2, 0);
        assert!(!player.discard_power(1)); // No power remaining in bowls

        // Claim federation token
        player.claim_federation_token(FederationToken::Fed4); // 7 VP, 2 ore, green
        assert_eq!(player.victory_points, 17);
        assert_eq!(player.ore, 6);
        assert_eq!(player.green_federation_tokens, 1);
    }
}



