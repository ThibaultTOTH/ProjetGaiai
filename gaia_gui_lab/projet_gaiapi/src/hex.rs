//! Hex cell model for the Gaia Project board.
//!
//! Compact, `Copy` representation with zero heap allocation.
//! Mirrors the logic from `gaia-hex.ts` in the TypeScript reference engine.

use serde::{Deserialize, Serialize};

use crate::rules::{Building, Planet, Spaceship, standard_building_value};

/// State of a single hex on the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hex {
    pub planet: Planet,
    pub building: Option<Building>,
    pub player: Option<u8>,
    pub additional_mine: Option<u8>,
    pub in_federation: [bool; 4],
    pub sector_id: u8,
    pub spaceship: Option<Spaceship>,
}

impl Default for Hex {
    fn default() -> Self {
        Self::empty()
    }
}

impl Hex {
    /// Constructs an empty space hex.
    pub const fn empty() -> Self {
        Self {
            planet: Planet::Empty,
            building: None,
            player: None,
            additional_mine: None,
            in_federation: [false; 4],
            sector_id: 0,
            spaceship: None,
        }
    }

    /// Constructs a hex with a planet and sector identifier.
    pub const fn new(planet: Planet, sector_id: u8) -> Self {
        Self {
            planet,
            building: None,
            player: None,
            additional_mine: None,
            in_federation: [false; 4],
            sector_id,
            spaceship: None,
        }
    }

    #[inline]
    pub const fn has_planet(&self) -> bool {
        !matches!(self.planet, Planet::Empty)
    }

    #[inline]
    pub const fn occupied(&self) -> bool {
        self.player.is_some()
    }

    /// Space stations and Gaia Formers are not structures; buildings with
    /// standard value > 0 (Mine, TS, Lab, PI, Academies) are structures.
    #[inline]
    pub fn has_structure(&self) -> bool {
        match self.building {
            Some(Building::GaiaFormer) | Some(Building::SpaceStation) | None => false,
            _ => self.occupied(),
        }
    }

    /// Building owned by a player on this hex, taking into account Lantids additional mine.
    pub fn building_of(&self, player: u8) -> Option<Building> {
        if self.additional_mine == Some(player) {
            return Some(Building::Mine);
        }
        if self.player == Some(player) {
            return self.building;
        }
        None
    }

    /// A planet is colonized by a player if they own a standard building on it.
    pub fn colonized_by(&self, player: u8) -> bool {
        match self.building_of(player) {
            Some(building) => standard_building_value(building) > 0,
            None => false,
        }
    }

    /// Whether this hex acts as a starting point for navigation range checks.
    pub fn is_range_starting_point(&self, player: u8) -> bool {
        self.colonized_by(player) || self.building_of(player) == Some(Building::SpaceStation)
    }

    /// Whether this hex is already part of a federation for the given player.
    #[inline]
    pub fn belongs_to_federation_of(&self, player: u8) -> bool {
        (player as usize) < 4 && self.in_federation[player as usize]
    }

    /// Adds this hex to a federation for the given player.
    pub fn add_to_federation(&mut self, player: u8) {
        if (player as usize) < 4 {
            self.in_federation[player as usize] = true;
        }
    }

    /// Building power value contributed to federations by the given player.
    pub fn value_for_federation(&self, player: u8) -> u8 {
        if self.belongs_to_federation_of(player) {
            return 0;
        }
        match self.building_of(player) {
            Some(building) => standard_building_value(building),
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_hex_has_no_planet_or_structure() {
        let hex = Hex::empty();
        assert!(!hex.has_planet());
        assert!(!hex.has_structure());
        assert!(!hex.occupied());
    }

    #[test]
    fn planet_hex_with_mine_is_colonized() {
        let mut hex = Hex::new(Planet::Terra, 1);
        hex.building = Some(Building::Mine);
        hex.player = Some(0);

        assert!(hex.has_planet());
        assert!(hex.has_structure());
        assert!(hex.colonized_by(0));
        assert!(!hex.colonized_by(1));
        assert_eq!(hex.value_for_federation(0), 1);
    }

    #[test]
    fn lantids_additional_mine_is_recognized() {
        let mut hex = Hex::new(Planet::Desert, 2);
        hex.building = Some(Building::TradingStation);
        hex.player = Some(0); // Main occupier
        hex.additional_mine = Some(1); // Lantids guest mine

        assert_eq!(hex.building_of(0), Some(Building::TradingStation));
        assert_eq!(hex.building_of(1), Some(Building::Mine));
        assert!(hex.colonized_by(0));
        assert!(hex.colonized_by(1));
        assert_eq!(hex.value_for_federation(0), 2);
        assert_eq!(hex.value_for_federation(1), 1);
    }

    #[test]
    fn gaia_former_is_not_a_structure_or_federation_value() {
        let mut hex = Hex::new(Planet::Transdim, 3);
        hex.building = Some(Building::GaiaFormer);
        hex.player = Some(2);

        assert!(!hex.has_structure());
        assert!(!hex.colonized_by(2));
        assert_eq!(hex.value_for_federation(2), 0);
    }
}



