//! Map representation, sector layout loader, and fast BFS pathfinding.
//!
//! Zero-allocation simulation suitable for high-throughput RL.

use serde::{Deserialize, Serialize};

use crate::board::HexCoord;
use crate::hex::Hex;
use crate::rules::{Planet, qic_for_distance};
use crate::sector::{
    BIG_CENTERS, SMALL_CENTERS, SectorId, place_sector,
};

/// Maximum map capacity (10 sectors * 19 hexes = 190 hexes for 4-player layout).
pub const MAP_SIZE: usize = 200;
pub const NO_NEIGHBOR: u8 = 255;

/// Sector placement specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectorPlacement {
    pub sector: SectorId,
    pub rotation: u8,
    pub center: Option<HexCoord>,
}

/// Map configuration describing sectors on the board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapConfiguration {
    pub sectors: Vec<SectorPlacement>,
    #[serde(default)]
    pub mirror: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Map {
    pub coords: [HexCoord; MAP_SIZE],
    pub hexes: [Hex; MAP_SIZE],
    /// Precomputed 6 neighbors per hex index. `NO_NEIGHBOR` (255) represents boundary.
    pub adjacency: [[u8; 6]; MAP_SIZE],
    pub count: usize,
}

impl Default for Map {
    fn default() -> Self {
        Self::empty()
    }
}

impl Map {
    /// Creates an empty map.
    pub const fn empty() -> Self {
        Self {
            coords: [HexCoord::origin(); MAP_SIZE],
            hexes: [Hex::empty(); MAP_SIZE],
            adjacency: [[NO_NEIGHBOR; 6]; MAP_SIZE],
            count: 0,
        }
    }

    /// Loads a map from an explicit configuration (matching TS `SpaceMap.load`).
    pub fn load_configuration(&mut self, conf: &MapConfiguration) {
        self.count = 0;
        let default_centers: &[HexCoord] = if conf.sectors.len() <= 7 {
            &SMALL_CENTERS
        } else {
            &BIG_CENTERS
        };

        for (i, placement) in conf.sectors.iter().enumerate() {
            let center = placement.center.unwrap_or(default_centers[i]);
            let placed = place_sector(placement.sector, center, placement.rotation, conf.mirror);

            for (coord, hex) in placed {
                if self.count < MAP_SIZE {
                    self.coords[self.count] = coord;
                    self.hexes[self.count] = hex;
                    self.count += 1;
                }
            }
        }

        self.sort_canonical();
    }

    /// Sorts hexes canonically by coordinate (ascending q, then r, then s).
    /// Guarantees that spatial index `i` maps to the exact same physical coordinates across all games.
    pub fn sort_canonical(&mut self) {
        let mut pairs: Vec<(HexCoord, Hex)> = (0..self.count)
            .map(|i| (self.coords[i], self.hexes[i]))
            .collect();
        pairs.sort_by_key(|&(c, _)| c);
        for (i, (c, h)) in pairs.into_iter().enumerate() {
            self.coords[i] = c;
            self.hexes[i] = h;
        }
        self.recompute_adjacency();
    }

    /// Generates a standard map layout from a player count and seed.
    pub fn generate_standard(players: usize, seed: u64) -> Self {
        let mut map = Self::empty();
        let nb_sectors = if players <= 2 { 7 } else { 10 };
        let mut available_sectors: Vec<SectorId> = if players <= 2 {
            SectorId::SMALL_CONFIG.to_vec()
        } else {
            SectorId::BIG_CONFIG.to_vec()
        };

        let mut rng = seed;
        let mut next_u64 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng
        };

        loop {
            // Fisher-Yates shuffle
            for i in (1..available_sectors.len()).rev() {
                let j = (next_u64() as usize) % (i + 1);
                available_sectors.swap(i, j);
            }

            let placements: Vec<SectorPlacement> = available_sectors[..nb_sectors]
                .iter()
                .map(|&sec| SectorPlacement {
                    sector: sec,
                    rotation: (next_u64() % 6) as u8,
                    center: None,
                })
                .collect();

            let conf = MapConfiguration {
                sectors: placements,
                mirror: false,
            };

            map.load_configuration(&conf);
            if map.is_valid(true) {
                break;
            }
        }

        map
    }

    /// Recomputes the 6-direction adjacency graph in O(N) without heap allocation.
    pub fn recompute_adjacency(&mut self) {
        for i in 0..self.count {
            let coord = self.coords[i];
            let nbs = coord.neighbours();
            for (d, &target) in nbs.iter().enumerate() {
                let mut found = NO_NEIGHBOR;
                for j in 0..self.count {
                    if self.coords[j] == target {
                        found = j as u8;
                        break;
                    }
                }
                self.adjacency[i][d] = found;
            }
        }
    }

    /// Checks validity of map: no two identical HOME planets adjacent (German rules).
    pub fn is_valid(&self, german_rules: bool) -> bool {
        for i in 0..self.count {
            let hex_i = self.hexes[i];
            if hex_i.planet == Planet::Empty {
                continue;
            }
            for &nb_idx in &self.adjacency[i] {
                if nb_idx == NO_NEIGHBOR {
                    continue;
                }
                let hex_nb = self.hexes[nb_idx as usize];
                if hex_nb.planet == Planet::Empty {
                    continue;
                }

                if german_rules {
                    // German rules only forbid HOME planets of the same type from touching;
                    // Gaia and Transdim planets touching each other or anything else is legal.
                    if hex_i.planet != Planet::Transdim
                        && hex_i.planet != Planet::Gaia
                        && hex_i.planet == hex_nb.planet
                    {
                        return false;
                    }
                } else if hex_i.sector_id != hex_nb.sector_id && hex_i.planet == hex_nb.planet {
                    return false;
                }
            }
        }
        true
    }

    /// Finds index of coordinate on the map (O(log N) if sorted, O(N) fallback).
    pub fn index_of(&self, coord: HexCoord) -> Option<usize> {
        self.coords[..self.count]
            .binary_search(&coord)
            .ok()
            .or_else(|| {
                self.coords[..self.count]
                    .iter()
                    .position(|&candidate| candidate == coord)
            })
    }

    pub fn get_hex(&self, coord: HexCoord) -> Option<&Hex> {
        self.index_of(coord).map(|idx| &self.hexes[idx])
    }

    pub fn get_hex_mut(&mut self, coord: HexCoord) -> Option<&mut Hex> {
        self.index_of(coord).map(|idx| &mut self.hexes[idx])
    }

    /// Breadth-First Search (BFS) distance between two hexes without heap allocation.
    /// Returns 255 if unreachable.
    pub fn distance(&self, start: usize, target: usize) -> u8 {
        if start >= self.count || target >= self.count {
            return 255;
        }
        if start == target {
            return 0;
        }

        let mut visited = [false; MAP_SIZE];
        let mut queue = [0usize; MAP_SIZE];
        let mut head = 0;
        let mut tail = 0;
        let mut distances = [255u8; MAP_SIZE];

        queue[tail] = start;
        tail += 1;
        visited[start] = true;
        distances[start] = 0;

        while head < tail {
            let current = queue[head];
            head += 1;

            if current == target {
                return distances[current];
            }

            for &neighbor in &self.adjacency[current] {
                if neighbor != NO_NEIGHBOR {
                    let n = neighbor as usize;
                    if n < self.count && !visited[n] {
                        visited[n] = true;
                        distances[n] = distances[current] + 1;
                        queue[tail] = n;
                        tail += 1;
                    }
                }
            }
        }
        255
    }

    /// Finds minimum distance from any range-starting hex owned by `player` to `target`.
    pub fn min_distance_from_player(&self, player: u8, target: usize) -> Option<u8> {
        let mut min_dist = 255u8;
        for i in 0..self.count {
            if self.hexes[i].is_range_starting_point(player) {
                let dist = self.distance(i, target);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        if min_dist == 255 {
            None
        } else {
            Some(min_dist)
        }
    }

    /// Calculates the required QIC to reach `target` from the player's closest colonized hex.
    pub fn qic_cost_to_reach(&self, player: u8, effective_range: u8, target: usize) -> Option<u8> {
        self.min_distance_from_player(player, target)
            .map(|dist| qic_for_distance(dist, effective_range, 0))
    }

    /// Returns the number of distinct sectors where `player` has at least one building.
    pub fn sectors_with_player(&self, player: u8) -> usize {
        let mut sectors = [false; 32];
        for hex in &self.hexes[..self.count] {
            if hex.colonized_by(player) {
                let sec = hex.sector_id as usize;
                if sec < sectors.len() {
                    sectors[sec] = true;
                }
            }
        }
        sectors.iter().filter(|&&present| present).count()
    }

    /// Returns the number of Gaia planets colonized by `player`.
    pub fn gaia_planets_colonized_by(&self, player: u8) -> usize {
        self.hexes[..self.count]
            .iter()
            .filter(|hex| hex.planet == Planet::Gaia && hex.colonized_by(player))
            .count()
    }

    /// Returns the number of distinct planet types colonized by `player`.
    pub fn distinct_planets_colonized_by(&self, player: u8) -> usize {
        let mut planets = [false; 16];
        for hex in &self.hexes[..self.count] {
            if hex.colonized_by(player) {
                planets[hex.planet as usize] = true;
            }
        }
        planets.iter().filter(|&&present| present).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Building;

    #[test]
    fn ts_engine_parity_load_configuration() {
        // Matches test in map.spec.ts line 38 from TypeScript engine:
        let conf = MapConfiguration {
            sectors: vec![
                SectorPlacement { sector: SectorId::S3, rotation: 5, center: None },
                SectorPlacement { sector: SectorId::S7B, rotation: 0, center: None },
                SectorPlacement { sector: SectorId::S2, rotation: 3, center: None },
                SectorPlacement { sector: SectorId::S5B, rotation: 2, center: None },
                SectorPlacement { sector: SectorId::S4, rotation: 4, center: None },
                SectorPlacement { sector: SectorId::S1, rotation: 0, center: None },
                SectorPlacement { sector: SectorId::S6B, rotation: 5, center: None },
            ],
            mirror: false,
        };

        let mut map = Map::empty();
        map.load_configuration(&conf);

        assert_eq!(map.count, 7 * 19);

        // (-2, 0, 2) is in sector 5B (id 5) and has Transdim planet
        let h1 = map.get_hex(HexCoord::new(-2, 0, 2).unwrap()).expect("hex at -2x0 exists");
        assert_eq!(h1.sector_id, 5);
        assert_eq!(h1.planet, Planet::Transdim);

        // (-3, 1, 2) is in sector 6B (id 6) and has Transdim planet
        let h2 = map.get_hex(HexCoord::new(-3, 1, 2).unwrap()).expect("hex at -3x1 exists");
        assert_eq!(h2.sector_id, 6);
        assert_eq!(h2.planet, Planet::Transdim);

        // (-4, 2, 2) has Terra planet
        let h3 = map.get_hex(HexCoord::new(-4, 2, 2).unwrap()).expect("hex at -4x2 exists");
        assert_eq!(h3.sector_id, 6);
        assert_eq!(h3.planet, Planet::Terra);
    }

    #[test]
    fn generate_standard_creates_valid_board() {
        let map_2p = Map::generate_standard(2, 42);
        assert_eq!(map_2p.count, 7 * 19);
        assert!(map_2p.is_valid(true));

        let map_4p = Map::generate_standard(4, 42);
        assert_eq!(map_4p.count, 10 * 19);
        assert!(map_4p.is_valid(true));
    }

    #[test]
    fn bfs_distance_computes_correctly() {
        let mut map = Map::empty();
        map.count = 3;

        map.coords[0] = HexCoord::new(0, 0, 0).unwrap();
        map.coords[1] = HexCoord::new(1, -1, 0).unwrap();
        map.coords[2] = HexCoord::new(2, -2, 0).unwrap();

        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;
        map.adjacency[1][0] = 2;
        map.adjacency[2][1] = 1;

        assert_eq!(map.distance(0, 0), 0);
        assert_eq!(map.distance(0, 1), 1);
        assert_eq!(map.distance(0, 2), 2);
        assert_eq!(map.distance(2, 0), 2);
    }

    #[test]
    fn player_reach_and_qic_calculation() {
        let mut map = Map::empty();
        map.count = 4;

        map.adjacency[0][0] = 1;
        map.adjacency[1][1] = 0;
        map.adjacency[1][0] = 2;
        map.adjacency[2][1] = 1;
        map.adjacency[2][0] = 3;
        map.adjacency[3][1] = 2;

        map.hexes[0].planet = Planet::Terra;
        map.hexes[0].building = Some(Building::Mine);
        map.hexes[0].player = Some(0);

        assert_eq!(map.min_distance_from_player(0, 3), Some(3));
        assert_eq!(map.qic_cost_to_reach(0, 1, 3), Some(1));
        assert_eq!(map.qic_cost_to_reach(0, 3, 3), Some(0));
    }
}


