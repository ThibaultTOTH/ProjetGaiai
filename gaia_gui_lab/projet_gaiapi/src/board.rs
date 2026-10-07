//! Hex-grid topology used by the Gaia Project board.
//!
//! This replaces the TypeScript `hexagrid` dependency for the operations needed
//! by range checks, federation searches and RL action generation.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

/// Cube coordinate (`q + r + s == 0`) with the six canonical Gaia neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HexCoord {
    pub q: i16,
    pub r: i16,
    pub s: i16,
}

impl HexCoord {
    pub const fn new(q: i16, r: i16, s: i16) -> Option<Self> {
        if q + r + s == 0 {
            Some(Self { q, r, s })
        } else {
            None
        }
    }
    pub const fn origin() -> Self {
        Self { q: 0, r: 0, s: 0 }
    }
    pub const fn neighbours(self) -> [Self; 6] {
        [
            Self {
                q: self.q + 1,
                r: self.r - 1,
                s: self.s,
            },
            Self {
                q: self.q + 1,
                r: self.r,
                s: self.s - 1,
            },
            Self {
                q: self.q,
                r: self.r + 1,
                s: self.s - 1,
            },
            Self {
                q: self.q - 1,
                r: self.r + 1,
                s: self.s,
            },
            Self {
                q: self.q - 1,
                r: self.r,
                s: self.s + 1,
            },
            Self {
                q: self.q,
                r: self.r - 1,
                s: self.s + 1,
            },
        ]
    }
    /// Geometric distance, independent of holes in a particular board.
    pub const fn geometric_distance(self, other: Self) -> u16 {
        ((self.q - other.q).unsigned_abs()
            + (self.r - other.r).unsigned_abs()
            + (self.s - other.s).unsigned_abs())
            / 2
    }

    /// Clockwise 60-degree rotation around origin (0, 0, 0).
    pub const fn rotate_cw(self) -> Self {
        Self {
            q: -self.r,
            r: -self.s,
            s: -self.q,
        }
    }

    /// Rotate `times` (0..5) steps clockwise around origin.
    pub const fn rotate_cw_times(self, times: u8) -> Self {
        match times % 6 {
            0 => self,
            1 => self.rotate_cw(),
            2 => self.rotate_cw().rotate_cw(),
            3 => self.rotate_cw().rotate_cw().rotate_cw(),
            4 => self.rotate_cw().rotate_cw().rotate_cw().rotate_cw(),
            5 => self.rotate_cw().rotate_cw().rotate_cw().rotate_cw().rotate_cw(),
            _ => unreachable!(),
        }
    }

    /// Rotate `times` steps clockwise around an arbitrary center.
    pub const fn rotate_around(self, center: Self, times: u8) -> Self {
        let rel = Self {
            q: self.q - center.q,
            r: self.r - center.r,
            s: self.s - center.s,
        };
        let rot = rel.rotate_cw_times(times);
        Self {
            q: center.q + rot.q,
            r: center.r + rot.r,
            s: center.s + rot.s,
        }
    }
}

/// Immutable sparse topology. A missing coordinate is not traversable.
#[derive(Debug, Clone)]
pub struct BoardTopology {
    cells: HashSet<HexCoord>,
}

impl BoardTopology {
    pub fn new(cells: impl IntoIterator<Item = HexCoord>) -> Self {
        Self {
            cells: cells.into_iter().collect(),
        }
    }
    pub fn contains(&self, coordinate: HexCoord) -> bool {
        self.cells.contains(&coordinate)
    }
    pub fn neighbours(&self, coordinate: HexCoord) -> impl Iterator<Item = HexCoord> + '_ {
        coordinate
            .neighbours()
            .into_iter()
            .filter(|candidate| self.cells.contains(candidate))
    }
    /// Walking distance through existing cells; `None` mirrors TypeScript's `-1` for unreachable locations.
    ///
    /// **Performance note**: Uses heap-allocated `HashMap`+`VecDeque` BFS. This is acceptable
    /// because `BoardTopology` is only used at setup time and for rule validation, NOT on the
    /// MCTS hot path (which uses the array-based `Map::adjacency` table instead).
    pub fn distance(&self, from: HexCoord, to: HexCoord) -> Option<u16> {
        if !self.contains(from) || !self.contains(to) {
            return None;
        }
        if from == to {
            return Some(0);
        }
        let mut distance = HashMap::from([(from, 0u16)]);
        let mut queue = VecDeque::from([from]);
        while let Some(current) = queue.pop_front() {
            let next_distance = distance[&current] + 1;
            for neighbour in self.neighbours(current) {
                if distance.insert(neighbour, next_distance).is_none() {
                    if neighbour == to {
                        return Some(next_distance);
                    }
                    queue.push_back(neighbour);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distance_respects_holes_like_grid_topology() {
        let origin = HexCoord::origin();
        let right = HexCoord::new(1, -1, 0).unwrap();
        let far = HexCoord::new(2, -2, 0).unwrap();
        assert_eq!(
            BoardTopology::new([origin, right, far]).distance(origin, far),
            Some(2)
        );
        assert_eq!(
            BoardTopology::new([origin, far]).distance(origin, far),
            None
        );
    }

    #[test]
    fn hex_coord_rotation_completes_full_cycle() {
        let coord = HexCoord::new(2, 0, -2).unwrap();
        assert_eq!(coord.rotate_cw(), HexCoord::new(0, 2, -2).unwrap());
        assert_eq!(coord.rotate_cw_times(6), coord);

        let center = HexCoord::new(5, -2, -3).unwrap();
        assert_eq!(center.rotate_around(center, 3), center);
        let pt = HexCoord::new(7, -2, -5).unwrap();
        assert_eq!(pt.rotate_around(center, 6), pt);
    }
}



