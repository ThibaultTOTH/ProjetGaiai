//! Sector definitions and layout geometry for Gaia Project.
//!
//! Mirrors `sector.ts` and the sector definitions from `map.ts` in the reference engine.

use serde::{Deserialize, Serialize};

use crate::board::HexCoord;
use crate::hex::Hex;
use crate::rules::Planet;

/// Available sector tiles in the base game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SectorId {
    #[serde(rename = "1", alias = "S1", alias = "s1")]
    S1,
    #[serde(rename = "2", alias = "S2", alias = "s2")]
    S2,
    #[serde(rename = "3", alias = "S3", alias = "s3")]
    S3,
    #[serde(rename = "4", alias = "S4", alias = "s4")]
    S4,
    #[serde(rename = "5A", alias = "5a", alias = "S5A", alias = "s5a")]
    S5A,
    #[serde(rename = "5B", alias = "5b", alias = "S5B", alias = "s5b")]
    S5B,
    #[serde(rename = "6A", alias = "6a", alias = "S6A", alias = "s6a")]
    S6A,
    #[serde(rename = "6B", alias = "6b", alias = "S6B", alias = "s6b")]
    S6B,
    #[serde(rename = "7A", alias = "7a", alias = "S7A", alias = "s7a")]
    S7A,
    #[serde(rename = "7B", alias = "7b", alias = "S7B", alias = "s7b")]
    S7B,
    #[serde(rename = "8", alias = "S8", alias = "s8")]
    S8,
    #[serde(rename = "9", alias = "S9", alias = "s9")]
    S9,
    #[serde(rename = "10", alias = "S10", alias = "s10")]
    S10,
}

impl SectorId {
    pub const ALL: [Self; 13] = [
        Self::S1,
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5A,
        Self::S5B,
        Self::S6A,
        Self::S6B,
        Self::S7A,
        Self::S7B,
        Self::S8,
        Self::S9,
        Self::S10,
    ];

    /// Default 7 sectors for 2-player games (using B-sides for 5, 6, 7).
    pub const SMALL_CONFIG: [Self; 7] = [
        Self::S1,
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5B,
        Self::S6B,
        Self::S7B,
    ];

    /// Default 10 sectors for 3-4 player games (using A-sides for 5, 6, 7).
    pub const BIG_CONFIG: [Self; 10] = [
        Self::S1,
        Self::S2,
        Self::S3,
        Self::S4,
        Self::S5A,
        Self::S6A,
        Self::S7A,
        Self::S8,
        Self::S9,
        Self::S10,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::S1 => "1",
            Self::S2 => "2",
            Self::S3 => "3",
            Self::S4 => "4",
            Self::S5A => "5A",
            Self::S5B => "5B",
            Self::S6A => "6A",
            Self::S6B => "6B",
            Self::S7A => "7A",
            Self::S7B => "7B",
            Self::S8 => "8",
            Self::S9 => "9",
            Self::S10 => "10",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        name.parse().ok()
    }
}

impl std::str::FromStr for SectorId {
    type Err = ();

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "1" => Ok(Self::S1),
            "2" => Ok(Self::S2),
            "3" => Ok(Self::S3),
            "4" => Ok(Self::S4),
            "5A" | "5a" => Ok(Self::S5A),
            "5B" | "5b" => Ok(Self::S5B),
            "6A" | "6a" => Ok(Self::S6A),
            "6B" | "6b" => Ok(Self::S6B),
            "7A" | "7a" => Ok(Self::S7A),
            "7B" | "7b" => Ok(Self::S7B),
            "8" => Ok(Self::S8),
            "9" => Ok(Self::S9),
            "10" => Ok(Self::S10),
            _ => Err(()),
        }
    }
}

impl SectorId {

    /// Numeric sector id (ignoring A/B suffix), fits into `Hex::sector_id`.
    pub const fn numeric_id(self) -> u8 {
        match self {
            Self::S1 => 1,
            Self::S2 => 2,
            Self::S3 => 3,
            Self::S4 => 4,
            Self::S5A | Self::S5B => 5,
            Self::S6A | Self::S6B => 6,
            Self::S7A | Self::S7B => 7,
            Self::S8 => 8,
            Self::S9 => 9,
            Self::S10 => 10,
        }
    }
}

/// The 19 canonical relative coordinates of a 2-radius hexagon.
/// Indices 0..11 = Ring A (outer, radius 2)
/// Indices 12..17 = Ring B (middle, radius 1)
/// Index 18 = Ring C (center, radius 0)
pub const SECTOR_OFFSETS: [HexCoord; 19] = [
    // Outer Ring A: A0 to A11
    HexCoord { q: 2, r: 0, s: -2 },
    HexCoord { q: 1, r: 1, s: -2 },
    HexCoord { q: 0, r: 2, s: -2 },
    HexCoord { q: -1, r: 2, s: -1 },
    HexCoord { q: -2, r: 2, s: 0 },
    HexCoord { q: -2, r: 1, s: 1 },
    HexCoord { q: -2, r: 0, s: 2 },
    HexCoord { q: -1, r: -1, s: 2 },
    HexCoord { q: 0, r: -2, s: 2 },
    HexCoord { q: 1, r: -2, s: 1 },
    HexCoord { q: 2, r: -2, s: 0 },
    HexCoord { q: 2, r: -1, s: -1 },
    // Middle Ring B: B0 to B5
    HexCoord { q: 1, r: 0, s: -1 },
    HexCoord { q: 0, r: 1, s: -1 },
    HexCoord { q: -1, r: 1, s: 0 },
    HexCoord { q: -1, r: 0, s: 1 },
    HexCoord { q: 0, r: -1, s: 1 },
    HexCoord { q: 1, r: -1, s: 0 },
    // Center: C
    HexCoord { q: 0, r: 0, s: 0 },
];

/// Official sector centers for the 2-player layout (7 sectors).
pub const SMALL_CENTERS: [HexCoord; 7] = [
    HexCoord { q: 5, r: -2, s: -3 },
    HexCoord { q: 2, r: 3, s: -5 },
    HexCoord { q: 3, r: -5, s: 2 },
    HexCoord { q: 0, r: 0, s: 0 },
    HexCoord { q: -3, r: 5, s: -2 },
    HexCoord { q: -2, r: -3, s: 5 },
    HexCoord { q: -5, r: 2, s: 3 },
];

/// Official sector centers for the 3-4 player layout (10 sectors).
pub const BIG_CENTERS: [HexCoord; 10] = [
    HexCoord { q: 5, r: -2, s: -3 },
    HexCoord { q: 2, r: 3, s: -5 },
    HexCoord { q: -1, r: 8, s: -7 },
    HexCoord { q: 3, r: -5, s: 2 },
    HexCoord { q: 0, r: 0, s: 0 },
    HexCoord { q: -3, r: 5, s: -2 },
    HexCoord { q: -6, r: 10, s: -4 },
    HexCoord { q: -2, r: -3, s: 5 },
    HexCoord { q: -5, r: 2, s: 3 },
    HexCoord { q: -8, r: 7, s: 1 },
];

const fn parse_planet_char(c: u8) -> Planet {
    match c {
        b'r' => Planet::Terra,
        b'd' => Planet::Desert,
        b's' => Planet::Swamp,
        b'o' => Planet::Oxide,
        b'v' => Planet::Volcanic,
        b't' => Planet::Titanium,
        b'i' => Planet::Ice,
        b'g' => Planet::Gaia,
        b'm' => Planet::Transdim,
        b'l' => Planet::Lost,
        _ => Planet::Empty,
    }
}

const fn parse_sector_def(s: &[u8]) -> [Planet; 19] {
    let mut planets = [Planet::Empty; 19];
    let mut idx = 0;
    let mut i = 0;
    while i < s.len() && idx < 19 {
        if s[i] != b',' {
            planets[idx] = parse_planet_char(s[i]);
            idx += 1;
        }
        i += 1;
    }
    planets
}

// 19-char raw sector strings directly from map.ts
const S1_MAP: [Planet; 19] = parse_sector_def(b"eeemevoeedee,ereees,e");
const S2_MAP: [Planet; 19] = parse_sector_def(b"teedemeoeeev,eieese,e");
const S3_MAP: [Planet; 19] = parse_sector_def(b"meeteedreeee,eeieeg,e");
const S4_MAP: [Planet; 19] = parse_sector_def(b"teeereeeeiee,oeseve,e");
const S5A_MAP: [Planet; 19] = parse_sector_def(b"iemoeedveeee,eeeeeg,e");
const S5B_MAP: [Planet; 19] = parse_sector_def(b"iemoeeeveeee,eeeeeg,e");
const S6A_MAP: [Planet; 19] = parse_sector_def(b"emeedmeeeeee,ereges,e");
const S6B_MAP: [Planet; 19] = parse_sector_def(b"emeedmeeeeee,eregee,e");
const S7A_MAP: [Planet; 19] = parse_sector_def(b"eseeeeteeeme,oegege,e");
const S7B_MAP: [Planet; 19] = parse_sector_def(b"eeeeeeteeeme,gesege,e");
const S8_MAP: [Planet; 19] = parse_sector_def(b"remeeeemeeee,ieteve,e");
const S9_MAP: [Planet; 19] = parse_sector_def(b"emieeeeeseev,eegete,e");
const S10_MAP: [Planet; 19] = parse_sector_def(b"emmeeeeoreee,eegeed,e");

/// Returns the 19 planets for the given sector, applying mirroring if requested.
pub fn sector_planets(sector: SectorId, mirror: bool) -> [Planet; 19] {
    let mut planets = match sector {
        SectorId::S1 => S1_MAP,
        SectorId::S2 => S2_MAP,
        SectorId::S3 => S3_MAP,
        SectorId::S4 => S4_MAP,
        SectorId::S5A => S5A_MAP,
        SectorId::S5B => S5B_MAP,
        SectorId::S6A => S6A_MAP,
        SectorId::S6B => S6B_MAP,
        SectorId::S7A => S7A_MAP,
        SectorId::S7B => S7B_MAP,
        SectorId::S8 => S8_MAP,
        SectorId::S9 => S9_MAP,
        SectorId::S10 => S10_MAP,
    };

    if mirror {
        // Reverse ring A (indices 1..12 in 1-based = indices 1..11 in 0-based)
        planets[1..12].reverse();
        // Reverse ring B (indices 13..18 in 1-based = indices 13..17 in 0-based)
        planets[13..18].reverse();
    }

    planets
}

/// Computes the 19 absolute coordinates and Hex cells for a sector placed on the map.
pub fn place_sector(
    sector: SectorId,
    center: HexCoord,
    rotation: u8,
    mirror: bool,
) -> [(HexCoord, Hex); 19] {
    let planets = sector_planets(sector, mirror);
    let sec_num = sector.numeric_id();
    let mut result = [(HexCoord::origin(), Hex::empty()); 19];

    for i in 0..19 {
        let rel = SECTOR_OFFSETS[i];
        let rotated_rel = rel.rotate_cw_times(rotation);
        let abs_coord = HexCoord {
            q: center.q + rotated_rel.q,
            r: center.r + rotated_rel.r,
            s: center.s + rotated_rel.s,
        };
        let hex = Hex::new(planets[i], sec_num);
        result[i] = (abs_coord, hex);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sector_offsets_are_all_valid_cube_coordinates() {
        for coord in SECTOR_OFFSETS {
            assert_eq!(coord.q + coord.r + coord.s, 0);
        }
    }

    #[test]
    fn small_and_big_centers_are_valid_cube_coordinates() {
        for center in SMALL_CENTERS {
            assert_eq!(center.q + center.r + center.s, 0);
        }
        for center in BIG_CENTERS {
            assert_eq!(center.q + center.r + center.s, 0);
        }
    }

    #[test]
    fn sector_name_parsing_round_trips() {
        for sector in SectorId::ALL {
            let name = sector.as_str();
            assert_eq!(SectorId::from_name(name), Some(sector));
            assert_eq!(name.parse::<SectorId>(), Ok(sector));
        }
    }
}



