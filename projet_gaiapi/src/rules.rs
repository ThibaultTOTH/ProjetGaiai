//! Rule primitives ported from the TypeScript engine.
//!
//! They are deliberately pure functions and static data, so both the HTTP API
//! and an in-process trainer share exactly the same rule calculations.

use crate::Faction;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Planet {
    Empty,
    Terra,
    Desert,
    Swamp,
    Oxide,
    Volcanic,
    Titanium,
    Ice,
    Gaia,
    Transdim,
    Lost,
    Protoplanet,
    Asteroid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Building {
    Mine,
    TradingStation,
    ResearchLab,
    PlanetaryInstitute,
    Academy1,
    Academy2,
    GaiaFormer,
    SpaceStation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResearchField {
    Terraforming,
    Navigation,
    Intelligence,
    GaiaProject,
    Economy,
    Science,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum PowerArea {
    Area1 = 0,
    Area2 = 1,
    Area3 = 2,
    Gaia = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum BoardAction {
    Power1 = 0, // 7pw -> 3k
    Power2 = 1, // 5pw -> 2step
    Power3 = 2, // 4pw -> 2o
    Power4 = 3, // 4pw -> 7c
    Power5 = 4, // 4pw -> 2k
    Power6 = 5, // 3pw -> 1step
    Power7 = 6, // 3pw -> 2t
    Qic1 = 7,   // 4q -> tech
    Qic2 = 8,   // 3q -> rescore fed
    Qic3 = 9,   // 2q -> 3vp + 1vp per planet type
}

impl BoardAction {
    pub const ALL: [Self; 10] = [
        Self::Power1,
        Self::Power2,
        Self::Power3,
        Self::Power4,
        Self::Power5,
        Self::Power6,
        Self::Power7,
        Self::Qic1,
        Self::Qic2,
        Self::Qic3,
    ];

    pub const POWER_ACTIONS: [Self; 7] = [
        Self::Power1,
        Self::Power2,
        Self::Power3,
        Self::Power4,
        Self::Power5,
        Self::Power6,
        Self::Power7,
    ];

    pub const QIC_ACTIONS: [Self; 3] = [
        Self::Qic1,
        Self::Qic2,
        Self::Qic3,
    ];

    #[inline]
    pub const fn power_cost(self) -> u8 {
        match self {
            Self::Power1 => 7,
            Self::Power2 => 5,
            Self::Power3 => 4,
            Self::Power4 => 4,
            Self::Power5 => 4,
            Self::Power6 => 3,
            Self::Power7 => 3,
            _ => 0,
        }
    }

    #[inline]
    pub const fn qic_cost(self) -> u8 {
        match self {
            Self::Qic1 => 4,
            Self::Qic2 => 3,
            Self::Qic3 => 2,
            _ => 0,
        }
    }
}

/// Standard Tech Tiles (9 base).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum TechTile {
    Tech1 = 0, // o,q (gain 1 ore, 1 QIC immediately)
    Tech2 = 1, // +k,c (income 1 knowledge, 1 credit)
    Tech3 = 2, // +o,t (income 1 ore, 1 power token)
    Tech4 = 3, // 7vp (gain 7 VP immediately)
    Tech5 = 4, // +4c (income 4 credits)
    Tech6 = 5, // +o,pw (income 1 ore, charge 1 power)
    Tech7 = 6, // mg >> 3vp (each mine on Gaia gives +3 VP)
    Tech8 = 7, // PA -> 4pw (PI and Academies count as 4 power for federations)
    Tech9 = 8, // => 4pw (action: charge 4 power 1x/round)
}

impl TechTile {
    pub const ALL: [Self; 9] = [
        Self::Tech1,
        Self::Tech2,
        Self::Tech3,
        Self::Tech4,
        Self::Tech5,
        Self::Tech6,
        Self::Tech7,
        Self::Tech8,
        Self::Tech9,
    ];
}

/// Advanced Tech Tiles (15 base).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum AdvTechTile {
    AdvTech1 = 0,   // fed | 3vp (pass: 3 VP per federation token)
    AdvTech2 = 1,   // a >> 2vp (each research advance: +2 VP)
    AdvTech3 = 2,   // => q,5c (action: +1 QIC, +5 credits 1x/round)
    AdvTech4 = 3,   // m > 2vp (immediate: 2 VP per mine)
    AdvTech5 = 4,   // lab | 3vp (pass: 3 VP per lab)
    AdvTech6 = 5,   // s > o (immediate: 1 ore per sector with building)
    AdvTech7 = 6,   // pt | vp (pass: 1 VP per planet type)
    AdvTech8 = 7,   // g > 2vp (immediate: 2 VP per Gaia planet colonized)
    AdvTech9 = 8,   // ts > 4vp (immediate: 4 VP per trading station)
    AdvTech10 = 9,  // s > 2vp (immediate: 2 VP per sector with building)
    AdvTech11 = 10, // => 3o (action: +3 ore 1x/round)
    AdvTech12 = 11, // fed > 5vp (immediate: 5 VP per federation token)
    AdvTech13 = 12, // => 3k (action: +3 knowledge 1x/round)
    AdvTech14 = 13, // m >> 3vp (each mine built: +3 VP)
    AdvTech15 = 14, // ts >> 3vp (each trading station built: +3 VP)
}

impl AdvTechTile {
    pub const ALL: [Self; 15] = [
        Self::AdvTech1,
        Self::AdvTech2,
        Self::AdvTech3,
        Self::AdvTech4,
        Self::AdvTech5,
        Self::AdvTech6,
        Self::AdvTech7,
        Self::AdvTech8,
        Self::AdvTech9,
        Self::AdvTech10,
        Self::AdvTech11,
        Self::AdvTech12,
        Self::AdvTech13,
        Self::AdvTech14,
        Self::AdvTech15,
    ];
}

/// Round Scoring Tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum ScoringTile {
    Score1 = 0,  // step >> 2vp
    Score2 = 1,  // a >> 2vp
    Score3 = 2,  // m >> 2vp
    Score4 = 3,  // fed >> 5vp
    Score5 = 4,  // ts >> 4vp
    Score6 = 5,  // mg >> 4vp
    Score7 = 6,  // PA >> 5vp
    Score8 = 7,  // ts >> 3vp
    Score9 = 8,  // mg >> 3vp
    Score10 = 9, // PA >> 5vp
    LfLab4 = 10,     // lab >> 4vp (Lost Fleet)
    LfSector3 = 11,  // newsector >> 3vp (Lost Fleet)
    LfPlanet3 = 12,  // newplanet >> 3vp (Lost Fleet)
}

impl ScoringTile {
    pub const BASE: [Self; 10] = [
        Self::Score1,
        Self::Score2,
        Self::Score3,
        Self::Score4,
        Self::Score5,
        Self::Score6,
        Self::Score7,
        Self::Score8,
        Self::Score9,
        Self::Score10,
    ];
}

/// Final Scoring Tiles (ranking at game end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum FinalTile {
    Structure = 0,
    StructureFed = 1,
    PlanetType = 2,
    Gaia = 3,
    Sector = 4,
    Satellite = 5,
    Asteroid = 6,
    PlanetaryInstituteAcademyDistance = 7,
    DeepSpaceSector = 8,
}

impl FinalTile {
    pub const BASE: [Self; 6] = [
        Self::Structure,
        Self::StructureFed,
        Self::PlanetType,
        Self::Gaia,
        Self::Sector,
        Self::Satellite,
    ];
}

/// Faction and tile special actions (typically usable once per round).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum SpecialAction {
    AmbasPiSwap = 0,
    FiraksDowngradeLab = 1,
    BescodsAdvanceLowest = 2,
    IvitsSpaceStation = 3,
    SpaceGiantsTerraform = 4,
    Tech9Charge4Power = 5,
    AdvTech3QicCredit = 6,
    AdvTech11Gain3Ore = 7,
    AdvTech13Gain3Knowledge = 8,
    Booster5TemporaryRange = 9,
}

/// Lost Fleet Spaceship Boards (4 generic exploration boards).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum Spaceship {
    Twilight = 0,
    Rebellion = 1,
    TFMars = 2,
    Eclipse = 3,
}

impl Spaceship {
    pub const ALL: [Self; 4] = [
        Self::Twilight,
        Self::Rebellion,
        Self::TFMars,
        Self::Eclipse,
    ];
}

/// Spaceship action types on spaceship exploration boards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum SpaceshipActionType {
    Qic = 0,
    Power = 1,
    Knowledge = 2,
    Credit = 3,
}

/// The 3 new Standard Tech tiles seeded onto Rebellion/TFMars/Eclipse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum SpaceshipTechTile {
    Range = 0,
    Terraform = 1,
    Resource = 2,
}

/// The 8 new Federation tokens distributed across spaceships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum SpaceshipFederation {
    Credit = 0,
    Knowledge = 1,
    OreQic = 2,
    PowerTokens = 3,
    Range = 4,
    Tech = 5,
    Terraform = 6,
    Vp = 7,
}

/// The 13 Artifact tokens in Lost Fleet (seeded onto Twilight spaceship).
/// Discarding 6 power allows a player who has explored Twilight to examine an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum Artefact {
    IncomeKnowledgeOre = 0,
    Credits3Ore3 = 1,
    Knowledge3Qic1 = 2,
    Credits5Ore2 = 3,
    ChargePower2 = 4,
    AsteroidVp = 5,
    ProtoplanetVp = 6,
    ResearchAreaLevel = 7,
    ResearchTracksCount = 8,
    RescoreFederation = 9,
    GaiaformingTrack = 10,
    PlanetTypes = 11,
    DeepSpace = 12,
}

pub const EXPLORATION_CHARGE_TRACK: [u8; 4] = [0, 2, 2, 3];


pub const MAX_ORE: i16 = 15;
pub const MAX_CREDIT: i16 = 30;
pub const MAX_KNOWLEDGE: i16 = 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameResource {
    Ore,
    Credit,
    Knowledge,
    Qic,
    ChargePower,
    PowerToken,
    GaiaToken,
    GaiaFormer,
    VictoryPoint,
    TechTile,
}

impl GameResource {
    /// Wire identifiers used by the TypeScript `Reward` grammar.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Ore => "o",
            Self::Credit => "c",
            Self::Knowledge => "k",
            Self::Qic => "q",
            Self::ChargePower => "pw",
            Self::PowerToken => "t",
            Self::GaiaToken => "tg",
            Self::GaiaFormer => "gf",
            Self::VictoryPoint => "vp",
            Self::TechTile => "tech",
        }
    }
    pub fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "o" => Self::Ore,
            "c" => Self::Credit,
            "k" => Self::Knowledge,
            "q" => Self::Qic,
            "pw" => Self::ChargePower,
            "t" | "ta3" => Self::PowerToken,
            "tg" => Self::GaiaToken,
            "gf" => Self::GaiaFormer,
            "vp" => Self::VictoryPoint,
            "tech" => Self::TechTile,
            _ => return None,
        })
    }
}

/// Type-safe form of the TypeScript `Reward` grammar (`4pw`, `q`, `-2vp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reward {
    pub amount: i16,
    pub resource: GameResource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewardParseError {
    Empty,
    InvalidAmount(String),
    UnknownResource(String),
}
impl std::fmt::Display for RewardParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("empty reward"),
            Self::InvalidAmount(value) => write!(f, "invalid reward amount: {value}"),
            Self::UnknownResource(value) => write!(f, "unknown reward resource: {value}"),
        }
    }
}
impl std::error::Error for RewardParseError {}
impl Reward {
    pub fn parse(specification: &str) -> Result<Self, RewardParseError> {
        let specification = specification.trim();
        if specification.is_empty() || specification == "~" {
            return Err(RewardParseError::Empty);
        }
        let numeric_end = specification
            .char_indices()
            .take_while(|(_, character)| *character == '-' || character.is_ascii_digit())
            .last()
            .map(|(index, character)| index + character.len_utf8())
            .unwrap_or(0);
        let (amount, code) = specification.split_at(numeric_end);
        let resource = GameResource::from_code(code)
            .ok_or_else(|| RewardParseError::UnknownResource(code.into()))?;
        let amount = if amount.is_empty() {
            1
        } else {
            amount
                .parse()
                .map_err(|_| RewardParseError::InvalidAmount(amount.into()))?
        };
        Ok(Self { amount, resource })
    }
    pub fn parse_list(specification: &str) -> Result<Vec<Self>, RewardParseError> {
        specification.split(',').map(Self::parse).collect()
    }
    pub fn merge(rewards: impl IntoIterator<Item = Self>) -> Vec<Self> {
        let mut result: Vec<Self> = Vec::new();
        for reward in rewards {
            if let Some(existing) = result
                .iter_mut()
                .find(|item| item.resource == reward.resource)
            {
                existing.amount += reward.amount;
            } else {
                result.push(reward);
            }
        }
        result.retain(|item| item.amount != 0);
        result
    }
}
impl std::fmt::Display for Reward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.amount == 1 {
            f.write_str(self.resource.code())
        } else {
            write!(f, "{}{}", self.amount, self.resource.code())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceAmount {
    pub resource: GameResource,
    pub amount: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionRule {
    pub cost: &'static [ResourceAmount],
    pub income: &'static [ResourceAmount],
}

/// Exact free-action identifiers from `src/actions.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreeAction {
    PowerToQic,
    PowerToKnowledge,
    PowerToOre,
    PowerToCredit,
    QicToOre,
    OreToToken,
    KnowledgeToCredit,
    OreToCredit,
    CreditToQic,
    CreditToOre,
    CreditToKnowledge,
    GaiaTokenToQic,
    GaiaTokenToKnowledge,
    GaiaTokenToOre,
    GaiaTokenToCredit,
    GaiaTokenToTech,
    PowerToGaiaForKnowledge,
    PowerToOreAndCredit,
    PowerTo2Credit,
    PowerTo2Ore,
    GaiaFormerToQic,
    PowerTo3Credit,
    OreToPowerTokenArea3,
}

/// Faction-aware availability of the TypeScript free-action tables. Affordability
/// is intentionally left to the environment's resource mask.
pub fn free_actions_for(faction: Faction) -> Vec<FreeAction> {
    let mut actions = vec![
        FreeAction::PowerToQic,
        FreeAction::PowerToKnowledge,
        FreeAction::PowerToOre,
        FreeAction::PowerToCredit,
        FreeAction::QicToOre,
        FreeAction::OreToToken,
        FreeAction::KnowledgeToCredit,
        FreeAction::OreToCredit,
    ];
    match faction {
        Faction::HadschHallas => actions.extend([
            FreeAction::CreditToQic,
            FreeAction::CreditToOre,
            FreeAction::CreditToKnowledge,
        ]),
        Faction::Terrans => actions.extend([
            FreeAction::GaiaTokenToQic,
            FreeAction::GaiaTokenToKnowledge,
            FreeAction::GaiaTokenToOre,
            FreeAction::GaiaTokenToCredit,
        ]),
        Faction::Itars => actions.push(FreeAction::GaiaTokenToTech),
        Faction::Nevlas => actions.extend([
            FreeAction::PowerToGaiaForKnowledge,
            FreeAction::PowerToOreAndCredit,
            FreeAction::PowerTo2Credit,
            FreeAction::PowerTo2Ore,
        ]),
        Faction::BalTaks => actions.push(FreeAction::GaiaFormerToQic),
        Faction::Taklons => actions.push(FreeAction::PowerTo3Credit),
        Faction::Xenos => actions.push(FreeAction::OreToPowerTokenArea3),
        _ => {}
    }
    actions
}

const POWER_TO_QIC: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Qic,
        amount: 1,
    }],
};
const POWER_TO_KNOWLEDGE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Knowledge,
        amount: 1,
    }],
};
const POWER_TO_ORE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 3,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
};
const POWER_TO_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 1,
    }],
};
const QIC_TO_ORE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Qic,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
};
const ORE_TO_TOKEN: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::PowerToken,
        amount: 1,
    }],
};
const KNOWLEDGE_TO_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Knowledge,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 1,
    }],
};
const ORE_TO_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 1,
    }],
};
const CREDIT_TO_QIC: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Qic,
        amount: 1,
    }],
};
const CREDIT_TO_ORE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 3,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
};
const CREDIT_TO_KNOWLEDGE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Knowledge,
        amount: 1,
    }],
};
const GAIA_TO_QIC: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::GaiaToken,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Qic,
        amount: 1,
    }],
};
const GAIA_TO_KNOWLEDGE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::GaiaToken,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Knowledge,
        amount: 1,
    }],
};
const GAIA_TO_ORE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::GaiaToken,
        amount: 3,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
};
const GAIA_TO_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::GaiaToken,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 1,
    }],
};
const GAIA_TO_TECH: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::GaiaToken,
        amount: 4,
    }],
    income: &[ResourceAmount {
        resource: GameResource::TechTile,
        amount: 1,
    }],
};
const POWER_TO_GAIA_KNOWLEDGE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::PowerToken,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Knowledge,
        amount: 1,
    }],
};
const POWER_TO_ORE_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 4,
    }],
    income: &[
        ResourceAmount {
            resource: GameResource::Ore,
            amount: 1,
        },
        ResourceAmount {
            resource: GameResource::Credit,
            amount: 1,
        },
    ],
};
const POWER_TO_2_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 2,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 2,
    }],
};
const POWER_TO_2_ORE: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 6,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 2,
    }],
};
const FORMER_TO_QIC: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::GaiaFormer,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Qic,
        amount: 1,
    }],
};
const POWER_TO_3_CREDIT: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::ChargePower,
        amount: 3,
    }],
    income: &[ResourceAmount {
        resource: GameResource::Credit,
        amount: 3,
    }],
};
const ORE_TO_AREA3_TOKEN: ConversionRule = ConversionRule {
    cost: &[ResourceAmount {
        resource: GameResource::Ore,
        amount: 1,
    }],
    income: &[ResourceAmount {
        resource: GameResource::PowerToken,
        amount: 1,
    }],
};

pub fn free_action_conversion(action: FreeAction) -> &'static ConversionRule {
    match action {
        FreeAction::PowerToQic => &POWER_TO_QIC,
        FreeAction::PowerToKnowledge => &POWER_TO_KNOWLEDGE,
        FreeAction::PowerToOre => &POWER_TO_ORE,
        FreeAction::PowerToCredit => &POWER_TO_CREDIT,
        FreeAction::QicToOre => &QIC_TO_ORE,
        FreeAction::OreToToken => &ORE_TO_TOKEN,
        FreeAction::KnowledgeToCredit => &KNOWLEDGE_TO_CREDIT,
        FreeAction::OreToCredit => &ORE_TO_CREDIT,
        FreeAction::CreditToQic => &CREDIT_TO_QIC,
        FreeAction::CreditToOre => &CREDIT_TO_ORE,
        FreeAction::CreditToKnowledge => &CREDIT_TO_KNOWLEDGE,
        FreeAction::GaiaTokenToQic => &GAIA_TO_QIC,
        FreeAction::GaiaTokenToKnowledge => &GAIA_TO_KNOWLEDGE,
        FreeAction::GaiaTokenToOre => &GAIA_TO_ORE,
        FreeAction::GaiaTokenToCredit => &GAIA_TO_CREDIT,
        FreeAction::GaiaTokenToTech => &GAIA_TO_TECH,
        FreeAction::PowerToGaiaForKnowledge => &POWER_TO_GAIA_KNOWLEDGE,
        FreeAction::PowerToOreAndCredit => &POWER_TO_ORE_CREDIT,
        FreeAction::PowerTo2Credit => &POWER_TO_2_CREDIT,
        FreeAction::PowerTo2Ore => &POWER_TO_2_ORE,
        FreeAction::GaiaFormerToQic => &FORMER_TO_QIC,
        FreeAction::PowerTo3Credit => &POWER_TO_3_CREDIT,
        FreeAction::OreToPowerTokenArea3 => &ORE_TO_AREA3_TOKEN,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildCost {
    pub credits: i16,
    pub ore: i16,
}

pub const MINE_COST: BuildCost = BuildCost { credits: 2, ore: 1 };
pub const TS_COST_ISOLATED: BuildCost = BuildCost { credits: 6, ore: 2 };
pub const TS_COST_NEIGHBOR: BuildCost = BuildCost { credits: 3, ore: 2 };
pub const PI_COST: BuildCost = BuildCost { credits: 6, ore: 4 };
pub const LAB_COST: BuildCost = BuildCost { credits: 5, ore: 3 };
pub const ACADEMY_COST: BuildCost = BuildCost { credits: 6, ore: 6 };

pub const TERRAFORMING_COST: i16 = 3;

/// Computes the number of terraforming steps required by a faction to colonize a planet.
/// Follows base game color wheel and Lost Fleet expansion rules (Protoplanets, Asteroids, Darkanians, Space Giants, etc.).
pub fn terraforming_steps(faction: Faction, target: Planet, cost3_planets: &[Planet]) -> u8 {
    if matches!(target, Planet::Gaia | Planet::Transdim | Planet::Asteroid | Planet::Empty | Planet::Lost) {
        return 0;
    }
    if target == Planet::Protoplanet {
        return 3;
    }
    // Lost Fleet factions:
    if faction == Faction::Darkanians {
        return 1;
    }
    if faction == Faction::SpaceGiants {
        return 2;
    }
    if faction == Faction::Tinkeroids || faction == Faction::Moweyds {
        return if cost3_planets.contains(&target) { 3 } else { 1 };
    }

    const CYCLE: [Planet; 7] = [
        Planet::Terra,
        Planet::Oxide,
        Planet::Volcanic,
        Planet::Desert,
        Planet::Swamp,
        Planet::Titanium,
        Planet::Ice,
    ];

    let home = faction_planet(faction);
    let home_idx = match CYCLE.iter().position(|&p| p == home) {
        Some(idx) => idx as i8,
        None => return 0,
    };
    let target_idx = match CYCLE.iter().position(|&p| p == target) {
        Some(idx) => idx as i8,
        None => return 0,
    };

    let mut diff = (target_idx - home_idx).abs();
    if diff > 3 {
        diff = 7 - diff;
    }
    diff as u8
}

pub fn terraforming_ore_cost(temporary_steps: i16, terraforming_discount: i16, steps: i16) -> i16 {
    (TERRAFORMING_COST - terraforming_discount) * (steps - temporary_steps).max(0)
}
pub fn qic_for_distance(distance: u8, effective_range: u8, temporary_range: u8) -> u8 {
    distance
        .saturating_sub(effective_range.saturating_add(temporary_range))
        .div_ceil(2)
}
pub fn is_academy(building: Building) -> bool {
    matches!(building, Building::Academy1 | Building::Academy2)
}
pub fn standard_building_value(building: Building) -> u8 {
    match building {
        Building::Mine => 1,
        Building::TradingStation | Building::ResearchLab => 2,
        Building::PlanetaryInstitute | Building::Academy1 | Building::Academy2 => 3,
        _ => 0,
    }
}
pub fn upgraded_buildings(building: Building, faction: Faction) -> &'static [Building] {
    match building {
        Building::GaiaFormer => &[Building::Mine],
        Building::Mine => &[Building::TradingStation],
        Building::TradingStation if faction == Faction::Bescods => &[
            Building::Academy1,
            Building::Academy2,
            Building::ResearchLab,
        ],
        Building::TradingStation => &[Building::PlanetaryInstitute, Building::ResearchLab],
        Building::ResearchLab if faction == Faction::Bescods => &[Building::PlanetaryInstitute],
        Building::ResearchLab => &[Building::Academy1, Building::Academy2],
        _ => &[],
    }
}
pub fn faction_planet(faction: Faction) -> Planet {
    match faction {
        Faction::Terrans | Faction::Lantids => Planet::Terra,
        Faction::Xenos | Faction::Gleens => Planet::Desert,
        Faction::Taklons | Faction::Ambas => Planet::Swamp,
        Faction::HadschHallas | Faction::Ivits => Planet::Oxide,
        Faction::Geodens | Faction::BalTaks => Planet::Volcanic,
        Faction::Firaks | Faction::Bescods => Planet::Titanium,
        Faction::Nevlas | Faction::Itars => Planet::Ice,
        Faction::Tinkeroids | Faction::Darkanians => Planet::Asteroid,
        Faction::Moweyds | Faction::SpaceGiants => Planet::Protoplanet,
    }
}

pub fn building_power_value(
    building: Building,
    faction: Faction,
    has_pi: bool,
    on_home_planet: bool,
) -> u8 {
    let base = match building {
        Building::Mine => 1,
        Building::TradingStation | Building::ResearchLab => 2,
        Building::PlanetaryInstitute | Building::Academy1 | Building::Academy2 => 3,
        Building::SpaceStation => 1,
        Building::GaiaFormer => 0,
    };
    if faction == Faction::Bescods && has_pi && on_home_planet && base > 0 {
        base + 1
    } else {
        base
    }
}

/// Federation tokens available in Gaia Project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FederationToken {
    /// 12 VP (gray on both sides)
    Fed1,
    /// 8 VP, 1 QIC (green / gray)
    Fed2,
    /// 8 VP, 2 Power Tokens (green / gray)
    Fed3,
    /// 7 VP, 2 Ore (green / gray)
    Fed4,
    /// 7 VP, 6 Credits (green / gray)
    Fed5,
    /// 6 VP, 2 Knowledge (green / gray)
    Fed6,
    /// Gleens special token: 1 ore, 1 knowledge, 2 credits (green / green)
    Gleens,
}

/// Instant rewards granted when claiming a federation token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FederationReward {
    pub vp: i16,
    pub credits: i16,
    pub ore: i16,
    pub knowledge: i16,
    pub qic: i16,
    pub power_tokens: i16,
    pub is_green: bool,
}

impl FederationToken {
    pub const BASE: [Self; 6] = [
        Self::Fed1,
        Self::Fed2,
        Self::Fed3,
        Self::Fed4,
        Self::Fed5,
        Self::Fed6,
    ];

    pub fn reward(self) -> FederationReward {
        match self {
            Self::Fed1 => FederationReward {
                vp: 12,
                credits: 0,
                ore: 0,
                knowledge: 0,
                qic: 0,
                power_tokens: 0,
                is_green: false,
            },
            Self::Fed2 => FederationReward {
                vp: 8,
                credits: 0,
                ore: 0,
                knowledge: 0,
                qic: 1,
                power_tokens: 0,
                is_green: true,
            },
            Self::Fed3 => FederationReward {
                vp: 8,
                credits: 0,
                ore: 0,
                knowledge: 0,
                qic: 0,
                power_tokens: 2,
                is_green: true,
            },
            Self::Fed4 => FederationReward {
                vp: 7,
                credits: 0,
                ore: 2,
                knowledge: 0,
                qic: 0,
                power_tokens: 0,
                is_green: true,
            },
            Self::Fed5 => FederationReward {
                vp: 7,
                credits: 6,
                ore: 0,
                knowledge: 0,
                qic: 0,
                power_tokens: 0,
                is_green: true,
            },
            Self::Fed6 => FederationReward {
                vp: 6,
                credits: 0,
                ore: 0,
                knowledge: 2,
                qic: 0,
                power_tokens: 0,
                is_green: true,
            },
            Self::Gleens => FederationReward {
                vp: 0,
                credits: 2,
                ore: 1,
                knowledge: 1,
                qic: 0,
                power_tokens: 0,
                is_green: true,
            },
        }
    }

    pub fn is_green(self) -> bool {
        self != Self::Fed1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conversion_matches_typescript_table() {
        let rule = free_action_conversion(FreeAction::PowerToOre);
        assert_eq!(
            rule.cost[0],
            ResourceAmount {
                resource: GameResource::ChargePower,
                amount: 3
            }
        );
        assert_eq!(
            rule.income[0],
            ResourceAmount {
                resource: GameResource::Ore,
                amount: 1
            }
        );
    }
    #[test]
    fn costs_match_typescript_formulas() {
        assert_eq!(terraforming_ore_cost(1, 1, 4), 6);
        assert_eq!(qic_for_distance(7, 3, 0), 2);
    }
    #[test]
    fn faction_and_upgrade_exceptions_are_preserved() {
        assert_eq!(faction_planet(Faction::Bescods), Planet::Titanium);
        assert_eq!(
            upgraded_buildings(Building::TradingStation, Faction::Bescods),
            &[
                Building::Academy1,
                Building::Academy2,
                Building::ResearchLab
            ]
        );
    }

    #[test]
    fn reward_grammar_and_merge_match_typescript() {
        assert_eq!(
            Reward::parse_list("4pw,q,-2vp")
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["4pw", "q", "-2vp"]
        );
        assert_eq!(
            Reward::merge([Reward::parse("o").unwrap(), Reward::parse("2o").unwrap()]),
            [Reward {
                amount: 3,
                resource: GameResource::Ore
            }]
        );
    }
    #[test]
    fn faction_free_actions_preserve_special_tables() {
        assert!(free_actions_for(Faction::Terrans).contains(&FreeAction::GaiaTokenToQic));
        assert!(free_actions_for(Faction::Xenos).contains(&FreeAction::OreToPowerTokenArea3));
        assert!(!free_actions_for(Faction::Geodens).contains(&FreeAction::CreditToQic));
    }

    #[test]
    fn terraforming_steps_matches_color_wheel_and_expansion() {
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Terra, &[]), 0);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Oxide, &[]), 1);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Ice, &[]), 1);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Volcanic, &[]), 2);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Titanium, &[]), 2);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Desert, &[]), 3);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Swamp, &[]), 3);

        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Protoplanet, &[]), 3);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Asteroid, &[]), 0);
        assert_eq!(terraforming_steps(Faction::Terrans, Planet::Gaia, &[]), 0);

        assert_eq!(terraforming_steps(Faction::Darkanians, Planet::Desert, &[]), 1);
        assert_eq!(terraforming_steps(Faction::SpaceGiants, Planet::Desert, &[]), 2);
    }

    #[test]
    fn federation_tokens_and_power_values_work() {
        assert_eq!(FederationToken::Fed1.reward().vp, 12);
        assert!(!FederationToken::Fed1.is_green());
        assert!(FederationToken::Fed2.is_green());
        assert_eq!(FederationToken::Fed2.reward().qic, 1);

        assert_eq!(building_power_value(Building::Mine, Faction::Terrans, false, false), 1);
        assert_eq!(building_power_value(Building::PlanetaryInstitute, Faction::Terrans, false, false), 3);
        // Bescods PI increases power value on gray home planets by 1
        assert_eq!(building_power_value(Building::Mine, Faction::Bescods, true, true), 2);
        assert_eq!(building_power_value(Building::Mine, Faction::Bescods, false, true), 1);
        assert_eq!(building_power_value(Building::Mine, Faction::Bescods, true, false), 1);
    }
}




