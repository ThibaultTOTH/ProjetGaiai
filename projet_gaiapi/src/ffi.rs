//! C-ABI Foreign Function Interface for gaiapi.
//!
//! Enables zero-overhead in-process embedding of the deterministic Gaia Project
//! RL simulation kernel into Python via `ctypes`.

use std::slice;

use crate::{
    Faction, GaiaEnv, GameConfig, ACTION_SPACE,
};

/// Instantiates a new Gaia Project RL environment.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_create(seed: u64, players: u32) -> *mut GaiaEnv {
    let p_count = (players as usize).clamp(2, 4);
    let config = GameConfig {
        players: p_count,
        max_rounds: 6,
        seed,
        factions: None,
    };
    match GaiaEnv::new(config) {
        Ok(env) => Box::into_raw(Box::new(env)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees an environment instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_destroy(env: *mut GaiaEnv) {
    if !env.is_null() {
        drop(unsafe { Box::from_raw(env) });
    }
}

/// Resets the environment with an optional new seed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_reset(env: *mut GaiaEnv, seed: u64) {
    if let Some(env) = unsafe { env.as_mut() } {
        env.config.seed = seed;
        env.reset();
    }
}

/// Overrides a player seat's faction (0..17 for all 18 factions including Lost Fleet).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_player_faction(env: *mut GaiaEnv, seat: u32, faction_id: u32) {
    if let Some(env) = unsafe { env.as_mut() } {
        if (seat as usize) < env.players.len() && (faction_id as usize) < Faction::ALL.len() {
            let faction = Faction::ALL[faction_id as usize];
            env.players[seat as usize].faction = faction;
        }
    }
}

/// Returns the exhaustive observation vector dimension (2276).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_obs_dim(_env: *mut GaiaEnv) -> u32 {
    crate::OBS_SPACE as u32
}

/// Returns the discrete action dimension (6).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_action_dim() -> u32 {
    ACTION_SPACE as u32
}

/// Copies current observation vector into `out_ptr`. Returns number of floats copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_observation(
    env: *mut GaiaEnv,
    out_ptr: *mut f32,
    max_len: u32,
) -> u32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if out_ptr.is_null() {
            return 0;
        }
        let obs = env.observe();
        let copy_len = obs.values.len().min(max_len as usize);
        let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, copy_len) };
        out_slice.copy_from_slice(&obs.values[..copy_len]);
        copy_len as u32
    } else {
        0
    }
}

/// Copies egocentric observation vector into `out_ptr`. Returns number of floats copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_observation_egocentric(
    env: *mut GaiaEnv,
    out_ptr: *mut f32,
    max_len: u32,
) -> u32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if out_ptr.is_null() {
            return 0;
        }
        let obs = env.observe_egocentric();
        let copy_len = obs.values.len().min(max_len as usize);
        let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, copy_len) };
        out_slice.copy_from_slice(&obs.values[..copy_len]);
        copy_len as u32
    } else {
        0
    }
}



/// Copies boolean legality mask into `out_ptr` as bytes (0 or 1). Returns elements written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_action_mask(
    env: *mut GaiaEnv,
    out_ptr: *mut u8,
    max_len: u32,
) -> u32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if out_ptr.is_null() {
            return 0;
        }
        let mask = env.action_mask();
        let copy_len = mask.len().min(max_len as usize);
        let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, copy_len) };
        for i in 0..copy_len {
            out_slice[i] = if mask[i] { 1 } else { 0 };
        }
        copy_len as u32
    } else {
        0
    }
}

/// Applies a discrete flattened action to the environment.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_step(
    env: *mut GaiaEnv,
    action_idx: u32,
    out_reward: *mut f32,
    out_done: *mut bool,
    out_round: *mut u32,
    out_current_player: *mut u32,
) -> bool {
    if let Some(env) = unsafe { env.as_mut() } {
        let cmd = match crate::action_space::decode_action(action_idx as usize, &env.map) {
            Some(a) => a,
            None => return false,
        };
        let actor = env.current_player;
        match env.execute_command_from_rl(actor, cmd) {
            Ok(step_res) => {
                unsafe {
                    if !out_reward.is_null() {
                        let r = if actor < step_res.rewards.len() {
                            step_res.rewards[actor]
                        } else {
                            0.0
                        };
                        *out_reward = r;
                    }
                    if !out_done.is_null() {
                        *out_done = step_res.terminated;
                    }
                    if !out_round.is_null() {
                        *out_round = step_res.round as u32;
                    }
                    if !out_current_player.is_null() {
                        *out_current_player = step_res.current_player as u32;
                    }
                }
                true
            }
            Err(_) => false,
        }
    } else {
        false
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_action_name(
    action_idx: u32,
    buf: *mut std::ffi::c_char,
    buf_len: usize,
) {
    if buf.is_null() || buf_len == 0 {
        return;
    }
    let dummy_map = crate::map::Map::empty();
    let name = crate::action_space::get_action_name(action_idx as usize, &dummy_map);
    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(buf_len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), buf as *mut u8, copy_len);
        *(buf.add(copy_len) as *mut u8) = 0; // null terminator
    }
}

/// Returns current victory points of a given player.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_player_vp(env: *mut GaiaEnv, seat: u32) -> f32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if (seat as usize) < env.players.len() {
            env.players[seat as usize].victory_points as f32
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Copies all players' victory points into `out_vps` (4 floats, one per seat).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_all_rewards(
    env: *mut GaiaEnv,
    out_vps: *mut f32,
    max_seats: u32,
) -> u32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if out_vps.is_null() {
            return 0;
        }
        let seats = env.players.len().min(max_seats as usize);
        let out_slice = unsafe { slice::from_raw_parts_mut(out_vps, seats) };
        for i in 0..seats {
            out_slice[i] = env.players[i].victory_points as f32;
        }
        seats as u32
    } else {
        0
    }
}

/// Clones an existing environment instance with zero serialization overhead.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_clone(env: *mut GaiaEnv) -> *mut GaiaEnv {
    if let Some(env) = unsafe { env.as_ref() } {
        Box::into_raw(Box::new(env.clone()))
    } else {
        std::ptr::null_mut()
    }
}

/// Copies the 6-neighbor adjacency table (200 hexes x 6 neighbors = 1200 bytes) into `out_adj`.
/// Returns the number of bytes written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_map_adjacency(
    env: *mut GaiaEnv,
    out_adj: *mut u8,
    max_len: u32,
) -> u32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if out_adj.is_null() {
            return 0;
        }
        let total = 200 * 6;
        let copy_len = total.min(max_len as usize);
        let slice = unsafe { slice::from_raw_parts_mut(out_adj, copy_len) };
        for i in 0..200 {
            for d in 0..6 {
                let idx = i * 6 + d;
                if idx < copy_len {
                    slice[idx] = env.map.adjacency[i][d];
                }
            }
        }
        copy_len as u32
    } else {
        0
    }
}





use std::ffi::CString;
use std::os::raw::c_char;

#[derive(serde::Serialize)]
pub struct GameStateJson<'a> {
    pub round: u8,
    pub current_player: usize,
    pub terminated: bool,
    pub players: &'a [crate::player::PlayerData],
    pub map: &'a crate::map::Map,
    pub research_level_5_claimed: [Option<u8>; 6],
    pub claimed_board_actions: [Option<u8>; 10],
    pub round_scoring_tiles: [crate::rules::ScoringTile; 6],
    pub final_scoring_tiles: [crate::rules::FinalTile; 2],
    pub pass_order: &'a [u8],
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_state_json(env: *mut GaiaEnv) -> *mut c_char {
    if let Some(env) = unsafe { env.as_ref() } {
        let summary = GameStateJson {
            round: env.round,
            current_player: env.current_player,
            terminated: env.terminated,
            players: &env.players,
            map: &env.map,
            research_level_5_claimed: env.research_level_5_claimed,
            claimed_board_actions: env.claimed_board_actions,
            round_scoring_tiles: env.round_scoring_tiles,
            final_scoring_tiles: env.final_scoring_tiles,
            pass_order: &env.pass_order,
        };
        if let Ok(json) = serde_json::to_string(&summary) {
            if let Ok(c_str) = CString::new(json) {
                return c_str.into_raw();
            }
        }
    }
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_action_name(env: *mut GaiaEnv, action_idx: usize) -> *mut c_char {
    if let Some(env) = unsafe { env.as_ref() } {
        let name = crate::action_space::get_action_name(action_idx, &env.map);
        if let Ok(c_str) = CString::new(name) {
            return c_str.into_raw();
        }
    }
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_map_json(env: *mut GaiaEnv) -> *mut c_char {
    if let Some(env) = unsafe { env.as_ref() } {
        if let Ok(json) = serde_json::to_string(&env.map) {
            return CString::new(json).unwrap().into_raw();
        }
    }
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_current_player(env: *mut GaiaEnv, seat: u32) {
    if let Some(env) = unsafe { env.as_mut() } {
        if (seat as usize) < env.players.len() {
            env.current_player = seat as usize;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_round(env: *mut GaiaEnv, round: u8) {
    if let Some(env) = unsafe { env.as_mut() } {
        env.round = round.clamp(1, 6);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_player_resources(
    env: *mut GaiaEnv,
    seat: u32,
    credits: i16,
    ore: i16,
    knowledge: i16,
    qic: i16,
    vp: i16,
    p1: u8,
    p2: u8,
    p3: u8,
) {
    if let Some(env) = unsafe { env.as_mut() } {
        if let Some(p) = env.players.get_mut(seat as usize) {
            p.credits = credits.clamp(0, 30);
            p.ore = ore.clamp(0, 15);
            p.knowledge = knowledge.clamp(0, 15);
            p.qic = qic.clamp(0, 15);
            p.victory_points = vp;
            p.power.area1 = p1;
            p.power.area2 = p2;
            p.power.area3 = p3;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_player_research(
    env: *mut GaiaEnv,
    seat: u32,
    field_idx: u32,
    level: u8,
) {
    if let Some(env) = unsafe { env.as_mut() } {
        if let Some(p) = env.players.get_mut(seat as usize) {
            if (field_idx as usize) < 6 {
                p.research[field_idx as usize] = level.min(5);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_hex_building(
    env: *mut GaiaEnv,
    q: i16,
    r: i16,
    s: i16,
    building_type: u8,
    player_seat: u8,
) {
    if let Some(env) = unsafe { env.as_mut() } {
        if let Some(idx) = env.map.coords[..env.map.count].iter().position(|c| c.q == q && c.r == r && c.s == s) {
            let hex = &mut env.map.hexes[idx];
            if building_type == 0 {
                hex.building = None;
                hex.player = None;
            } else {
                let bldg = match building_type {
                    1 => crate::rules::Building::Mine,
                    2 => crate::rules::Building::TradingStation,
                    3 => crate::rules::Building::ResearchLab,
                    4 => crate::rules::Building::PlanetaryInstitute,
                    5 => crate::rules::Building::Academy1,
                    6 => crate::rules::Building::Academy2,
                    7 => crate::rules::Building::GaiaFormer,
                    _ => crate::rules::Building::SpaceStation,
                };
                hex.building = Some(bldg);
                hex.player = Some(player_seat.min(3));
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_set_hex_planet(
    env: *mut GaiaEnv,
    q: i16,
    r: i16,
    s: i16,
    planet_id: u8,
) {
    if let Some(env) = unsafe { env.as_mut() } {
        if let Some(idx) = env.map.coords[..env.map.count].iter().position(|c| c.q == q && c.r == r && c.s == s) {
            let planet = match planet_id {
                0 => crate::rules::Planet::Empty,
                1 => crate::rules::Planet::Terra,
                2 => crate::rules::Planet::Desert,
                3 => crate::rules::Planet::Swamp,
                4 => crate::rules::Planet::Oxide,
                5 => crate::rules::Planet::Volcanic,
                6 => crate::rules::Planet::Titanium,
                7 => crate::rules::Planet::Ice,
                8 => crate::rules::Planet::Gaia,
                9 => crate::rules::Planet::Transdim,
                10 => crate::rules::Planet::Asteroid,
                11 => crate::rules::Planet::Protoplanet,
                _ => crate::rules::Planet::Lost,
            };
            env.map.hexes[idx].planet = planet;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_free_string(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe { let _ = CString::from_raw(s); }
}

/// Loads a map configuration from a JSON string and updates the environment's map.
/// Returns true if successful, false otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_load_map_json(env: *mut GaiaEnv, json_str: *const c_char) -> bool {
    if env.is_null() || json_str.is_null() {
        return false;
    }
    let c_str = unsafe { std::ffi::CStr::from_ptr(json_str) };
    let Ok(str_slice) = c_str.to_str() else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<crate::map::MapConfiguration>(str_slice) else {
        return false;
    };
    if let Some(env) = unsafe { env.as_mut() } {
        env.map.load_configuration(&config);
        true
    } else {
        false
    }
}

/// Returns the canonical 0-based spatial hex index for coordinate (q, r, s), or -1 if not on map.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiapi_get_hex_index(env: *mut GaiaEnv, q: i16, r: i16, s: i16) -> i32 {
    if let Some(env) = unsafe { env.as_ref() } {
        if let Some(idx) = env.map.coords[..env.map.count].iter().position(|c| c.q == q && c.r == r && c.s == s) {
            idx as i32
        } else {
            -1
        }
    } else {
        -1
    }
}

