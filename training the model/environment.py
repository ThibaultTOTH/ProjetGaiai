"""Gaia Project Environment Bridge for Reinforcement Learning.

Provides:
1. NativeGaiaEnv: Direct in-process C-ABI ctypes bridge to the compiled Rust gaiapi.dll.
2. RestGaiaEnv: Connects to the compiled Rust gaiapi Axum server (http://127.0.0.1:3000).
3. make_gaia_env: Automatic cascade factory: Native DLL -> REST.
"""

import copy
import ctypes
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests

_RUST_MISSING_WARNED = False


def warn_rust_not_detected(detail: Optional[str] = None) -> None:
    """Displays a prominent warning banner when the compiled Rust gaiapi library is missing."""
    global _RUST_MISSING_WARNED
    if _RUST_MISSING_WARNED or os.environ.get("GAIAPI_NO_RUST_WARN", "0") == "1":
        return
    _RUST_MISSING_WARNED = True

    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    banner_width = 80
    border = "!" * banner_width
    msg = f"""
{border}
  ⚠️   ATTENTION CRITIQUE : MOTEUR RUST NON COMPILÉ / NON DÉTECTÉ   ⚠️
{border}
[GAIA RL ENGINE] La bibliothèque native compilée Rust est introuvable.
(Fichier attendu : libgaiapi.so sous Linux, gaiapi.dll sous Windows)

Emplacements scannés :
  • projet_gaiapi/target/release/libgaiapi.so (ou .dll)
  • training the model/libgaiapi.so (ou .dll)
  • Variables d'environnement GAIAPI_LIB / GAIAPI_DLL
{f"  • Détail : {detail}" if detail else ""}

⚡ LE MOTEUR RUST EST INDISPENSABLE POUR LES ALGORITHMES SOTA (AlphaZero / MuZero).

🛠️  POUR COMPILER SUR LINUX (Pop!_OS / Ubuntu) EN 1 CLIC :
  1. Ouvrez un terminal dans le dossier ProjetGaiai :
       cd ~/ProjetGaiai
  2. Lancez le script officiel :
       ./compile_linux.sh

  Si Rust / Cargo n'est pas encore installé sur votre système Linux :
       sudo apt update && sudo apt install -y cargo rustc
       (ou : curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)
       puis relancez : ./compile_linux.sh
{border}
"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    warnings.warn(
        "Moteur Rust gaiapi non détecté. Veuillez compiler la bibliothèque native (libgaiapi.so / gaiapi.dll).",
        category=RuntimeWarning,
        stacklevel=2,
    )


class GaiaEnvStepResult:
    def __init__(
        self,
        obs: np.ndarray,
        reward: float,
        done: bool,
        action_mask: np.ndarray,
        info: Dict[str, Any],
    ):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.action_mask = action_mask
        self.info = info


class NativeGaiaEnv:
    """Ultra-high performance C-ABI in-process bridge to compiled Rust gaiapi.dll."""

    ACTION_NAMES = []

    _dll_instance = None
    _dll_path = None

    @classmethod
    def _find_dll(cls) -> Optional[str]:
        if cls._dll_path and os.path.exists(cls._dll_path):
            try:
                ctypes.CDLL(cls._dll_path)
                return cls._dll_path
            except OSError:
                cls._dll_path = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        cwd = os.getcwd()

        # Strict platform-specific shared library extensions
        if sys.platform.startswith("win"):
            lib_names = ["gaiapi.dll", "libgaiapi.dll"]
        elif sys.platform.startswith("darwin"):
            lib_names = ["libgaiapi.dylib", "gaiapi.dylib"]
        else:  # Linux, Unix, BSD (NEVER load a .dll on Linux!)
            lib_names = ["libgaiapi.so", "gaiapi.so"]

        candidates = [
            os.environ.get("GAIAPI_LIB"),
            os.environ.get("GAIAPI_DLL"),
        ]

        # Scan comprehensive list of standard locations
        search_dirs = [
            os.path.join(project_root, "projet_gaiapi", "target", "release"),
            os.path.join(cwd, "projet_gaiapi", "target", "release"),
            base_dir,
            os.path.join(project_root, "projet_gaiapi", "target", "debug"),
            os.path.join(project_root, "target", "release"),
            os.path.join(project_root, "target", "debug"),
            os.path.join(cwd, "target", "release"),
            cwd,
        ]

        for sdir in search_dirs:
            for name in lib_names:
                candidates.append(os.path.join(sdir, name))

        for c in candidates:
            if c and os.path.exists(c):
                try:
                    # Test if it can actually be loaded on this OS
                    test_dll = ctypes.CDLL(c)
                    cls._dll_path = os.path.abspath(c)
                    return cls._dll_path
                except OSError:
                    # e.g. invalid ELF header if trying to load Windows DLL on Linux
                    continue

        # If still not found, attempt automatic build via Cargo if available on the system
        import shutil
        import subprocess

        cargo_candidates = [
            shutil.which("cargo"),
            os.path.expanduser("~/.cargo/bin/cargo"),
            os.path.expanduser("~/.cargo/bin/cargo.exe"),
            "/usr/bin/cargo",
            "/usr/local/bin/cargo",
            "/opt/cargo/bin/cargo",
        ]
        cargo_bin = None
        for cand in cargo_candidates:
            if cand and os.path.exists(cand) and (sys.platform.startswith("win") or os.access(cand, os.X_OK)):
                cargo_bin = cand
                break

        rust_crate_dir = os.path.join(project_root, "projet_gaiapi")
        cargo_toml = os.path.join(rust_crate_dir, "Cargo.toml")
        if not os.path.exists(cargo_toml) and os.path.exists(os.path.join(cwd, "projet_gaiapi", "Cargo.toml")):
            rust_crate_dir = os.path.join(cwd, "projet_gaiapi")
            cargo_toml = os.path.join(rust_crate_dir, "Cargo.toml")

        if cargo_bin and os.path.exists(cargo_toml):
            print(f"⚙️  [RUST ENGINE] Cargo détecté ({cargo_bin}) : compilation automatique du moteur Rust (cargo build --release)...")
            try:
                cargo_dir = os.path.dirname(cargo_bin)
                build_env = os.environ.copy()
                if cargo_dir:
                    build_env["PATH"] = f"{cargo_dir}{os.pathsep}{build_env.get('PATH', '')}"

                res = subprocess.run(
                    [cargo_bin, "build", "--release"],
                    cwd=rust_crate_dir,
                    env=build_env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    print("✓ [RUST ENGINE] Compilation Rust terminée avec succès !")
                    target_release = os.path.join(rust_crate_dir, "target", "release")
                    for name in lib_names:
                        candidate = os.path.join(target_release, name)
                        if os.path.exists(candidate):
                            try:
                                test_dll = ctypes.CDLL(candidate)
                                cls._dll_path = os.path.abspath(candidate)
                                try:
                                    shutil.copy2(cls._dll_path, os.path.join(base_dir, name))
                                except Exception:
                                    pass
                                return cls._dll_path
                            except OSError:
                                continue
                else:
                    err_msg = res.stderr.strip() or res.stdout.strip()
                    print(f"⚠️  [RUST ENGINE] Erreur lors de la compilation Cargo :\n{err_msg}")
            except Exception as e:
                print(f"⚠️  [RUST ENGINE] Échec de l'exécution Cargo : {e}")

        return None

    @classmethod
    def is_available(cls) -> bool:
        return cls._find_dll() is not None

    @classmethod
    def _get_dll(cls):
        if cls._dll_instance is None:
            dll_path = cls._find_dll()
            if not dll_path:
                warn_rust_not_detected(detail="Tentative d'accès direct à la DLL Rust sans binaire disponible.")
                raise RuntimeError("gaiapi library (gaiapi.dll / libgaiapi.so) was not found.")
            dll = ctypes.CDLL(dll_path)

            dll.gaiapi_create.restype = ctypes.c_void_p
            dll.gaiapi_create.argtypes = [ctypes.c_uint64, ctypes.c_uint32]

            dll.gaiapi_destroy.argtypes = [ctypes.c_void_p]

            dll.gaiapi_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

            dll.gaiapi_set_player_faction.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]

            dll.gaiapi_get_obs_dim.argtypes = [ctypes.c_void_p]
            dll.gaiapi_get_obs_dim.restype = ctypes.c_uint32

            dll.gaiapi_get_action_dim.restype = ctypes.c_uint32

            dll.gaiapi_get_observation.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint32,
            ]
            dll.gaiapi_get_observation.restype = ctypes.c_uint32

            if hasattr(dll, "gaiapi_get_observation_egocentric"):
                dll.gaiapi_get_observation_egocentric.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint32,
                ]
                dll.gaiapi_get_observation_egocentric.restype = ctypes.c_uint32

            dll.gaiapi_get_action_mask.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_uint32,
            ]
            dll.gaiapi_get_action_mask.restype = ctypes.c_uint32

            dll.gaiapi_step.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_bool),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            dll.gaiapi_step.restype = ctypes.c_bool

            dll.gaiapi_get_player_vp.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            dll.gaiapi_get_player_vp.restype = ctypes.c_float

            dll.gaiapi_clone.argtypes = [ctypes.c_void_p]
            dll.gaiapi_clone.restype = ctypes.c_void_p

            dll.gaiapi_get_map_adjacency.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_uint32,
            ]
            dll.gaiapi_get_map_adjacency.restype = ctypes.c_uint32

            if hasattr(dll, "gaiapi_step_target"):
                dll.gaiapi_step_target.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_bool),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                dll.gaiapi_step_target.restype = ctypes.c_bool

            cls._dll_instance = dll
        return cls._dll_instance

    def __init__(self, players: Any = 4, max_rounds: int = 6, seed: int = 42, egocentric: bool = True):
        if hasattr(players, "model") and hasattr(players.model, "num_players"):
            players = players.model.num_players
        elif isinstance(players, dict) and "model" in players:
            players = players["model"].get("num_players", 4)
        elif not isinstance(players, int):
            try:
                players = int(players)
            except (TypeError, ValueError):
                players = 4

        self.num_players = int(players)
        self.max_rounds = int(max_rounds)
        self.egocentric = bool(egocentric)
        self.dll = self._get_dll()
        self.env_ptr = self.dll.gaiapi_create(ctypes.c_uint64(int(seed)), ctypes.c_uint32(int(self.num_players)))
        if not self.env_ptr:
            raise RuntimeError("Failed to allocate native GaiaEnv.")
        self.obs_dim = int(self.dll.gaiapi_get_obs_dim(self.env_ptr))
        self.action_dim = int(self.dll.gaiapi_get_action_dim())

        # Pre-allocated ctypes buffers
        self._obs_buf = (ctypes.c_float * self.obs_dim)()
        self._mask_buf = (ctypes.c_uint8 * self.action_dim)()
        self._r_buf = ctypes.c_float(0.0)
        self._d_buf = ctypes.c_bool(False)
        self._round_buf = ctypes.c_uint32(1)
        self._cp_buf = ctypes.c_uint32(0)

        self.round = 1
        self.current_player = 0
        self.terminated = False
        self._faction_ids = [float(i) for i in range(players)]
        self.reset(seed)

    def __del__(self):
        if hasattr(self, "env_ptr") and self.env_ptr and hasattr(self, "dll"):
            try:
                self.dll.gaiapi_destroy(self.env_ptr)
                self.env_ptr = None
            except Exception:
                pass

    def set_player_faction(self, seat: int, faction_id: int):
        self._faction_ids[seat] = float(faction_id)
        self.dll.gaiapi_set_player_faction(self.env_ptr, seat, faction_id)

    def reset(self, seed: Optional[int] = None, egocentric: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        actual_seed = seed if seed is not None else int(np.random.randint(0, 1000000))
        self.dll.gaiapi_reset(self.env_ptr, actual_seed)
        self.round = 1
        self.current_player = 0
        self.terminated = False
        return self._get_obs(egocentric), self.get_action_mask()

    def _get_obs(self, egocentric: Optional[bool] = None) -> np.ndarray:
        use_ego = self.egocentric if egocentric is None else egocentric
        if use_ego and hasattr(self.dll, "gaiapi_get_observation_egocentric"):
            self.dll.gaiapi_get_observation_egocentric(self.env_ptr, self._obs_buf, self.obs_dim)
        else:
            self.dll.gaiapi_get_observation(self.env_ptr, self._obs_buf, self.obs_dim)
        return np.array(self._obs_buf, dtype=np.float32)

    def get_observation(self, egocentric: Optional[bool] = None) -> np.ndarray:
        """Public helper to retrieve the current observation tensor (standard or egocentric)."""
        return self._get_obs(egocentric)

    def get_action_mask(self) -> np.ndarray:
        self.dll.gaiapi_get_action_mask(self.env_ptr, self._mask_buf, self.action_dim)
        return np.array(self._mask_buf, dtype=bool)

    @property
    def players_state(self) -> List[Dict[str, Any]]:
        states = []
        for seat in range(self.num_players):
            vp = float(self.dll.gaiapi_get_player_vp(self.env_ptr, seat))
            states.append({
                "faction": self._faction_ids[seat] if seat < len(self._faction_ids) else float(seat),
                "vp": vp,
            })
        return states

    def step(self, action: int, target: Optional[int] = None, egocentric: Optional[bool] = None) -> GaiaEnvStepResult:
        if self.terminated:
            return GaiaEnvStepResult(
                obs=self._get_obs(egocentric),
                reward=0.0,
                done=True,
                action_mask=self.get_action_mask(),
                info={"round": self.round, "player_vp": [p["vp"] for p in self.players_state]},
            )

        flat_action = action
        if target is not None:
            if action == 3:  # BuildMine
                flat_action = target
            elif action == 4:  # StartGaiaProject
                flat_action = 200 + target
            elif action == 5:  # UpgradeTradingStation
                flat_action = 400 + target
            elif action == 6:  # UpgradeResearchLab
                flat_action = 600 + target
            elif action == 7:  # UpgradePlanetaryInstitute
                flat_action = 800 + target
            elif action == 8:  # UpgradeAcademy
                flat_action = 1000 + target
            elif action == 9:  # FormFederation
                flat_action = 1400 + target
            elif action == 10:  # AdvanceResearch
                flat_action = 1406 + target
            elif action == 15:  # Pass
                flat_action = 1412 + target

        actor = getattr(self, "current_player", 0)
        prev_vp = float(self.dll.gaiapi_get_player_vp(self.env_ptr, actor)) if hasattr(self.dll, "gaiapi_get_player_vp") else 0.0

        if target is not None and hasattr(self.dll, "gaiapi_step_target"):
            success = self.dll.gaiapi_step_target(
                self.env_ptr,
                int(action),
                int(target),
                ctypes.byref(self._r_buf),
                ctypes.byref(self._d_buf),
                ctypes.byref(self._round_buf),
                ctypes.byref(self._cp_buf),
            )
        else:
            success = self.dll.gaiapi_step(
                self.env_ptr,
                int(flat_action),
                ctypes.byref(self._r_buf),
                ctypes.byref(self._d_buf),
                ctypes.byref(self._round_buf),
                ctypes.byref(self._cp_buf),
            )
        if not success:
            return GaiaEnvStepResult(
                obs=self._get_obs(egocentric),
                reward=-5.0,
                done=self.terminated,
                action_mask=self.get_action_mask(),
                info={"error": f"Illegal action {flat_action}"},
            )

        self.round = int(self._round_buf.value)
        self.current_player = int(self._cp_buf.value)
        self.terminated = bool(self._d_buf.value) or (self.round > self.max_rounds)

        new_vp = float(self.dll.gaiapi_get_player_vp(self.env_ptr, actor)) if hasattr(self.dll, "gaiapi_get_player_vp") else 0.0
        raw_r = float(self._r_buf.value)
        delta_vp = max(0.0, new_vp - prev_vp)
        # Defend against cumulative VP returned by legacy binary
        step_r = delta_vp if (abs(raw_r - new_vp) < 1e-4 and new_vp > delta_vp) else raw_r

        return GaiaEnvStepResult(
            obs=self._get_obs(egocentric),
            reward=step_r,
            done=self.terminated,
            action_mask=self.get_action_mask(),
            info={
                "round": self.round,
                "current_player": self.current_player,
                "player_vp": [p["vp"] for p in self.players_state],
            },
        )

    def get_micro_candidates(self, action: int) -> List[int]:
        """Returns valid target candidate indices for micro-action dispatch."""
        actor = getattr(self, "current_player", 0)
        mask = self.get_action_mask()
        if action == 3:  # BuildMine: legal mine hexes according to action mask
            legal_mines = [k for k in range(200) if mask[k]]
            if legal_mines:
                return legal_mines[:8]
            obs = self._get_obs()
            base_map = 88 + 388
            candidates = []
            for k in range(200):
                pl = obs[base_map + k * 9 + 0]
                bld = obs[base_map + k * 9 + 1]
                if pl > 0.0 and bld == 0.0:
                    candidates.append(k)
            return candidates[:8] if candidates else [0]
        elif action in (5, 6, 7, 8):  # Upgrades
            target_bld = 1.0 if action == 5 else (2.0 if action in (6, 7) else 3.0)
            obs = self._get_obs()
            base_map = 88 + 388
            candidates = []
            expected_p = (float(actor) + 1.0) / 4.0
            expected_bld = (target_bld + 1.0) / 9.0
            for k in range(200):
                p_owner = obs[base_map + k * 9 + 2]
                bld = obs[base_map + k * 9 + 1]
                if abs(p_owner - expected_p) < 1e-3 and abs(bld - expected_bld) < 1e-3:
                    candidates.append(k)
            return candidates if candidates else [0]
        elif action == 9:  # FormFederation
            return [0, 1, 2, 3, 4, 5, 6]
        elif action == 10:  # AdvanceResearch
            return [0, 1, 2, 3, 4, 5]
        elif action == 11:  # ClaimTechTile
            return [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif action == 13:  # ExploreSpaceship
            return [0, 1, 2, 3]
        elif action == 15:  # Pass
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return [0]

    def get_initial_mine_candidates(self, seat: int) -> List[int]:
        obs = self._get_obs()
        base_map = 88 + 388
        candidates = []
        for k in range(200):
            pl = obs[base_map + k * 9 + 0]
            bld = obs[base_map + k * 9 + 1]
            if pl > 0.0 and bld == 0.0:
                candidates.append(k)
        return candidates[:8] if candidates else [0]

    def get_available_boosters(self) -> List[int]:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def clone(self) -> "NativeGaiaEnv":
        """Zero-overhead clone of the native C-ABI simulation kernel for MCTS lookahead."""
        cloned_ptr = self.dll.gaiapi_clone(self.env_ptr)
        if not cloned_ptr:
            raise RuntimeError("Failed to clone native GaiaEnv.")
        new_env = NativeGaiaEnv.__new__(NativeGaiaEnv)
        new_env.num_players = self.num_players
        new_env.max_rounds = self.max_rounds
        new_env.dll = self.dll
        new_env.env_ptr = cloned_ptr
        new_env.obs_dim = self.obs_dim
        new_env.action_dim = self.action_dim
        new_env.round = self.round
        new_env.current_player = self.current_player
        new_env.terminated = self.terminated
        new_env.egocentric = getattr(self, "egocentric", True)
        new_env._faction_ids = list(self._faction_ids)
        new_env._obs_buf = (ctypes.c_float * self.obs_dim)()
        new_env._mask_buf = (ctypes.c_uint8 * self.action_dim)()
        new_env._r_buf = ctypes.c_float(0.0)
        new_env._d_buf = ctypes.c_bool(False)
        new_env._round_buf = ctypes.c_uint32(0)
        new_env._cp_buf = ctypes.c_uint32(0)
        return new_env

    def get_map_adjacency(self) -> np.ndarray:
        """Returns the (200, 6) neighbor index table (255 for boundary/no neighbor)."""
        adj_buf = (ctypes.c_uint8 * 1200)()
        n = self.dll.gaiapi_get_map_adjacency(self.env_ptr, adj_buf, 1200)
        if n > 0:
            arr = np.frombuffer(adj_buf, dtype=np.uint8, count=1200).reshape((200, 6))
            return arr.copy()
        return build_canonical_hex_adjacency()


def build_canonical_hex_adjacency() -> np.ndarray:
    """Constructs a canonical (200, 6) neighbor adjacency matrix on a hexagonal grid."""
    coords = [(0, 0, 0)]
    visited = {(0, 0, 0): 0}
    directions = [
        (1, -1, 0), (1, 0, -1), (0, 1, -1),
        (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
    ]

    ring = 1
    while len(coords) < 200:
        q, r, s = -ring, 0, ring
        for dq, dr, ds in directions:
            for _ in range(ring):
                if len(coords) >= 200:
                    break
                coord = (q, r, s)
                if coord not in visited:
                    visited[coord] = len(coords)
                    coords.append(coord)
                q += dq
                r += dr
                s += ds
            if len(coords) >= 200:
                break
        ring += 1

    adj = np.full((200, 6), 255, dtype=np.uint8)
    for i, (q, r, s) in enumerate(coords):
        for d, (dq, dr, ds) in enumerate(directions):
            target = (q + dq, r + dr, s + ds)
            if target in visited:
                adj[i, d] = visited[target]
    return adj


class RestGaiaEnv:
    """Connects to the compiled Rust gaiapi Axum REST server (127.0.0.1:3000)."""

    def __init__(self, base_url: str = "http://127.0.0.1:3000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.env_id: Optional[int] = None
        self.obs_dim = 2476
        self.action_dim = 3130

    def is_available(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.env_id is None:
            res = self.session.post(
                f"{self.base_url}/v1/environments",
                json={"config": {"players": 2, "max_rounds": 6}},
                timeout=2.0,
            ).json()
            self.env_id = res["id"]
            obs_data = res["observation"]
        else:
            res = self.session.post(
                f"{self.base_url}/v1/environments/{self.env_id}/reset",
                timeout=2.0,
            ).json()
            obs_data = res

        obs = np.array(obs_data["values"], dtype=np.float32)
        mask = np.array(obs_data["action_mask"], dtype=bool)
        return obs, mask

    def step(self, action: int, target: Optional[int] = None) -> GaiaEnvStepResult:
        action_names = []
        action_str = f"Action {action}"

        resp = self.session.post(
            f"{self.base_url}/v1/environments/{self.env_id}/step",
            json={"action": action_str},
            timeout=2.0,
        ).json()

        obs = np.array(resp["observation"]["values"], dtype=np.float32)
        mask = np.array(resp["observation"]["action_mask"], dtype=bool)
        reward = (
            float(resp["rewards"][resp["current_player"]])
            if resp.get("rewards")
            else 0.0
        )
        done = bool(resp.get("terminated", False))

        return GaiaEnvStepResult(
            obs=obs,
            reward=reward,
            done=done,
            action_mask=mask,
            info={"round": resp.get("round", 1)},
        )


def format_flat_action(action_id: int) -> str:
    """Decodes a flat action index (0..3129) into human-readable text."""
    if 0 <= action_id < 200:
        return f"BuildMine(Hex {action_id})"
    elif 200 <= action_id < 400:
        return f"StartGaiaProject(Hex {action_id - 200})"
    elif 400 <= action_id < 1400:
        b_idx = (action_id - 400) // 200
        h_idx = (action_id - 400) % 200
        b_names = ["TradingStation", "ResearchLab", "PlanetaryInst", "Academy", "LostFleetDeepSpace"]
        b_name = b_names[b_idx] if b_idx < len(b_names) else f"Bldg_{b_idx}"
        return f"Upgrade{b_name}(Hex {h_idx})"
    elif 1400 <= action_id < 1406:
        return f"FormFederation(Token {action_id - 1400 + 1})"
    elif 1406 <= action_id < 1412:
        tracks = ["Terraforming", "Navigation", "ArtificialIntel", "Gaiaforming", "Economy", "Science"]
        t_idx = action_id - 1406
        return f"AdvanceResearch({tracks[t_idx]})"
    elif 1412 <= action_id < 1422:
        return f"Pass(Booster {action_id - 1412 + 1})"
    elif 1422 <= action_id < 1424:
        return "AcceptLeechPower" if action_id == 1423 else "DeclineLeechPower"
    elif 1424 <= action_id < 1434:
        return f"BoardAction(Slot {action_id - 1424 + 1})"
    elif 1434 <= action_id < 1444:
        return f"SpecialAction(Slot {action_id - 1434 + 1})"
    elif 1444 <= action_id < 1498:
        tile_id = (action_id - 1444) // 6
        track_id = (action_id - 1444) % 6
        return f"ClaimTechTile(Tile {tile_id + 1}, Track {track_id + 1})"
    elif 1498 <= action_id < 2308:
        adv_idx = (action_id - 1498) // (9 * 6)
        return f"ClaimAdvTechTile(AdvTile {adv_idx + 1})"
    elif 2308 <= action_id < 3108:
        ship_idx = (action_id - 2308) // 200
        h_idx = (action_id - 2308) % 200
        return f"ExploreSpaceship(Ship {ship_idx + 1}, Hex {h_idx})"
    elif 3108 <= action_id < 3124:
        ship_idx = (action_id - 3108) // 4
        act_sub = (action_id - 3108) % 4
        return f"SpaceshipBoardAction(Ship {ship_idx + 1}, Act {act_sub + 1})"
    elif 3124 <= action_id < 3130:
        free_names = [
            "Power4->Ore1", "Power3->Credit1", "Power4->Knowledge1",
            "Power1->Credit1", "Qic1->Ore1", "Ore1->Credit1",
        ]
        sub = action_id - 3124
        return f"FreeAction({free_names[sub]})"
    return f"Action_{action_id}"


def make_gaia_env(
    players: Any = 4,
    prefer_native: bool = True,
    use_rest: bool = False,
    rest_url: str = "http://127.0.0.1:3000",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Factory creating NativeGaiaEnv (DLL/.so) > RestGaiaEnv."""
    mode = kwargs.pop("mode", None)
    config = kwargs.pop("config", None)
    if hasattr(players, "model") and hasattr(players.model, "num_players"):
        players = players.model.num_players
    elif isinstance(players, dict) and "model" in players:
        players = players["model"].get("num_players", 4)
    elif not isinstance(players, int):
        try:
            players = int(players)
        except (TypeError, ValueError):
            players = 4

    actual_seed = seed if seed is not None else 42

    if mode == "rest":
        use_rest = True
        prefer_native = False
    if prefer_native:
        if NativeGaiaEnv.is_available():
            try:
                return NativeGaiaEnv(players=players, seed=actual_seed, **kwargs)
            except Exception as e:
                print(f"❌ [make_gaia_env] Erreur lors de l'instanciation de NativeGaiaEnv: {e}")
                raise e
        else:
            warn_rust_not_detected(detail="make_gaia_env(prefer_native=True) n'a trouvé aucune bibliothèque native compilée.")

    if use_rest:
        rest_env = RestGaiaEnv(rest_url)
        if rest_env.is_available():
            return rest_env

    raise RuntimeError("Native Gaia DLL / .so not found. Please compile the Rust engine with ./compile_linux.sh.")


# Automatic check on import: immediately warn developer if Rust engine is missing
if not NativeGaiaEnv.is_available():
    warn_rust_not_detected(detail="Vérification automatique au chargement du module environment.py.")
