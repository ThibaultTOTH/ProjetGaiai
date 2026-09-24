"""
Projet Gaïa AI Lab - Station Complète d'Analyse, de Jeu & Éditeur de Scénario Arbitraire (Sandbox)
Branché sur le moteur Rust natif (3129 actions).
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ctypes
import math
import json

# ---------------------------------------------------------------------------
# Importation Qt Universelle
# ---------------------------------------------------------------------------
QT_LIB = None
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
        QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
        QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel,
        QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
        QGroupBox, QTabWidget, QTextEdit, QSplitter, QFrame, QGridLayout,
        QScrollArea, QProgressBar, QMessageBox, QRadioButton, QButtonGroup
    )
    from PyQt5.QtGui import QPolygonF, QBrush, QPen, QColor, QFont, QPainter
    from PyQt5.QtCore import Qt, QPointF, QRectF
    QT_LIB = "PyQt5"
except ImportError:
    try:
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
            QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
            QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel,
            QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
            QGroupBox, QTabWidget, QTextEdit, QSplitter, QFrame, QGridLayout,
            QScrollArea, QProgressBar, QMessageBox, QRadioButton, QButtonGroup
        )
        from PySide6.QtGui import QPolygonF, QBrush, QPen, QColor, QFont, QPainter
        from PySide6.QtCore import Qt, QPointF, QRectF
        QT_LIB = "PySide6"
    except ImportError:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
            QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem,
            QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel,
            QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
            QGroupBox, QTabWidget, QTextEdit, QSplitter, QFrame, QGridLayout,
            QScrollArea, QProgressBar, QMessageBox, QRadioButton, QButtonGroup
        )
        from PyQt6.QtGui import QPolygonF, QBrush, QPen, QColor, QFont, QPainter
        from PyQt6.QtCore import Qt, QPointF, QRectF
        QT_LIB = "PyQt6"

print(f"[GaiaLab] Initialisé avec: {QT_LIB}")


# ---------------------------------------------------------------------------
# Recherche Dynamique Cross-Platform de la Bibliothèque Rust gaiapi
# ---------------------------------------------------------------------------
def find_gaiapi_library():
    """Localise gaiapi.dll (Windows), libgaiapi.so (Linux / Pop!_OS) ou libgaiapi.dylib (macOS)."""
    lib_names = ["gaiapi.dll", "libgaiapi.so", "libgaiapi.dylib", "gaiapi.so"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("GAIAPI_LIB"),
        os.environ.get("GAIAPI_DLL"),
    ]
    for name in lib_names:
        candidates.extend([
            os.path.join(base_dir, "..", "projet_gaiapi", "target", "release", name),
            os.path.join(base_dir, "..", "projet_gaiapi", "target", "debug", name),
            os.path.join(base_dir, name),
            os.path.join(base_dir, "..", "training the model", name),
        ])
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


class RustEngineBridge:
    def __init__(self):
        dll_path = find_gaiapi_library()
        if not dll_path:
            raise FileNotFoundError(
                "Bibliothèque native gaiapi (gaiapi.dll / libgaiapi.so) introuvable.\n"
                "Veuillez exécuter 'cargo build --release' dans le dossier projet_gaiapi."
            )
        print(f"[RustBridge] Chargement du moteur natif : {dll_path}")
        self.dll = ctypes.CDLL(dll_path)
        
        # Moteur principal
        self.dll.gaiapi_create.restype = ctypes.c_void_p
        self.dll.gaiapi_create.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
        
        self.dll.gaiapi_reset.restype = None
        self.dll.gaiapi_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        self.dll.gaiapi_destroy.restype = None
        self.dll.gaiapi_destroy.argtypes = [ctypes.c_void_p]
        
        self.dll.gaiapi_step.restype = ctypes.c_bool
        self.dll.gaiapi_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        
        self.dll.gaiapi_get_obs_dim.restype = ctypes.c_uint32
        self.dll.gaiapi_get_obs_dim.argtypes = [ctypes.c_void_p]

        self.dll.gaiapi_get_action_dim.restype = ctypes.c_uint32

        self.dll.gaiapi_get_observation.restype = ctypes.c_uint32
        self.dll.gaiapi_get_observation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
        ]

        self.dll.gaiapi_get_action_mask.restype = ctypes.c_uint32
        self.dll.gaiapi_get_action_mask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
        ]
        
        self.dll.gaiapi_get_state_json.restype = ctypes.c_void_p
        self.dll.gaiapi_get_state_json.argtypes = [ctypes.c_void_p]
        
        self.dll.gaiapi_get_action_name.restype = ctypes.c_void_p
        self.dll.gaiapi_get_action_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        
        self.dll.gaiapi_free_string.restype = None
        self.dll.gaiapi_free_string.argtypes = [ctypes.c_void_p]

        # Fonctions Sandbox (Édition de situation donnée)
        self.dll.gaiapi_set_current_player.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.dll.gaiapi_set_round.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        self.dll.gaiapi_set_player_resources.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int16, ctypes.c_int16,
            ctypes.c_int16, ctypes.c_int16, ctypes.c_int16, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8
        ]
        self.dll.gaiapi_set_player_research.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint8]
        self.dll.gaiapi_set_hex_building.argtypes = [ctypes.c_void_p, ctypes.c_int16, ctypes.c_int16, ctypes.c_int16, ctypes.c_uint8, ctypes.c_uint8]
        self.dll.gaiapi_set_hex_planet.argtypes = [ctypes.c_void_p, ctypes.c_int16, ctypes.c_int16, ctypes.c_int16, ctypes.c_uint8]

    def create_env(self, seed=42, players=4):
        return self.dll.gaiapi_create(seed, players)

    def destroy_env(self, env_ptr):
        if env_ptr:
            try:
                self.dll.gaiapi_destroy(ctypes.c_void_p(env_ptr))
            except Exception:
                pass

    def reset_env(self, env_ptr, seed):
        self.dll.gaiapi_reset(ctypes.c_void_p(env_ptr), seed)

    def step(self, env_ptr, action_idx):
        r = ctypes.c_float(0.0)
        d = ctypes.c_bool(False)
        rnd = ctypes.c_uint32(1)
        cp = ctypes.c_uint32(0)
        self.dll.gaiapi_step(
            env_ptr,
            ctypes.c_uint32(action_idx),
            ctypes.byref(r),
            ctypes.byref(d),
            ctypes.byref(rnd),
            ctypes.byref(cp)
        )
        return r.value

    def get_state(self, env_ptr):
        ptr = self.dll.gaiapi_get_state_json(env_ptr)
        if not ptr: return {}
        try:
            c_str = ctypes.cast(ptr, ctypes.c_char_p).value
            return json.loads(c_str.decode('utf-8'))
        finally:
            self.dll.gaiapi_free_string(ptr)

    def get_action_name(self, env_ptr, action_idx):
        ptr = self.dll.gaiapi_get_action_name(env_ptr, action_idx)
        if not ptr: return f"Action #{action_idx}"
        try:
            c_str = ctypes.cast(ptr, ctypes.c_char_p).value
            return c_str.decode('utf-8', errors='replace')
        finally:
            self.dll.gaiapi_free_string(ptr)

    def get_observation(self, env_ptr):
        dim = int(self.dll.gaiapi_get_obs_dim(env_ptr))
        buf = (ctypes.c_float * dim)()
        count = self.dll.gaiapi_get_observation(env_ptr, buf, dim)
        import numpy as np
        return np.array(buf[:count], dtype=np.float32)

    def get_action_mask(self, env_ptr):
        dim = int(self.dll.gaiapi_get_action_dim())
        buffer = (ctypes.c_uint8 * dim)()
        count = self.dll.gaiapi_get_action_mask(env_ptr, buffer, dim)
        import numpy as np
        return np.array(buffer[:count], dtype=bool)

    def get_legal_actions(self, env_ptr):
        mask = self.get_action_mask(env_ptr)
        return [i for i, val in enumerate(mask) if val]

    # Méthodes d'édition de situation arbitraire
    def set_current_player(self, env_ptr, seat):
        self.dll.gaiapi_set_current_player(env_ptr, seat)

    def set_round(self, env_ptr, round_num):
        self.dll.gaiapi_set_round(env_ptr, round_num)

    def set_player_resources(self, env_ptr, seat, c, o, k, q, vp, p1, p2, p3):
        self.dll.gaiapi_set_player_resources(env_ptr, seat, c, o, k, q, vp, p1, p2, p3)

    def set_player_research(self, env_ptr, seat, field_idx, lvl):
        self.dll.gaiapi_set_player_research(env_ptr, seat, field_idx, lvl)

    def set_hex_building(self, env_ptr, q, r, s, building_id, player_seat):
        self.dll.gaiapi_set_hex_building(env_ptr, q, r, s, building_id, player_seat)

    def set_hex_planet(self, env_ptr, q, r, s, planet_id):
        self.dll.gaiapi_set_hex_planet(env_ptr, q, r, s, planet_id)


# ---------------------------------------------------------------------------
# Évaluateur Neuronal IA (DualGaiaAgent & ActionOptimizerNet)
# ---------------------------------------------------------------------------
class AIEvaluator:
    """Connecte le modèle neuronal PyTorch (Policy & Value Networks) directement au Lab."""

    def __init__(self, bridge: RustEngineBridge):
        self.bridge = bridge
        self.agent = None
        self.is_loaded = False
        
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F

            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training the model"))
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            
            from models import DualGaiaAgent
            from config import ModelConfig
            
            cfg = ModelConfig(obs_dim=2476, action_dim=3130)
            self.agent = DualGaiaAgent(config=cfg)
            self.agent.eval()
            
            # Recherche de checkpoints entraînés
            ckpt_paths = [
                os.path.join(model_dir, "checkpoints", "gaia_latest.pt"),
                os.path.join(model_dir, "checkpoints", "gaia_best_elo.pt"),
                os.path.join(model_dir, "checkpoints", "best_model.pt"),
                os.path.join(model_dir, "checkpoints", "checkpoint_latest.pt"),
                os.path.join(model_dir, "best_model.pt"),
            ]
            for p in ckpt_paths:
                if os.path.exists(p):
                    try:
                        data = torch.load(p, map_location="cpu", weights_only=False)
                        if isinstance(data, dict) and "model_state_dict" in data:
                            self.agent.load_state_dict(data["model_state_dict"], strict=False)
                        elif isinstance(data, dict):
                            self.agent.load_state_dict(data, strict=False)
                        print(f"[AIEvaluator] Modèle RL chargé : {p}")
                        break
                    except Exception as e:
                        print(f"[AIEvaluator] Avertissement checkpoint {p}: {e}")
            self.is_loaded = True
            print("[AIEvaluator] Modèle neuronal IA (obs: 2476, actions: 3130) prêt pour l'analyse en temps réel.")
        except Exception as e:
            print(f"[AIEvaluator] IA non chargée (mode heuristique actif) : {e}")

    def evaluate(self, env_ptr):
        """Évalue l'état actuel (2476 floats) et le masque d'actions (3130 bools).
        Retourne : {
            'predicted_vp': float,
            'top_actions': [(idx, name, prob_pct), ...],
            'best_action': int
        }
        """
        if not self.is_loaded or self.agent is None:
            return None
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F

            obs = self.bridge.get_observation(env_ptr)
            mask = self.bridge.get_action_mask(env_ptr)
            if len(obs) < 2476 or len(mask) < 3130:
                return None
            
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            mask_t = torch.from_numpy(mask.astype(bool)).unsqueeze(0)
            
            with torch.no_grad():
                act, log_prob, pred_val, probs = self.agent.act_and_evaluate(obs_t, mask_t)
            
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                return {"predicted_vp": float(pred_val), "top_actions": [], "best_action": None}
            
            probs_np = probs.cpu().numpy()
            legal_ranked = [(idx, float(probs_np[idx])) for idx in legal_indices]
            legal_ranked.sort(key=lambda x: x[1], reverse=True)
            
            top_actions = []
            for idx, prob in legal_ranked[:5]:
                name = self.bridge.get_action_name(env_ptr, idx)
                top_actions.append((idx, name, prob * 100.0))
            
            best_action = top_actions[0][0] if top_actions else None
            return {
                "predicted_vp": float(pred_val),
                "top_actions": top_actions,
                "best_action": best_action,
            }
        except Exception as e:
            print(f"[AIEvaluator] Erreur d'inférence IA : {e}")
            return None


# ---------------------------------------------------------------------------
# Couleurs et Styles Graphiques
# ---------------------------------------------------------------------------
PLANET_STYLES = {
    "terra": {"fill": "#0284c7", "border": "#38bdf8", "name": "Terra (Bleu)", "disk": "#0ea5e9"},
    "desert": {"fill": "#ca8a04", "border": "#fde047", "name": "Désert (Jaune)", "disk": "#eab308"},
    "swamp": {"fill": "#365314", "border": "#84cc16", "name": "Marécage (Brun)", "disk": "#4d7c0f"},
    "oxide": {"fill": "#991b1b", "border": "#f87171", "name": "Oxyde (Rouge)", "disk": "#dc2626"},
    "volcanic": {"fill": "#9a3412", "border": "#fb923c", "name": "Volcanique (Orange)", "disk": "#ea580c"},
    "titanium": {"fill": "#581c87", "border": "#c084fc", "name": "Titane (Violet)", "disk": "#7e22ce"},
    "ice": {"fill": "#64748b", "border": "#f8fafc", "name": "Glace (Blanc)", "disk": "#cbd5e1"},
    "gaia": {"fill": "#065f46", "border": "#34d399", "name": "Gaïa (Vert)", "disk": "#10b981"},
    "transdim": {"fill": "#312e81", "border": "#818cf8", "name": "Transdim (Violet)", "disk": "#4338ca"},
    "asteroid": {"fill": "#334155", "border": "#94a3b8", "name": "Astéroïde (Gris)", "disk": "#475569"},
    "protoplanet": {"fill": "#075985", "border": "#7dd3fc", "name": "Protoplanète", "disk": "#0284c7"},
    "lost": {"fill": "#1e293b", "border": "#475569", "name": "Flotte Perdue", "disk": "#334155"},
    "empty": {"fill": "#070b14", "border": "#1e293b", "name": "Espace", "disk": None}
}

PLAYER_COLORS = ["#38bdf8", "#f87171", "#facc15", "#4ade80"]
PLAYER_LABELS = ["J0 (Bleu)", "J1 (Rouge)", "J2 (Jaune)", "J3 (Vert)"]

BUILDING_TYPES = [
    (0, "Vide (Supprimer bâtiment)", "❌"),
    (1, "Mine", "⛏️"),
    (2, "Station Commerciale", "🏛️"),
    (3, "Laboratoire de Recherche", "🧪"),
    (4, "Institut Planétaire", "🏰"),
    (5, "Académie 1", "🎓"),
    (6, "Académie 2", "🎓"),
    (7, "Gaiaformer", "🔺")
]


# ---------------------------------------------------------------------------
# Composant Tuile Hexagonale
# ---------------------------------------------------------------------------
class HexTile(QGraphicsPolygonItem):
    def __init__(self, q, r, s, hex_data, on_click_callback, radius=32):
        super().__init__()
        self.q = q
        self.r = r
        self.s = s
        self.hex_data = hex_data
        self.on_click_callback = on_click_callback
        self.radius = radius

        self.center_x = radius * math.sqrt(3) * (q + r / 2.0)
        self.center_y = radius * 1.5 * r

        points = []
        for i in range(6):
            angle_rad = math.radians(60 * i - 30)
            points.append(QPointF(
                self.center_x + radius * math.cos(angle_rad),
                self.center_y + radius * math.sin(angle_rad)
            ))
        self.setPolygon(QPolygonF(points))

        planet_key = str(hex_data.get("planet", "empty")).lower()
        style = PLANET_STYLES.get(planet_key, PLANET_STYLES["empty"])
        self.setBrush(QBrush(QColor(style["fill"])))
        self.setPen(QPen(QColor(style["border"]), 1.2 if planet_key != "empty" else 0.5))

    def mousePressEvent(self, event):
        if self.on_click_callback:
            self.on_click_callback(self.q, self.r, self.s, self.hex_data)
        super().mousePressEvent(event)


class SpaceMapView(QGraphicsView):
    def __init__(self, on_hex_selected):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.on_hex_selected = on_hex_selected
        
        self.setBackgroundBrush(QBrush(QColor("#030712")))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        # Zoom molette fluide
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(factor, factor)

    def render_galaxy(self, map_data):
        self.scene.clear()
        hexes = map_data.get("hexes", [])
        coords = map_data.get("coords", [])
        count = min(len(coords), len(hexes))
        
        for i in range(count):
            c = coords[i]
            h = hexes[i]
            q, r, s = c.get("q", 0), c.get("r", 0), c.get("s", 0)
            planet_key = str(h.get("planet", "empty")).lower()
            style = PLANET_STYLES.get(planet_key, PLANET_STYLES["empty"])

            # Hexagone
            tile = HexTile(q, r, s, h, self.on_hex_selected)
            self.scene.addItem(tile)
            cx, cy = tile.center_x, tile.center_y

            # Disque de la planète
            if style["disk"] is not None:
                planet_radius = 17
                self.scene.addEllipse(
                    cx - planet_radius, cy - planet_radius,
                    planet_radius * 2, planet_radius * 2,
                    QPen(QColor(style["border"]), 1.6),
                    QBrush(QColor(style["disk"]))
                )
                lbl = planet_key[:2].upper()
                txt = self.scene.addText(lbl)
                txt.setFont(QFont("Segoe UI", 7, QFont.Bold))
                txt.setDefaultTextColor(QColor("#ffffff"))
                txt.setPos(cx - 7, cy - 8)

            # Rendu Bâtiments
            bldg = h.get("building")
            p_seat = h.get("player")
            if bldg and p_seat is not None:
                p_col = QColor(PLAYER_COLORS[p_seat % len(PLAYER_COLORS)])
                b_str = str(bldg).lower()

                if "mine" in b_str:
                    size = 12
                    self.scene.addRect(cx - size/2, cy - size/2, size, size, QPen(QColor("#ffffff"), 1.2), QBrush(p_col))
                elif "trading_station" in b_str:
                    w, h_size = 10, 18
                    self.scene.addRect(cx - w/2, cy - h_size/2, w, h_size, QPen(QColor("#ffffff"), 1.2), QBrush(p_col))
                elif "research_lab" in b_str:
                    r_lab = 9
                    self.scene.addEllipse(cx - r_lab, cy - r_lab, r_lab*2, r_lab*2, QPen(QColor("#ffffff"), 1.5), QBrush(p_col))
                elif "planetary_institute" in b_str:
                    size = 18
                    self.scene.addRect(cx - size/2, cy - size/2, size, size, QPen(QColor("#facc15"), 2.2), QBrush(p_col))
                elif "academy" in b_str:
                    r_acad = 11
                    self.scene.addEllipse(cx - r_acad, cy - r_acad, r_acad*2, r_acad*2, QPen(QColor("#facc15"), 2.2), QBrush(p_col))
                elif "gaia_former" in b_str:
                    pts = [QPointF(cx, cy - 10), QPointF(cx - 9, cy + 8), QPointF(cx + 9, cy + 8)]
                    self.scene.addPolygon(QPolygonF(pts), QPen(QColor("#ffffff"), 1.2), QBrush(p_col))

            # Rendu Satellites
            in_fed = h.get("in_federation", [False]*4)
            if planet_key == "empty":
                for idx, sat in enumerate(in_fed):
                    if sat:
                        col = QColor(PLAYER_COLORS[idx % len(PLAYER_COLORS)])
                        self.scene.addEllipse(cx - 5, cy - 5, 10, 10, QPen(QColor("#ffffff"), 1.2), QBrush(col))


# ---------------------------------------------------------------------------
# Fenêtre Principale Ultra-Ergonomique avec Mode Sandbox & Analyse
# ---------------------------------------------------------------------------
class GaiaLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projet Gaïa - Station IA, Simulateur & Éditeur de Scénario (Sandbox)")
        self.resize(1560, 960)
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QLabel { color: #e2e8f0; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
            QTabWidget::pane { border: 1px solid #1e293b; background-color: #0f172a; border-radius: 6px; }
            QTabBar::tab { background-color: #1e293b; color: #94a3b8; padding: 10px 18px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 3px; }
            QTabBar::tab:selected { background-color: #2563eb; color: #ffffff; }
            QGroupBox { color: #38bdf8; font-weight: bold; border: 1px solid #1e293b; border-radius: 8px; margin-top: 10px; padding-top: 14px; }
            QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2563eb; border-color: #3b82f6; }
            QTableWidget { background-color: #030712; color: #f1f5f9; gridline-color: #1e293b; border: 1px solid #1e293b; border-radius: 6px; }
            QHeaderView::section { background-color: #1e293b; color: #94a3b8; font-weight: bold; border: none; padding: 6px; }
            QComboBox, QSpinBox { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 5px; font-weight: bold; }
            QTextEdit { background-color: #030712; color: #38bdf8; border: 1px solid #1e293b; font-family: 'Consolas', monospace; }
        """)

        self.bridge = RustEngineBridge()
        self.ai_evaluator = AIEvaluator(self.bridge)
        self.best_ai_action = None
        self.current_seed = 42
        self.env_ptr = self.bridge.create_env(self.current_seed, 4)
        
        self.cached_state = {}
        self.cached_legal_actions = []
        self.selected_hex_coord = None
        
        # Mode : 'PLAY' (Jeu normal) ou 'SANDBOX' (Édition arbitraire)
        self.app_mode = "PLAY"
        
        # Outils du mode Édition
        self.sandbox_selected_building = 1  # 1 = Mine
        self.sandbox_selected_player = 0    # J0

        self.setup_ui()
        self.refresh_game_state()

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 10, 12, 12)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. BARRE DE COMMANDE SUPÉRIEURE (MODES & EN-TÊTE)
        # -------------------------------------------------------------
        top_bar = QHBoxLayout()
        
        self.lbl_title = QLabel("🌌 <b>PROJET GAÏA</b> | Station Tactique & IA")
        self.lbl_title.setStyleSheet("font-size: 15px; color: #38bdf8;")
        top_bar.addWidget(self.lbl_title)
        
        top_bar.addSpacing(25)
        
        # Sélecteur de Mode
        top_bar.addWidget(QLabel("<b>Mode actif :</b>"))
        self.btn_mode_play = QPushButton("🎮 Mode Jeu / Déroulé")
        self.btn_mode_play.setCheckable(True)
        self.btn_mode_play.setChecked(True)
        self.btn_mode_play.setStyleSheet("background-color: #2563eb; color: white;")
        self.btn_mode_play.clicked.connect(lambda: self.switch_mode("PLAY"))
        top_bar.addWidget(self.btn_mode_play)

        self.btn_mode_sandbox = QPushButton("🛠️ Mode Éditeur de Scénario (Sandbox)")
        self.btn_mode_sandbox.setCheckable(True)
        self.btn_mode_sandbox.clicked.connect(lambda: self.switch_mode("SANDBOX"))
        top_bar.addWidget(self.btn_mode_sandbox)

        top_bar.addStretch()

        self.lbl_active_info = QLabel("Manche 1 / 6 | Tour du Joueur 0 (Bleu)")
        self.lbl_active_info.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #1e293b; padding: 6px 12px; border-radius: 4px;")
        top_bar.addWidget(self.lbl_active_info)

        self.btn_reseed = QPushButton("🎲 Carte Aléatoire")
        self.btn_reseed.clicked.connect(self.generate_new_map)
        top_bar.addWidget(self.btn_reseed)

        main_layout.addLayout(top_bar)

        # -------------------------------------------------------------
        # 2. ZONE CENTRALE : SPLIT CARTE (GAUCHE) & PANNEAUX (DROITE)
        # -------------------------------------------------------------
        content_layout = QHBoxLayout()

        # --- A. COLONNE GAUCHE : CARTE ET ACTIONS DIRECTES ---
        map_col = QVBoxLayout()
        
        self.map_view = SpaceMapView(self.on_hex_clicked)
        map_col.addWidget(self.map_view, stretch=4)

        # Inspecteur d'Hexagone & Action directe
        grp_inspect = QGroupBox("📍 Inspecteur & Actions Immédiates sur l'Hexagone")
        self.inspect_layout = QHBoxLayout()
        self.lbl_hex_details = QLabel("<i>Cliquez sur un hexagone pour agir immédiatement.</i>")
        self.inspect_layout.addWidget(self.lbl_hex_details, stretch=2)
        
        self.context_actions_box = QHBoxLayout()
        self.inspect_layout.addLayout(self.context_actions_box, stretch=3)
        grp_inspect.setLayout(self.inspect_layout)
        map_col.addWidget(grp_inspect)

        content_layout.addLayout(map_col, stretch=3)

        # --- B. COLONNE DROITE : CENTRE DE CONTRÔLE OU SANDBOX ---
        right_col = QVBoxLayout()

        self.right_stack = QTabWidget()

        # ONGLET 1 : COMMANDES & ANALYSE DE L'IA (MODE JEU)
        self.tab_commands = QWidget()
        cmd_layout = QVBoxLayout(self.tab_commands)

        # Bloc Recommandation Majeure de l'IA
        grp_ai_rec = QGroupBox("🧠 Décision Stratégique Majeure de l'IA")
        ai_rec_layout = QVBoxLayout()
        
        self.lbl_ai_best = QLabel("Recherche de la meilleure opportunité en cours...")
        self.lbl_ai_best.setStyleSheet("font-size: 13px; font-weight: bold; color: #4ade80; padding: 8px; background-color: #030712; border-radius: 4px;")
        ai_rec_layout.addWidget(self.lbl_ai_best)

        btn_ai_exec = QPushButton("⚡ Valider & Exécuter ce Coup Recommandé")
        btn_ai_exec.setStyleSheet("background-color: #16a34a; font-size: 13px; padding: 10px; color: white;")
        btn_ai_exec.clicked.connect(self.play_ai_move)
        ai_rec_layout.addWidget(btn_ai_exec)
        grp_ai_rec.setLayout(ai_rec_layout)
        cmd_layout.addWidget(grp_ai_rec)

        # Filtre Catégories d'Actions
        cat_filter_box = QHBoxLayout()
        cat_filter_box.addWidget(QLabel("<b>Catégorie d'actions :</b>"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems([
            "✨ Toutes les Actions Légales",
            "🪐 Construire une Mine",
            "🏢 Améliorer Bâtiment",
            "🔬 Recherche & Technologies",
            "⚡ Plateau de Puissance (QIC & Énergie)",
            "🛑 Passer son Tour & Prendre Booster"
        ])
        self.cmb_filter.currentIndexChanged.connect(self.filter_actions)
        cat_filter_box.addWidget(self.cmb_filter, stretch=2)
        cmd_layout.addLayout(cat_filter_box)

        # Tableau des coups possibles
        self.tbl_actions = QTableWidget(0, 3)
        self.tbl_actions.setHorizontalHeaderLabels(["ID", "Description Détaillée", "Jouer"])
        self.tbl_actions.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_actions.verticalHeader().setVisible(False)
        cmd_layout.addWidget(self.tbl_actions)

        self.right_stack.addTab(self.tab_commands, "🎮 Commandes & IA")

        # ONGLET 2 : CONFIGURATEUR DE SITUATION DONNÉE (MODE SANDBOX)
        self.tab_sandbox = QWidget()
        sb_layout = QVBoxLayout(self.tab_sandbox)

        # Outil Pinceau de Carte
        grp_brush = QGroupBox("🖌️ Pinceau Bâtiment sur la Carte")
        brush_layout = QGridLayout()

        brush_layout.addWidget(QLabel("Bâtiment à poser au clic :"), 0, 0)
        self.cmb_brush_bldg = QComboBox()
        for bid, bname, emoji in BUILDING_TYPES:
            self.cmb_brush_bldg.addItem(f"{emoji} {bname}", bid)
        self.cmb_brush_bldg.setCurrentIndex(1)  # Mine par défaut
        brush_layout.addWidget(self.cmb_brush_bldg, 0, 1)

        brush_layout.addWidget(QLabel("Appartenance :"), 1, 0)
        self.cmb_brush_player = QComboBox()
        for idx, name in enumerate(PLAYER_LABELS):
            self.cmb_brush_player.addItem(name, idx)
        brush_layout.addWidget(self.cmb_brush_player, 1, 1)

        lbl_brush_hint = QLabel("<i>Cliquez sur n'importe quel hexagone de la galaxie pour y poser ou retirer le bâtiment choisi !</i>")
        lbl_brush_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        brush_layout.addWidget(lbl_brush_hint, 2, 0, 1, 2)
        grp_brush.setLayout(brush_layout)
        sb_layout.addWidget(grp_brush)

        # Édition Tour & Joueur Actif
        grp_turn = QGroupBox("⏱️ Manche & Joueur Actif")
        turn_layout = QHBoxLayout()
        turn_layout.addWidget(QLabel("Manche :"))
        self.spn_round = QSpinBox()
        self.spn_round.setRange(1, 6)
        turn_layout.addWidget(self.spn_round)

        turn_layout.addWidget(QLabel("Joueur dont c'est le tour :"))
        self.cmb_turn_player = QComboBox()
        self.cmb_turn_player.addItems(PLAYER_LABELS)
        turn_layout.addWidget(self.cmb_turn_player)

        btn_apply_turn = QPushButton("Appliquer")
        btn_apply_turn.clicked.connect(self.apply_sandbox_turn)
        turn_layout.addWidget(btn_apply_turn)
        grp_turn.setLayout(turn_layout)
        sb_layout.addWidget(grp_turn)

        # Édition des Ressources d'un Joueur
        grp_res = QGroupBox("💰 Réglage Direct des Ressources & Puissance")
        res_layout = QGridLayout()

        res_layout.addWidget(QLabel("Éditer joueur :"), 0, 0)
        self.cmb_edit_p = QComboBox()
        self.cmb_edit_p.addItems(PLAYER_LABELS)
        self.cmb_edit_p.currentIndexChanged.connect(self.load_sandbox_player_fields)
        res_layout.addWidget(self.cmb_edit_p, 0, 1, 1, 3)

        res_layout.addWidget(QLabel("Crédits (0-30) :"), 1, 0)
        self.spn_c = QSpinBox(); self.spn_c.setRange(0, 30); res_layout.addWidget(self.spn_c, 1, 1)

        res_layout.addWidget(QLabel("Minerai (0-15) :"), 1, 2)
        self.spn_o = QSpinBox(); self.spn_o.setRange(0, 15); res_layout.addWidget(self.spn_o, 1, 3)

        res_layout.addWidget(QLabel("Savoir (0-15) :"), 2, 0)
        self.spn_k = QSpinBox(); self.spn_k.setRange(0, 15); res_layout.addWidget(self.spn_k, 2, 1)

        res_layout.addWidget(QLabel("Q.I.C. (0-15) :"), 2, 2)
        self.spn_q = QSpinBox(); self.spn_q.setRange(0, 15); res_layout.addWidget(self.spn_q, 2, 3)

        res_layout.addWidget(QLabel("Points Victoire :"), 3, 0)
        self.spn_vp = QSpinBox(); self.spn_vp.setRange(-50, 300); res_layout.addWidget(self.spn_vp, 3, 1)

        res_layout.addWidget(QLabel("Puissance Bol 3 (Dépensable) :"), 3, 2)
        self.spn_p3 = QSpinBox(); self.spn_p3.setRange(0, 20); res_layout.addWidget(self.spn_p3, 3, 3)

        btn_apply_res = QPushButton("💾 Sauvegarder les Ressources du Joueur")
        btn_apply_res.setStyleSheet("background-color: #2563eb; color: white;")
        btn_apply_res.clicked.connect(self.apply_sandbox_resources)
        res_layout.addWidget(btn_apply_res, 4, 0, 1, 4)

        grp_res.setLayout(res_layout)
        sb_layout.addWidget(grp_res)

        # Édition des Niveaux de Recherche
        grp_research_edit = QGroupBox("🔬 Réglage de Recherche Technologique")
        re_layout = QGridLayout()
        self.spn_research = []
        for i, (name, _) in enumerate([("Terra", 0), ("Navig", 1), ("Intel", 2), ("Gaïa", 3), ("Éco", 4), ("Scien", 5)]):
            re_layout.addWidget(QLabel(name), 0, i)
            spn = QSpinBox()
            spn.setRange(0, 5)
            self.spn_research.append(spn)
            re_layout.addWidget(spn, 1, i)

        btn_apply_res_track = QPushButton("💾 Sauvegarder Recherche")
        btn_apply_res_track.clicked.connect(self.apply_sandbox_research)
        re_layout.addWidget(btn_apply_res_track, 2, 0, 1, 6)
        grp_research_edit.setLayout(re_layout)
        sb_layout.addWidget(grp_research_edit)

        # Gros bouton Évaluation IA
        btn_eval_scenario = QPushButton("🚀 Lancer l'Analyse IA sur cette Situation")
        btn_eval_scenario.setStyleSheet("background-color: #16a34a; font-size: 13px; padding: 12px; color: white;")
        btn_eval_scenario.clicked.connect(self.analyze_sandbox_situation)
        sb_layout.addWidget(btn_eval_scenario)

        self.right_stack.addTab(self.tab_sandbox, "🛠️ Éditeur de Scénario (Sandbox)")

        # ONGLET 3 : PLATEAUX JOUEURS & RÉSERVES
        self.tab_factions = QWidget()
        fac_layout = QVBoxLayout(self.tab_factions)

        self.tbl_players_summary = QTableWidget(4, 8)
        self.tbl_players_summary.setHorizontalHeaderLabels(["Siège", "Faction", "Points", "Crédits", "Minerai", "Savoir", "QIC", "Satellites"])
        self.tbl_players_summary.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_players_summary.verticalHeader().setVisible(False)
        self.tbl_players_summary.setFixedHeight(140)
        fac_layout.addWidget(self.tbl_players_summary)

        self.fac_inspect_box = QLabel()
        self.fac_inspect_box.setStyleSheet("background-color: #030712; padding: 15px; border-radius: 6px; border: 1px solid #1e293b; font-size: 12px; line-height: 1.6;")
        fac_layout.addWidget(self.fac_inspect_box)

        self.right_stack.addTab(self.tab_factions, "🪐 Plateaux de Faction")

        # ONGLET 4 : RECHERCHE & SCORING
        self.tab_tech = QWidget()
        tech_layout = QVBoxLayout(self.tab_tech)

        self.tbl_research_view = QTableWidget(6, 6)
        self.tbl_research_view.setHorizontalHeaderLabels(["Piste Technologique", "J0", "J1", "J2", "J3", "Clé Niveau 5"])
        self.tbl_research_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_research_view.verticalHeader().setVisible(False)
        tech_layout.addWidget(self.tbl_research_view)

        self.lbl_scoring_view = QLabel()
        self.lbl_scoring_view.setStyleSheet("background-color: #030712; padding: 12px; border-radius: 6px; border: 1px solid #1e293b;")
        tech_layout.addWidget(self.lbl_scoring_view)

        self.right_stack.addTab(self.tab_tech, "🔬 Recherche & Scoring")

        right_col.addWidget(self.right_stack, stretch=4)

        # Journal Moteur
        grp_log = QGroupBox("📜 Journal d'Événements")
        log_box = QVBoxLayout()
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(90)
        log_box.addWidget(self.txt_log)
        grp_log.setLayout(log_box)
        right_col.addWidget(grp_log, stretch=1)

        content_layout.addLayout(right_col, stretch=2)
        main_layout.addLayout(content_layout)
        self.setCentralWidget(main_widget)

    def log(self, msg):
        self.txt_log.append(f">> {msg}")

    # =============================================================
    # Basculement de Mode (Jeu vs Sandbox)
    # =============================================================
    def switch_mode(self, mode):
        self.app_mode = mode
        if mode == "PLAY":
            self.btn_mode_play.setChecked(True)
            self.btn_mode_sandbox.setChecked(False)
            self.btn_mode_play.setStyleSheet("background-color: #2563eb; color: white;")
            self.btn_mode_sandbox.setStyleSheet("background-color: #1e293b; color: #f8fafc;")
            self.right_stack.setCurrentIndex(0)
            self.log("Mode JEU activé. L'IA analyse les coups légaux disponibles.")
        else:
            self.btn_mode_play.setChecked(False)
            self.btn_mode_sandbox.setChecked(True)
            self.btn_mode_sandbox.setStyleSheet("background-color: #d97706; color: white;")
            self.btn_mode_play.setStyleSheet("background-color: #1e293b; color: #f8fafc;")
            self.right_stack.setCurrentIndex(1)
            self.load_sandbox_player_fields()
            self.log("Mode ÉDITEUR (Sandbox) activé. Cliquez sur la carte pour poser des bâtiments ou modifiez les ressources.")
        self.refresh_game_state()

    # =============================================================
    # Mise à Jour de l'État
    # =============================================================
    def refresh_game_state(self):
        state = self.bridge.get_state(self.env_ptr)
        if not state: return
        self.cached_state = state

        # 1. Rendu Galaxie
        map_data = state.get("map", {})
        self.map_view.render_galaxy(map_data)

        # 2. En-tête
        round_num = state.get("round", 1)
        cur_p = state.get("current_player", 0)
        players = state.get("players", [])
        cur_faction = players[cur_p].get("faction", "") if cur_p < len(players) else ""
        
        self.lbl_active_info.setText(f"Manche {round_num} / 6 | Tour du Joueur {cur_p} ({str(cur_faction).capitalize()})")

        # 3. Tableau Joueurs & Factions
        for i, p in enumerate(players):
            self.tbl_players_summary.setItem(i, 0, QTableWidgetItem(f"J{i}"))
            self.tbl_players_summary.setItem(i, 1, QTableWidgetItem(str(p.get("faction", "")).capitalize()))
            self.tbl_players_summary.setItem(i, 2, QTableWidgetItem(f"{p.get('victory_points', 0)} VP"))
            self.tbl_players_summary.setItem(i, 3, QTableWidgetItem(f"{p.get('credits', 0)} C"))
            self.tbl_players_summary.setItem(i, 4, QTableWidgetItem(f"{p.get('ore', 0)} O"))
            self.tbl_players_summary.setItem(i, 5, QTableWidgetItem(f"{p.get('knowledge', 0)} K"))
            self.tbl_players_summary.setItem(i, 6, QTableWidgetItem(f"{p.get('qic', 0)} Q"))
            self.tbl_players_summary.setItem(i, 7, QTableWidgetItem(f"{p.get('satellites', 0)} Sat"))

        # Détail Faction active
        if cur_p < len(players):
            p = players[cur_p]
            pw = p.get("power", {})
            bldgs = p.get("buildings", [0]*8)
            self.fac_inspect_box.setText(
                f"<h3>🪐 Détails Faction {str(p.get('faction')).upper()} (Joueur {cur_p}) :</h3>"
                f"<b>⚡ Réserve d'Énergie :</b> Bol 1: {pw.get('area1',0)} | Bol 2: {pw.get('area2',0)} | "
                f"<span style='color:#4ade80;'><b>Bol 3 Dépensable: {pw.get('area3',0)}</b></span> | Zone Gaïa: {pw.get('gaia',0)}<br>"
                f"<b>🏢 Structures Déployées :</b> Mines: {bldgs[0]}/8 | Stations: {bldgs[1]}/4 | Labos: {bldgs[2]}/3 | "
                f"Institut: {'Construit' if bldgs[3]>0 else 'Disponible'}<br>"
                f"<b>🛰️ Satellites et Gaiaformers :</b> Satellites placés: {p.get('satellites',0)} | Gaiaformers débloqués: {p.get('gaiaformers_unlocked',0)}"
            )

        # 4. Pistes Recherche
        claimed_l5 = state.get("research_level_5_claimed", [None]*6)
        r_names = ["🌍 Terraformation", "🚀 Navigation", "🧠 Intelligence (QIC)", "🌀 Projet Gaïa", "💰 Économie", "🧪 Science"]
        for row, rname in enumerate(r_names):
            self.tbl_research_view.setItem(row, 0, QTableWidgetItem(rname))
            for p_idx, p in enumerate(players):
                lvl = p.get("research", [0]*6)[row]
                it = QTableWidgetItem(f"Niv {lvl}")
                it.setTextAlignment(Qt.AlignCenter)
                if lvl == 5:
                    it.setForeground(QBrush(QColor("#facc15")))
                    it.setFont(QFont("Segoe UI", 9, QFont.Bold))
                self.tbl_research_view.setItem(row, p_idx + 1, it)
            l5_win = claimed_l5[row]
            self.tbl_research_view.setItem(row, 5, QTableWidgetItem(f"J{l5_win}" if l5_win is not None else "Libre"))

        # Scoring
        r_tiles = state.get("round_scoring_tiles", [])
        self.lbl_scoring_view.setText(f"<b>🎯 Objectif Manche {round_num} :</b> {r_tiles[round_num-1].upper() if round_num<=len(r_tiles) else 'Fin'}")

        # 5. Actions Légales & Meilleur Coup IA Neuronal
        self.cached_legal_actions = self.bridge.get_legal_actions(self.env_ptr)
        self.best_ai_action = None
        if self.cached_legal_actions:
            eval_res = self.ai_evaluator.evaluate(self.env_ptr) if hasattr(self, "ai_evaluator") else None
            if eval_res and eval_res.get("best_action") is not None:
                self.best_ai_action = eval_res["best_action"]
                top = eval_res["top_actions"]
                best_name = top[0][1] if top else self.bridge.get_action_name(self.env_ptr, self.best_ai_action)
                best_pct = top[0][2] if top else 100.0
                pred_vp = eval_res["predicted_vp"]
                
                text = f"⭐ <b>Recommandation IA ({best_pct:.1f}%) :</b> {best_name} | <i>Score projeté : {pred_vp:.1f} VP</i>"
                if len(top) > 1:
                    alts = " • ".join([f"{name.split(' en ')[0] if ' en ' in name else name} ({p:.1f}%)" for _, name, p in top[1:4]])
                    text += f"<br><span style='color: #94a3b8; font-size: 11px;'>Alternatives : {alts}</span>"
                self.lbl_ai_best.setText(text)
            else:
                self.best_ai_action = self.cached_legal_actions[0]
                best_action_desc = self.bridge.get_action_name(self.env_ptr, self.best_ai_action)
                self.lbl_ai_best.setText(f"⭐ Coup Heuristique Détecté : {best_action_desc}")
        else:
            self.lbl_ai_best.setText("Aucune action légale (Manche ou partie terminée).")

        self.filter_actions()

        # 6. Rafraîchir inspecteur d'hexagone
        if self.selected_hex_coord:
            q, r, s = self.selected_hex_coord
            coords = map_data.get("coords", [])
            hexes = map_data.get("hexes", [])
            for i in range(min(len(coords), len(hexes))):
                c = coords[i]
                if c.get("q") == q and c.get("r") == r and c.get("s") == s:
                    self.on_hex_clicked(q, r, s, hexes[i], trigger_refresh=False)
                    break

    # =============================================================
    # Clic sur Hexagone : Jeu Contextuel ou Pinceau Sandbox
    # =============================================================
    def on_hex_clicked(self, q, r, s, hex_data, trigger_refresh=True):
        self.selected_hex_coord = (q, r, s)

        # SI ON EST EN MODE SANDBOX : APPLIQUER LE PINCEAU DIRECTEMENT !
        if self.app_mode == "SANDBOX" and trigger_refresh:
            bldg_id = self.cmb_brush_bldg.currentData()
            p_seat = self.cmb_brush_player.currentIndex()
            self.bridge.set_hex_building(self.env_ptr, q, r, s, bldg_id, p_seat)
            self.log(f"Sandbox : Bâtiment #{bldg_id} assigné au Joueur {p_seat} sur [{q}, {r}, {s}].")
            self.refresh_game_state()
            return

        # MODE JEU NORMAL : INSPECTER ET PROPOSER LES ACTIONS DIRECTES
        planet = str(hex_data.get("planet", "empty")).capitalize()
        bldg = hex_data.get("building") or "Aucun"
        owner = hex_data.get("player")
        owner_str = f"Joueur {owner}" if owner is not None else "Inoccupé"

        self.lbl_hex_details.setText(
            f"<b>Hexagone [Q:{q}, R:{r}, S:{s}]</b><br>"
            f"Planète : <b>{planet}</b> | Structure : <b>{bldg}</b> ({owner_str})"
        )

        for i in reversed(range(self.context_actions_box.count())):
            w = self.context_actions_box.itemAt(i).widget()
            if w: w.setParent(None)

        # Recherche des coups légaux ciblant cet hex
        target_pattern = f"q:{q}, r:{r}, s:{s}"
        matches = []
        for act_id in self.cached_legal_actions:
            name = self.bridge.get_action_name(self.env_ptr, act_id)
            if target_pattern in name:
                matches.append((act_id, name))

        if matches:
            for act_id, act_name in matches[:3]:
                short_name = act_name.split(" en ")[0] if " en " in act_name else act_name
                btn = QPushButton(f"✨ {short_name}")
                btn.setStyleSheet("background-color: #16a34a; color: white; padding: 8px 12px; font-weight: bold;")
                btn.clicked.connect(lambda checked, a=act_id: self.execute_action(a))
                self.context_actions_box.addWidget(btn)
        else:
            lbl_none = QLabel("<i>Aucune action directe légale sur cette case.</i>")
            lbl_none.setStyleSheet("color: #64748b;")
            self.context_actions_box.addWidget(lbl_none)

    # =============================================================
    # Méthodes de Gestion Sandbox
    # =============================================================
    def load_sandbox_player_fields(self):
        p_idx = self.cmb_edit_p.currentIndex()
        players = self.cached_state.get("players", [])
        if p_idx >= len(players): return
        p = players[p_idx]
        
        self.spn_c.setValue(int(p.get("credits", 0)))
        self.spn_o.setValue(int(p.get("ore", 0)))
        self.spn_k.setValue(int(p.get("knowledge", 0)))
        self.spn_q.setValue(int(p.get("qic", 0)))
        self.spn_vp.setValue(int(p.get("victory_points", 0)))
        self.spn_p3.setValue(int(p.get("power", {}).get("area3", 0)))

        research = p.get("research", [0]*6)
        for i, spn in enumerate(self.spn_research):
            if i < len(research):
                spn.setValue(int(research[i]))

    def apply_sandbox_turn(self):
        rnd = self.spn_round.value()
        seat = self.cmb_turn_player.currentIndex()
        self.bridge.set_round(self.env_ptr, rnd)
        self.bridge.set_current_player(self.env_ptr, seat)
        self.log(f"Sandbox : Manche fixée à {rnd}, Tour passé au Joueur {seat}.")
        self.refresh_game_state()

    def apply_sandbox_resources(self):
        seat = self.cmb_edit_p.currentIndex()
        c = self.spn_c.value()
        o = self.spn_o.value()
        k = self.spn_k.value()
        q = self.spn_q.value()
        vp = self.spn_vp.value()
        p3 = self.spn_p3.value()
        self.bridge.set_player_resources(self.env_ptr, seat, c, o, k, q, vp, 0, 0, p3)
        self.log(f"Sandbox : Ressources appliquées pour Joueur {seat}.")
        self.refresh_game_state()

    def apply_sandbox_research(self):
        seat = self.cmb_edit_p.currentIndex()
        for field_idx, spn in enumerate(self.spn_research):
            lvl = spn.value()
            self.bridge.set_player_research(self.env_ptr, seat, field_idx, lvl)
        self.log(f"Sandbox : Niveaux technologiques appliqués pour Joueur {seat}.")
        self.refresh_game_state()

    # =============================================================
    # Exécution & Filtrage des Actions
    # =============================================================
    def filter_actions(self):
        cat_idx = self.cmb_filter.currentIndex()
        self.tbl_actions.setRowCount(0)
        
        row = 0
        for act_id in self.cached_legal_actions:
            desc = self.bridge.get_action_name(self.env_ptr, act_id)
            desc_lower = desc.lower()

            if cat_idx == 1 and "construire mine" not in desc_lower: continue
            if cat_idx == 2 and "améliorer vers" not in desc_lower: continue
            if cat_idx == 3 and "recherche" not in desc_lower and "tech" not in desc_lower: continue
            if cat_idx == 4 and "puissance" not in desc_lower and "projet gaïa" not in desc_lower: continue
            if cat_idx == 5 and "passer" not in desc_lower: continue

            self.tbl_actions.insertRow(row)
            self.tbl_actions.setItem(row, 0, QTableWidgetItem(f"#{act_id}"))
            self.tbl_actions.setItem(row, 1, QTableWidgetItem(desc))

            btn = QPushButton("Valider")
            btn.clicked.connect(lambda checked, a=act_id: self.execute_action(a))
            self.tbl_actions.setCellWidget(row, 2, btn)
            row += 1
            if row >= 25: break

    def execute_action(self, action_idx):
        action_name = self.bridge.get_action_name(self.env_ptr, action_idx)
        reward = self.bridge.step(self.env_ptr, action_idx)
        self.log(f"Coup exécuté : {action_name} (Reward: {reward:+.2f})")
        self.refresh_game_state()

    def play_ai_move(self):
        if self.best_ai_action is not None:
            self.execute_action(self.best_ai_action)
        elif self.cached_legal_actions:
            self.execute_action(self.cached_legal_actions[0])
        else:
            self.log("Aucune action légale disponible.")

    def analyze_sandbox_situation(self):
        self.switch_mode("PLAY")
        self.log("🚀 Analyse neuronale lancée sur la situation configurée !")

    def generate_new_map(self):
        self.current_seed = (self.current_seed * 1664525 + 1013904223) % 1000000
        self.bridge.reset_env(self.env_ptr, self.current_seed)
        self.log(f"Nouvelle carte générée avec Seed #{self.current_seed}")
        self.refresh_game_state()


# ---------------------------------------------------------------------------
# Lancement de l'Application
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GaiaLabWindow()
    window.show()
    sys.exit(app.exec_() if QT_LIB == "PyQt5" else app.exec())
