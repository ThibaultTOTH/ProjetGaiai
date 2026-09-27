"""Tkinter Graphical User Interface for Gaia Project Deep RL Studio.

Features:
- Live Matplotlib training curves (Policy Loss, Value Loss, VP accuracy, Win Rate).
- 1-Click Training Presets: [Fast (5 min)], [Pro Compétitif (1h30)], [Grandmaster Top 1% (4-6h)].
- Full SOTA 2026 Configuration Panel with 13 scrollable technology sections:
  * Hardware & GPU (RTX 5070 / CUDA / CPU, AMP FP16/BF16, TF32 Tensor Cores)
  * Deep RL Architectures (Pre-LN, SwiGLU, Bottleneck, MLP, SiLU/GELU, Dropout)
  * Spatial HexGNN Hexagonal Map Encoder (Message Passing on 200 hex tiles)
  * PPO Actor-Critic (LRs, Batch sizes, Rollout buffers, Gamma, GAE, Clip)
  * Dynamic Cosine Warmup & Annealing Schedules (LR & Entropy)
  * Potential-based Annealed Reward Shaping
  * Random Network Distillation (RND) Intrinsic Curiosity
  * Multi-Player Auxiliary Opponent Action Modeling
  * Regularized Nash Dynamics (R-NaD) for multi-agent equilibrium
  * Regret-Guided Search Control (RGSC / Go-Exploit) counterfactual crisis puzzles
  * Population-Based League Training with Gaussian Elo Matchmaking (ZPD)
  * Asynchronous Distributed Actor-Learner Architecture (APPO)
  * Monte-Carlo Tree Search (Gumbel GAZ 2026 Sequential Halving, MC-Dropout epistemic guidance, entropy gating)
- Automated Hyperparameter Search (NAS) with live trials table.
- Interactive Match Simulator with Gumbel GAZ / PUCT and rich telemetry logs.
- Cross-platform Linux & Windows execution with graceful headless fallback.
"""

from copy import deepcopy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch

from analytics import generate_strategy_pdf
from config import AppConfig, get_training_preset
from environment import format_flat_action, make_gaia_env
from hyperopt import HyperoptTrial, HyperparameterOptimizer
from mcts import MultiPlayerMCTS
from async_trainer import AsyncRLTrainer
from alphazero_trainer import AlphaZeroTrainer
from muzero import MuZeroTrainer
from models import DualGaiaAgent
from trainer import RLTrainer, TrainingMetrics

matplotlib.use("TkAgg")


class GaiaRLStudioGUI:
    """Main Application Window for Gaia Project RL Training, MCTS Analysis & Tuning."""

    def _create_trainer(self) -> Any:
        algo = getattr(self, 'var_training_algo', None)
        algo_name = algo.get() if algo else ("AlphaZero" if getattr(self.config.alphazero, "enabled", False) else "PPO")
        if algo_name == "AlphaZero":
            return AlphaZeroTrainer(self.config)
        if algo_name == "MuZero":
            return MuZeroTrainer(self.config)
        if getattr(self.config.async_dist, "enabled", False):
            return AsyncRLTrainer(self.config)
        return RLTrainer(self.config)

    def __init__(self, root: tk.Tk, config: Optional[AppConfig] = None):
        self.root = root
        self.root.title("🌌 Gaia Project — Deep RL Studio (SOTA 2026 / Linux & Windows)")
        self.root.geometry("1240x860")
        self.root.minsize(1020, 720)

        self.config = config or get_training_preset("pretrain")
        self.trainer = self._create_trainer()
        self.hyperopt = HyperparameterOptimizer(self.config)

        # Thread-safe event queues for UI updates
        self.metrics_queue: queue.Queue = queue.Queue()
        self.hyperopt_queue: queue.Queue = queue.Queue()

        # History tracking for plotting
        self.history_epochs: List[int] = []
        self.history_policy_loss: List[float] = []
        self.history_value_loss: List[float] = []
        self.history_pred_scores: List[float] = []
        self.history_real_scores: List[float] = []

        self._build_style()
        self._build_header()
        self._build_tabs()
        self._sync_config_to_ui()
        if hasattr(self, "lbl_preset_status"):
            self.lbl_preset_status.config(
                text="✓ Préréglage actif : 🚀 1. Pré-entraînement Fondation (150 VP)",
                fg="#38bdf8",
            )
        self._start_polling()

    # -------------------------------------------------------------
    # STYLES & THEMING
    # -------------------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Header.TFrame", background="#0f172a")
        style.configure("Header.TLabel", background="#0f172a", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("SubHeader.TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 9))

        style.configure("Card.TFrame", background="#f8fafc", relief="ridge")
        style.configure("CardTitle.TLabel", font=("Segoe UI", 8, "bold"), foreground="#475569")

        style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), padding=5)
        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.configure("PresetFast.TButton", font=("Segoe UI", 8, "bold"))
        style.configure("PresetPro.TButton", font=("Segoe UI", 8, "bold"))
        style.configure("PresetGM.TButton", font=("Segoe UI", 8, "bold"))

    # -------------------------------------------------------------
    # HEADER & PRESETS TOOLBAR
    # -------------------------------------------------------------
    def _build_header(self) -> None:
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 8))
        header_frame.pack(fill=tk.X)

        title_box = ttk.Frame(header_frame, style="Header.TFrame")
        title_box.pack(side=tk.LEFT)

        ttk.Label(title_box, text="🌌 GAIA PROJECT — REINFORCEMENT LEARNING STUDIO (SOTA 2026)", style="Header.TLabel").pack(anchor=tk.W)

        hw_info = self.trainer.hw_info
        if hw_info["cuda_available"] and self.trainer.device.type == "cuda":
            dev_str = f"🚀 GPU: {hw_info['device_name']} ({hw_info['vram_total_gb']} GB) | Mixed Precision: {'ON' if hw_info['mixed_precision'] else 'OFF'} | TF32: ON"
            color = "#38bdf8"
        else:
            dev_str = "💻 Mode: CPU (GPU non détecté par PyTorch — installez le wheel CUDA pour activer votre carte)"
            color = "#f59e0b"

        self.lbl_hw_status = tk.Label(
            title_box,
            text=dev_str,
            bg="#0f172a",
            fg=color,
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_hw_status.pack(anchor=tk.W, pady=(2, 0))

        # Status badge & Presets Toolbar on right
        right_box = ttk.Frame(header_frame, style="Header.TFrame")
        right_box.pack(side=tk.RIGHT)

        preset_box = ttk.Frame(right_box, style="Header.TFrame")
        preset_box.pack(side=tk.TOP, anchor=tk.E, pady=(0, 4))

        tk.Label(preset_box, text="⚡ Préréglages :", bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=4)

        btn_pretrain = tk.Button(
            preset_box,
            text="🚀 1. Pré-entraînement Fondation (150 VP)",
            bg="#047857",
            fg="#a7f3d0",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=3,
            command=lambda: self._apply_preset_by_name("pretrain"),
        )
        btn_pretrain.pack(side=tk.LEFT, padx=3)

        btn_finetune = tk.Button(
            preset_box,
            text="👑 2. Fine-Tuning Grand Maître (230+ VP)",
            bg="#6b21a8",
            fg="#f3e8ff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=3,
            command=lambda: self._apply_preset_by_name("finetune"),
        )
        btn_finetune.pack(side=tk.LEFT, padx=3)

        status_row = ttk.Frame(right_box, style="Header.TFrame")
        status_row.pack(side=tk.BOTTOM, anchor=tk.E, pady=(2, 0))

        self.lbl_preset_status = tk.Label(
            status_row,
            text="⚡ Préréglages disponibles (charge uniquement les HP)",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 8, "italic"),
        )
        self.lbl_preset_status.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_train_status = tk.Label(
            status_row,
            text="● Prêt (Idle)",
            bg="#334155",
            fg="#94a3b8",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=3,
        )
        self.lbl_train_status.pack(side=tk.RIGHT)

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 1: Dashboard & Plots
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="📊 Tableau de bord & Courbes")
        self._setup_dashboard_tab()

        # Tab 2: Hyperparameters Configuration (Scrollable)
        self.tab_config = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_config, text="⚙️ Paramétrage & SOTA 2026")
        self._setup_config_tab()

        # Tab 3: Auto Hyperparameter Optimization
        self.tab_hyperopt = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hyperopt, text="🔍 Auto-Tuning Hyperparamètres")
        self._setup_hyperopt_tab()

        # Tab 4: Match Simulator
        self.tab_eval = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_eval, text="🎮 Simulateur de Matchs & MCTS")
        self._setup_eval_tab()

        # Tab 5: Gaia AI Lab & Sandbox
        self.tab_arena = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_arena, text="🚀 Gaia AI Lab & Sandbox")
        self._setup_arena_tab()

    # -------------------------------------------------------------
    # TAB 1: DASHBOARD
    # -------------------------------------------------------------
    def _setup_dashboard_tab(self) -> None:
        toolbar = ttk.Frame(self.tab_dashboard, padding=8)
        toolbar.pack(fill=tk.X)

        self.btn_start = ttk.Button(toolbar, text="▶ Démarrer l'entraînement", style="Action.TButton", command=self._on_start_training)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_pause = ttk.Button(toolbar, text="⏸ Pause", style="Action.TButton", state=tk.DISABLED, command=self._on_pause_training)
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(toolbar, text="⏹ Arrêter", style="Action.TButton", state=tk.DISABLED, command=self._on_stop_training)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="💾 Sauvegarder Modèles", command=self._on_save_checkpoint).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="📂 Charger Modèles", command=self._on_load_checkpoint).pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.btn_report = ttk.Button(toolbar, text="📄 Générer Rapport PDF", command=self._on_generate_pdf_report)
        self.btn_report.pack(side=tk.LEFT, padx=4)

        self.var_auto_report = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Générer PDF en fin", variable=self.var_auto_report).pack(side=tk.LEFT, padx=8)

        # 9 KPI Metrics Cards across 2 rows
        kpi_frame = ttk.Frame(self.tab_dashboard, padding=4)
        kpi_frame.pack(fill=tk.X)

        row1_cards = [
            ("ÉPOQUE / TOTAL", "0 / 1000", "kpi_epochs"),
            ("CADENCE (STEPS/S)", "0", "kpi_speed"),
            ("PERTE PPO / SCORE", "0.000 / 0.000", "kpi_losses"),
            ("R-NaD KL DIV", "0.000", "kpi_rnad"),
            ("RND CURIOSITÉ", "0.000", "kpi_rnd"),
        ]
        row2_cards = [
            ("RGSC PUZZLES", "0 / 200", "kpi_rgsc"),
            ("SCORE MOYEN (VP)", "0.0 (R: 0.0)", "kpi_vp"),
            ("TAUX DE VICTOIRE", "0%", "kpi_winrate"),
            ("ELO LIGUE / POOL", "1200 / 2", "kpi_elo"),
        ]

        self.kpi_labels: Dict[str, tk.Label] = {}

        kpi_r1 = ttk.Frame(kpi_frame)
        kpi_r1.pack(fill=tk.X, pady=2)
        for title, initial, key in row1_cards:
            c = ttk.Frame(kpi_r1, style="Card.TFrame", padding=(8, 4))
            c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            ttk.Label(c, text=title, style="CardTitle.TLabel").pack(anchor=tk.W)
            lbl = tk.Label(c, text=initial, bg="#f8fafc", font=("Segoe UI", 12, "bold"), fg="#0f172a")
            lbl.pack(anchor=tk.W, pady=(1, 0))
            self.kpi_labels[key] = lbl

        kpi_r2 = ttk.Frame(kpi_frame)
        kpi_r2.pack(fill=tk.X, pady=2)
        for title, initial, key in row2_cards:
            c = ttk.Frame(kpi_r2, style="Card.TFrame", padding=(8, 4))
            c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            ttk.Label(c, text=title, style="CardTitle.TLabel").pack(anchor=tk.W)
            lbl = tk.Label(c, text=initial, bg="#f8fafc", font=("Segoe UI", 12, "bold"), fg="#0f172a")
            lbl.pack(anchor=tk.W, pady=(1, 0))
            self.kpi_labels[key] = lbl

        # Live Matplotlib Canvas
        plot_frame = ttk.Frame(self.tab_dashboard, padding=6)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, (self.ax_loss, self.ax_score) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        self.ax_loss_twin = self.ax_loss.twinx()
        self.fig.patch.set_facecolor("#f8fafc")

        self.ax_loss.set_facecolor("#ffffff")
        self.ax_loss.set_title("Pertes d'apprentissage (Politique & Score)", fontsize=10, fontweight="bold", color="#1e293b")
        self.ax_loss.set_xlabel("Époques", fontsize=8)
        self.ax_loss.set_ylabel("Perte Politique", color="#2563eb", fontsize=8, fontweight="bold")
        self.ax_loss.tick_params(axis="y", labelcolor="#2563eb")
        self.ax_loss.grid(True, linestyle="--", alpha=0.5)

        self.ax_loss_twin.set_ylabel("Perte Valeur (VP)", color="#e11d48", fontsize=8, fontweight="bold")
        self.ax_loss_twin.tick_params(axis="y", labelcolor="#e11d48")

        self.ax_score.set_facecolor("#ffffff")
        self.ax_score.set_title("Score moyen : Prédit vs Réel (VP)", fontsize=10, fontweight="bold", color="#1e293b")
        self.ax_score.set_xlabel("Époques", fontsize=8)
        self.ax_score.set_ylabel("Points de Victoire (VP)", fontsize=8, fontweight="bold")
        self.ax_score.grid(True, linestyle="--", alpha=0.5)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------
    # TAB 2: CONFIGURATION (SCROLLABLE FRAME - ALL 14 SOTA TECHNOLOGIES)
    # -------------------------------------------------------------
    def _setup_config_tab(self) -> None:
        outer_container = ttk.Frame(self.tab_config)
        outer_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=12)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mousewheel scroll support (Windows & Linux)
        def _on_mousewheel(event):
            if event.num == 5 or getattr(event, "delta", 0) < 0:
                canvas.yview_scroll(1, "units")
            elif event.num == 4 or getattr(event, "delta", 0) > 0:
                canvas.yview_scroll(-1, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        c = self.config
        col_w = 12

        # 1. Hardware & GPU Acceleration
        sec_hw = ttk.LabelFrame(scrollable_frame, text="1. Accélération Matérielle (NVIDIA RTX / CUDA / CPU)", padding=8)
        sec_hw.pack(fill=tk.X, pady=4)

        ttk.Label(sec_hw, text="Périphérique cible :").grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_device = tk.StringVar(value=c.hardware.device_override)
        ttk.Combobox(sec_hw, textvariable=self.var_device, values=["auto", "cuda:0", "cpu"], state="readonly", width=col_w).grid(row=0, column=1, sticky=tk.W, padx=4)

        self.var_fp16 = tk.BooleanVar(value=c.hardware.use_mixed_precision)
        ttk.Checkbutton(sec_hw, text="Activer la précision mixte (FP16/BF16 AMP)", variable=self.var_fp16).grid(row=0, column=2, padx=12, sticky=tk.W)

        self.var_tf32 = tk.BooleanVar(value=c.hardware.enable_tf32)
        ttk.Checkbutton(sec_hw, text="Activer TF32 (Tensor Cores Ampere/Ada/Blackwell)", variable=self.var_tf32).grid(row=0, column=3, padx=12, sticky=tk.W)

        ttk.Label(sec_hw, text="Algorithme :").grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        default_algo = "AlphaZero" if getattr(c.alphazero, "enabled", False) else ("MuZero" if getattr(c.muzero, "enabled", False) else "PPO")
        self.var_training_algo = tk.StringVar(value=default_algo)
        ttk.Combobox(sec_hw, textvariable=self.var_training_algo, values=["PPO", "AlphaZero", "MuZero"], state="readonly", width=col_w).grid(row=1, column=1, sticky=tk.W, padx=4)
        ttk.Label(sec_hw, text="PPO = Classique | AlphaZero = MCTS+Self-Play | MuZero = Latent MCTS (10x plus rapide)", font=("Segoe UI", 8)).grid(row=1, column=2, columnspan=3, sticky=tk.W, padx=8)

        # 2. Modern Deep RL Architecture
        sec_arch = ttk.LabelFrame(scrollable_frame, text="2. Architecture Réseau & Blocs Modernes", padding=8)
        sec_arch.pack(fill=tk.X, pady=4)

        ttk.Label(sec_arch, text="Type de Bloc :").grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_block_type = tk.StringVar(value=c.model.block_type)
        ttk.Combobox(sec_arch, textvariable=self.var_block_type, values=["pre_ln", "swiglu", "bottleneck", "mlp"], state="readonly", width=col_w).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_arch, text="Activation :").grid(row=0, column=2, sticky=tk.W, padx=8, pady=3)
        self.var_activation = tk.StringVar(value=c.model.policy_activation)
        ttk.Combobox(sec_arch, textvariable=self.var_activation, values=["silu", "gelu", "relu"], state="readonly", width=8).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_arch, text="Dropout :").grid(row=0, column=4, sticky=tk.W, padx=8, pady=3)
        self.var_dropout = tk.DoubleVar(value=c.model.policy_dropout)
        ttk.Entry(sec_arch, textvariable=self.var_dropout, width=6).grid(row=0, column=5, sticky=tk.W, padx=4)

        ttk.Label(sec_arch, text="Couches cachées :").grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        default_layers = ", ".join(str(x) for x in c.model.policy_hidden_layers)
        self.var_layers = tk.StringVar(value=default_layers)
        ttk.Combobox(sec_arch, textvariable=self.var_layers, values=["1024, 1024, 512, 256", "512, 512, 512, 256", "512, 512, 256", "256, 256, 128", "128, 128"], width=24).grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=4)

        self.var_finetune = tk.BooleanVar(value=getattr(c.model, "finetune_mode", False))
        ttk.Checkbutton(sec_arch, text="Mode Fine-Tuning (Gèle le Shared Backbone)", variable=self.var_finetune).grid(row=1, column=3, columnspan=3, sticky=tk.W, padx=12)

        # 3. Spatial HexGNN Map Encoder
        sec_gnn = ttk.LabelFrame(scrollable_frame, text="3. Encodeur Spatial HexGNN (Plateau Hexagonal 200 hexes)", padding=8)
        sec_gnn.pack(fill=tk.X, pady=4)

        self.var_gnn_map = tk.BooleanVar(value=c.model.use_gnn_map)
        ttk.Checkbutton(sec_gnn, text="Activer HexGNN (Message Passing spatial)", variable=self.var_gnn_map).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_gnn, text="Couches GNN :").grid(row=0, column=2, sticky=tk.W, padx=8)
        self.var_gnn_layers = tk.IntVar(value=c.model.gnn_layers)
        ttk.Spinbox(sec_gnn, from_=1, to=6, textvariable=self.var_gnn_layers, width=4).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_gnn, text="Dim. cachée GNN :").grid(row=0, column=4, sticky=tk.W, padx=8)
        self.var_gnn_dim = tk.IntVar(value=c.model.gnn_hidden_dim)
        ttk.Combobox(sec_gnn, textvariable=self.var_gnn_dim, values=[32, 64, 128], state="readonly", width=6).grid(row=0, column=5, sticky=tk.W, padx=4)

        # 4. PPO Actor-Critic Base Parameters
        sec_ppo = ttk.LabelFrame(scrollable_frame, text="4. Algorithme PPO & Paramètres d'Apprentissage", padding=8)
        sec_ppo.pack(fill=tk.X, pady=4)

        ttk.Label(sec_ppo, text="LR Politique :").grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_policy_lr = tk.StringVar(value=str(c.model.policy_lr))
        ttk.Entry(sec_ppo, textvariable=self.var_policy_lr, width=col_w).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="LR Valeur :").grid(row=0, column=2, sticky=tk.W, padx=8)
        self.var_score_lr = tk.StringVar(value=str(c.model.score_lr))
        ttk.Entry(sec_ppo, textvariable=self.var_score_lr, width=col_w).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="Taille Mini-Batch :").grid(row=0, column=4, sticky=tk.W, padx=8)
        self.var_batch = tk.IntVar(value=c.training.batch_size)
        ttk.Combobox(sec_ppo, textvariable=self.var_batch, values=[64, 128, 256, 512, 1024], state="readonly", width=8).grid(row=0, column=5, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="Buffer Rollout :").grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_rollout = tk.IntVar(value=c.training.rollout_steps_per_epoch)
        ttk.Entry(sec_ppo, textvariable=self.var_rollout, width=col_w).grid(row=1, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="Gamma (γ) :").grid(row=1, column=2, sticky=tk.W, padx=8)
        self.var_gamma = tk.DoubleVar(value=c.training.gamma)
        ttk.Entry(sec_ppo, textvariable=self.var_gamma, width=col_w).grid(row=1, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="GAE Lambda (λ) :").grid(row=1, column=4, sticky=tk.W, padx=8)
        self.var_gae = tk.DoubleVar(value=c.training.gae_lambda)
        ttk.Entry(sec_ppo, textvariable=self.var_gae, width=8).grid(row=1, column=5, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="Clip Epsilon (ε) :").grid(row=2, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_clip = tk.DoubleVar(value=c.training.clip_epsilon)
        ttk.Entry(sec_ppo, textvariable=self.var_clip, width=col_w).grid(row=2, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_ppo, text="Époques Cibles :").grid(row=2, column=2, sticky=tk.W, padx=8)
        self.var_max_epochs = tk.IntVar(value=c.training.total_episodes)
        ttk.Entry(sec_ppo, textvariable=self.var_max_epochs, width=col_w).grid(row=2, column=3, sticky=tk.W, padx=4)

        # 5. Dynamic Warmup & Annealing Schedules
        sec_sched = ttk.LabelFrame(scrollable_frame, text="5. Schedules Dynamiques (Warmup & Décroissance Cosinus)", padding=8)
        sec_sched.pack(fill=tk.X, pady=4)

        ttk.Label(sec_sched, text="Schedule LR :").grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_lr_sched = tk.StringVar(value=c.training.lr_schedule_type)
        ttk.Combobox(sec_sched, textvariable=self.var_lr_sched, values=["cosine", "exponential", "linear", "constant"], state="readonly", width=col_w).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_sched, text="Warmup Ratio :").grid(row=0, column=2, sticky=tk.W, padx=8)
        self.var_warmup = tk.DoubleVar(value=c.training.warmup_ratio)
        ttk.Entry(sec_sched, textvariable=self.var_warmup, width=8).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_sched, text="Schedule Entropie :").grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_ent_sched = tk.StringVar(value=c.training.entropy_schedule_type)
        ttk.Combobox(sec_sched, textvariable=self.var_ent_sched, values=["cosine", "exponential", "linear", "constant"], state="readonly", width=col_w).grid(row=1, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_sched, text="Entropie Début / Fin :").grid(row=1, column=2, sticky=tk.W, padx=8)
        self.var_ent_start = tk.DoubleVar(value=c.training.entropy_start)
        self.var_ent_end = tk.DoubleVar(value=c.training.entropy_end)
        ent_sub = ttk.Frame(sec_sched)
        ent_sub.grid(row=1, column=3, columnspan=2, sticky=tk.W)
        ttk.Entry(ent_sub, textvariable=self.var_ent_start, width=6).pack(side=tk.LEFT)
        ttk.Label(ent_sub, text=" -> ").pack(side=tk.LEFT)
        ttk.Entry(ent_sub, textvariable=self.var_ent_end, width=6).pack(side=tk.LEFT)

        # 6. Potential-Based Annealed Reward Shaping
        sec_shaping = ttk.LabelFrame(scrollable_frame, text="6. Annealed Reward Shaping (Milestones de Progression)", padding=8)
        sec_shaping.pack(fill=tk.X, pady=4)

        self.var_shaping_enabled = tk.BooleanVar(value=c.training.shaping_enabled)
        ttk.Checkbutton(sec_shaping, text="Activer le Shaping Décroissant", variable=self.var_shaping_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_shaping, text="Poids Initial :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_shaping_w = tk.DoubleVar(value=c.training.shaping_initial_weight)
        ttk.Entry(sec_shaping, textvariable=self.var_shaping_w, width=8).grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(sec_shaping, text="Taux de Décroissance :").grid(row=0, column=3, sticky=tk.W, padx=8)
        self.var_shaping_decay = tk.DoubleVar(value=c.training.shaping_decay_rate)
        ttk.Entry(sec_shaping, textvariable=self.var_shaping_decay, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        # 7. Random Network Distillation (RND) Intrinsic Curiosity
        sec_rnd = ttk.LabelFrame(scrollable_frame, text="7. Motivation Intrinsèque RND (Random Network Distillation)", padding=8)
        sec_rnd.pack(fill=tk.X, pady=4)

        self.var_rnd_enabled = tk.BooleanVar(value=c.training.rnd_enabled)
        ttk.Checkbutton(sec_rnd, text="Activer Curiosité RND", variable=self.var_rnd_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_rnd, text="Poids RND Initial :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_rnd_w = tk.DoubleVar(value=c.training.rnd_initial_weight)
        ttk.Entry(sec_rnd, textvariable=self.var_rnd_w, width=8).grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(sec_rnd, text="Taux Décroissance RND :").grid(row=0, column=3, sticky=tk.W, padx=8)
        self.var_rnd_decay = tk.DoubleVar(value=c.training.rnd_decay_rate)
        ttk.Entry(sec_rnd, textvariable=self.var_rnd_decay, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        # 8. Opponent Action Modeling Auxiliary Head
        sec_opp = ttk.LabelFrame(scrollable_frame, text="8. Modélisation Prédictive des Adversaires (Auxiliary Head)", padding=8)
        sec_opp.pack(fill=tk.X, pady=4)

        self.var_opp_enabled = tk.BooleanVar(value=c.training.use_opponent_modeling)
        ttk.Checkbutton(sec_opp, text="Activer la Tête Auxiliaire Adversaire (14 actions)", variable=self.var_opp_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_opp, text="Coefficient de Perte :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_opp_coef = tk.DoubleVar(value=c.training.opponent_loss_coef)
        ttk.Entry(sec_opp, textvariable=self.var_opp_coef, width=8).grid(row=0, column=2, sticky=tk.W, padx=4)

        # 9. Regularized Nash Dynamics (R-NaD) Equilibrium
        sec_rnad = ttk.LabelFrame(scrollable_frame, text="9. Régularisation à l'Équilibre de Nash R-NaD (Multi-Player 1v1v1v1)", padding=8)
        sec_rnad.pack(fill=tk.X, pady=4)

        self.var_rnad_enabled = tk.BooleanVar(value=c.training.rnad_enabled)
        ttk.Checkbutton(sec_rnad, text="Activer R-NaD", variable=self.var_rnad_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_rnad, text="Alpha KL :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_rnad_alpha = tk.DoubleVar(value=c.training.rnad_alpha)
        ttk.Entry(sec_rnad, textvariable=self.var_rnad_alpha, width=8).grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(sec_rnad, text="Polyak Beta (β) :").grid(row=0, column=3, sticky=tk.W, padx=8)
        self.var_rnad_beta = tk.DoubleVar(value=c.training.rnad_polyak_beta)
        ttk.Entry(sec_rnad, textvariable=self.var_rnad_beta, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        # 10. Regret-Guided Search Control (RGSC / Go-Exploit)
        sec_rgsc = ttk.LabelFrame(scrollable_frame, text="10. Recherche de Regret RGSC / Go-Exploit (Puzzles de Crise)", padding=8)
        sec_rgsc.pack(fill=tk.X, pady=4)

        self.var_rgsc_enabled = tk.BooleanVar(value=c.training.rgsc_enabled)
        ttk.Checkbutton(sec_rgsc, text="Activer RGSC Go-Exploit", variable=self.var_rgsc_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_rgsc, text="Seuil de Regret (ΔV) :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_rgsc_thresh = tk.DoubleVar(value=c.training.rgsc_regret_threshold)
        ttk.Entry(sec_rgsc, textvariable=self.var_rgsc_thresh, width=8).grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(sec_rgsc, text="Capacité Buffer :").grid(row=0, column=3, sticky=tk.W, padx=8)
        self.var_rgsc_cap = tk.IntVar(value=c.training.rgsc_buffer_capacity)
        ttk.Entry(sec_rgsc, textvariable=self.var_rgsc_cap, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        ttk.Label(sec_rgsc, text="Proba Reset Crise :").grid(row=0, column=5, sticky=tk.W, padx=8)
        self.var_rgsc_prob = tk.DoubleVar(value=c.training.rgsc_reset_prob)
        ttk.Entry(sec_rgsc, textvariable=self.var_rgsc_prob, width=8).grid(row=0, column=6, sticky=tk.W, padx=4)

        # 11. Population League & Gaussian Matchmaking (ZPD)
        sec_league = ttk.LabelFrame(scrollable_frame, text="11. Entraînement en Ligue & Matchmaking Gaussien ZPD", padding=8)
        sec_league.pack(fill=tk.X, pady=4)

        self.var_league_enabled = tk.BooleanVar(value=c.league.enabled)
        ttk.Checkbutton(sec_league, text="Activer la Ligue", variable=self.var_league_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_league, text="Matchmaking :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_league_mode = tk.StringVar(value=c.league.matchmaking_type)
        ttk.Combobox(sec_league, textvariable=self.var_league_mode, values=["gaussian", "uniform"], state="readonly", width=10).grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(sec_league, text="Fenêtre Elo (σ) :").grid(row=0, column=3, sticky=tk.W, padx=8)
        self.var_league_sigma = tk.DoubleVar(value=c.league.matchmaking_elo_window)
        ttk.Entry(sec_league, textvariable=self.var_league_sigma, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        ttk.Label(sec_league, text="Intervalle Snapshots :").grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_league_interval = tk.IntVar(value=c.league.snapshot_interval_epochs)
        ttk.Entry(sec_league, textvariable=self.var_league_interval, width=8).grid(row=1, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_league, text="Max Snapshots :").grid(row=1, column=2, sticky=tk.W, padx=8)
        self.var_league_max = tk.IntVar(value=c.league.max_snapshots)
        ttk.Entry(sec_league, textvariable=self.var_league_max, width=8).grid(row=1, column=3, sticky=tk.W, padx=4)

        # 12. Asynchronous Distributed APPO Architecture
        sec_appo = ttk.LabelFrame(scrollable_frame, text="12. Entraînement Asynchrone Distribué APPO (Multi-Processus)", padding=8)
        sec_appo.pack(fill=tk.X, pady=4)

        self.var_appo_enabled = tk.BooleanVar(value=c.async_dist.enabled)
        ttk.Checkbutton(sec_appo, text="Activer Entraînement Asynchrone APPO", variable=self.var_appo_enabled).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_appo, text="Nombre d'Actors (CPU) :").grid(row=0, column=1, sticky=tk.W, padx=8)
        self.var_appo_actors = tk.IntVar(value=c.async_dist.num_actors)
        ttk.Spinbox(sec_appo, from_=2, to=16, textvariable=self.var_appo_actors, width=6).grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(sec_appo, text="Taille File IPC :").grid(row=0, column=3, sticky=tk.W, padx=8)
        self.var_appo_queue = tk.IntVar(value=c.async_dist.queue_max_size)
        ttk.Entry(sec_appo, textvariable=self.var_appo_queue, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        # 13. MCTS Lookahead SOTA 2026 (Gumbel GAZ & Epistemic Uncertainty)
        sec_mcts = ttk.LabelFrame(scrollable_frame, text="13. MCTS Lookahead SOTA 2026 (Inférence & Analyse Stratégique)", padding=8)
        sec_mcts.pack(fill=tk.X, pady=4)

        ttk.Label(sec_mcts, text="Algorithme MCTS :").grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_mcts_algo = tk.StringVar(value=c.mcts.algorithm)
        ttk.Combobox(sec_mcts, textvariable=self.var_mcts_algo, values=["gumbel", "puct"], state="readonly", width=10).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(sec_mcts, text="Simulations :").grid(row=0, column=2, sticky=tk.W, padx=8)
        self.var_mcts_sims_cfg = tk.IntVar(value=c.mcts.num_simulations)
        ttk.Combobox(sec_mcts, textvariable=self.var_mcts_sims_cfg, values=[8, 16, 25, 50, 100], state="readonly", width=6).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_mcts, text="Candidats Gumbel :").grid(row=0, column=4, sticky=tk.W, padx=8)
        self.var_mcts_cand = tk.IntVar(value=c.mcts.gumbel_candidates)
        ttk.Combobox(sec_mcts, textvariable=self.var_mcts_cand, values=[2, 4, 8], state="readonly", width=4).grid(row=0, column=5, sticky=tk.W, padx=4)

        self.var_mcts_unc = tk.BooleanVar(value=c.mcts.use_epistemic_uncertainty)
        ttk.Checkbutton(sec_mcts, text="Guidage par Incertitude Épistémique (MC-Dropout)", variable=self.var_mcts_unc).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

        self.var_mcts_gating = tk.BooleanVar(value=c.mcts.adaptive_budget_enabled)
        ttk.Checkbutton(sec_mcts, text="Budget Adaptatif (Entropy Gating)", variable=self.var_mcts_gating).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=8)

        # 14. Value-Guided Micro-Dispatch SOTA 2026 (Sélection Fine Hexagone / Recherche / Booster)
        sec_micro = ttk.LabelFrame(scrollable_frame, text="14. Micro-Dispatch Guidé par la Valeur (Placement d'Hexagone & Recherche)", padding=8)
        sec_micro.pack(fill=tk.X, pady=4)

        micro_cfg = getattr(c, "micro_dispatch", None)
        self.var_micro_enabled = tk.BooleanVar(value=getattr(micro_cfg, "enabled", True))
        ttk.Checkbutton(sec_micro, text="Activer le Micro-Dispatch Guidé par la Valeur (ScorePredictorNet)", variable=self.var_micro_enabled).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=4, pady=3)

        ttk.Label(sec_micro, text="Candidats Évalués :").grid(row=0, column=2, sticky=tk.W, padx=8)
        self.var_micro_candidates = tk.IntVar(value=getattr(micro_cfg, "num_candidates", 4))
        ttk.Combobox(sec_micro, textvariable=self.var_micro_candidates, values=[2, 4, 8], state="readonly", width=4).grid(row=0, column=3, sticky=tk.W, padx=4)

        ttk.Label(sec_micro, text="Température Exploration :").grid(row=0, column=4, sticky=tk.W, padx=8)
        self.var_micro_temp = tk.DoubleVar(value=getattr(micro_cfg, "temperature", 0.0))
        ttk.Combobox(sec_micro, textvariable=self.var_micro_temp, values=[0.0, 0.05, 0.10, 0.20], state="readonly", width=6).grid(row=0, column=5, sticky=tk.W, padx=4)

        ttk.Label(sec_micro, text="Setup Initial :").grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        self.var_micro_setup = tk.StringVar(value=getattr(micro_cfg, "setup_mode", "value_guided"))
        ttk.Combobox(sec_micro, textvariable=self.var_micro_setup, values=["value_guided", "random", "deterministic"], state="readonly", width=14).grid(row=1, column=1, sticky=tk.W, padx=4)

        self.var_micro_booster = tk.BooleanVar(value=getattr(micro_cfg, "initial_booster_draft_enabled", True))
        ttk.Checkbutton(sec_micro, text="Draft Initial Boosters (Ordre Inversé)", variable=self.var_micro_booster).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=8)

        self.var_micro_tech = tk.BooleanVar(value=getattr(micro_cfg, "tech_tile_dispatch_enabled", True))
        ttk.Checkbutton(sec_micro, text="Micro-Dispatch Tuiles Techno & Fédérations", variable=self.var_micro_tech).grid(row=1, column=4, columnspan=2, sticky=tk.W, padx=8)

        # Apply button
        btn_apply = ttk.Button(scrollable_frame, text="✔ Appliquer et Synchroniser tous les Paramètres", command=self._on_apply_config, style="Primary.TButton")
        btn_apply.pack(pady=12, anchor=tk.W)

    def _sync_config_to_ui(self) -> None:
        """Populates all UI input variables from current self.config."""
        c = self.config
        self.var_device.set(c.hardware.device_override)
        self.var_fp16.set(c.hardware.use_mixed_precision)
        self.var_tf32.set(c.hardware.enable_tf32)

        if getattr(c.alphazero, "enabled", False):
            self.var_training_algo.set("AlphaZero")
        elif getattr(c.muzero, "enabled", False):
            self.var_training_algo.set("MuZero")
        else:
            self.var_training_algo.set("PPO")

        self.var_block_type.set(c.model.block_type)
        self.var_activation.set(c.model.policy_activation)
        self.var_dropout.set(c.model.policy_dropout)
        self.var_layers.set(", ".join(str(x) for x in c.model.policy_hidden_layers))
        self.var_finetune.set(getattr(c.model, "finetune_mode", False))

        self.var_gnn_map.set(c.model.use_gnn_map)
        self.var_gnn_layers.set(c.model.gnn_layers)
        self.var_gnn_dim.set(c.model.gnn_hidden_dim)

        self.var_policy_lr.set(str(c.model.policy_lr))
        self.var_score_lr.set(str(c.model.score_lr))
        self.var_batch.set(c.training.batch_size)
        self.var_rollout.set(c.training.rollout_steps_per_epoch)
        self.var_gamma.set(c.training.gamma)
        self.var_gae.set(c.training.gae_lambda)
        self.var_clip.set(c.training.clip_epsilon)
        self.var_max_epochs.set(c.training.total_episodes)

        self.var_lr_sched.set(c.training.lr_schedule_type)
        self.var_warmup.set(c.training.warmup_ratio)
        self.var_ent_sched.set(c.training.entropy_schedule_type)
        self.var_ent_start.set(c.training.entropy_start)
        self.var_ent_end.set(c.training.entropy_end)

        self.var_shaping_enabled.set(c.training.shaping_enabled)
        self.var_shaping_w.set(c.training.shaping_initial_weight)
        self.var_shaping_decay.set(c.training.shaping_decay_rate)

        self.var_rnd_enabled.set(c.training.rnd_enabled)
        self.var_rnd_w.set(c.training.rnd_initial_weight)
        self.var_rnd_decay.set(c.training.rnd_decay_rate)

        self.var_opp_enabled.set(c.training.use_opponent_modeling)
        self.var_opp_coef.set(c.training.opponent_loss_coef)

        self.var_rnad_enabled.set(c.training.rnad_enabled)
        self.var_rnad_alpha.set(c.training.rnad_alpha)
        self.var_rnad_beta.set(c.training.rnad_polyak_beta)

        self.var_rgsc_enabled.set(c.training.rgsc_enabled)
        self.var_rgsc_thresh.set(c.training.rgsc_regret_threshold)
        self.var_rgsc_cap.set(c.training.rgsc_buffer_capacity)
        self.var_rgsc_prob.set(c.training.rgsc_reset_prob)

        self.var_league_enabled.set(c.league.enabled)
        self.var_league_mode.set(c.league.matchmaking_type)
        self.var_league_sigma.set(c.league.matchmaking_elo_window)
        self.var_league_interval.set(c.league.snapshot_interval_epochs)
        self.var_league_max.set(c.league.max_snapshots)

        self.var_appo_enabled.set(c.async_dist.enabled)
        self.var_appo_actors.set(c.async_dist.num_actors)
        self.var_appo_queue.set(c.async_dist.queue_max_size)

        self.var_mcts_algo.set(c.mcts.algorithm)
        self.var_mcts_sims_cfg.set(c.mcts.num_simulations)
        self.var_mcts_cand.set(c.mcts.gumbel_candidates)
        self.var_mcts_unc.set(c.mcts.use_epistemic_uncertainty)
        self.var_mcts_gating.set(c.mcts.adaptive_budget_enabled)

        micro_cfg = getattr(c, "micro_dispatch", None)
        self.var_micro_enabled.set(getattr(micro_cfg, "enabled", True))
        self.var_micro_candidates.set(getattr(micro_cfg, "num_candidates", 4))
        self.var_micro_temp.set(getattr(micro_cfg, "temperature", 0.0))
        if hasattr(self, "var_micro_setup"):
            self.var_micro_setup.set(getattr(micro_cfg, "setup_mode", "value_guided"))
            self.var_micro_booster.set(getattr(micro_cfg, "initial_booster_draft_enabled", True))
            self.var_micro_tech.set(getattr(micro_cfg, "tech_tile_dispatch_enabled", True))

    def _apply_preset_by_name(self, name: str) -> None:
        """Loads and syncs a pre-calibrated preset's hyperparameters into UI without blocking popups or resetting models."""
        preset = get_training_preset(name)
        self.config = preset
        self._sync_config_to_ui()
        self.hyperopt.base_config = deepcopy(self.config)

        labels = {
            "pretrain": "🚀 1. Pré-entraînement Fondation (150 VP)",
            "finetune": "👑 2. Fine-Tuning Grand Maître (230+ VP)",
            "fast": "⚡ Test Rapide",
        }
        name_str = labels.get(name, name.upper())
        if hasattr(self, "lbl_preset_status"):
            self.lbl_preset_status.config(
                text=f"✓ Préréglage appliqué : {name_str}",
                fg="#38bdf8",
            )

    def _save_ui_to_config(self) -> None:
        """Saves current UI variable states into self.config and updates hyperopt baseline."""
        c = self.config
        c.hardware.device_override = self.var_device.get()
        c.hardware.use_mixed_precision = self.var_fp16.get()
        c.hardware.enable_tf32 = self.var_tf32.get()

        algo = self.var_training_algo.get()
        c.alphazero.enabled = (algo == "AlphaZero")
        c.muzero.enabled = (algo == "MuZero")

        c.model.block_type = self.var_block_type.get()
        c.model.policy_activation = self.var_activation.get()
        c.model.score_activation = self.var_activation.get()
        c.model.policy_dropout = self.var_dropout.get()
        c.model.score_dropout = self.var_dropout.get()
        raw_layers = self.var_layers.get().split(",")
        layers = [int(x.strip()) for x in raw_layers if x.strip().isdigit()]
        if layers:
            c.model.policy_hidden_layers = layers
            c.model.score_hidden_layers = layers
            
        c.model.finetune_mode = self.var_finetune.get()

        c.model.use_gnn_map = self.var_gnn_map.get()
        c.model.gnn_layers = self.var_gnn_layers.get()
        c.model.gnn_hidden_dim = self.var_gnn_dim.get()

        c.model.policy_lr = float(self.var_policy_lr.get())
        c.model.score_lr = float(self.var_score_lr.get())
        c.training.batch_size = self.var_batch.get()
        c.training.rollout_steps_per_epoch = self.var_rollout.get()
        c.training.gamma = self.var_gamma.get()
        c.training.gae_lambda = self.var_gae.get()
        c.training.clip_epsilon = self.var_clip.get()
        c.training.total_episodes = self.var_max_epochs.get()

        c.training.lr_schedule_type = self.var_lr_sched.get()
        c.training.warmup_ratio = self.var_warmup.get()
        c.training.entropy_schedule_type = self.var_ent_sched.get()
        c.training.entropy_start = self.var_ent_start.get()
        c.training.entropy_end = self.var_ent_end.get()

        c.training.shaping_enabled = self.var_shaping_enabled.get()
        c.training.shaping_initial_weight = self.var_shaping_w.get()
        c.training.shaping_decay_rate = self.var_shaping_decay.get()

        c.training.rnd_enabled = self.var_rnd_enabled.get()
        c.training.rnd_initial_weight = self.var_rnd_w.get()
        c.training.rnd_decay_rate = self.var_rnd_decay.get()

        c.training.use_opponent_modeling = self.var_opp_enabled.get()
        c.training.opponent_loss_coef = self.var_opp_coef.get()

        c.training.rnad_enabled = self.var_rnad_enabled.get()
        c.training.rnad_alpha = self.var_rnad_alpha.get()
        c.training.rnad_polyak_beta = self.var_rnad_beta.get()

        c.training.rgsc_enabled = self.var_rgsc_enabled.get()
        c.training.rgsc_regret_threshold = self.var_rgsc_thresh.get()
        c.training.rgsc_buffer_capacity = self.var_rgsc_cap.get()
        c.training.rgsc_reset_prob = self.var_rgsc_prob.get()

        c.league.enabled = self.var_league_enabled.get()
        c.league.matchmaking_type = self.var_league_mode.get()
        c.league.matchmaking_elo_window = self.var_league_sigma.get()
        c.league.snapshot_interval_epochs = self.var_league_interval.get()
        c.league.max_snapshots = self.var_league_max.get()

        c.async_dist.enabled = self.var_appo_enabled.get()
        c.async_dist.num_actors = self.var_appo_actors.get()
        c.async_dist.queue_max_size = self.var_appo_queue.get()

        c.mcts.algorithm = self.var_mcts_algo.get()
        c.mcts.num_simulations = self.var_mcts_sims_cfg.get()
        c.mcts.gumbel_candidates = self.var_mcts_cand.get()
        c.mcts.use_epistemic_uncertainty = self.var_mcts_unc.get()
        c.mcts.adaptive_budget_enabled = self.var_mcts_gating.get()

        if hasattr(c, "micro_dispatch"):
            c.micro_dispatch.enabled = self.var_micro_enabled.get()
            c.micro_dispatch.num_candidates = self.var_micro_candidates.get()
            c.micro_dispatch.temperature = self.var_micro_temp.get()
            if hasattr(self, "var_micro_setup"):
                c.micro_dispatch.setup_mode = self.var_micro_setup.get()
                c.micro_dispatch.initial_booster_draft_enabled = self.var_micro_booster.get()
                c.micro_dispatch.tech_tile_dispatch_enabled = self.var_micro_tech.get()
                c.micro_dispatch.federation_dispatch_enabled = self.var_micro_tech.get()

        # Synchronize hyperopt baseline directly with current UI variables
        self.hyperopt.base_config = deepcopy(self.config)

    def _on_apply_config(self) -> None:
        try:
            self._save_ui_to_config()
            self.trainer = self._create_trainer()
            messagebox.showinfo("Succès", "Toutes les configurations SOTA 2026 ont été appliquées avec succès !")
        except Exception as e:
            messagebox.showerror("Erreur de configuration", f"Valeur invalide détectée : {e}")

    # -------------------------------------------------------------
    # TAB 3: HYPEROPT
    # -------------------------------------------------------------
    def _setup_hyperopt_tab(self) -> None:
        ctrl_frame = ttk.Frame(self.tab_hyperopt, padding=12)
        ctrl_frame.pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="Nombre d'essais (Trials) :").pack(side=tk.LEFT, padx=4)
        self.var_opt_trials = tk.IntVar(value=5)
        ttk.Spinbox(ctrl_frame, from_=2, to=30, textvariable=self.var_opt_trials, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl_frame, text="Époques par essai :").pack(side=tk.LEFT, padx=(16, 4))
        self.var_opt_sprint = tk.IntVar(value=6)
        ttk.Spinbox(ctrl_frame, from_=2, to=20, textvariable=self.var_opt_sprint, width=5).pack(side=tk.LEFT, padx=4)

        self.btn_run_opt = ttk.Button(ctrl_frame, text="🚀 Lancer la recherche", command=self._on_run_hyperopt, style="Action.TButton")
        self.btn_run_opt.pack(side=tk.LEFT, padx=16)

        self.btn_stop_opt = ttk.Button(ctrl_frame, text="⏹ Interrompre", command=self._on_stop_hyperopt, state=tk.DISABLED)
        self.btn_stop_opt.pack(side=tk.LEFT, padx=4)

        self.btn_apply_best = ttk.Button(ctrl_frame, text="🏆 Appliquer le meilleur essai", command=self._on_apply_best_trial, state=tk.DISABLED)
        self.btn_apply_best.pack(side=tk.RIGHT, padx=4)

        hint_banner = tk.Label(
            self.tab_hyperopt,
            text="💡 L'Essai #1 teste automatiquement vos HP actifs (préréglage ou configuration enregistrée). Les essais suivants optimisent par raffinement Bayésien (TPE) autour de cette base pour un gain de temps maximal.",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9),
            padx=10,
            pady=4,
            relief=tk.FLAT,
        )
        hint_banner.pack(fill=tk.X, padx=12, pady=(0, 6))

        table_frame = ttk.Frame(self.tab_hyperopt, padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("id", "arch", "policy_lr", "batch", "sched", "val_loss", "win_rate", "score_vp", "score_obj", "status")
        self.tree_trials = ttk.Treeview(table_frame, columns=cols, show="headings", height=15)
        self.tree_trials.heading("id", text="Essai #")
        self.tree_trials.heading("arch", text="Architecture (Bloc + Couches + Act)")
        self.tree_trials.heading("policy_lr", text="LR")
        self.tree_trials.heading("batch", text="Batch")
        self.tree_trials.heading("sched", text="Schedules LR / H")
        self.tree_trials.heading("val_loss", text="Perte Val.")
        self.tree_trials.heading("win_rate", text="Victoires")
        self.tree_trials.heading("score_vp", text="Score VP")
        self.tree_trials.heading("score_obj", text="Score NAS")
        self.tree_trials.heading("status", text="Statut")

        col_widths = {
            "id": 55,
            "arch": 220,
            "policy_lr": 75,
            "batch": 60,
            "sched": 125,
            "val_loss": 75,
            "win_rate": 70,
            "score_vp": 75,
            "score_obj": 75,
            "status": 115,
        }
        for c_name in cols:
            self.tree_trials.column(c_name, width=col_widths.get(c_name, 90), anchor=tk.CENTER)

        self.tree_trials.pack(fill=tk.BOTH, expand=True)

    def _on_run_hyperopt(self) -> None:
        self._save_ui_to_config()
        self.btn_run_opt.config(state=tk.DISABLED)
        self.btn_stop_opt.config(state=tk.NORMAL)
        for row in self.tree_trials.get_children():
            self.tree_trials.delete(row)

        num_trials = self.var_opt_trials.get()
        sprint = self.var_opt_sprint.get()

        def _on_trial_update(t: HyperoptTrial):
            self.hyperopt_queue.put(("update", t))

        def _on_finished(best: Optional[HyperoptTrial]):
            self.hyperopt_queue.put(("finished", best))

        self.hyperopt.run_optimization(
            num_trials=num_trials,
            sprint_epochs=sprint,
            on_trial_update=_on_trial_update,
            on_finished=_on_finished,
        )

    def _on_stop_hyperopt(self) -> None:
        self.hyperopt.stop()
        self.btn_run_opt.config(state=tk.NORMAL)
        self.btn_stop_opt.config(state=tk.DISABLED)

    def _on_apply_best_trial(self) -> None:
        if not self.hyperopt.best_trial:
            return
        p = self.hyperopt.best_trial.params
        self.var_policy_lr.set(str(p["policy_lr"]))
        self.var_score_lr.set(str(p["score_lr"]))
        self.var_batch.set(p["batch_size"])
        self.var_ent_start.set(p.get("entropy_start", p.get("entropy_coef", 0.04)))
        self.var_clip.set(p["clip_epsilon"])
        self.var_layers.set(", ".join(map(str, p["hidden_layers"])))

        self.config.model.block_type = p.get("block_type", "pre_ln")
        self.config.model.policy_activation = p.get("activation", "silu")
        self.config.model.score_activation = p.get("activation", "silu")
        self.config.model.policy_dropout = p.get("dropout", 0.05)
        self.config.model.score_dropout = p.get("dropout", 0.05)

        self.config.training.lr_schedule_type = p.get("lr_schedule_type", "cosine")
        self.config.training.warmup_ratio = p.get("warmup_ratio", 0.05)
        self.config.training.lr_final_factor = p.get("lr_final_factor", 0.1)
        self.config.training.entropy_schedule_type = p.get("entropy_schedule_type", "cosine")
        self.config.training.entropy_start = p.get("entropy_start", 0.05)
        self.config.training.entropy_end = p.get("entropy_end", 0.005)

        # SOTA Components sync
        if "use_gnn_map" in p and hasattr(self, "var_gnn_map"):
            self.var_gnn_map.set(p["use_gnn_map"])
            self.config.model.use_gnn_map = p["use_gnn_map"]
        if "gnn_layers" in p and hasattr(self, "var_gnn_layers"):
            self.var_gnn_layers.set(p["gnn_layers"])
            self.config.model.gnn_layers = p["gnn_layers"]
        if "gnn_hidden_dim" in p and hasattr(self, "var_gnn_dim"):
            self.var_gnn_dim.set(p["gnn_hidden_dim"])
            self.config.model.gnn_hidden_dim = p["gnn_hidden_dim"]
        if "rnad_alpha" in p and hasattr(self, "var_rnad_alpha"):
            self.var_rnad_alpha.set(p["rnad_alpha"])
            self.config.training.rnad_alpha = p["rnad_alpha"]
        if "rnad_polyak_beta" in p and hasattr(self, "var_rnad_beta"):
            self.var_rnad_beta.set(p["rnad_polyak_beta"])
            self.config.training.rnad_polyak_beta = p["rnad_polyak_beta"]
        if "rnd_initial_weight" in p and hasattr(self, "var_rnd_w"):
            self.var_rnd_w.set(p["rnd_initial_weight"])
            self.config.training.rnd_initial_weight = p["rnd_initial_weight"]
        if "rgsc_regret_threshold" in p and hasattr(self, "var_rgsc_thresh"):
            self.var_rgsc_thresh.set(p["rgsc_regret_threshold"])
            self.config.training.rgsc_regret_threshold = p["rgsc_regret_threshold"]
        if "rgsc_reset_prob" in p and hasattr(self, "var_rgsc_prob"):
            self.var_rgsc_prob.set(p["rgsc_reset_prob"])
            self.config.training.rgsc_reset_prob = p["rgsc_reset_prob"]

        self._on_apply_config()
        arch = self.hyperopt.best_trial.architecture_name
        messagebox.showinfo(
            "Architecture & Paramètres Appliqués",
            f"L'architecture optimale '{arch}' et tous ses paramètres ont été synchronisés !",
        )

    # -------------------------------------------------------------
    # TAB 4: EVALUATION & SIMULATOR WITH MCTS 2026
    # -------------------------------------------------------------
    def _setup_eval_tab(self) -> None:
        top = ttk.Frame(self.tab_eval, padding=10)
        top.pack(fill=tk.X)

        ttk.Button(top, text="▶ Jouer un match de démonstration", command=self._on_simulate_match, style="Primary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="🏆 Classement Ligue (Elo)", command=self._on_show_league_leaderboard).pack(side=tk.LEFT, padx=6)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # MCTS Lookahead controls
        ttk.Label(top, text="Moteur :").pack(side=tk.LEFT, padx=2)
        self.var_sim_engine = tk.StringVar(value="gumbel")
        ttk.Combobox(top, textvariable=self.var_sim_engine, values=["gumbel", "puct", "direct"], state="readonly", width=8).pack(side=tk.LEFT, padx=3)

        ttk.Label(top, text="Sims :").pack(side=tk.LEFT, padx=2)
        self.var_sim_budget = tk.IntVar(value=16)
        ttk.Combobox(top, textvariable=self.var_sim_budget, values=[8, 16, 25, 50, 100], width=4, state="readonly").pack(side=tk.LEFT, padx=3)

        self.var_sim_unc = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="MC-Dropout Guidage", variable=self.var_sim_unc).pack(side=tk.LEFT, padx=6)

        self.var_sim_gating = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Entropy Gating", variable=self.var_sim_gating).pack(side=tk.LEFT, padx=6)

        self.lbl_sim_result = ttk.Label(top, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_sim_result.pack(side=tk.RIGHT, padx=8)

        # Move-by-move log box
        log_frame = ttk.Frame(self.tab_eval, padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_eval_log = tk.Text(log_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#0f172a", fg="#f8fafc", padx=10, pady=10)
        self.txt_eval_log.pack(fill=tk.BOTH, expand=True)

    def _on_show_league_leaderboard(self) -> None:
        self.txt_eval_log.delete("1.0", tk.END)
        leaderboard = self.trainer.league_manager.get_leaderboard()
        self.txt_eval_log.insert(tk.END, "=== CLASSEMENT DE LA LIGUE (POPULATION & HISTORIQUE GAUSSIEN) ===\n\n")
        self.txt_eval_log.insert(
            tk.END,
            f"{'Rang':<6} {'Agent / Snapshot':<32} {'Elo':<10} {'Parties':<10} {'Victoires':<12} {'Ratio':<10}\n"
        )
        self.txt_eval_log.insert(tk.END, "-" * 82 + "\n")
        for rank, m in enumerate(leaderboard, start=1):
            name_disp = m["name"]
            if m.get("is_current"):
                name_disp += " (Actif)"
            elif m.get("is_random"):
                name_disp += " (Baseline)"
            self.txt_eval_log.insert(
                tk.END,
                f"#{rank:<5} {name_disp:<32} {m['elo']:<10.1f} {m['games']:<10} {m['wins']:<12} {m['win_rate']:<10}\n"
            )
        self.txt_eval_log.insert(tk.END, "\n" + "=" * 82 + "\n")

    def _on_simulate_match(self) -> None:
        self.txt_eval_log.delete("1.0", tk.END)
        env = make_gaia_env(players=self.trainer.config.model.num_players)
        obs, mask = env.reset()

        engine_choice = self.var_sim_engine.get()
        sim_budget = self.var_sim_budget.get()
        use_unc = self.var_sim_unc.get()
        use_gating = self.var_sim_gating.get()

        mcts_engine = None
        if engine_choice in ["gumbel", "puct"]:
            mcts_cfg = deepcopy(self.config.mcts)
            mcts_cfg.algorithm = engine_choice
            mcts_cfg.num_simulations = sim_budget
            mcts_cfg.use_epistemic_uncertainty = use_unc
            mcts_cfg.adaptive_budget_enabled = use_gating
            mcts_engine = MultiPlayerMCTS(self.trainer.agent, config=mcts_cfg, device=self.trainer.device)

        p_count = len(env.players_state)
        algo_title = f"MCTS {engine_choice.upper()} ({sim_budget} sims, unc={use_unc})" if mcts_engine else "Politique Réflexe Directe"
        self.txt_eval_log.insert(tk.END, f"=== DÉBUT DU MATCH ({p_count} Joueurs | IA P0 : {algo_title}) ===\n\n")

        step_count = 0
        while not env.terminated and step_count < 150:
            step_count += 1
            curr_p = env.current_player
            round_num = env.round

            obs_t = torch.from_numpy(obs).float().to(self.trainer.device)
            mask_t = torch.from_numpy(mask).bool().to(self.trainer.device)

            if curr_p == 0:
                if mcts_engine is not None:
                    action, probs, meta = mcts_engine.search(env, num_simulations=sim_budget, temperature=0.0)
                    with torch.no_grad():
                        if use_unc:
                            pred_vp, unc = self.trainer.agent.predict_score_with_uncertainty(obs_t.unsqueeze(0), num_passes=3)
                        else:
                            pred_vp = float(self.trainer.agent.score_net(obs_t.unsqueeze(0)).item())
                            unc = 0.0
                    act_name = format_flat_action(action)
                    child_summary = ", ".join([f"{format_flat_action(a)[:10]}:{v}" for a, v in sorted(meta.get('child_visits', {}).items()) if v > 0])
                    unc_str = f" | Incertitude σ={unc:.3f}" if use_unc else ""
                    self.txt_eval_log.insert(
                        tk.END,
                        f"[Manche {round_num} | Tour {step_count:02d}] 🧠 IA (P0) [{engine_choice.upper()} {meta.get('root_visits', sim_budget)} sims{unc_str}] -> '{act_name}'\n"
                        f"    └─ Score estimé : {pred_vp:.1f} VP | Arbre : [{child_summary}]\n\n"
                    )
                else:
                    action, _, pred_vp, probs = self.trainer.agent.act_and_evaluate(obs_t, mask_t, deterministic=True)
                    act_name = format_flat_action(action)
                    top_acts = sorted(enumerate(probs), key=lambda x: float(x[1]), reverse=True)[:3]
                    prob_str = ", ".join([f"{format_flat_action(i)[:12]}:{float(p):.2f}" for i, p in top_acts if float(p) > 0.005])
                    self.txt_eval_log.insert(
                        tk.END,
                        f"[Manche {round_num} | Tour {step_count:02d}] 🤖 IA (P0) choisit '{act_name}'\n"
                        f"    └─ Score estimé : {pred_vp:.1f} VP | Probas : [{prob_str}]\n\n"
                    )
            else:
                legal_indices = np.where(mask)[0]
                action = int(np.random.choice(legal_indices))
                act_name = format_flat_action(action)
                self.txt_eval_log.insert(
                    tk.END,
                    f"[Manche {round_num} | Tour {step_count:02d}] 🎲 Aléatoire (P{curr_p}) joue '{act_name}'\n\n"
                )

            res = env.step(action)
            obs = res.obs
            mask = res.action_mask

        vps = [p["vp"] for p in env.players_state]
        best_vp = max(vps) if vps else 0.0
        best_seats = [i for i, v in enumerate(vps) if v == best_vp]
        if 0 in best_seats and len(best_seats) == 1:
            winner = "IA (Joueur 0)"
        elif 0 in best_seats and len(best_seats) > 1:
            winner = "Égalité (avec IA P0)"
        else:
            winner = f"Joueur {best_seats[0]}"

        score_lines = " | ".join([f"P{i}: {vps[i]:.0f} VP" for i in range(len(vps))])
        self.txt_eval_log.insert(
            tk.END,
            f"=== FIN DE LA PARTIE ===\n"
            f"Scores finaux : {score_lines}\n"
            f"Vainqueur : {winner} !\n"
        )
        self.lbl_sim_result.config(text=f"Résultat : {winner} ({score_lines})")

    # -------------------------------------------------------------
    # TRAINING CONTROLS & EVENT LOOP
    # -------------------------------------------------------------
    def _on_start_training(self) -> None:
        self._save_ui_to_config()
        self.trainer = self._create_trainer()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="⏸ Pause")
        self.btn_stop.config(state=tk.NORMAL)
        algo_name = self.var_training_algo.get()
        self.lbl_train_status.config(text=f"● Entraînement {algo_name} en cours...", bg="#16a34a", fg="#ffffff")

        def _on_metrics(m):
            self.metrics_queue.put(m)

        def _on_finish():
            self.metrics_queue.put("FINISHED")

        max_epochs = self.var_max_epochs.get()

        if algo_name in ("AlphaZero", "MuZero"):
            # AlphaZero/MuZero use run_training_loop with a callback
            import threading
            env = make_gaia_env(players=self.config.model.num_players)
            def _az_thread():
                try:
                    self.trainer.run_training_loop(env, max_epochs=max_epochs, callback=_on_metrics)
                finally:
                    _on_finish()
            t = threading.Thread(target=_az_thread, daemon=True)
            t.start()
        else:
            self.trainer.start_background_training(
                on_metrics=_on_metrics,
                on_finished=_on_finish,
                max_epochs=max_epochs,
            )

    def _on_pause_training(self) -> None:
        if self.trainer.is_paused():
            self.trainer.resume()
            self.btn_pause.config(text="⏸ Pause")
            self.lbl_train_status.config(text="● Entraînement en cours...", bg="#16a34a", fg="#ffffff")
        else:
            self.trainer.pause()
            self.btn_pause.config(text="▶ Reprendre")
            self.lbl_train_status.config(text="⏸ En pause", bg="#eab308", fg="#000000")

    def _on_stop_training(self) -> None:
        self.trainer.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_train_status.config(text="● Arrêté", bg="#334155", fg="#94a3b8")

    def _on_save_checkpoint(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch Model", "*.pt")])
        if path:
            self.trainer.save_current_checkpoint(path)
            messagebox.showinfo("Sauvegarde", f"Modèles sauvegardés dans {os.path.basename(path)}")

    def _on_load_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if path:
            filename = os.path.basename(path).lower()
            if "az_" in filename or "alphazero" in filename:
                if hasattr(self, "var_training_algo"):
                    self.var_training_algo.set("AlphaZero")
                self.trainer = self._create_trainer()
            elif "muzero" in filename:
                if hasattr(self, "var_training_algo"):
                    self.var_training_algo.set("MuZero")
                self.trainer = self._create_trainer()
            epoch = self.trainer.resume_from_checkpoint(path)
            if hasattr(self.trainer, "agent") and hasattr(self.trainer.agent, "config"):
                self.config.model = self.trainer.agent.config
            if hasattr(self.trainer, "config"):
                self.config = self.trainer.config
            self._sync_config_to_ui()
            if hasattr(self, "hyperopt"):
                self.hyperopt.base_config = deepcopy(self.config)

            if hasattr(self, "kpi_labels") and "kpi_epochs" in self.kpi_labels:
                self.kpi_labels["kpi_epochs"].config(text=f"{epoch} / {self.var_max_epochs.get()}")
            messagebox.showinfo("Chargement", f"Modèles chargés avec succès (Reprise à l'Époque {epoch})")

    def _start_polling(self) -> None:
        self._process_queues()
        self.root.after(100, self._start_polling)

    def _process_queues(self) -> None:
        # 1. Training metrics queue
        while not self.metrics_queue.empty():
            item = self.metrics_queue.get_nowait()
            if item == "FINISHED":
                self._on_stop_training()
                if getattr(self, "var_auto_report", None) and self.var_auto_report.get():
                    self._on_generate_pdf_report()
                continue

            m = item
            # Update KPI cards
            if "kpi_epochs" in self.kpi_labels:
                self.kpi_labels["kpi_epochs"].config(text=f"{m.epoch} / {self.var_max_epochs.get()}")
            if "kpi_speed" in self.kpi_labels:
                self.kpi_labels["kpi_speed"].config(text=f"{getattr(m, 'steps_per_sec', 0.0):.0f}")
            if "kpi_losses" in self.kpi_labels:
                self.kpi_labels["kpi_losses"].config(text=f"P:{m.policy_loss:.3f} V:{m.value_loss:.3f}")
            if "kpi_rnad" in self.kpi_labels:
                self.kpi_labels["kpi_rnad"].config(text=f"{getattr(m, 'rnad_kl', 0.0):.4f}")
            if "kpi_rnd" in self.kpi_labels:
                rnd_val = getattr(self.trainer.rnd, "last_intrinsic_reward", 0.0) if getattr(self.trainer, "rnd", None) else 0.0
                self.kpi_labels["kpi_rnd"].config(text=f"{rnd_val:.4f}")
            if "kpi_rgsc" in self.kpi_labels:
                rgsc_buf = getattr(self.trainer, "state_buffer", getattr(self.trainer, "rgsc_buffer", None))
                buf_len = len(rgsc_buf) if rgsc_buf else 0
                cap = getattr(self.config.training, "rgsc_buffer_capacity", 200)
                self.kpi_labels["kpi_rgsc"].config(text=f"{buf_len} / {cap}")
            if "kpi_vp" in self.kpi_labels:
                self.kpi_labels["kpi_vp"].config(text=f"{m.avg_predicted_score:.1f} (R: {m.avg_real_score:.1f})")
            if "kpi_winrate" in self.kpi_labels:
                self.kpi_labels["kpi_winrate"].config(text=f"{getattr(m, 'win_rate', 0.5) * 100:.0f}%")
            if "kpi_elo" in self.kpi_labels:
                self.kpi_labels["kpi_elo"].config(
                    text=f"{getattr(m, 'league_elo', 1200.0):.0f} ({getattr(m, 'league_size', 2)})"
                )

            # Update Plot histories
            self.history_epochs.append(m.epoch)
            self.history_policy_loss.append(m.policy_loss)
            self.history_value_loss.append(m.value_loss)
            self.history_pred_scores.append(m.avg_predicted_score)
            self.history_real_scores.append(m.avg_real_score)

            # Redraw Matplotlib plots periodically
            if m.epoch % 2 == 0 or m.epoch == 1:
                self.ax_loss.clear()
                self.ax_loss_twin.clear()
                algo_str = getattr(self, "var_training_algo", None)
                algo_label = algo_str.get() if algo_str else "Policy"

                # Left Y-axis: Policy Loss (Blue)
                l1 = self.ax_loss.plot(self.history_epochs, self.history_policy_loss, label=f"Policy Loss ({algo_label})", color="#2563eb", lw=1.8)
                self.ax_loss.set_ylabel("Perte Politique (Policy)", color="#2563eb", fontsize=8, fontweight="bold")
                self.ax_loss.tick_params(axis="y", labelcolor="#2563eb")
                self.ax_loss.set_xlabel("Époques", fontsize=8)
                self.ax_loss.grid(True, linestyle="--", alpha=0.4)

                # Right Y-axis: Value Loss (Red)
                l2 = self.ax_loss_twin.plot(self.history_epochs, self.history_value_loss, label="Value Loss (Score VP)", color="#e11d48", lw=1.8, linestyle="--")
                self.ax_loss_twin.set_ylabel("Perte Valeur (VP)", color="#e11d48", fontsize=8, fontweight="bold")
                self.ax_loss_twin.tick_params(axis="y", labelcolor="#e11d48")

                lines = l1 + l2
                labels = [l.get_label() for l in lines]
                self.ax_loss.legend(lines, labels, fontsize=7, loc="upper right")
                self.ax_loss.set_title(f"Pertes d'apprentissage ({algo_label} & Score)", fontsize=9, fontweight="bold")

                self.ax_score.clear()
                self.ax_score.plot(self.history_epochs, self.history_pred_scores, label="Score Prédit (VP)", color="#059669", lw=1.8)
                self.ax_score.plot(self.history_epochs, self.history_real_scores, label="Score Réel (VP)", color="#d97706", lw=1.8, linestyle="--")
                self.ax_score.set_ylabel("Points de Victoire (VP)", fontsize=8, fontweight="bold")
                self.ax_score.set_xlabel("Époques", fontsize=8)
                self.ax_score.legend(fontsize=7, loc="upper right")
                self.ax_score.set_title("Score moyen : Prédit vs Réel (VP)", fontsize=9, fontweight="bold")
                self.ax_score.grid(True, linestyle="--", alpha=0.4)

                self.canvas.draw_idle()

        # 2. Hyperopt queue
        while not self.hyperopt_queue.empty():
            action, payload = self.hyperopt_queue.get_nowait()
            if action == "update":
                t: HyperoptTrial = payload
                sched_str = f"{t.params.get('lr_schedule_type', 'cos')[:3]} | {t.params.get('entropy_schedule_type', 'cos')[:3]}"
                row_vals = (
                    t.trial_id,
                    t.architecture_name or "ResNet",
                    f"{t.params.get('policy_lr', 3e-4):.1e}",
                    t.params.get("batch_size", 256),
                    sched_str,
                    f"{t.val_loss:.3f}",
                    f"{t.win_rate * 100:.0f}%",
                    f"{t.avg_score:.1f}",
                    f"{t.objective_score:.2f}",
                    t.status,
                )
                found = False
                for item in self.tree_trials.get_children():
                    if str(self.tree_trials.item(item, "values")[0]) == str(t.trial_id):
                        self.tree_trials.item(item, values=row_vals)
                        found = True
                        break
                if not found:
                    self.tree_trials.insert("", tk.END, values=row_vals)
            elif action == "finished":
                self.btn_run_opt.config(state=tk.NORMAL)
                self.btn_stop_opt.config(state=tk.DISABLED)
                if payload is not None:
                    self.btn_apply_best.config(state=tk.NORMAL)
                    messagebox.showinfo("Fin d'optimisation NAS", f"Recherche terminée ! Meilleure architecture : #{payload.trial_id} {payload.architecture_name} avec score NAS {payload.objective_score:.2f}")

    def _on_generate_pdf_report(self) -> None:
        self.lbl_train_status.config(text="● Génération PDF...", bg="#6366f1", fg="#ffffff")
        default_path = os.path.join(os.getcwd(), "Gaia_Project_Strategy_Report.pdf")

        def _worker():
            try:
                metrics_hist = {
                    "epochs": list(self.history_epochs),
                    "policy_loss": list(self.history_policy_loss),
                    "value_loss": list(self.history_value_loss),
                    "pred_scores": list(self.history_pred_scores),
                    "real_scores": list(self.history_real_scores),
                }
                buf = getattr(self.trainer, "replay_buffer", getattr(self.trainer, "buffer", None))
                out = generate_strategy_pdf(
                    agent=self.trainer.agent,
                    output_path=default_path,
                    device=self.trainer.device,
                    trainer=self.trainer,
                    replay_buffer=buf,
                    metrics_history=metrics_hist,
                )
                self.root.after(0, lambda: self._on_pdf_generated_success(out))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur PDF", f"Impossible de générer le rapport : {e}"))
            finally:
                self.root.after(0, lambda: self.lbl_train_status.config(
                    text="● Prêt (Idle)", bg="#334155", fg="#94a3b8"
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_pdf_generated_success(self, pdf_path: str) -> None:
        answer = messagebox.askyesno(
            "Rapport PDF Prêt !",
            f"Le rapport stratégique a été généré avec succès :\n{pdf_path}\n\nSouhaitez-vous l'ouvrir maintenant ?"
        )
        if answer:
            try:
                import webbrowser
                webbrowser.open(f"file:///{os.path.abspath(pdf_path)}")
            except Exception as e:
                messagebox.showinfo("Fichier disponible", f"Fichier enregistré sous :\n{pdf_path}")

    def _setup_arena_tab(self) -> None:
        container = ttk.Frame(self.tab_arena, padding=25)
        container.pack(fill=tk.BOTH, expand=True)

        card = ttk.Frame(container, style="Card.TFrame", padding=20)
        card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            card,
            text="🌌 Gaia AI Lab & Éditeur Tactique Sandbox",
            font=("Segoe UI", 16, "bold"),
            foreground="#0f172a",
        ).pack(anchor=tk.W, pady=(0, 10))

        desc_text = (
            "L'ancienne arène simplifiée a été remplacée par Gaia AI Lab, la station tactique complète\n"
            "connectée directement au moteur natif Rust (3130 actions discrètes plates) et à l'IA neuronale.\n\n"
            "Fonctionnalités avancées incluses :\n"
            "  • 🪐 Carte galactique interactive avec visualisation complète des 200 hexagones.\n"
            "  • 🧠 Inférence neuronale en temps réel (DualGaiaAgent) : meilleur coup, % de confiance et VP projetés.\n"
            "  • 🛠️ Éditeur de situation donnée (Sandbox) : pose de structures au pinceau, configuration des ressources\n"
            "    et des pistes de recherche pour tester instantanément des fins de partie ou des énigmes complexes.\n"
            "  • ⚡ Exécution directe des coups dans le moteur Rust officiel en un clic."
        )
        ttk.Label(
            card,
            text=desc_text,
            font=("Segoe UI", 10),
            foreground="#334155",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 20))

        btn_box = ttk.Frame(card)
        btn_box.pack(anchor=tk.W)

        btn_launch = ttk.Button(
            btn_box,
            text="🚀 Lancer Gaia AI Lab & Sandbox (PyQt)",
            style="Primary.TButton",
            command=self._launch_gaia_lab,
        )
        btn_launch.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_lab_status = ttk.Label(btn_box, text="", font=("Segoe UI", 9, "italic"), foreground="#64748b")
        self.lbl_lab_status.pack(side=tk.LEFT)

    def _launch_gaia_lab(self) -> None:
        import subprocess
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lab_script = os.path.join(root_dir, "gaia_gui_lab", "main.py")
        if os.path.exists(lab_script):
            try:
                subprocess.Popen([sys.executable, lab_script])
                self.lbl_lab_status.config(text="✓ Gaia AI Lab lancé en arrière-plan !")
            except Exception as e:
                messagebox.showerror("Erreur de lancement", f"Impossible de lancer Gaia AI Lab : {e}")
        else:
            messagebox.showerror("Fichier introuvable", f"Script introuvable : {lab_script}")



def launch_gui():
    """Launches GUI with headless display fallback for Linux servers."""
    if os.environ.get("DISPLAY") is None and os.name != "nt":
        print("=" * 76)
        print("⚠️  AVERTISSEMENT : Aucun serveur d'affichage X11/Wayland détecté ($DISPLAY non défini).")
        print("    Vous êtes sur un serveur distant headless (SSH ou VM sans interface graphique).")
        print("    Pour exécuter l'entraînement de production SOTA en ligne de commande :")
        print("       python train.py --preset grandmaster")
        print("    Ou pour tester rapidement :")
        print("       python train.py --preset fast")
        print("=" * 76)
        return

    root = tk.Tk()
    app = GaiaRLStudioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
