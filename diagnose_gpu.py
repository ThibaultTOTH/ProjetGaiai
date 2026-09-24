#!/usr/bin/env python3
"""Root launcher for GPU & CUDA diagnostic script."""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_SCRIPT = os.path.join(SCRIPT_DIR, "training the model", "diagnose_gpu.py")

if os.path.exists(DIAG_SCRIPT):
    with open(DIAG_SCRIPT, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, DIAG_SCRIPT, "exec"))
else:
    print(f"Error: {DIAG_SCRIPT} not found.")
