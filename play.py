"""Root launcher for Gaia Project Game & Tactical Lab GUI (PyQt).

Allows running `python play.py` directly from project root.
Automatically loads `gaia_gui_lab/main.py` connected to the native Rust engine and neural AI.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.join(SCRIPT_DIR, "gaia_gui_lab")
MAIN_SCRIPT = os.path.join(LAB_DIR, "main.py")


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if os.path.exists(MAIN_SCRIPT):
        if LAB_DIR not in sys.path:
            sys.path.insert(0, LAB_DIR)
        import importlib.util
        spec = importlib.util.spec_from_file_location("gaia_gui_lab_main", MAIN_SCRIPT)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            subprocess.run([sys.executable, MAIN_SCRIPT])
    else:
        print(f"Erreur : {MAIN_SCRIPT} introuvable.")


if __name__ == "__main__":
    main()
