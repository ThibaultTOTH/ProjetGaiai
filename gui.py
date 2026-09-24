"""Root launcher for Gaia Project Deep RL Studio GUI.

Allows running `python gui.py` directly from project root.
Automatically configures sys.path and passes control to `training the model/gui.py`.
"""

import importlib.util
import os
import sys

# Ensure 'training the model' directory is at the beginning of sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "training the model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# Dynamically load the GUI script from the subdirectory without module name collisions
def main():
    gui_script = os.path.join(MODEL_DIR, "gui.py")
    spec = importlib.util.spec_from_file_location("gui_main_script", gui_script)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "launch_gui"):
            module.launch_gui()
    else:
        raise ImportError(f"Could not load GUI module from {gui_script}")


if __name__ == "__main__":
    main()

