#!/usr/bin/env python
"""Gaia Project Deep RL — Root Training Entrypoint."""
import importlib.util
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(root_dir, "training the model")
if train_dir not in sys.path:
    sys.path.insert(0, train_dir)

train_script = os.path.join(train_dir, "train.py")

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("trainer_main_script", train_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
