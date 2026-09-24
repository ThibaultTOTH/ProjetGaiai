#!/usr/bin/env python3
"""Diagnostic Script for NVIDIA GPU & CUDA Detection (RTX 5070 Blackwell / Linux & Windows)."""

import os
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print("=" * 65)
    print("  [*] GAIA PROJECT - DIAGNOSTIC GPU & ACCELERATION CUDA")
    print("=" * 65)

    # 1. Verification systeme & nvidia-smi
    smi_path = shutil.which("nvidia-smi")
    print(f"\n[1] Outil NVIDIA Systeme (nvidia-smi) :")
    if smi_path:
        print(f"    [OK] Trouve : {smi_path}")
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                print(f"    [GPU Physique] : {res.stdout.strip()}")
            else:
                print(f"    [!] nvidia-smi a retourne une erreur :\n{res.stderr.strip()}")
                print("    -> Le pilote NVIDIA est installe mais ne communique pas avec le noyau.")
        except Exception as e:
            print(f"    [!] Erreur execution nvidia-smi : {e}")
    else:
        print("    [X] 'nvidia-smi' est introuvable sur ce systeme.")
        print("    -> Le pilote NVIDIA n'est pas installe ou n'est pas dans le PATH.")

    # 2. Verification PyTorch
    print(f"\n[2] Environnement PyTorch :")
    try:
        import torch
        print(f"    PyTorch Version : {torch.__version__}")
        print(f"    Backend CUDA compile dans PyTorch : {torch.version.cuda}")

        if torch.version.cuda is None:
            print("\n    [X] CAUSE DETECTEE : Version CPU-only de PyTorch installee !")
            print("    [SOLUTION] : Reinstallez PyTorch avec les dependances CUDA 12 via :")
            print("       pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            return

        cuda_ok = torch.cuda.is_available()
        print(f"    torch.cuda.is_available() : {cuda_ok}")

        if cuda_ok:
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
            print(f"    [OK] GPU Accessible par PyTorch : {name} (Total: {count} GPU)")
            print(f"    VRAM Totale : {vram_gb} GB")
            print(f"    Support BF16 Natif : {torch.cuda.is_bf16_supported()}")
            print(f"    Compute Capability : {props.major}.{props.minor}")
            print("\n  ==> TOUT EST PARFAITEMENT CONFIGURE POUR L'ENTRAINEMENT GPU !")
        else:
            print("\n    [!] CAUSE DETECTEE : PyTorch contient CUDA, mais ne peut pas communiquer avec le GPU.")
            print("    Causes frequentes sous Linux pour la RTX 5070 (Blackwell) :")
            print("    1. Pilote NVIDIA trop ancien (Blackwell requiert nvidia-driver-570+).")
            print("       Installez le pilote recent : sudo apt install nvidia-driver-570")
            print("    2. Secure Boot actif dans l'UEFI qui bloque le module noyau nvidia.")
            print("       Verifiez avec : mokutil --sb-state (desactiver dans le BIOS si actif)")
            print("    3. Droits d'acces /dev/nvidia* : sudo usermod -aG video $USER")
    except ImportError:
        print("    [X] PyTorch n'est pas installe dans cet environnement Python.")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
