import torch
import os

path = r"C:\Users\Thibault\OneDrive - Université Paris Sciences et Lettres\Documents\poroject_gaiapi\league_checkpoints\Snapshot_Epoch_0001.pt"
if os.path.exists(path):
    print("Loading checkpoint...")
    data = torch.load(path, map_location='cpu')
    st = data.get('state_dict', data.get('model_state_dict', data))
    
    def check_dict(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                check_dict(v, prefix + k + ".")
            elif hasattr(v, 'shape'):
                if len(v.shape) > 0 and (v.shape[0] == 16 or v.shape[-1] == 16):
                    print(f"Found action_dim=16 in tensor: {prefix}{k}, shape: {v.shape}")

    check_dict(st)
    print("Inspection complete.")
else:
    print("Path does not exist")
