import sys
sys.path.insert(0, 'training the model')
import torch

d = torch.load('checkpoints/gaia_latest.pt', map_location='cpu', weights_only=False)
print('Keys in checkpoint:', list(d.keys()))
print('Epoch:', d.get('epoch'))
metrics = d.get('metrics', {})
print('Metrics:', metrics)
if 'config' in d:
    cfg = d['config']
    print('Block type:', getattr(getattr(cfg, 'model', None), 'block_type', None))
    print('Action dim:', getattr(getattr(cfg, 'model', None), 'action_dim', None))
    print('Obs dim:', getattr(getattr(cfg, 'model', None), 'obs_dim', None))
    print('Policy hidden layers:', getattr(getattr(cfg, 'model', None), 'policy_hidden_layers', None))
    print('Sims:', getattr(getattr(cfg, 'alphazero', None), 'num_simulations', None))
    print('Games per epoch:', getattr(getattr(cfg, 'alphazero', None), 'games_per_epoch', None))
    print('Replay buffer size:', getattr(getattr(cfg, 'alphazero', None), 'replay_buffer_size', None))
    print('LR:', getattr(getattr(cfg, 'model', None), 'policy_lr', None))
