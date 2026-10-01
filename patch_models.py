import re

with open("training the model/models.py", "r") as f:
    content = f.read()

# 1. Update HexGNNEncoder
content = content.replace(
    """    def forward(self, map_flat: torch.Tensor, adj_norm: Optional[torch.Tensor] = None) -> torch.Tensor:""",
    """    def forward(self, map_flat: torch.Tensor, adj_norm: Optional[torch.Tensor] = None, return_nodes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:"""
)
content = content.replace(
    """        pool = torch.cat([mean_p, max_p], dim=-1)\n        return self.proj(pool)""",
    """        pool = torch.cat([mean_p, max_p], dim=-1)\n        pooled = self.proj(pool)\n        if return_nodes:\n            return pooled, x\n        return pooled"""
)

# 2. Update DualStreamBackbone
content = content.replace(
    """    def forward(self, obs: torch.Tensor, adj_norm: Optional[torch.Tensor] = None) -> torch.Tensor:""",
    """    def forward(self, obs: torch.Tensor, adj_norm: Optional[torch.Tensor] = None, return_map: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:"""
)
content = content.replace(
    """            e_map = self.map_encoder(map_x, adj_norm=adj_norm)\n            fused = torch.cat([e_scalar, e_map], dim=-1)\n            return self.trunk(fused)""",
    """            if return_map:\n                e_map, node_feat = self.map_encoder(map_x, adj_norm=adj_norm, return_nodes=True)\n            else:\n                e_map = self.map_encoder(map_x, adj_norm=adj_norm)\n                node_feat = None\n            fused = torch.cat([e_scalar, e_map], dim=-1)\n            out = self.trunk(fused)\n            if return_map:\n                return out, node_feat\n            return out"""
)
content = content.replace(
    """            x = self.flat_proj(x)\n            return self.trunk(x)""",
    """            x = self.flat_proj(x)\n            out = self.trunk(x)\n            if return_map:\n                return out, None\n            return out"""
)

# 3. Update ScorePredictorNet
content = content.replace(
    """            nn.Linear(128, 1),""",
    """            nn.Linear(128, 4),"""
)
content = content.replace(
    """        score = self.head(feat)\n        return score.squeeze(-1)""",
    """        score = self.head(feat)\n        return score"""
)
content = content.replace(
    """    def predict_score(self, obs: torch.Tensor) -> float:\n        self.eval()\n        if obs.dim() == 1:\n            obs = obs.unsqueeze(0)\n        val = self.forward(obs)\n        return float(val.item())""",
    """    def predict_score(self, obs: torch.Tensor) -> np.ndarray:\n        self.eval()\n        if obs.dim() == 1:\n            obs = obs.unsqueeze(0)\n        val = self.forward(obs)\n        return val.cpu().numpy()[0]"""
)
content = content.replace(
    """    def predict_score_with_uncertainty(\n        self, obs: torch.Tensor, num_passes: int = 4\n    ) -> Tuple[float, float]:""",
    """    def predict_score_with_uncertainty(\n        self, obs: torch.Tensor, num_passes: int = 4\n    ) -> Tuple[np.ndarray, np.ndarray]:"""
)
content = content.replace(
    """        if num_passes <= 1:\n            val = self.predict_score(obs)\n            return val, 0.0""",
    """        if num_passes <= 1:\n            val = self.predict_score(obs)\n            return val, np.zeros_like(val)"""
)
content = content.replace(
    """            scores = self.forward(repeated)  # shape (num_passes,)\n            mean_val = float(scores.mean().item())\n            std_val = float(scores.std().item())""",
    """            scores = self.forward(repeated)  # shape (num_passes, 4)\n            mean_val = scores.mean(dim=0).cpu().numpy()\n            std_val = scores.std(dim=0).cpu().numpy()"""
)

with open("training the model/models.py", "w") as f:
    f.write(content)
