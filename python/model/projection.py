
import torch
import torch.nn as nn


class projection(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        bottleneck_dim
    ):
        super().__init__()

        self.project = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                     nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.GELU(),
                                     nn.Linear(hidden_dim, bottleneck_dim))
            
        
    def forward(self, x):
        x = self.project(x)
        return x