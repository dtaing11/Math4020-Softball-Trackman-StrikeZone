import torch
import torch.nn as nn
import torch.nn.functional as F

class StrikeZonePredictionModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, output_dim)  
        )

    def forward(self, x):
        return self.model(x)
