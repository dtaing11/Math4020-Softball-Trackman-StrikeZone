import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModel(nn.Module): 
    def __init__(self):
        super().__init__()
        nn.Linear()
        nn.ReLU()
        nn.Linear()
        nn.Sigmoid()
        nn.Linear()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x) 
        return x
        

        
