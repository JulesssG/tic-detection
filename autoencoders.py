import numpy as np # for prod
import torch
from torch import nn

class PCAAutoEncoder(nn.Module):
    def __init__(self, shape, ncomp):
        super().__init__()
        infeatures = np.prod(shape)
        self.shape = shape
        self.to_lower_rep = nn.Linear(infeatures, ncomp)
        self.from_lower_rep = nn.Linear(ncomp, infeatures)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.from_lower_rep(self.to_lower_rep(x))
        
        return x.view(x.shape[0], *self.shape)