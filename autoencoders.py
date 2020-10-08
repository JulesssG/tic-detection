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

class OneAutoEncoder(nn.Module):
    def __init__(self, shape, ncomp, nl=nn.ReLU):
        super().__init__()
        infeatures = np.prod(shape)
        self.shape = shape
        self.ncomp = ncomp
        self.hidden_dim = 200
        self.to_lower_rep = nn.Sequential(nn.Linear(infeatures, self.hidden_dim),
                                          nl(), 
                                          nn.Linear(self.hidden_dim, ncomp))
        self.from_lower_rep = nn.Sequential(nn.Linear(ncomp, self.hidden_dim),
                                           nl(),
                                           nn.Linear(self.hidden_dim, infeatures))
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.from_lower_rep(self.to_lower_rep(x))
        
        return x.view(x.shape[0], *self.shape)
    
class SpatialConvAE(nn.Module):
    def __init__(self, inchannels, ncomp, nl=nn.ReLU):
        super().__init__()
        self.ncomp = ncomp
        
        self.encoder_convs = nn.Sequential(nn.Conv2d(inchannels, 128, kernel_size=26, stride=5), nl(), # 47
                                           nn.Conv2d(128, 128, kernel_size=11, stride=3), nl(), # 13
                                           nn.Conv2d(128, 64, kernel_size=6), nl()) # 8
        
        self.encoder_lin = nn.Linear(64*8*8, ncomp)
        self.decoder_lin = nn.Linear(ncomp, 64*8*8)
        
        self.decoder_convs = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=6), nl(),
                                           nn.ConvTranspose2d(128, 128, kernel_size=11, stride=3), nl(),
                                           nn.ConvTranspose2d(128, inchannels, kernel_size=26, stride=5))
        
        
    def forward(self, x):
        x = self.encoder_convs(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = x.view(x.shape[0], 64, 8, 8)
        x = self.decoder_convs(x)
        
        return x