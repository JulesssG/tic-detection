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

class OneHAutoEncoder(nn.Module):
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
    
    def encode(self, x):
        x = x.view(x.shape[0], -1)
        
        return  self.to_lower_rep(x)

    def decode(self, x):
        return self.from_lower_rep(x).view(x.shape[0], *self.shape)

class SpatialConvAE(nn.Module):
    def __init__(self, inchannels, ncomp, nl=nn.ReLU, chans=[128, 128, 64]):
        super().__init__()
        self.ncomp = ncomp
        self.chans = chans

        self.encoder_convs = nn.Sequential(nn.Conv2d(inchannels, chans[0], kernel_size=26, stride=5), nl(), # 47
                                           nn.Conv2d(chans[0], chans[1], kernel_size=11, stride=3), nl(), # 13
                                           nn.Conv2d(chans[1], chans[2], kernel_size=6), nl()) # 8

        self.encoder_lin = nn.Linear(chans[2]*8*8, ncomp)
        self.decoder_lin = nn.Linear(ncomp, chans[2]*8*8)

        self.decoder_convs = nn.Sequential(nn.ConvTranspose2d(chans[2], chans[1], kernel_size=6), nl(),
                                           nn.ConvTranspose2d(chans[1], chans[0], kernel_size=11, stride=3), nl(),
                                           nn.ConvTranspose2d(chans[0], inchannels, kernel_size=26, stride=5))


    def forward(self, x):
        x = self.encoder_convs(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = x.view(x.shape[0], self.chans[2], 8, 8)
        x = self.decoder_convs(x)

        return x

class TemporalConvAE(nn.Module):
    def __init__(self, inchannels, nlayers, layerchans, low_dim_rep=None):
        super().__init__()
        self.inchannels = inchannels
        self.layerchans = layerchans
        c1 = c2 = c3 = c4 = c5 = layerchans
        self.low_dim_rep = low_dim_rep

        conv_params = {2: [(inchannels, c1, 8, 2),          # (1, c1, 5, 125, 125)
                           (c1, c2, (5, 7, 7), (1, 2, 2))], # (1, c2, 1, 60, 60)

                       3: [(inchannels, c1, 8, 2),          # (1, c1, 5, 125, 125)
                           (c1, c2, (3, 7, 7), (1, 2, 2)),  # (1, c2, 3, 60, 60)
                           (c2, c3, (3, 8, 8), (1, 2, 2))]  # (1, c3, 1, 27, 27)
        }

        encoder_modules = []
        for params in conv_params[nlayers]:
            encoder_modules.append(nn.Conv3d(params[0], params[1], kernel_size=params[2], stride=params[3]))
            encoder_modules.append(nn.ReLU())
        self.encoder_convs = nn.Sequential(*encoder_modules)
        self.end_shape = (c2, 1, 60, 60) if nlayers==2 else (c3, 1, 27, 27)
        if self.low_dim_rep is not None:
            self.low_dim_mapping = nn.Sequential(nn.Linear(np.prod(self.end_shape), self.low_dim_rep), 
                                               nn.Linear(self.low_dim_rep, np.prod(self.end_shape)))

        decoder_modules = []
        for params in conv_params[nlayers][::-1]:
            decoder_modules.append(nn.ConvTranspose3d(params[1], params[0], kernel_size=params[2], stride=params[3]))
            decoder_modules.append(nn.ReLU())
        self.decoder_convs = nn.Sequential(*decoder_modules)
        
    def forward(self, x):
        x = self.encoder_convs(x)
        if self.low_dim_rep is not None:
          x = self.low_dim_mapping(x.view(x.shape[0], -1)).view(-1, *self.end_shape)
        x = self.decoder_convs(x)
        
        return x

class TemporalConvAE2(nn.Module):
    def __init__(self, inchannels, nlayers, layerchans, hidden_dim):
        super().__init__()
        self.inchannels = inchannels
        self.layerchans = layerchans
        c1 = c2 = c3 = c4 = c5 = layerchans

        conv_params = {2: [(inchannels, c1, 8, 2),          # (1, c1, 5, 125, 125)
                           (c1, c2, (5, 7, 7), (1, 2, 2))], # (1, c2, 1, 60, 60)

                       3: [(inchannels, c1, 8, 2),          # (1, c1, 5, 125, 125)
                           (c1, c2, (3, 7, 7), (1, 2, 2)),  # (1, c2, 3, 60, 60)
                           (c2, c3, (3, 8, 8), (1, 2, 2))]  # (1, c3, 1, 27, 27)
        }

        encoder_modules = []
        for params in conv_params[nlayers]:
            encoder_modules.append(nn.Conv3d(params[0], params[1], kernel_size=params[2], stride=params[3]))
            encoder_modules.append(nn.ReLU())
        self.encoder_convs = nn.Sequential(*encoder_modules)
        self.end_shape = (c2, 1, 60, 60) if nlayers==2 else (c3, 1, 27, 27)
        self.encoder_lin = nn.Sequential(nn.Linear(np.prod(self.end_shape), 2*hidden_dim),
                                         nn.Linear(2*hidden_dim, hidden_dim))

        decoder_modules = []
        for params in conv_params[nlayers][::-1]:
            decoder_modules.append(nn.ConvTranspose3d(params[1], params[0], kernel_size=params[2], stride=params[3]))
            decoder_modules.append(nn.ReLU())
        self.decoder_convs = nn.Sequential(*decoder_modules)
        self.decoder_lin   = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                           nn.Linear(2*hidden_dim, np.prod(self.end_shape)))

    def forward(self, x):
        x = self.encoder_convs(x)
        x = self.encoder_lin(x.view(x.shape[0], -1))

        x = self.decoder_lin(x)
        x = x.view(-1, *self.end_shape)
        x = self.decoder_convs(x)

        return x

    def transform(self, x):
        x = self.encoder_convs(x)
        x = self.encoder_lin(x.view(x.shape[0], -1))

        return x

    def inverse_transform(self, x):
        x = self.decoder_lin(x).view(-1, *self.end_shape)
        x = self.decoder_convs(x)

        return x
