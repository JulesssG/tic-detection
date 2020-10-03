import torch
from torch import nn

class BasicAutoEncoder(nn.Module):
    def __init__(self, inchannels, ncomp):
        super().__init__()
        """
        # After that: B x 16 x 13 x 13
        self.transform_convs = nn.Sequential(nn.Conv2d(inchannels, 32, kernel_size=4), nn.ReLU(), # 253
                                            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.ReLU(), # 125
                                            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.ReLU(), # 61
                                            nn.Conv2d(32, 32, kernel_size=5), nn.ReLU(), # 57
                                            nn.Conv2d(32, 16, kernel_size=3, stride=2), nn.ReLU(), # 28
                                            nn.Conv2d(16, 16, kernel_size=4, stride=2), nn.ReLU()) # 13
        """
        
        self.transform_convs = nn.Sequential(nn.Conv2d(inchannels, 8, kernel_size=4), nn.ReLU(), # 253
                                             nn.Conv2d(8,  16, kernel_size=6, stride=3, padding=1), nn.ReLU(), # 84
                                             nn.Conv2d(16, 16, kernel_size=6, stride=2), nn.ReLU(), # 40
                                             nn.Conv2d(16, 16, kernel_size=6, stride=2), nn.ReLU(), # 18
                                             nn.Conv2d(16, 8,  kernel_size=6), nn.ReLU()) # 13
        low_channel = 8
        """
            to_lower_dim: map each sample from low_channel x 13 x 13 to ncomp dimensions
            from_lower_dim: does the inverse mapping
        """
        self.ncomp = ncomp
        if ncomp == 200:
            self.to_lower_dim = nn.Sequential(nn.Conv2d(low_channel, 8, kernel_size=5, stride=2), nn.ReLU())
            self.from_lower_dim = nn.Sequential(nn.ConvTranspose2d(8, low_channel, kernel_size=2, stride=2), nn.ReLU())
        elif ncomp == 150:
            self.to_lower_dim = nn.Sequential(nn.Conv2d(low_channel, 6, kernel_size=5, stride=2), nn.ReLU())
            self.from_lower_dim = nn.Sequential(nn.ConvTranspose2d(6, low_channel, kernel_size=5, stride=2), nn.ReLU())
        elif ncomp == 100:
            self.to_lower_dim = nn.Sequential(nn.Conv2d(low_channel, kernel_size=5, stride=2), nn.ReLU())
            self.from_lower_dim = nn.Sequential(nn.ConvTranspose2d(4, low_channel, kernel_size=5, stride=2), nn.ReLU())
        elif ncomp == 50:
            self.to_lower_dim = nn.Sequential(nn.Conv2d(low_channel, 2, kernel_size=5, stride=2), nn.ReLU())
            self.from_lower_dim = nn.Sequential(nn.ConvTranspose2d(2, low_channel, kernel_size=5, stride=2), nn.ReLU())
        elif ncomp == 25:
            self.to_lower_dim = nn.Sequential(nn.Conv2d(low_channel, 1, kernel_size=5, stride=2), nn.ReLU())
            self.from_lower_dim = nn.Sequential(nn.ConvTranspose2d(1, low_channel, kernel_size=5, stride=2), nn.ReLU())
        elif ncomp == 16:
            self.to_lower_dim = nn.Sequential(nn.Conv2d(low_channel, 1, kernel_size=7, stride=2), nn.ReLU())
            self.from_lower_dim = nn.Sequential(nn.ConvTranspose2d(1, low_channel, kernel_size=7, stride=2), nn.ReLU())
        else:
            print('The lower dimension must be one of:', self.ncomps())
            return
            
        """
        # Inverse of the transform's convolutions
        self.inv_transform_convs = nn.Sequential(nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2), nn.ReLU(),
                                                 nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
                                                 nn.ConvTranspose2d(32, 32, kernel_size=5), nn.ReLU(),
                                                 nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2), nn.ReLU(),
                                                 nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2), nn.ReLU(),
                                                 nn.ConvTranspose2d(32, inchannels, kernel_size=4), nn.ReLU())
        """
        
        self.inv_transform_convs = nn.Sequential(nn.ConvTranspose2d(8,  16, kernel_size=6), nn.ReLU(),
                                                   nn.ConvTranspose2d(16, 16, kernel_size=6, stride=2), nn.ReLU(),
                                                   nn.ConvTranspose2d(16, 16, kernel_size=6, stride=2), nn.ReLU(),
                                                   nn.ConvTranspose2d(16, 8, kernel_size=6, stride=3, padding=1), nn.ReLU(),
                                                   nn.ConvTranspose2d(8, inchannels,  kernel_size=4))
        
    def transform(self, x):
        x = self.transform_convs(x)
        x = self.to_lower_dim(x)
        
        return x.view(x.shape[0], -1), x.shape
    
    def inverse_transform(self, x, shape):
        x = x.view(shape)
        x = self.from_lower_dim(x)
        x = self.inv_transform_convs(x)
        
        return x
    
    @staticmethod
    def ncomps():
        return [16, 25, 50, 100, 150, 200]