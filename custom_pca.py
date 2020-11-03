import numpy as np
from sklearn.utils.extmath import randomized_svd

class custom_pca():
    def __init__(self, ncomp=10):
        self.ncomp = ncomp
        
    def fit(self, frames):
        self.mean = np.mean(frames)
        self.std = np.std(frames)
        frames = (frames - self.mean) / self.std
        frames = frames.reshape(frames.shape[0], -1)
        self.pc, _, _ = randomized_svd(frames.T, self.ncomp)
        
    def transform(self, frames):
        shape = frames.shape[1:]
        if len(shape) > 1:
            frames = frames.reshape(frames.shape[0], -1)
        frames = (frames - self.mean) / self.std
        frames_reduced = self.pc.T @ frames.T
        
        return frames_reduced.T, shape
        
    def inverse_transform(self, frames, shape=None, cast=True):
        nframes = frames.shape[0]
        frames_reconstructed = (self.pc @ frames.T).T
        frames_reconstructed = (frames_reconstructed * self.std) + self.mean
        
        if cast:
            frames_reconstructed = np.clip(frames_reconstructed, 0, 255).astype(np.uint8)
        if shape:
            frames_reconstructed = frames_reconstructed.reshape(nframes, *shape)
        
        return frames_reconstructed