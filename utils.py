import torch
import cv2
import numpy as np
from torchvision.io import read_video

def video2tensor(filename, vid_root='data/', tensor_root='data/tensors/'):
    vframes = read_video(vid_root+filename)[0]
    vframes = vframes.view(vframes.shape[0], -1).contiguous()
    torch.save(vframes, tensor_root+ '%s.pt' % filename[:-4])
    
class VideoIterator:
    def __init__(self, filename, batch_size=64):
        self.cap = cv2.VideoCapture(filename)
        self.batch_size = batch_size
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def __iter__(self):
        self.current_frame = 0
        self.stop = False
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration()
        
        frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
                self.current_frame += 1
            else:
                self.cap.release()
                self.stop = True
                break
            
            if self.current_frame % self.batch_size == 0:
                break
        
        return np.array(frames)