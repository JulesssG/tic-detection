import torch
import cv2
import numpy as np
from torchvision.io import read_video

def video2tensor(filename, vid_root='data/', tensor_root='data/tensors/'):
    vframes = read_video(vid_root+filename)[0]
    vframes = vframes.view(vframes.shape[0], -1).contiguous()
    torch.save(vframes, tensor_root+ '%s.pt' % filename[:-4])
    
class VideoIterator:
    def __init__(self, filename, duration=np.inf, batch_size=64, grayscale=False):
        self.cap = cv2.VideoCapture(filename)
        self.gray = grayscale
        self.batch_size = batch_size
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.duration_frames = min(self.total_frames, np.ceil(duration*self.fps/batch_size)*batch_size)
        self.duration = self.duration_frames/self.fps
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
                if self.gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
                self.current_frame += 1
            else:
                self.cap.release()
                self.stop = True
                break
            
            if self.current_frame % self.batch_size == 0:
                break
        
        if self.current_frame >= self.duration*self.fps:
            self.stop = True
            
        return np.array(frames)
    
def read_video(filename, nframes=np.inf):
    """
        Read the given number of frames of a video using 
        the opencv library
        
        same as list(VideoIterator(filename, batch_size=np.inf))[0]
    """
    frames = []
    cap = cv2.VideoCapture(filename)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    i = 0
    while cap.isOpened() and i < nframes:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            i += 1
        else:
            break
    
    cap.release()
    return np.array(frames), fps, width, height

def reconstruction_error(frames1, frames2):
    return np.sum((frames1 - frames2)**2)