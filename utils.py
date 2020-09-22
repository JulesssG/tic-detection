import torch
import cv2
import numpy as np
from torchvision.io import read_video
from sklearn.utils.extmath import randomized_svd

def video2tensor(filename, vid_root='data/', tensor_root='data/tensors/'):
    vframes = read_video(vid_root+filename)[0]
    vframes = vframes.view(vframes.shape[0], -1).contiguous()
    torch.save(vframes, tensor_root+ '%s.pt' % filename[:-4])
    
class VideoLoader:
    def __init__(self, filename, duration=np.inf, batch_size=64, grayscale=False, **kwargs):
        self.filename = filename
        self.gray = grayscale
        self.batch_size = batch_size
        cap = cv2.VideoCapture(filename)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.duration_frames = min(self.total_frames, np.ceil(duration*self.fps/batch_size)*batch_size)
        self.duration = self.duration_frames/self.fps
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if 'scale' in kwargs:
            self.scale = True
            self.original_width  = self.width
            self.original_height = self.height
            self.width, self.height = kwargs['scale']
        else:
            self.scale = False
        
    def get_all_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(self.frame_transform(frame))
            else:
                cap.release()
        
        return np.array(frames)
    
    def get_random_frames(self, frames_ratio):
        nframes = int(self.total_frames * frames_ratio)
        frames = []
        cap = cv2.VideoCapture(self.filename)
        frame_ids = np.random.choice(np.arange(self.total_frames), 
                                     size=nframes, 
                                     replace=False)
        while cap.isOpened():
            ret, frame = cap.read()
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret:
                if current_frame in frame_ids:
                    frames.append(self.frame_transform(frame))
            else:
                cap.release()
        
        return np.array(frames)
            

    def frame_transform(self, frame):
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            
        return frame
    
    def __iter__(self):
        self.__cap = cv2.VideoCapture(self.filename)
        self.current_frame = 0
        self.stop = False
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration()
        
        frames = []
        while self.__cap.isOpened():
            ret, frame = self.__cap.read()
            if ret:
                frames.append(self.frame_transform(frame))
                self.current_frame += 1
            else:
                self.__cap.release()
                self.stop = True
                break
            
            if self.current_frame % self.batch_size == 0:
                break
        
        if self.current_frame >= self.duration*self.fps:
            self.stop = True
            
        return np.array(frames)
    
def read_video(filename, nframes=np.inf, random=False):
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
    
    if random:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = np.random.choice(np.arange(total_frames), 
                                     size=nframes, 
                                     replace=False)
    frame_condition = True
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        if random:
            if current_frame not in frame_ids:
                continue
        elif current_frame > nframes:
            break
            
        frames.append(frame)
    
    cap.release()
    return np.array(frames), {'fps': fps, 'width': width, 'height': height}

def write_video(filename, frames, width, height, fps, grayscale=False):
    if grayscale:
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), 0)
    else:
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    
    for frame in np.clip(np.around(frames), 0, 255).astype(np.uint8):
        writer.write(frame)
    writer.release()

def show_video(frames, imduration=int(1000/24.0)):
    for frame in frames:
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(imduration) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()

def reconstruction_error(frames1, frames2):
    if frames1.shape != frames2.shape:
        print("Shapes don't match")
        return -1
    return np.mean((frames1 - frames2)**2)

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
        if len(frames.shape) > 2:
            frames = frames.reshape(frames.shape[0], -1)
        frames = (frames - self.mean) / self.std
        frames_reduced = self.pc.T @ frames.T
        
        return frames_reduced.T
        
    def inverse_transform(self, frames, convert=True):
        frames_reconstructed = (self.pc @ frames.T).T
        frames_reconstructed = (frames_reconstructed * self.std) + self.mean
        
        if convert:
            frames_reconstructed = np.clip(frames_reconstructed, 0, 255).astype(np.uint8)
        
        return frames_reconstructed

def normalize_frames(frames, **kwargs):
    mean = kwargs['mean'] if 'mean' in kwargs else np.mean(frames)
    frames = frames - mean
    std  = kwargs['std'] if 'std' in kwargs else np.std(frames)
    frames = frames / std
    
    return frames