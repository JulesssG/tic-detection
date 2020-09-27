import numpy as np
import cv2

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
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if current_frame >= self.duration*self.fps:
                cap.release()
                break
            if ret:
                frames.append(self.frame_transform(frame))
                current_frame += 1
            else:
                cap.release()
        
        return np.array(frames)
    
    def get_random_frames(self, frames_ratio, seed=42):
        nframes = int(self.total_frames * frames_ratio)
        frames = []
        cap = cv2.VideoCapture(self.filename)
        np.random.seed(seed)
        frame_ids = np.random.choice(np.arange(self.total_frames), 
                                     size=nframes, 
                                     replace=False, )
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