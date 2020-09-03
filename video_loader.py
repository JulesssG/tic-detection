import torch
from torchvision.io import read_video

class video_loader:
    """
        This is a class to load videos into pytorch tensor, either eagerly as a batch
        of videos or lazily (todo)
        Each video will be of the form Channels x Frames x Height x Width
        
        Attributes:
            video_paths: Iterator of string, each is a location of a video
            duration: duration of the chunk of video to load in seconds (will start at the beginning of it)
            create_batch: Boolean, describing if the returned value will be an iterator or a Tensor 
            with the batch of videos
            normalize: Boolean, whether to normalize the data or not
    """
    def __init__(self, video_paths, duration, create_batch=True, normalize=True):
        self.batch = create_batch
        self.normalize = normalize
        self.duration = duration
        self.videos = video_paths
        self.nvideos = len(video_paths)
    
    def create_frames(self):
        allvframes = [read_video(vid, end_pts=self.duration, pts_unit='sec')[0] for vid in self.videos]

        if not self.batch:
            if self.normalize:
                temp=0 #TODO
            
            # TODO: return iterator instead of full load of data
            #       to be able to deal with large amount of videos
            self.vframes = allvframes
            return allvframes
        
        # Make their shape consistent and create batch
        self.nframes = min(map(lambda x: x.shape[0], allvframes))
        for i, f in enumerate(allvframes):
            allvframes[i] = f[:self.nframes].unsqueeze(0)
        batchvframes = allvframes[0]
        for nextvframes in allvframes[1:]:
            batchvframes = torch.cat((batchvframes, nextvframes), 0)

        # Reshape videos frames into Channels x Frames x Height x Width
        batchvframes = batchvframes.float().unsqueeze(1).transpose(1, 5).squeeze()

        if self.normalize:
            vmeans, vstds = batchvframes.mean((0,2,3,4)), batchvframes.std((0,2,3,4))
            batchvframes = ((batchvframes.transpose(1, 4) - vmeans) / vstds).transpose(1, 4)
            
        self.vframes = batchvframes
        return batchvframes
