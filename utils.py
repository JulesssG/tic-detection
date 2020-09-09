import torch
from torchvision.io import read_video

def video2tensor(filename, vid_root='data/', tensor_root='data/tensors/'):
    vframes = read_video(vid_root+filename)[0]
    vframes = vframes.view(vframes.shape[0], -1).contiguous()
    torch.save(vframes, tensor_root+ '%s.pt' % filename[:-4])