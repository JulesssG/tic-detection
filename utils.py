import cv2
import numpy as np
import torch
from torch import nn
import datetime
from matplotlib import pyplot as plt
from scipy.linalg import solve_sylvester
    
def write_video(filename, frames, meta):
    if meta['gray']:
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP4V"), meta['fps'], (meta['width'], meta['height']), 0)
    else:
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP4V"), meta['fps'], (meta['width'], meta['height']))
    
    for frame in np.clip(np.around(frames), 0, 255).astype(np.uint8):
        writer.write(frame)
    writer.release()

def show_video(frames, imduration=int(1000/24.0)):
    for frame in frames:
        cv2.imshow('frame',frame)
        if cv2.waitKey(imduration) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def reconstruction_error(frames1, frames2):
    if frames1.shape != frames2.shape:
        return -1
    return np.sqrt(np.mean((frames1 - frames2)**2))

def crit(output, gt):
    output = output.view(output.shape[0], -1)
    gt = gt.view(gt.shape[0], -1)
    return torch.clip(torch.sqrt(torch.mean((output - gt)**2)), 0, 255)

def standardize_frames(frames, **kwargs):
    mean = kwargs['mean'] if 'mean' in kwargs else torch.mean(frames)
    frames = frames - mean.item()
    std  = kwargs['std'] if 'std' in kwargs else torch.std(frames)
    frames = frames / std.item()
    
    return frames

def plot(xs, ys, **kwargs):
    if 'styles' in kwargs:
        styles = kwargs['styles']
    else:
        styles = ['C'+str(c)+'-'+s for s in ['', '.', 'o', '^'] 
                  for c in [0, 1, 2, 3, 6, 8, 9] ]
    
    if len(ys) > len(styles):
        print('Duplicate styles')
    
    if 'fontsize' in kwargs:
        plt.rcParams.update({'font.size': kwargs['fontsize']})
    else:
        plt.rcParams.update({'font.size': 12})
        
        
    if 'figsize' in kwargs:
        plt.figure(figsize=kwargs['figsize'])
    else:
        plt.figure(figsize=(15,10))
        
    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])
        
    if 'yrange' in kwargs:
        low, high = kwargs['yrange']
        plt.ylim(low, high)
    
    if 'bound_to_plot' in kwargs:
        epoch, max_error = kwargs['bound_to_plot']
        ys = list(filter(lambda x: max(x[epoch:]) < max_error, ys))
        
        
    if 'labels' in kwargs:
        for i in range(len(ys)):
            if not str(xs[0]).isnumeric():
                plt.plot(xs[i], ys[i], styles[i], label=kwargs['labels'][i])
            else:
                plt.plot(xs, ys[i], styles[i], label=kwargs['labels'][i])
        plt.legend()
    else:
        if len(np.shape(xs)) > 1:
            for x, y in zip(xs, ys):
                plt.plot(x, y)
        else:
            for y in ys:
                plt.plot(xs, y)

    if 'title' in kwargs:
        plt.title(kwargs['title'])


def sec2string(sec):
    if sec <= 60:
        return round(sec, 2)
    secr = round(sec)
    
    return str(datetime.timedelta(seconds=secr)).strip("00:")

def subspace_angles(A1, A2, C1=None, C2=None, **kwargs):
    """
    Expect C1 and C2 tensor or numpy array or None. If both are None expect a
    named argument p to know the dimension of the frames.
    
    Expect A1, A2 to be numpy array or tensors. If Ai is passed as model such
    as FramePredictor it expects a named argument Ai_key, the key in the 
    state_dict of the model.
    """
    # Extract A's
    if isinstance(A1, nn.Module):
        A1 = A1.state_dict[kwargs['A1_key']]
    if isinstance(A2, nn.Module):
        A2 = A2.state_dict[kwargs['A2_key']]
    n1, n2 = A1.shape, A2.shape
    if n1 != n2 or n1[0] != n1[1]:
        print('Matrix A must be of same order and square.')
        return
    n = n1[0]
    if isinstance(A1, torch.Tensor):
        A1 = A1.detach().cpu().numpy()
    if isinstance(A2, torch.Tensor):
        A2 = A2.detach().cpu().numpy()
    
    # Extract C's
    p = C1.shape[0] if C1 is not None else C2.shape[0] if C2 is not None else kwargs['p']
    if C1 is None:
        C1 = np.eye(p, M=n)
    elif isinstance(C1, torch.Tensor):
        C1 = C1.detach().cpu().numpy()
    if C2 is None:
        C2 = np.eye(p, M=n)
    elif isinstance(C2, torch.Tensor):
        C2 = C2.detach().cpu().numpy()
    Cs = np.array([C1, C2])
    
    # Normalize A's
    norms = np.linalg.norm(np.stack((A1, A2)), ord=2, axis=(1,2))
    if norms[0] > 1:
        A1 = 0.98*A1/norms[0]
    if norms[1] > 1:
        A2 = 0.98*A2/norms[1]
    As = np.array([A1, A2])
    
    Ps, Ps_ = np.zeros((2, 2, n, n)), np.zeros((2, n, n))
    for i in range(2):
        for j in range(2):
            A_sylv = np.linalg.pinv(As[i]).T # Inverse or pseudo-inverse?
            B_sylv = -As[j]
            C_sylv = A_sylv @ Cs[i].T @ Cs[j]
            Ps[i,j] = solve_sylvester(A_sylv, B_sylv, C_sylv)
    for i,j in [(0,1), (1,0)]:
        Ps_[i] = (np.linalg.pinv(Ps[i,i]) @ Ps[i,j]
                    @ np.linalg.pinv(Ps[j,j]) @ Ps[j,i])
    
    
    eigens = np.concatenate((np.linalg.eig(Ps_[0])[0], np.linalg.eig(Ps_[1])[0]))
    eigens = np.arccos(np.sqrt(eigens))
    
    return eigens
        