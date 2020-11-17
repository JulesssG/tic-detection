import cv2
import numpy as np
import torch
from torch import nn
import datetime
from matplotlib import pyplot as plt
from scipy.linalg import solve_sylvester

from custom_pca import *

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
    if isinstance(frames1, torch.Tensor):
        frames1 = frames1.detach().cpu().numpy()
    if isinstance(frames2, torch.Tensor):
        frames2 = frames2.detach().cpu().numpy()

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

    return str(datetime.timedelta(seconds=secr))

def subspace_angles(model1, model2, **kwargs):
    """
    Expect two models as input, each model is a tuple (m, m_ds):
        - m is the compression model, or the projection matrix C
        - m_ds is the dynamical system model or the transition matrix A
    If both models are non-linear projections, a named argument p is required with
    the dimensions of the frames.

    If a model is given instead of the matrix the key associated with the matrix
    must be provided (except if there is an attribute with same name as the required
    matrix; A or C). For example if A1 is a torch module describing the dynamical
    system a named argument A1_key corresponding to the attribute containing the matrix
    A must be provided.
    """
    # Extract A's
    A1, A2 = model1[1], model2[1]
    try:
        if isinstance(A1, nn.Module):
            A1 = A1.state_dict()[kwargs['A1_key']].T
            A1 = A1.detach().cpu().numpy()
        elif isinstance(A1, torch.Tensor):
            A1 = A1.detach().cpu().numpy()
        if isinstance(A2, nn.Module):
            A2 = A2.state_dict()[kwargs['A2_key']].T
            A2 = A2.detach().cpu().numpy()
        elif isinstance(A2, torch.Tensor):
            A2 = A2.detach().cpu().numpy()
    except KeyError:
        raise KeyError('You must provide the associated key for the dynamical system model.')
    n1, n2 = A1.shape, A2.shape
    if n1 != n2 or n1[0] != n1[1]:
        raise Error('Matrix A must be of same order and square.')
    n = n1[0]

    # Extract C's
    C1, C2 = model1[0], model2[0]
    if isinstance(C1, (custom_pca, nn.Module)):
        if isinstance(C1, custom_pca):
            C1 = C1.C
        elif isinstance(C1, nn.Module):
            C1 = C1.state_dict()[kwargs['C1_key']]
        else:
            C1 = None
    if isinstance(C2, (custom_pca, nn.Module)):
        if isinstance(C2, custom_pca):
            C2 = C2.C
        elif isinstance(C2, nn.Module):
            C2 = C2.state_dict()[kwargs['C2_key']]
        else:
            C2 = None
    try:
        p = (C1.shape[0] if C1 is not None
             else C2.shape[0] if C2 is not None else kwargs['p'])
    except KeyError:
        raise KeyError('No C matrix detected, you must provide the dimension of the frames as named parameter p.')
    if C1 is None:
        print("Using identity for model1's projection.")
        C1 = np.eye(p, M=n)
    elif isinstance(C1, torch.Tensor):
        C1 = C1.detach().cpu().numpy()
    if C2 is None:
        print("Using identity for model2's projection.")
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

def martin_dist(thetas):
    return -np.log(np.prod(np.cos(thetas)**2))

def frob_dist(thetas):
    return 2*(np.sum(np.sin(thetas)**2))