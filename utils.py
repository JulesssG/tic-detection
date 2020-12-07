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
    frames1, frames2 = np.clip(frames1, 0, 255).reshape(frames1.shape[0], -1), np.clip(frames2, 0, 255).reshape(frames2.shape[0], -1)

    return np.sqrt(np.mean((frames1 - frames2)**2))

def crit(output, gt):
    output = output.view(output.shape[0], -1)
    gt = gt.view(gt.shape[0], -1)
    output, gt = torch.clip(output, 0, 255), torch.clip(gt, 0, 255)
    return torch.sqrt(torch.mean((output - gt)**2))

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
        fontsize = kwargs['fontsize']
    else:
        fontsize = 12
    plt.rcParams.update({'font.size': fontsize})

    if 'figsize' in kwargs:
        plt.figure(figsize=kwargs['figsize'])
    else:
        plt.figure(figsize=(15,10))

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=fontsize)
    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=fontsize)

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

def custom_sylvester(P1, P2, P3):
    """
        Solve a sylvester equation of the form:
        X = P1.T @ X @ P2 + P3
    """
    A_sylv = np.linalg.pinv(P1).T # Inverse or pseudo-inverse?
    B_sylv = -P2
    C_sylv = A_sylv @ P3

    return solve_sylvester(A_sylv, B_sylv, C_sylv)

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
    A_default_key = 'predictor.weight'
    A1, A2 = model1[1], model2[1]
    try:
        if isinstance(A1, nn.Module):
            if 'A1_key' in kwargs:
                A1 = A1.state_dict()[kwargs['A1_key']].T
            else:
                A1 = A1.state_dict()[A_default_key]
            A1 = A1.detach().cpu().numpy()
        elif isinstance(A1, torch.Tensor):
            A1 = A1.detach().cpu().numpy()
        if isinstance(A2, nn.Module):
            if 'A2_key' in kwargs:
                A2 = A2.state_dict()[kwargs['A2_key']].T
            else:
                A2 = A2.state_dict()[A_default_key]
            A2 = A2.detach().cpu().numpy()
        elif isinstance(A2, torch.Tensor):
            A2 = A2.detach().cpu().numpy()
    except KeyError:
        raise KeyError('You must provide the associated key for the dynamical system model.')
    n1, n2 = A1.shape, A2.shape
    if n1 != n2 or n1[0] != n1[1]:
        raise ValueError(f'Matrix A must be of same order and square but were {n1} and {n2}.')
    n = n1[0]

    # Extract C's
    if 'C_key' in kwargs:
        kwargs['C1_key'] = kwargs['C_key']
        kwargs['C2_key'] = kwargs['C_key']
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
        #print("Using identity for model1's projection.")
        C1 = np.eye(p, M=n)
    elif isinstance(C1, torch.Tensor):
        C1 = C1.detach().cpu().numpy()
    if C2 is None:
        #print("Using identity for model2's projection.")
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
            Ps[i,j] = custom_sylvester(As[i], As[j], Cs[i].T@Cs[j])
    for i,j in [(0,1), (1,0)]:
        Ps_[i] = (np.linalg.pinv(Ps[i,i]) @ Ps[i,j]
                    @ np.linalg.pinv(Ps[j,j]) @ Ps[j,i])


    eigens = np.concatenate((np.linalg.eig(Ps_[0])[0], np.linalg.eig(Ps_[1])[0]))

    if np.any(eigens < 0):
        raise ValueError("Negative eigen values present: "+str(eigens))


    return eigens

def grad_martin_dist(Ai, A):
    """
        Compute gradient of the martin distance for two stable
        transition matrices with respect to the second argument.

        Multiplying a matrix A with J_{jk} results in:
         - just the line a_k^t at line j if J_{jk}A
         - just the column a_j at line k if AJ_{jk}
    """
    # For now assume numpy
    if Ai.shape != A.shape:
        return None
    n = A.shape[0]
    X =  custom_sylvester(A, A, np.identity(n))
    Xi = custom_sylvester(Ai, Ai, np.identity(n))

    dA = np.zeros(A.shape)
    for j in range(n):
        for k in range(n):
            P3 = np.zeros(A.shape)
            P3[k,:] = X[j, :] @ A
            P3[:,k] = A.T @ X[:,j]
            Xd = custom_sylvester(A, A, P3)

            P3 = np.zeros(A.shape)
            P3[k,:] = Xi[j, :] @ Ai
            Xid = custom_sylvester(A, Ai, P3)

            dA[j,k] = np.sum(np.linalg.inv(X)*Xd) - 2*np.sum(np.linalg.pinv(Xi)*Xid)

    return dA

def martin_dist(model1, model2, **kwargs):
    return -np.log(np.prod(subspace_angles(model1, model2, **kwargs)))

def frob_dist(model1, model2, **kwargs):
    return 2*np.sum(1-subspace_angles(model1, model2, **kwargs))
