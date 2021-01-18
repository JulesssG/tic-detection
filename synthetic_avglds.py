#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:29:04 2020

@author: bejar
"""

# ============================================================================
# Import modules
# ============================================================================
import numpy as np
from scipy.linalg import solve_sylvester as sylv
from scipy.linalg import solve_discrete_lyapunov as dlyap
from numpy.linalg import svd, det
from numpy import log
from numpy.random import randn, rand

# ============================================================================
# MAIN - module executed as a script
# ============================================================================
if __name__ == "__main__":
    
    # ========================================================================
    # parameters
    # ========================================================================

    # fix seed of random number generator
    np.random.seed(10)
    
    # maximum number of iterations
    maxiter = 1000;
    
    # data dimensions
    n = 3; p = 5

    # number of models
    m = 5

    # noise variance
    sigma = 0.5

    # step size
    h = 1e-4
    
    # ========================================================================
    # generation of data points (LDSs)
    # ========================================================================
    
    # random system generation
    Z = np.random.randn(p,n)

    # singular value decomposition
    [U,S,V] = svd(Z,full_matrices=False)

    # true "average" LDS generating model
    C = U
    A = rand(n,n)
    A = rand(1)*A/np.linalg.norm(A)

    # generate a number of random models around the average model
    Ca = np.zeros((p,n,m+3))
    Aa = np.zeros((n,n,m+3))
    # loop for generation
    for mm in range(m):
        # random model around true one
        Z = C + sigma*randn(p,n)
        (U,S,Vh) = svd(Z,full_matrices=False);
        # (A,C) LDS model
        Ca[:,:,mm] = np.dot(U,Vh)
        Aa[:,:,mm] = A + sigma*randn(n,n);
        # re-normalize if needed
        na = np.linalg.norm(Aa[:,:,mm]);
        if na >= 1:
            Aa[:,:,mm] = rand(1)*Aa[:,:,mm]/na;

    # ========================================================================
    # LDS averaging
    # ========================================================================
    
    # Euclidean average
    (U,S,Vh) = svd(np.mean(Ca[:,:,:m],axis=2))
    Ce = np.dot(U[:,:n],Vh)
    Ae = np.mean(Aa[:,:,:m],axis=2)
    
    # average w.r.t. Martin distance
    # ========================================================================
    
    # random initialization
    (U,S,Vh) = svd(randn(p,n),full_matrices=False)
    Chat = np.dot(U,Vh)
    Ahat = rand(n,n);
    Ahat = rand(1)*Ahat/np.linalg.norm(Ahat);

    # initial S and offset
    S = dlyap(Ahat.T,np.eye(n));
    logdetS = log(det(S));
    logdetQ = 0;
    for mm in range(m):
        logdetQ = logdetQ + log(det(dlyap(Aa[:,:,mm].T,np.eye(n))));

    # inverse of As (this requires the transition matrices to be non-singular!)
    iA = np.zeros_like(Aa)
    # precompute P = Chat'*C*A_inv
    P = np.zeros((n,n,m))
    for mm in range(m):
        iA[:,:,mm] = np.linalg.pinv(Aa[:,:,mm])
        P[:,:,mm]  = np.dot(np.dot(Chat.T,Ca[:,:,mm]),iA[:,:,mm])

    # aux variable
    Q = np.zeros((n,n))

    # initial objective function
    fobj = np.zeros(maxiter)
    
    # optimize over A
    # ========================================================================
    for kk in range(maxiter):
        
        # compute S and its inverse
        S  = dlyap(Ahat.T,np.eye(n))
        iS = np.linalg.pinv(S)
      
        # evaluate objective function
        fobj[kk] = m*log(det(S)) + logdetQ
      
        # compute derivative
        dA = np.zeros((n,n))
        for mm in range(m):
            Xn = sylv(Ahat.T,-iA[:,:,mm],P[:,:,mm])
            iX = np.linalg.pinv(Xn)
            # derivative w.r.t. A of -2*sum(logdet( Xn ))
            for ii in range(n):
                for jj in range(n):
                    Q[:] = 0; Q[jj,:] = np.dot(Xn[ii,:].T,Aa[:,:,mm])
                    dXn = sylv(Ahat.T,-iA[:,:,mm],np.dot(Q,iA[:,:,mm]))
                    dA[ii,jj] = dA[ii,jj]-2*np.sum(iX.T*dXn)
            # update objective function
            fobj[kk] = fobj[kk] - log(det(np.dot(Xn.T,Xn)))
    
        # derivative w.r.t. A of logdet( S )
        for ii in range(n):
            for jj in range(n):
                # logdet S
                Q[:] = 0; Q[:,jj] = np.dot(Ahat.T,S[:,ii]); Q = Q + Q.T;
                dS = dlyap(Ahat.T,Q)
                # update derivative
                dA[ii,jj] = dA[ii,jj] + m*np.sum(iS.T*dS)
          
        # scale objective function
        fobj[kk] = fobj[kk]/m
          
        # update of A
        Ahat = Ahat - h*dA;
          
        # exit condition
        if kk>=1:
            if (fobj[kk-1]-fobj[kk])/fobj[kk-1] < 1e-5:
                fobj = fobj[:kk]
                break

    
    
    