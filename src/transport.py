import numpy as np
import matplotlib.pyplot as plt
import novosparc
from ot.bregman import sinkhorn
from ot.utils import dist

import torch
import torch.optim as optim

import sys
sys.path.append("ManifoldWarping/python")

from alignment import (
    ManifoldLinear)
from correspondence import Correspondence
from distance import SquaredL2
from neighborhood import neighbor_graph
from util import pairwise_error, Timer
from viz import show_alignment

def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=10):
    '''Solve the entropy regularized optimal transport problem.'''
    
    m, n = M.shape
    if r is None: r = torch.ones((m, 1)) / m
    if c is None: c = torch.ones((n, 1)) / n
    assert r.shape == (m, 1) and c.shape == (n, 1)
    
    P = torch.exp(-1.0 * gamma * M)
    for i in range(maxiters):
        alpha = torch.sum(P, 1).reshape(m, 1)
        P = r / alpha * P
        
        beta = torch.sum(P, 0)
        if torch.all(torch.isclose(beta, c, atol=eps, rtol=0.0)):
            break
        P = P * c.T / beta.T
        P = (P.T/P.T.sum(0)).T
        
    return P

def learnM(fcn, M_init, P_true, iters=500, device='cpu'):
    """Find an M such that sinkhorn(M) matches P_true. Return M and the learning curve."""
    M_init.float().to(device)
    P_true.float().to(device)
    fcn.to(device)
    M = M_init.clone()
    M.float().to(device)
    M.requires_grad = True

    optimizer = optim.AdamW([M], lr=1.0e-1)

    h = []
    for i in range(iters):
        optimizer.zero_grad(set_to_none=True)
        P = fcn(M)
        P = (P.T/P.T.sum(0)).T
        # J = torch.linalg.norm(P - P_true)#, ord='nuc')
        loss = torch.nn.CosineEmbeddingLoss()
        J = loss(P, P_true, torch.ones(P.shape[0])).float().to(device)
        print(J)
        h.append(J.item())
        J.backward()
        optimizer.step()

    return M, h

def learnM_v2(fcn, M_init, ct, P_true, iters=500, device='cpu'):
    """Find an M such that sinkhorn(M) matches P_true. Return M and the learning curve."""
    M_init.float().to(device)
    P_true.float().to(device)
    fcn.to(device)
    M = M_init.clone()
    M.float().to(device)
    M.requires_grad = True

    optimizer = optim.AdamW([M], lr=1.0e-1)

    h = []
    for i in range(iters):
        optimizer.zero_grad(set_to_none=True)
        P = fcn(M)
        P = (P.T/P.T.sum(0)).T
        # J = torch.linalg.norm(P - P_true)#, ord='nuc')
        loss = torch.nn.CosineEmbeddingLoss()
        J = loss(P@ct, P_true, torch.ones(P.shape[0])).float().to(device)
        print(J)
        h.append(J.item())
        J.backward()
        optimizer.step()

    return M, h


def tensor_square_loss_adjusted(C1, C2, T):
    """
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the square loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        T : A coupling between those two spaces

    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            f1(a)=(a^2)/2
            f2(b)=(b^2)/2
            h1(a)=a
            h2(b)=b

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T : ndarray, shape (ns, nt)
         Coupling between source and target spaces

    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a**2) / 2

    def f2(b):
        return (b**2) / 2

    def h1(a):
        return a

    def h2(b):
        return b

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()

    return tens

def cell_graph(sc_adata, st_adata, coupling, k=10, save=None):
    sc_adata = sc_adata
    st_adata = st_adata
    coupling = coupling
    k = k
    save = save

    X = np.asarray(sc_adata.X)
    Y = np.asarray(st_adata.X)
    corr = Correspondence(matrix=coupling)
    Wx = neighbor_graph(X,k=10)
    Wy = neighbor_graph(st_adata.obsm['spatial'],k=6)
    d = 2
    lin_aligners = (
        ('linear manifold',  lambda: ManifoldLinear(X,Y,corr,d,Wx,Wy)),
    )
    for name, aln in lin_aligners:
        plt.figure()
        with Timer(name):
          Xnew,Ynew = aln().project(X, Y)
        print(' sum sq. error =', pairwise_error(Xnew, Ynew, metric=SquaredL2))
        show_alignment(Xnew,Ynew,name)
    A = neighbor_graph(Xnew, k=10)
    np.save(save, A) if save != None else None

    return A