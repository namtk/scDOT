import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix, issparse
# from sklearn.utils.validation import check_array as check_arrays

import pandas as pd
import networkx as nx

def safe_fro(X, squared=False):
    if issparse(X):
        nrm = np.sum(X.data ** 2)
    else:
        if hasattr(X, 'A'):
            X = X.A
        nrm = np.sum(X ** 2)
    return nrm if squared else np.sqrt(nrm)

# Author: Vlad Niculae 
#         Lars Buitinck 
# Author: Chih-Jen Lin, National Taiwan University (original projected gradient
#     NMF implementation)
# Author: Anthony Di Franco (original Python and NumPy port)
# License: BSD 3 clause
def nls_projgrad(X, Y, W_init=None, l1_reg=0, l2_reg=0, tol=1e-3, max_iter=5000,
                 sigma=0.01, beta=0.1, callback=None):
    """Non-negative least square solver

    Solves a non-negative least squares subproblem using the
    projected gradient descent algorithm.
    min 0.5 * || XW - Y ||^2_F + l1_reg * sum(W) + 0.5 * l2_reg * * ||W||^2_F

    Parameters
    ----------
    Y, X : array-like
        Constant matrices.

    W_init : array-like
        Initial guess for the solution.

    l1_reg, l2_reg : float,
        Regularization factors

    tol : float
        Tolerance of the stopping condition.

    max_iter : int
        Maximum number of iterations before timing out.

    sigma : float
        Constant used in the sufficient decrease condition checked by the line
        search.  Smaller values lead to a looser sufficient decrease condition,
        thus reducing the time taken by the line search, but potentially
        increasing the number of iterations of the projected gradient
        procedure. 0.01 is a commonly used value in the optimization
        literature.

    beta : float
        Factor by which the step size is decreased (resp. increased) until
        (resp. as long as) the sufficient decrease condition is satisfied.
        Larger values allow to find a better step size but lead to longer line
        search. 0.1 is a commonly used value in the optimization literature.

    Returns
    -------
    W : array-like
        Solution to the non-negative least squares problem.

    grad : array-like
        The gradient.
    
    n_iter : int
        The number of iterations done by the algorithm.

    Reference
    ---------

    C.-J. Lin. Projected gradient methods
    for non-negative matrix factorization. Neural
    Computation, 19(2007), 2756-2779.
    http://www.csie.ntu.edu.tw/~cjlin/nmf/

    """
    # X, Y = check_arrays(X, Y)
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    XY = safe_sparse_dot(X.T, Y)
    G = np.dot(X.T, X)

    if W_init is None:
        W = np.zeros((n_features, n_targets), dtype=np.float64)
    else:
        W = W_init.copy()
    init_grad_norm = safe_fro(np.dot(G, W) - XY + l2_reg * W + l1_reg)
    # values justified in the paper
    alpha = 1
    for n_iter in range(1, max_iter + 1):
        grad = np.dot(G, W) - XY  # X'(XW - Y) using precomputation
        if l2_reg:
            grad += l2_reg * W
        if l1_reg:
            grad += l1_reg

        # The following multiplication with a boolean array is more than twice
        # as fast as indexing into grad.
        proj_grad_norm = np.linalg.norm(grad * np.logical_or(grad < 0, W > 0))
        if proj_grad_norm / init_grad_norm < tol:
            break

        W_prev = W

        for inner_iter in range(20):
            # Gradient step.
            W_next = W - alpha * grad
            # Projection step.
            W_next *= W_next > 0
            d = W_next - W
            gradd = np.dot(grad.ravel(), d.ravel())
            dGd = np.dot(np.dot((G + l2_reg), d).ravel(), d.ravel())
            suff_decr = (1 - sigma) * gradd + 0.5 * dGd < 0
            if inner_iter == 0:
                decr_alpha = not suff_decr

            if decr_alpha:
                if suff_decr:
                    W = W_next
                    break
                else:
                    alpha *= beta
            elif not suff_decr or (W_prev == W_next).all():
                W = W_prev
                break
            else:
                alpha /= beta
                W_prev = W_next
        if callback:
            callback(W)

    if n_iter == max_iter:
        print("PG failed to converge")
    residual = safe_fro(np.dot(X, W) - Y)
    return W, residual

def dec_byot(sc_adata, st_adata, annotations, coupling):
    sc_adata = sc_adata
    st_adata = st_adata
    annotations = annotations
    coupling = coupling

    ctxc = pd.DataFrame(sc_adata.obs[annotations]).reset_index()
    G = nx.from_pandas_edgelist(
        ctxc,
        source=annotations,
        target='index',
        # create_using=nx.DiGraph()
    )
    adj_df = pd.DataFrame(
        nx.adjacency_matrix(G).todense(),
        index=G.nodes,
        columns=G.nodes
    )
    ct = adj_df.loc[np.unique(ctxc[annotations]).tolist()]\
        [np.unique(ctxc['index']).tolist()]
    ct = ct.T.reindex(sc_adata.obs_names)

    P = coupling.T@ct.to_numpy()
    P = P/P.sum(1)[:, np.newaxis]
    P_df = pd.DataFrame(P, columns = ct.columns,
                        index = st_adata.obs_names)
    st_adata.obsm['deconvolution'] = P_df

    # also copy as single field in the anndata for visualization
    for ct in st_adata.obsm["deconvolution"].columns:
        st_adata.obs[ct] = st_adata.obsm["deconvolution"][ct]
    
    return st_adata