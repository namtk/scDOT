import numpy as np
import torch
import torch.nn as nn
import ot

from transport import OptimalTransportLayer

class scDOT(nn.Module):
    def __init__(self, sc_adata, st_adata):
        super(scDOT, self).__init__()
        self.NNLS = NNLS(sc_adata, st_adata)
        self.OT = OptimalTransportLayer(method='approx')
        self.M = nn.Parameter(torch.from_numpy(
            ot.dist(st_adata.X, sc_adata.X, metric='cosine'))
        )

    def forward(self, sc_adata, st_adata):
        NNLS_output = self.NNLS(sc_adata, st_adata)
        OT_output = self.OT(self.M)
        OT_output = OT_output/OT_output.sum(0) # col (cell) sum to 1
        return NNLS_output, OT_output

class NNLS(torch.nn.Module):
    def __init__(self, sc_adata, st_adata):
        super(NNLS, self).__init__()
        markers = st_adata.uns['markers'].to_numpy()
        st = st_adata.X
        self.W = nn.Parameter(torch.randn(st.shape[0], markers.shape[0]))

    def forward(self, sc_adata, st_adata):
        markers = st_adata.uns['markers'].to_numpy()
        st = st_adata.X
        W_nnls, _ = nls_projgrad(markers.T, st.T)
        W_nnls = W_nnls.T
        W_nnls = W_nnls/W_nnls.sum(1)[:,None]
        self.W.data.copy_(torch.tensor(W_nnls).float())
        return self.W