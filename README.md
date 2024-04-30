# scDOT
Optimal transport for mapping senescent cells in spatial transcriptomics
## Abstract
Spatial transcriptomics (ST) provides a unique opportunity to study cellular organization and cell-cell interactions at the molecular level. However, due to the low resolution of the sequencing data additional information is required to utilize this technology, especially for cases where only a few cells are present for important cell types. To enable the use of ST to study senescence we developed scDOT, which combines ST and single cell RNA-Sequencing (scRNA-Seq) to improve the ability to reconstruct single cell resolved spatial maps. scDOT integrates optimal transport and expression deconvolution to learn non-linear couplings between cells and spots and to infer cell placements. Application of scDOT to existing and new lung ST data improves on prior methods and allows the identification of the spatial organization of senescent cells, the identification of their neighboring cells and the identification of novel genes involved in cell-cell interactions that may be driving senescence.
# Installation
This script need no installation, but has the following requirements:
- PyTorch 1.9.0
- POT 0.9.3
# Usage
nputs, including spatial transcriptomics data and single-cell transcriptomics data, must be formatted as AnnData objects. The basic usage of the script is as follows:
```python
model = scDOT(sc_adata, st_adata)
optimizer = optim.Adam(model.parameters(), lr=1.0e-1)

iters = 10
for i in range(iters):
    # Forward pass
    NNLS_output, OT_output = model(sc_adata, st_adata)

    ct = torch.tensor(sc_adata.obsm['cell_type'].values.T)
    P_true = NNLS_output @ ct.float() # spots by cells
    P = P_true/P_true.sum(0) # col (cell) sum to 1
    loss_fn = torch.nn.CosineEmbeddingLoss()
    loss = loss_fn(OT_output, P_true, torch.ones(P.shape[0]))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update the parameters
    optimizer.step()
```
# License
MIT License

Copyright (c) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
