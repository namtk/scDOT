import pandas as pd
import anndata as ad
import scanpy as sc
import networkx as nx
import numpy as np
from ast import literal_eval
from collections import Counter
from scipy.sparse import identity


def load_data(data, path):
    path = path
    if data=='xenium':
        adata = sc.read_10x_h5(
            filename=path + "Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5"
        )
        df = pd.read_csv(
            path + "Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv"
        )
        df.set_index(adata.obs_names, inplace=True)
        adata.obs = df.copy()
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
        sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)
        
        cprobes = (
            adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100
        )
        cwords = (
            adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100
        )

        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_genes(adata, min_cells=5)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        
        # Get observation names
        obs_names = adata.obs_names
        # Create sparse identity matrix
        sparse_identity_matrix = identity(len(obs_names))
        # Convert the sparse matrix to a DataFrame
        mapping_df = pd.DataFrame.sparse.from_spmatrix(
            sparse_identity_matrix, index=obs_names, columns=obs_names
        )
        # Assign the DataFrame to adata.obsm['mapping']
        adata.obsm['mapping'] = mapping_df
        
        # Define the number of observations to keep in the final subsample
        remaining_observations = 5000

        # Count observations in each group
        group_counts = adata.obs['leiden'].value_counts()

        # Calculate the number of observations to sample from each group to maintain proportionality
        sampling_probs = group_counts / group_counts.sum()
        observations_to_sample = (sampling_probs * remaining_observations).astype(int)

        # Subsample from each group
        subsampled_indices = []
        for group, count in observations_to_sample.items():
            group_indices = adata.obs.index[adata.obs['leiden'] == group]
            sampled_indices = np.random.choice(group_indices, count, replace=False)
            subsampled_indices.extend(sampled_indices)

        # Create a new Anndata object with subsampled data
        subsampled_adata = adata[subsampled_indices, :]
        # Subsample the columns in subsampled_adata.obsm['mapping']
        subsampled_mapping_df = subsampled_adata.obsm['mapping'].loc[:, subsampled_adata.obs_names]
        # Assign the subsampled mapping DataFrame to subsampled_adata
        subsampled_adata.obsm['mapping'] = subsampled_mapping_df

        sc_adata = subsampled_adata.copy()
        st_adata = subsampled_adata.copy()
        
    
    if data=='synthetic':
        corr = pd.read_csv(path + 'corr.csv', index_col=0) # cell metadata
        st_ = pd.read_csv(path + 'st.csv', index_col=0) # st expr
        sc_ = pd.read_csv(path + 'sc.csv', index_col=0) # sc expr
        sc_ = sc_[sc_.index.isin(corr.index)] # sc expr in cell metadata
        ctp = pd.read_csv(path + 'ctp.csv', index_col=0) # true cell type proportion
        cellcounts = pd.read_csv(path + 'cellcounts.csv')\
            .set_index('Unnamed: 0').drop('counts', axis=1)

        ctxc = corr['Cell_class'].reset_index() # cell type adj list
        G = nx.from_pandas_edgelist(
            ctxc,
            source= 'Cell_class', #'annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['Cell_class']).tolist()]\
            [np.unique(ctxc['index']).tolist()] # cell type matrix
        ct = ct.T.reindex(sc_.index) # reindex to match cell data

        ctxc = corr['patch_id'].reset_index() # cell to spot adj list
        G = nx.from_pandas_edgelist(
            ctxc,
            source= 'patch_id', #'annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        mapping = adj_df.loc[np.unique(ctxc['patch_id']).tolist()]\
            [np.unique(ctxc['index']).tolist()] # cell to spot matrix
        mapping = mapping[sc_.index].reindex(st_.index) # reindex to match both cell and spot data

        st_adata = ad.AnnData(st_.to_numpy())
        st_adata.obs_names = st_.index
        st_adata.var_names = st_.columns

        st_adata.obsm['spatial'] = cellcounts.to_numpy()
        st_adata.obsm['true_deconvolution'] = ctp
        st_adata.obsm['mapping'] = mapping # true cell-spot mapping

        sc_adata = ad.AnnData(sc_.to_numpy())
        sc_adata.obs_names = sc_.index
        sc_adata.var_names = sc_.columns

        sc_adata.obsm['cell_type'] = ct
        sc_adata.obsm['mapping'] = mapping.T
        
        sc_adata.obs['annotations'] = corr['Cell_class']
        
        # markers x cell types
        markers = pd.read_csv(path + 'marker.csv', index_col=0)
        st_adata.uns['markers'] = markers
        
        sc_adata.var_names_make_unique()
        st_adata.var_names_make_unique()

    if data=='ipf_lung':
        sc_adata = ad.read(path + 'store/oliver/OliverEC.h5ad')
        sc_adata.var_names_make_unique()

        st_adata = sc.read_visium(path=path + 'data/osu/STB01S1_preproccesed/outs')
        st_adata.var_names_make_unique()

        ctp = pd.read_csv(path + 'output/stereoscope_CTP-5.csv', index_col=0, header=1).\
            drop(['max_value','Unnamed: 26'], axis=1)

        ctxc = pd.DataFrame(sc_adata.obs['annotations']).reset_index()
        G = nx.from_pandas_edgelist(
            ctxc,
            source='annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['annotations']).tolist()]\
            [np.unique(ctxc['index']).tolist()]
        ct = ct.T.reindex(sc_adata.obs_names) # reindex to match cell id

        
        # preprocessing (spots/cells) st_adata just to match with stereoscope_CTP-5.csv in ctp,
        # then asign ctp and ct
        st_adata.var["mt"] = st_adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_adata, qc_vars = ["mt"], inplace = True)
        sc.pp.filter_cells(st_adata, min_counts = 500)
        sc.pp.filter_cells(st_adata, min_genes = 500)
        ctp = ctp.reindex(st_adata.obs_names) # reindex to match spot id
        st_adata.obsm['true_deconvolution'] = ctp
        
        sc_adata.obsm['cell_type'] = ct

        # preprocessing (genes) both sc_adata and st_adata then taking intersection
        sc.pp.filter_genes(sc_adata, min_counts = 10)
        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]
        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum = 1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes = 7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span = 1
        )

        sc.pp.filter_genes(st_adata, min_cells=3)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        sc.pp.highly_variable_genes(st_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(st_adata, n_top_genes=2000)

        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()
        
    if data=='paired_ul':
        sc_adata = ad.read(path + 'store/paired/sc_ul.h5ad')
        sc_adata.var_names_make_unique()

        st_adata = sc.read_visium(path=path + 'data/osu/SL70A10S2_preproccesed/outs')
        st_adata.obsm['spatial'] = st_adata.obsm['spatial'].astype(int)
        st_adata.var_names_make_unique()

        # ctp = pd.read_csv(path + 'output/stereoscope_CTP-5.csv', index_col=0, header=1).\
        #     drop(['max_value','Unnamed: 26'], axis=1)

        ctxc = pd.DataFrame(sc_adata.obs['annotations']).reset_index()
        G = nx.from_pandas_edgelist(
            ctxc,
            source='annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['annotations']).tolist()]\
            [np.unique(ctxc['index']).tolist()]
        ct = ct.T.reindex(sc_adata.obs_names) # reindex to match cell id

        
        # preprocessing (spots/cells) st_adata just to match with stereoscope_CTP-5.csv in ctp,
        # then asign ctp and ct
        st_adata.var["mt"] = st_adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_adata, qc_vars = ["mt"], inplace = True)
        sc.pp.filter_cells(st_adata, min_counts = 500)
        sc.pp.filter_cells(st_adata, min_genes = 500)
        # ctp = ctp.reindex(st_adata.obs_names) # reindex to match spot id
        # st_adata.obsm['true_deconvolution'] = ctp
        
        sc_adata.obsm['cell_type'] = ct

        # preprocessing (genes) both sc_adata and st_adata then taking intersection
        sc.pp.filter_genes(sc_adata, min_counts = 10)
        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]
        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum = 1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes = 7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span = 1
        )

        sc.pp.filter_genes(st_adata, min_cells=3)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        sc.pp.highly_variable_genes(st_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(st_adata, n_top_genes=2000)

        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()
        
    if data=='paired_ll':
        sc_adata = ad.read(path + 'store/paired/sc_ll.h5ad')
        sc_adata.var_names_make_unique()

        st_adata = sc.read_visium(path=path + 'data/osu/SL71A5S1_preproccesed/outs')
        st_adata.obsm['spatial'] = st_adata.obsm['spatial'].astype(int)
        st_adata.var_names_make_unique()

        # ctp = pd.read_csv(path + 'output/stereoscope_CTP-5.csv', index_col=0, header=1).\
        #     drop(['max_value','Unnamed: 26'], axis=1)

        ctxc = pd.DataFrame(sc_adata.obs['annotations']).reset_index()
        G = nx.from_pandas_edgelist(
            ctxc,
            source='annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['annotations']).tolist()]\
            [np.unique(ctxc['index']).tolist()]
        ct = ct.T.reindex(sc_adata.obs_names) # reindex to match cell id

        
        # preprocessing (spots/cells) st_adata just to match with stereoscope_CTP-5.csv in ctp,
        # then asign ctp and ct
        st_adata.var["mt"] = st_adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_adata, qc_vars = ["mt"], inplace = True)
        sc.pp.filter_cells(st_adata, min_counts = 500)
        sc.pp.filter_cells(st_adata, min_genes = 500)
        # ctp = ctp.reindex(st_adata.obs_names) # reindex to match spot id
        # st_adata.obsm['true_deconvolution'] = ctp
        
        sc_adata.obsm['cell_type'] = ct

        # preprocessing (genes) both sc_adata and st_adata then taking intersection
        sc.pp.filter_genes(sc_adata, min_counts = 10)
        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]
        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum = 1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes = 7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span = 1
        )

        sc.pp.filter_genes(st_adata, min_cells=3)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        sc.pp.highly_variable_genes(st_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(st_adata, n_top_genes=2000)

        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()

    if data=='oliver_simu':
        sc_adata = ad.read_h5ad(path + 'sc_simu_OliverEC.h5ad')
        st_adata = ad.read_h5ad(path + 'st_simu_OliverEC.h5ad')
        
        sc_adata.var_names_make_unique()
        st_adata.var_names_make_unique()
        
        ctxc = pd.DataFrame(sc_adata.obs['cell_type']).reset_index()
        G = nx.from_pandas_edgelist(
            ctxc,
            source='cell_type',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['cell_type']).tolist()]\
            [np.unique(ctxc['index']).tolist()]
        ct = ct.T.reindex(sc_adata.obs_names) # reindex to match cell id
        
        ctxc = sc_adata.obs['mapping'].reset_index() # cell to spot adj list
        G = nx.from_pandas_edgelist(
            ctxc,
            source= 'mapping', #'annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        mapping = adj_df.loc[np.unique(ctxc['mapping']).tolist()]\
            [np.unique(ctxc['index']).tolist()] # cell to spot matrix
        mapping = mapping[sc_adata.obs_names].reindex(st_adata.obs_names) # reindex to match both cell and spot data
        
        ctp = pd.DataFrame(data=None, index=st_adata.obs_names, columns=ct.columns)
        for loc in ctp.index:
            for k, v in literal_eval(st_adata.obs['cell_mixture'].loc[loc]).items():
                ctp[k].loc[loc] = v
        ctp = ctp.fillna(0)
        
        sc_adata.obsm['cell_type'] = ct
        sc_adata.obsm['mapping'] = mapping.T
        
        st_adata.obsm['true_deconvolution'] = ctp
        st_adata.obsm['mapping'] = mapping
        
        st_adata.var["mt"] = st_adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_adata, qc_vars = ["mt"], inplace = True)
        
        # preprocessing (genes) both sc_adata and st_adata then taking intersection
        sc.pp.filter_genes(sc_adata, min_counts = 10)
        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]
        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum = 1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes = 7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span = 1
        )

        sc.pp.filter_genes(st_adata, min_cells=3)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        sc.pp.highly_variable_genes(st_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(st_adata, n_top_genes=2000)

        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()
        
    if data=='simu_spatial':
        sc_adata = ad.read_h5ad(path + 'sc_simu_spatialcorr_T=5.h5ad')
        st_adata = ad.read_h5ad(path + 'st_simu_spatialcorr_T=5.h5ad')
        
        sc_adata.var_names_make_unique()
        st_adata.var_names_make_unique()
        
        ctxc = pd.DataFrame(sc_adata.obs['cell_type']).reset_index()
        G = nx.from_pandas_edgelist(
            ctxc,
            source='cell_type',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['cell_type']).tolist()]\
            [np.unique(ctxc['index']).tolist()]
        ct = ct.T.reindex(sc_adata.obs_names) # reindex to match cell id
        
        ctxc = sc_adata.obs['cell2spot_tag'].reset_index() # cell to spot adj list
        G = nx.from_pandas_edgelist(
            ctxc,
            source= 'cell2spot_tag', #'annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        mapping = adj_df.loc[np.unique(ctxc['cell2spot_tag']).tolist()]\
            [np.unique(ctxc['index']).tolist()] # cell to spot matrix
        mapping = mapping[sc_adata.obs_names].reindex(st_adata.obs_names) # reindex to match both cell and spot data
        
        ctp = pd.DataFrame(data=None, index=st_adata.obs_names, columns=ct.columns)
        for loc in ctp.index:
            items = st_adata.obs['cell_types'].loc[loc]
            item_list = items.split(',')
            item_counts = dict(Counter(item_list))
            for k, v in item_counts.items():
                ctp[k].loc[loc] = v
        ctp = ctp.fillna(0)
        
        sc_adata.obsm['cell_type'] = ct
        sc_adata.obsm['mapping'] = mapping.T
        
        st_adata.obsm['true_deconvolution'] = ctp
        st_adata.obsm['mapping'] = mapping
        
        st_adata.var["mt"] = st_adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_adata, qc_vars = ["mt"], inplace = True)
        
        # preprocessing (genes) both sc_adata and st_adata then taking intersection
        sc.pp.filter_genes(sc_adata, min_counts = 10)
        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]
        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum = 1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes = 7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span = 1
        )

        sc.pp.filter_genes(st_adata, min_cells=3)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        sc.pp.highly_variable_genes(st_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(st_adata, n_top_genes=2000)

        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()
        
    if data=='simu_spatial_rare':
        sc_adata = ad.read_h5ad(path + 'sc_simu_spatialcorr_T=5.h5ad')
        
        # Identify cells of the specific cell type, e.g., 'B cell lineage'
        b_cell_indices = np.where(sc_adata.obs['cell_type'] == 'B cell lineage')[0]

        # Randomly select 5% of cells of the specific cell type
        num_b_cells_to_select = int(0.005 * len(b_cell_indices))
        selected_b_cells_indices = np.random.choice(b_cell_indices, size=num_b_cells_to_select, replace=False)

        # Identify indices of cells of other cell types
        other_cell_indices = np.where(sc_adata.obs['cell_type'] != 'B cell lineage')[0]

        # Combine selected B cells with cells of other types
        selected_indices = np.concatenate([selected_b_cells_indices, other_cell_indices])

        # Subset the anndata object to include only the selected cells
        sc_adata = sc_adata[selected_indices, :]
        
        st_adata = ad.read_h5ad(path + 'st_simu_spatialcorr_T=5.h5ad')
        
        # Identify cells of the specific cell type, e.g., 'B cell lineage'
        b_cell_indices = np.where(st_adata.obs['cell_types'] == 'B cell lineage')[0]

        # Randomly select 5% of cells of the specific cell type
        num_b_cells_to_select = int(0.005 * len(b_cell_indices))
        selected_b_cells_indices = np.random.choice(b_cell_indices, size=num_b_cells_to_select, replace=False)

        # Identify indices of cells of other cell types
        other_cell_indices = np.where(st_adata.obs['cell_types'] != 'B cell lineage')[0]

        # Combine selected B cells with cells of other types
        selected_indices = np.concatenate([selected_b_cells_indices, other_cell_indices])

        # Subset the anndata object to include only the selected cells
        st_adata = st_adata[selected_indices, :]        
        
        # sc_adata = adata_subset#.copy()
        # st_adata = adata_subset#.copy()
        
        sc_adata.var_names_make_unique()
        st_adata.var_names_make_unique()
        
        ctxc = pd.DataFrame(sc_adata.obs['cell_type']).reset_index()
        G = nx.from_pandas_edgelist(
            ctxc,
            source='cell_type',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        ct = adj_df.loc[np.unique(ctxc['cell_type']).tolist()]\
            [np.unique(ctxc['index']).tolist()]
        ct = ct.T.reindex(sc_adata.obs_names) # reindex to match cell id
        
        ctxc = sc_adata.obs['cell2spot_tag'].reset_index() # cell to spot adj list
        G = nx.from_pandas_edgelist(
            ctxc,
            source= 'cell2spot_tag', #'annotations',
            target='index',
            # create_using=nx.DiGraph()
        )
        adj_df = pd.DataFrame(
            nx.adjacency_matrix(G).todense(),
            index=G.nodes,
            columns=G.nodes
        )
        mapping = adj_df.loc[np.unique(ctxc['cell2spot_tag']).tolist()]\
            [np.unique(ctxc['index']).tolist()] # cell to spot matrix
        mapping = mapping[sc_adata.obs_names].reindex(st_adata.obs_names) # reindex to match both cell and spot data
        
        ctp = pd.DataFrame(data=None, index=st_adata.obs_names, columns=ct.columns)
        for loc in ctp.index:
            items = st_adata.obs['cell_types'].loc[loc]
            item_list = items.split(',')
            item_counts = dict(Counter(item_list))
            for k, v in item_counts.items():
                ctp[k].loc[loc] = v
        ctp = ctp.fillna(0)
        
        sc_adata.obsm['cell_type'] = ct
        sc_adata.obsm['mapping'] = mapping.T
        
        st_adata.obsm['true_deconvolution'] = ctp
        st_adata.obsm['mapping'] = mapping
        
        st_adata.var["mt"] = st_adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_adata, qc_vars = ["mt"], inplace = True)
        
        # preprocessing (genes) both sc_adata and st_adata then taking intersection
        sc.pp.filter_genes(sc_adata, min_counts = 10)
        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]
        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum = 1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes = 7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span = 1
        )

        sc.pp.filter_genes(st_adata, min_cells=3)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        sc.pp.highly_variable_genes(st_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(st_adata, n_top_genes=2000)

        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()

    return sc_adata, st_adata

