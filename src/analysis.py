import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def senescence(data, sc_adata, senescent_gene_list, rec=False):
    data = data
    sc_adata = sc_adata.copy()
    senescent_gene_list = senescent_gene_list
    rec = rec
    
    if data == 'ipf_lung' and rec == False:
        sc_adata.X = sc_adata.layers["counts"]
    if data != 'ipf_lung' and rec == True:
        sc_adata.X = sc_adata.layers["counts"]
    
    cell_data = sc_adata.to_df()
    cell_data['cell_type'] = sc_adata.obs['annotations']
    # Get the genes in the list that are present in the expression data
    senescent_marker_genes = cell_data.columns[cell_data.columns.isin(senescent_gene_list)].tolist()
    # 1. Extract the expression profiles of the marker genes for the "senescent" state
    senescent_expression = cell_data.loc[:, senescent_marker_genes]
    # 2. Calculate the average expression of the marker genes across all cells
    senescent_expression_mean = senescent_expression.mean(axis=1)
    # 3. Identify the cells that have a higher than average expression of the marker genes
    senescent_cells = cell_data.index[senescent_expression_mean > np.percentile(senescent_expression_mean, 95)]
    # 4. Count the number of cells in each of the 24 cell types that have a "senescent" state
    senescent_counts = cell_data.loc[senescent_cells, 'cell_type'].value_counts()

    return senescent_cells, senescent_counts

def reannotate(sc_adata, most_common, senescent_cells, most_common_rec, senescent_cells_rec):
    sc_adata = sc_adata
    most_common = most_common
    senescent_cells = senescent_cells
    most_common_rec = most_common_rec
    senescent_cells_rec = senescent_cells_rec
    
    sc_adata.obs['annotations_updated'] = sc_adata.obs['annotations'].astype(str)
    for cell_type in most_common:
        cells_in_type = sc_adata.obs['annotations'].loc[sc_adata.obs['annotations'] == cell_type].index
        for cell in cells_in_type:
            if cell in senescent_cells:
                sc_adata.obs.loc[cell, 'annotations_updated'] = cell_type + ':senescent'
            else:
                sc_adata.obs.loc[cell, 'annotations_updated'] = cell_type + ':non-senescent'
    if most_common_rec != None:
        for cell_type in most_common_rec:
            cells_in_type = sc_adata.obs['annotations'].loc[sc_adata.obs['annotations'] == cell_type].index
            for cell in cells_in_type:
                if cell in senescent_cells_rec:
                    sc_adata.obs.loc[cell, 'annotations_updated'] = cell_type + ':senescent'
                else:
                    sc_adata.obs.loc[cell, 'annotations_updated'] = cell_type + ':non-senescent'
    sc_adata.obs['annotations_updated'] = sc_adata.obs['annotations_updated'].astype('category')
    
    return sc_adata

def pl_senesence(sc_adata, senescent_cells, senescent_cells_rec, most_common_rec, save=None):
    sc_adata = sc_adata
    senescent_cells = senescent_cells
    senescent_cells_rec = senescent_cells_rec
    most_common_rec = most_common_rec
    
    counts_df = sc_adata.obs['annotations'].value_counts().to_frame().rename(columns={'annotations': 'total counts'}).join(
        sc_adata.obs['annotations'].loc[senescent_cells].value_counts().to_frame().rename(columns={'annotations': 'senescent counts'})
    )
    if most_common_rec != None:
        counts_df = counts_df.join(
            sc_adata.obs['annotations'].loc[senescent_cells_rec].value_counts().to_frame().rename(columns={'annotations': 'senescent counts 2'})
        )

        def calculate_max_value(df, column1, column2, index_list):
            """
            Calculate the maximum value between two columns of a DataFrame for specific indices,
            returning the values of the first column if the index is not matched.

            Args:
                df (pandas.DataFrame): The input DataFrame.
                column1 (str): The name of the first column.
                column2 (str): The name of the second column.
                index_list (list): List of indices to check for matches.

            Returns:
                pandas.DataFrame: The DataFrame with the 'MaxValue' column added.
            """
            # Create 'MaxValue' column
            df['senescent counts'] = df.apply(lambda row: max(row[column1], row[column2]) if row.name in index_list else row[column1], axis=1)

            return df

        counts_df = calculate_max_value(counts_df, 'senescent counts', 'senescent counts 2', most_common_rec)

    # Define the data for the chart
    categories = counts_df.index.tolist()
    scenescent_counts = counts_df['senescent counts']
    total_counts = counts_df['total counts']

    # Calculate the remaining counts
    remaining_counts = np.subtract(total_counts, scenescent_counts)

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Create an array to use as the x-axis
    x = np.arange(len(categories))

    # Create the stacked bars
    ax.bar(x, remaining_counts, label='Non-Scenescent Counts')
    ax.bar(x, scenescent_counts, bottom=remaining_counts, label='Scenescent Counts')

    # Set the labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=90)
    ax.set_ylabel('Number of Cells')
    ax.set_title('')

    # Add a legend
    ax.legend()

    plt.savefig(save, bbox_inches='tight') if save != None else None
    # Show the chart
    plt.show()
    
def pl_volcano(result, FDR=0.01, LOG_FOLD_CHANGE=1.5, gene_names_to_show=None, save=None):
    result = result
    FDR = FDR
    LOG_FOLD_CHANGE = LOG_FOLD_CHANGE
    gene_names_to_show = gene_names_to_show

    result["-logQ"] = -np.log(result["pvals"].astype("float"))
    # lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    lowqval_de_pov = result.loc[result["logfoldchanges"] > LOG_FOLD_CHANGE]
    lowqval_de_neg = result.loc[result["logfoldchanges"] < -LOG_FOLD_CHANGE]
    other_de = result.loc[abs(result["logfoldchanges"]) <= LOG_FOLD_CHANGE]

    fig, ax = plt.subplots(figsize=(6,5))
    sns.regplot(
        x=other_de["logfoldchanges"],
        y=other_de["-logQ"],
        fit_reg=False,
        scatter_kws={"color": "grey", "alpha": 0.5, "s": 6},
    )
    sns.regplot(
        x=lowqval_de_pov["logfoldchanges"],
        y=lowqval_de_pov["-logQ"],
        fit_reg=False,
        scatter_kws={"color": "#ff7f0e", "alpha": 0.5, "s": 6},
    )
    sns.regplot(
        x=lowqval_de_neg["logfoldchanges"],
        y=lowqval_de_neg["-logQ"],
        fit_reg=False,
        scatter_kws={"color": "#1f77b4", "alpha": 0.5, "s": 6},
    )
        
    # Add specified gene names to the plot
    if gene_names_to_show != None:
        for i, row in result.iterrows():
            if row["names"] in gene_names_to_show:
                x = row["logfoldchanges"]
                y = row["-logQ"]
                gene_name = row["names"]
                ax.text(x, y, gene_name, fontsize=8)
                
                # Draw an arrow from (x, y) to the label
                # ax.annotate(gene_name, xy=(x, y), xytext=(x-15, y+250),
                #     arrowprops=dict(arrowstyle='->'), fontsize=8)

    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log Q-value")

    # if title is None:
    #     title = group_key.replace("_", " ")
    # plt.title(title)

    plt.savefig(save) if save else None
    plt.show()