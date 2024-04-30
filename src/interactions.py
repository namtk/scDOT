import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def type_table(adj, node_annotations, loaded_table=None, save_table=None, nodes_of_interest=None, weight_threshold=0.0, layout='spring', save_plot=None):
    adj = adj
    node_annotations = node_annotations
    loaded_table = loaded_table
    save_table = save_table
    nodes_of_interest = nodes_of_interest
    weight_threshold = weight_threshold
    layout = layout
    save_plot = save_plot

    if loaded_table == None:
        graph = nx.from_numpy_matrix(adj)
        # Create the empty table
        idx = list(set(node_annotations.values()))
        type_table = pd.DataFrame(
            0, 
            index=idx, 
            columns=idx
        )

        # Iterate over all edges in the graph and increment the table entries accordingly
        for u, v in graph.edges():
            u_type = node_annotations[u]
            v_type = node_annotations[v]
            type_table.loc[u_type, v_type] += 1
            type_table.loc[v_type, u_type] += 1
        type_table_normed = type_table.div(type_table.sum(axis=1), axis=0)
        type_table_normed.to_csv(save_table) if save_table != None else None
    else:
        type_table_normed = pd.read_csv(loaded_table, index_col=0)

    G = nx.from_pandas_adjacency(type_table_normed)
    sr = pd.Series(node_annotations).value_counts()
    # Define the nodes of interest
    nodes_of_interest = G.nodes() if nodes_of_interest == None else nodes_of_interest
    # Define the weight threshold

    # Get the edges that connect the nodes of interest and meet the weight threshold
    edges = [(u, v, d) for u, v, d in G.edges(data=True)
                if (u in nodes_of_interest or v in nodes_of_interest) and d['weight'] >= weight_threshold]

    # Create a subgraph with only the edges that connect the nodes of interest
    subgraph = nx.Graph()
    subgraph.add_edges_from(edges)

    # Set the node sizes from the series
    sizes = [sr[node] for node in subgraph.nodes]
    pos = nx.spring_layout(subgraph) if layout == 'spring' else nx.circular_layout(subgraph)
    sizes_scaled = [i*.1 for i in sizes]
    nx.draw_networkx_nodes(
        subgraph, 
        pos=pos, 
        node_size=sizes_scaled, 
        linewidths=.5, 
        alpha=0.5
    )
    nx.draw_networkx_labels(subgraph, pos=pos, font_size=5)
    widths = nx.get_edge_attributes(subgraph, 'weight')
    widths_scaled = {k: v*5 for (k,v) in widths.items()}
    nx.draw_networkx_edges(
        subgraph, 
        pos, 
        edgelist = widths_scaled.keys(), 
        width=list(widths_scaled.values()), 
        edge_color='gray', 
        alpha=0.6
    )

    # Show the plot
    plt.savefig(save_plot) if save_plot != None else None
    plt.show()

    return type_table_normed if loaded_table == None else None

def neigh_of_interest(adj, loaded_subgraph, node_annotations, desired_category, n_sizes, n_alphas, clrmap, save_graph=None, save_plot=None):
    adj = adj
    loaded_subgraph = loaded_subgraph
    node_annotations = node_annotations
    desired_category = desired_category
    n_sizes = n_sizes
    n_alphas = n_alphas
    clrmap = clrmap
    save_graph = save_graph
    save_plot = save_plot

    if loaded_subgraph == None:
        # Identify the nodes of the desired category and their neighbors
        nodes_to_keep = np.array([k for k, v in node_annotations.items() if v in desired_category])
        neighbors_to_keep = np.unique(np.nonzero(adj[nodes_to_keep, :])[1])

        # Create a subgraph consisting of the nodes of the desired category and their neighbors
        node_list = list(nodes_to_keep) + list(neighbors_to_keep)
        edge_list = []
        for i in node_list:
            for j in node_list:
                if adj[i, j] == 1:
                    edge_list.append((i, j))
        subgraph = nx.Graph()
        subgraph.add_nodes_from(node_list)
        subgraph.add_edges_from(edge_list)
        nx.write_gpickle(subgraph, save_graph) if save_graph != None else None
    elif isinstance(loaded_subgraph, str):
        subgraph = nx.read_gpickle(subgraph)
    else:
        subgraph = loaded_subgraph
    
    node_sizes = [n_sizes[0] 
                  if node_annotations[node] in desired_category 
                  else n_sizes[1] for node in subgraph.nodes()] if isinstance(n_sizes, list) else n_sizes
    node_alphas = [n_alphas[0] 
                   if node_annotations[node] in desired_category 
                   else n_alphas[1] for node in subgraph.nodes()] if isinstance(n_alphas, list) else n_alphas

    # Get unique node values from the annotations dictionary
    unique_values = set(node_annotations.values())

    if clrmap == 'binary':
        node_colors = ['red' if node_annotations[node] in desired_category else 'grey' for node in subgraph.nodes()]
    else:
        if isinstance(clrmap, list):
            # Create a custom colormap based on the provided RGB values
            custom_colormap = mcolors.ListedColormap(clrmap)
            # Create a color mapping dictionary based on unique node values
            color_mapping = {value: custom_colormap(i) for i, value in enumerate(unique_values)}
        else:
            # Create a colormap based on the number of unique values
            color_map = cm.get_cmap(clrmap, len(unique_values))
            # Create a color mapping dictionary based on unique node values
            color_mapping = {value: color_map(i) for i, value in enumerate(unique_values)}
        # Create a list of colors based on node values
        node_colors = [color_mapping[node_annotations[node]] for node in subgraph.nodes()]
    
    # Plot the subgraph with different node colors
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(subgraph)  # Specify the layout algorithm
    nx.draw(
        subgraph, 
        pos, 
        node_color=node_colors, 
        with_labels=False, 
        font_color='white',
        node_size=node_sizes, 
        width=0.1, 
        alpha=node_alphas
    )

    # Create legend
    if clrmap == 'binary':
        red_patch = plt.Line2D([], [], marker='o', markersize=10, markerfacecolor='red', linestyle='', label=desired_category)
        white_patch = plt.Line2D([], [], marker='o', markersize=10, markerfacecolor='grey', linestyle='', label='Other cell types')
        plt.legend(handles=[red_patch, white_patch], loc='best')
    else:
        # Create legend only for nodes present in the subgraph
        legend_labels = []
        legend_handles = []
        for node in subgraph.nodes():
            value = node_annotations[node]
            if value not in legend_labels:
                legend_labels.append(value)
                legend_handles.append(plt.Line2D([], [], marker='o', markersize=8, linestyle='', color=color_mapping[value], label=value))
        plt.legend(handles=legend_handles)

    plt.savefig(save_plot) if save_plot != None else None
    plt.show()

    return subgraph if loaded_subgraph == None else None

def cell_in_region(coupling, region, adj, subgraph):
    coupling = coupling
    region = region
    adj = adj
    subgraph = subgraph

    cell_list = coupling@region
    cell_in_region = np.argsort(cell_list)[::-1][:region.sum() * 10] # 10 cells per spot
    cell_in_region = np.intersect1d(subgraph.nodes(), cell_in_region) if subgraph != None else cell_in_region

    edge_list = []
    for i in cell_in_region:
        for j in cell_in_region:
            if adj[i, j] == 1:
                edge_list.append((i, j))
    G = nx.Graph()
    G.add_nodes_from(cell_in_region)
    G.add_edges_from(edge_list)

    return G