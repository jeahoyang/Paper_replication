import numpy as np
from scipy.sparse import load_npz

# 1. Load sparse adjacency matrix
adj_matrix = load_npz("C:\\Users\\ADS_Lab\\Desktop\\Jaeho\\git\\GATOmics\\GATOmics\\data\\pan-cancer\\PP.adj.npz")

# 2. Convert to COO format (row, col, value)
adj_coo = adj_matrix.tocoo()

# 3. Save to TXT as edge list: from_node, to_node, weight
with open("PP.adj.txt", "w") as f:
    for i, j, v in zip(adj_coo.row, adj_coo.col, adj_coo.data):
        f.write(f"{i}\t{j}\t{v}\n")
