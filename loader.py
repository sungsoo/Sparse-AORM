import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sps
# from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, csc_matrix
from scipy.sparse.csgraph import connected_components

def load_networkx_pickle(pickle_path, sparse=True):
    graph = nx.read_gpickle(pickle_path)
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    adj_sparse_matrix = nx.adjacency_matrix(graph)

    if (sparse == True):
        adj_matrix = coo_matrix(adj_sparse_matrix)
        return adj_matrix, nodes, edges
    else:
        adj_dense_matrix = np.array(adj_sparse_matrix.todense())
        return adj_dense_matrix.astype(np.bool), nodes, edges

def is_disconnected_graph(adj_matrix):
    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    if n_components > 1:
        return True
    else:
        return False
        
def make_undirected(A):
    return np.heaviside(np.maximum(A, A.T), 0).astype(np.bool)

def get_delimiter(input_file_path):
    delimiter = " "
    if ".csv" in input_file_path:
        delimiter = ","
    elif ".tsv" in input_file_path:
        delimiter = "\t"
    elif ".txt" in input_file_path:
        delimiter = " "
    elif ".edges" in input_file_path:
        delimiter = " "
    elif ".mtx" in input_file_path:
        delimiter = " "        
    else:
        sys.exit('Format not supported.')

    return delimiter

def load_networks(filepath, sparse=False):

    delimiter = get_delimiter(filepath)
    df = pd.read_csv(filepath, comment='#', delimiter=delimiter, usecols=[0, 1], names=['FromNodeId', 'ToNodeId'])
    vals = np.unique(df[['FromNodeId', 'ToNodeId']])
    
    df2 = pd.DataFrame(0, index=vals, columns=vals)
    f = df2.index.get_indexer
    df2.values[f(df.FromNodeId), f(df.ToNodeId)] = 1

    if sparse == False:
        A = df2.values
        A = np.array(A, dtype=np.uint16)
        n, e = len(A), np.count_nonzero(A)
    else:
        D = np.array(df2.values, dtype=np.uint16)
        A = csr_matrix(D)
        n, e = A.shape[0], A.nnz

    return A, n, e  

def preprocess(input_file_path, sparse=True):  
    delimiter = get_delimiter(input_file_path)

    raw = np.loadtxt(input_file_path, comments=['#', '%', '@'], delimiter=delimiter, dtype=int)
  
    COL = raw.shape[1]

    if COL < 2:
        sys.exit('[Input format error.]')
    elif COL == 2:
        srcs = raw[:,0]
        dest = raw[:,1]
        weis = np.ones(len(srcs))
    elif COL == 3:
        print('[weighted graph detected.]')
        srcs = raw[:,0]
        dest = raw[:,1]
        weis = raw[:,2]

    max_id = int(max(max(srcs), max(dest)))
    num_nodes = max_id + 1
    nodes_to_embed = range(int(max_id)+1)

    if max(srcs) != max(dest):
        srcs = np.append(srcs,max(max(srcs), max(dest)))
        dest = np.append(dest,max(max(srcs), max(dest)))
        weis = np.append(weis, 0)

    adj_matrix = sps.coo_matrix(sps.csr_matrix((weis, (srcs, dest))))
    adj_matrix.setdiag(0)

    if (sparse == False):
        return adj_matrix.todense(), adj_matrix.shape[0], adj_matrix.nnz
    else:
        return adj_matrix, adj_matrix.shape[0], adj_matrix.nnz