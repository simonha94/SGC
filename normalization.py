import numpy as np
import scipy.sparse as sp
import torch

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalized_adjacency(adj):
   """graph project, sh 2021"""
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def random_walk_adjacency(adj):
   """graph project, sh 2021"""
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -1).flatten()
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).tocoo()

def aug_random_walk_adjacency(adj):
   """graph project, sh 2021"""
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -1).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).tocoo()

def first_order_cheby_adjacency(adj):
   """graph project, sh 2021"""
   adj = normalized_adjacency(adj)
   return sp.eye(adj.shape[0]) + adj

def fetch_normalization(type):
   switcher = {                                 # It is not distinguished between "hat" and "not hat" in the comment formulars!!!
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'AugRandAdj': aug_random_walk_adjacency, # A' = (D + I)^-1 * (A + I)
       'NormAdj': normalized_adjacency,         # A' =  D^-1/2 * A * D^-1/2
       'RandAdj': random_walk_adjacency,        # A' =  D^-1 * A
       'ChebyAdj': first_order_cheby_adjacency  # A' = (I + D^-1/2 * A * D^-1/2)
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
