"""Module containing useful functions such as dataloader function and data normalize function."""

import tensorflow as tf
import h5py
import numpy as np


EPSILON=1e-8

def normalize_curvature(data):
    """Normalize data."""
   
    mean_curv = data[:, 25:]  # First 25 columns (H)
    gauss_curv = data[:, :25]  # Last 25 columns (K)
    
    # Compute min and max for each curvature type
    h_max = tf.reduce_max(mean_curv, axis=0)
    h_min = tf.reduce_min(mean_curv, axis=0)
    k_max = tf.reduce_max(gauss_curv, axis=0)
    k_min = tf.reduce_min(gauss_curv, axis=0)
    
    # Normalize each curvature type separately
    mean_curv_norm = (mean_curv - h_min) / (h_max - h_min + EPSILON)
    gauss_curv_norm = (gauss_curv - k_min) / (k_max - k_min + EPSILON)
    
    # Concatenate back together
    return tf.concat([ gauss_curv_norm,mean_curv_norm], axis=1)


def normalize_attr(data):
    """Normalize data."""
    data_max=tf.reduce_max(data,axis=0)
    data_min=tf.reduce_min(data,axis=0)
    
    data_norm = (data - data_min) / (data_max - data_min + EPSILON)
    return data_norm


def normalize_adj(adj_dense):
    """Adajcent matrix with self loop"""
  
    if len(adj_dense.shape) != 2:
        raise ValueError("Adj shape is not correct!")
    
    N = adj_dense.shape[0]  
    
    #A+I
    adj_with_self_loops = adj_dense + tf.eye(N, dtype=adj_dense.dtype)
    
    #D
    degrees = tf.reduce_sum(adj_with_self_loops, axis=1) 
    
    #D^(-1/2)
    d_inv_sqrt = tf.pow(degrees, -0.5)
    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), 0.0, d_inv_sqrt)  
    
    # D^(-1/2)
    D_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    
    # D^(-1/2) (A+I) D^(-1/2)
    normalized_adj = tf.matmul(tf.matmul(D_inv_sqrt, adj_with_self_loops), D_inv_sqrt)
    
    return normalized_adj


def dataloader(file_path):
    """Load dataset with edge convexity information."""
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = tf.Variable(np.array(group.get("V_1")), dtype=tf.dtypes.float32, name="V_1")

        labels = np.array(group.get("labels"), dtype=np.int16)

        names=np.array(group.get("names"),dtype='S')
        
        #brep adjancy 
        A_1_idx = np.array(group.get("A_1_idx"))
        A_1_values = np.array(group.get("A_1_values"))
        A_1_shape = np.array(group.get("A_1_shape"))
        A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape) 
        A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")
        
        E_1_idx = np.array(group.get("E_1_idx"))
        E_1_values = np.array(group.get("E_1_values"))
        E_1_shape = np.array(group.get("E_1_shape"))
        E_1_sparse = tf.SparseTensor(E_1_idx, E_1_values, E_1_shape)
        E_1 = tf.Variable(tf.sparse.to_dense(E_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_1")

        E_2_idx = np.array(group.get("E_2_idx"))
        E_2_values = np.array(group.get("E_2_values"))
        E_2_shape = np.array(group.get("E_2_shape"))
        E_2_sparse = tf.SparseTensor(E_2_idx, E_2_values, E_2_shape)
        E_2 = tf.Variable(tf.sparse.to_dense(E_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_2")

        E_3_idx = np.array(group.get("E_3_idx"))
        E_3_values = np.array(group.get("E_3_values"))
        E_3_shape = np.array(group.get("E_3_shape"))
        E_3_sparse = tf.SparseTensor(E_3_idx, E_3_values, E_3_shape)
        E_3 = tf.Variable(tf.sparse.to_dense(E_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_3")
        
        A_1_normalize=normalize_adj(A_1)
        
        
        V_1_curvature=tf.abs(V_1[:,:50])
        V_1_attr=V_1[:,50:51] #surface width
        V_1_topo=V_1[:,51:52] #adjacent surface angle

        
        yield [V_1_curvature, V_1_attr, V_1_topo, A_1_normalize, E_1, E_2, E_3], labels
        

    hf.close()
    