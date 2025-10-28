"""Module containing classes for different neural network layers for Hierarchical CADNet neural architecture."""


import tensorflow as tf
import math
import math
from typing import Optional, Tuple

class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0



class GraphCNNLayer(tf.keras.layers.Layer):
    """Graph convolutional layer that uses the adjacency matrix."""
    def __init__(self, filters, name="GCNN", **kwargs):
        super(GraphCNNLayer, self).__init__(name=name, **kwargs)

        self.num_filters = filters
        self.weight_decay = 0.0005
        self.W = None
        self.W_I = None
        self.b = None
    
    #defer weights creation until the input_shape is known
    def build(self, input_shape): 
        V_shape, _ = input_shape
        num_features = V_shape[1]
        
        W_dim = [num_features, self.num_filters]
        
        b_dim = [self.num_filters]

        W_stddev = math.sqrt(1.0 / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
      
        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev), #standard deviation
            regularizer=tf.keras.regularizers.l2(self.weight_decay), #weight_decay is lambda
            trainable=True,
            name="W")


        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")
        
    #D^(-1/2) (A+I) D^(-1/2)VW+b
    def call(self, input, training=None):
        V, A = input
        n = tf.matmul(A, V)
        output = tf.matmul(n, self.W) +  self.b

        return output


class GraphEmbeddingLayer(tf.keras.layers.Layer):
    """Graph embedding layer for summarizing learned information."""
    def __init__(self, filters, name="GEmbed", **kwargs):
        super(GraphEmbeddingLayer, self).__init__(name=name, **kwargs)

        self.num_filters = filters
        self.weight_decay = 0.0005
        self.W = None
        self.b = None

    def build(self, input_shape):
        V_shape = input_shape
        num_features = V_shape[1]
        W_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]
        W_stddev = 1.0 / math.sqrt(num_features)

        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="weight")

        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")

    def call(self, V, training=None):
        output = tf.matmul(V, self.W) + self.b
        return output


class PreProcessingAttrLayer(tf.keras.layers.Layer):
    """Graph embedding layer for summarizing learned information."""
    def __init__(self, filters, name="GEmbed", **kwargs):
        super(PreProcessingAttrLayer, self).__init__(name=name, **kwargs)

        self.num_filters = filters
        self.weight_decay = 0.0005
        self.W = None
        self.b = None

    def build(self, input_shape):
        V_shape = input_shape
        num_features = V_shape[1]
        W_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]
        W_stddev = 1.0 / math.sqrt(num_features)

        self.W = self.add_weight(
            shape=W_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=W_stddev),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            name="weight")

        self.b = self.add_weight(
            shape=b_dim,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name="bias")

    def call(self, V, training=None):
        output = tf.matmul(V, self.W) + self.b
        return output



class PreProcessingCurvLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 kernel_size=3,
                 activation='relu',
                 name="PreprocessingCur",
                 **kwargs):
        super(PreProcessingCurvLayer, self).__init__(name=name, **kwargs)
    
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='same', 
            name=f"{name}_conv2d"
        )
        
 
        self.global_pool = tf.keras.layers.GlobalAvgPool2D(
            name=f"{name}_global_pool"
        )
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        
        gaussian_curv = inputs[:, :25]  # (batch_size, 25)
        mean_curv = inputs[:, 25:]      # (batch_size, 25)
        
        gaussian_curv = tf.reshape(gaussian_curv, (batch_size, 5, 5, 1))  # (batch_size, 5, 5, 1)
        mean_curv = tf.reshape(mean_curv, (batch_size, 5, 5, 1))          # (batch_size, 5, 5, 1)
        
        #  (batch_size, 5, 5, 2)
        curvatures = tf.concat([gaussian_curv, mean_curv], axis=-1)
        
        #  (batch_size, 5, 5, filters)
        x = self.conv2d(curvatures)
        
        #  (batch_size, filters)
        output = self.global_pool(x)
        
        return output