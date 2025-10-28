"""FilletRec network architecture"""

from src.layers import *


class FilletRecGCN(tf.keras.Model):

    def __init__(self, units, out_channel,filter, rate, num_classes, num_layers=3):
        super(FilletRecGCN, self).__init__()
        self.num_layers = num_layers

        self.pre_curvature=PreProcessingAttrLayer(filters=out_channel,name="Pre_curvature")
        self.bn_curvature = tf.keras.layers.BatchNormalization(name="BN_curvature")
        
        self.pre_attr=PreProcessingAttrLayer(filters=out_channel,name="Pre_attr") 
        self.bn_attr=tf.keras.layers.BatchNormalization(name="BN_attr")
        
        
        self.pre_topo=PreProcessingAttrLayer(filters=out_channel,name="Pre_topo") 
        self.bn_topo=tf.keras.layers.BatchNormalization(name="BN_topo")
        

        self.ge_start = GraphEmbeddingLayer(filters=units, name="GE_start")
        self.bn_start = tf.keras.layers.BatchNormalization(name="BN_start")
       

        for i in range(1, self.num_layers + 1):
            setattr(self, f"gcnn_1_{i}", GraphCNNLayer(filters=filter[i-1], name=f"GCNN_1_{i}"))
            setattr(self, f"bn_1_{i}", tf.keras.layers.BatchNormalization(name=f"BN_1_{i}"))
            setattr(self, f"dp_1_{i}", tf.keras.layers.Dropout(rate=rate, name=f"DP_1_{i}"))
            
            
        self.ge_1 = GraphEmbeddingLayer(filters=out_channel, name="GE_1")
        self.bn_1 = tf.keras.layers.BatchNormalization(name="BN_1")
        self.dp_1 = tf.keras.layers.Dropout(rate=rate, name="DP_1")
        
        self.ge_2 = GraphEmbeddingLayer(filters=out_channel, name="GE_2")
        self.bn_2 = tf.keras.layers.BatchNormalization(name="BN_2")
        self.dp_2 = tf.keras.layers.Dropout(rate=rate, name="DP_2")
        
        self.ge_3 = GraphEmbeddingLayer(filters=out_channel, name="GE_3")
        self.bn_3 = tf.keras.layers.BatchNormalization(name="BN_3")
        self.dp_3 = tf.keras.layers.Dropout(rate=rate, name="DP_3")
     
    
        self.ge_final = GraphEmbeddingLayer(filters=num_classes, name="GE_final")
        
       
        self.softmax = tf.keras.layers.Softmax()
        self.tanh=tf.keras.activations.tanh
        self.sigmoid=tf.keras.activations.sigmoid
        
    
    
    def call(self, inputs, training=False):   
        V_1_curvature, V_1_attr,V_1_topo, A_1, E_1, E_2, E_3=inputs
        
        x_1_curvature=self.pre_curvature(V_1_curvature)
        x_1_curvature=tf.nn.relu(x_1_curvature)
        
        x_1_attr=self.pre_attr(V_1_attr)
        x_1_attr=tf.nn.relu(x_1_attr)
        
        x_1_topo=self.pre_topo(V_1_topo)
        x_1_topo=tf.nn.relu(x_1_topo)
        
        
        x_1 = self.ge_start(x_1_curvature)
        x_1 = self.bn_start(x_1, training=training)
        x_1 = tf.nn.relu(x_1)

        for i in range(1, self.num_layers + 1):
            r_1 = getattr(self, f"gcnn_1_{i}")([x_1,A_1 ])
            r_1 = getattr(self, f"bn_1_{i}")(r_1, training=training)
            r_1 = tf.nn.tanh(r_1)
            r_1 = getattr(self, f"dp_1_{i}")(r_1, training=training)
            x_1+=r_1
        
        
        x_1=self.ge_1(x_1)
        x_1 = self.bn_1(x_1, training=training)
        x_1 = tf.nn.relu(x_1)
        x_1 = self.dp_1(x_1, training=training)
        
        
        
        x_1=tf.concat([x_1,x_1_attr],axis=1)
        x_1=self.ge_2(x_1)
        x_1=self.bn_2(x_1)
        x_1=tf.nn.relu(x_1)
        x_1 = self.dp_2(x_1, training=training)
        
        
        x_1=tf.concat([x_1,x_1_topo],axis=1)
        x_1=self.ge_3(x_1)
        x_1=self.bn_3(x_1)
        x_1=tf.nn.relu(x_1)
        x_1 = self.dp_3(x_1, training=training)
        
        x=self.ge_final(x_1)
        x = self.softmax(x)

        return x
    