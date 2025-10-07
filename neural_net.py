import random
import numpy as np

class node():
    def __init__ (self, layer_arr, weight_arr):
        self.layer_arr = layer_arr
        self.weight_arr = weight_arr

def construct_layers():
    node_arr = np.array([])
    for i in range(1024): # first layer generation for input of 1024 words
        np.append(node_arr,
                  node(np.array([512])),
                  np.array([random.random() for i in range(512)]))
    
    # last_layer = ()
    # for i in range(10):
    #     last_layer[i] = node(any, any) #last layer does not have a next layer

    # layer_two = ()
    # for i in range(100): # generate a hidden layer with 100 nodes and random weights
    #     layer_two[i] = node(last_layer, tuple(random.random() for i in range(10)))

    # first_layer = ()
    # for i in range(10): # first layer of 10 nodes with random weights
    #     first_layer[i] = node(layer_two, tuple(random.random() for i in range(100)))
