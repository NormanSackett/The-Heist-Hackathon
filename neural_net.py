import random
import numpy as np

class node():
    def __init__ (self, layer_tuple, weight_tuple):
        self.layer_tuple = layer_tuple
        self.weight_tuple = weight_tuple

def layers():
    last_layer = ()
    for i in range(10):
        last_layer[i] = node(any, any) #last layer does not have a next layer

    layer_two = ()
    for i in range(100): # generate a hidden layer with 100 nodes and random weights
        layer_two[i] = node(last_layer, tuple(random.random() for i in range(10)))

    first_layer = ()
    for i in range(10): # first layer of 10 nodes with random weights
        first_layer[i] = node(layer_two, tuple(random.random() for i in range(100)))
