from .trainableLayer import TrainableLayer
from .layer import Layer
from .Conv import Conv
from .Dense import Dense
from .flatten import Flatten
from .MaxPool import MaxPool
import pickle
from typing import Self
import numpy as np
from .globalVariables import *

# also acts as a record of which layers aren't trainable, as those are 'None' here
activ_funcs =   [
                (ReLU, dReLU), None, 
                (ReLU, dReLU), None,
                (ReLU, dReLU), None, 
                None, 
                (ReLU, dReLU), (ReLU, dReLU), 
                (linear, dlinear)
                ]

class Network():
    def __init__(self, input_shape):
        self.layers : list[Layer] = []
        self.input_shape = input_shape # -> (100, 3, 3, 250, 250)

        self.layers.append(Conv((16, self.input_shape[2], 3, 3)))       # -> (100, 3, 16, 250, 250)
        self.layers.append(MaxPool(3))                                  # -> (100, 3, 16, 83, 83)
        self.layers.append(Conv((32, 16, 5, 5)))                        # -> (100, 3, 32, 83, 83)
        self.layers.append(MaxPool(3))                                  # -> (100, 3, 32, 27, 27)
        self.layers.append(Conv((64, 32, 7, 7)))                        # -> (100, 3, 64, 27, 27)
        self.layers.append(MaxPool(3))                                  # -> (100, 3, 64, 9, 9)
        self.layers.append(Flatten())                                   # -> (100, 3, 5184)
        self.layers.append(Dense((1024, 64*((self.input_shape[-1]//27)**2))))# -> (100, 3, 1024)
        self.layers.append(Dense((256, 1024)))                          # -> (100, 3, 256)
        self.layers.append(Dense((feature_vector_size, 256)))           # -> (100, 3, 128)

        self.set_activ_funcs()
    
    def feedforward(self, input : np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.forward(output).output_activations
            print(f"layer: {output.shape}")
        return output
    
    def backprop_step(self, input : np.ndarray, backprop_step_number : int) -> np.ndarray:
        forwardContexts = []

        output = input
        for l in self.layers:
            forwardContext = l.forward(output)
            forwardContexts.append(forwardContext)
            output = forwardContext.output_activations

        cost = self.cost(output)

        dL_dOutput = self.dLoss(output)
        for i in range(len(self.layers)-1, -1, -1):
            dL_dOutput = self.layers[i].backprop(dL_dOutput, forwardContexts[i], backprop_step_number)
            forwardContexts[i] = None
        
        return cost

    def cost(self, data : np.ndarray) -> np.float64:
        return np.mean([self.triplet_loss(A, P, N) for A, P, N in data])
    
    def triplet_loss(self, Anchor : np.ndarray, Positive : np.ndarray, Negative : np.ndarray, tolerance=(t+0.1)) -> np.float64:
        return max(self.euclid_dist(Anchor, Positive)-self.euclid_dist(Anchor, Negative)+tolerance, 0.0)
    
    def dLoss(self, vector):
        result = np.zeros((self.input_shape[:2]+(feature_vector_size,)))
        for i, APN_triplet in enumerate(vector):
            if (self.requires_learning(APN_triplet)):
                result[i] = self.dLossdAPN(APN_triplet)
        return result
    
    def requires_learning(self, APN_triplet) -> bool:
        return self.triplet_loss(APN_triplet[0], APN_triplet[1], APN_triplet[2]) != 0
    def dLossdAPN(self, APN_triplet):
        A, P, N = APN_triplet
        #return np.array([((A-P)/self.euclid_dist(A, P))-((A-N)/self.euclid_dist(A, N)), 
        #                 (P-A)/(self.euclid_dist(A, P)), 
        #                 (A-N)/(self.euclid_dist(A, N))])
        return np.array([2*(N-P),
                         -2*(A-P),
                         2*(A-N)])
    
    def euclid_dist(self, v1, v2) -> np.float64:
        return (np.linalg.norm(v1-v2)**2)
    
    def saveNetwork(self):
        with open("./NetworkState", 'wb') as f:
            pickle.dump(self, f)

    def loadNetwork() -> Self:
        try:
            with open("./NetworkState", 'rb') as f:
                net : Self = pickle.load(f)
            net.set_activ_funcs()
            return net
        except IOError:
            return None
        
    def set_activ_funcs(self):
        for i,l in enumerate(self.layers):
            if isinstance(l, TrainableLayer) and activ_funcs[i] is not None:
                l.set_activ_func(activ_funcs[i][0], activ_funcs[i][1])