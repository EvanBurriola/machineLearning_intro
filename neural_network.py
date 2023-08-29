# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:14:49 2023
NOTES : https://www.youtube.com/watch?v=QUCzvlgvk6I&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=6
IMPLEMENTATION: https://www.youtube.com/watch?v=0oWnheK-gGk&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=6
@author: Evan
"""

# Net input: h = Vec_X DOT Vec_W 

# Components of an Artificial Neural Network(ANN)
# - Neurons
# - Input, hidden, output layers
# - Weighted connections
# - Activation Function

# Multilayer Perceptron (MLP):
# 1 input layer, 1 or multiple hidden layers, 1 output layer.
# Computation travels left to right ->
# Computation aspects in MLP:
# - Weights, Net inputs, Activations(output of neurons to next layer)

#Computation in MLP
# -1st layer: Create input vector(X)
# -2nd layer: Create net Input h_2: X matrix_mult W_1 THEN create activation vector(output) of 2nd layer a_2 = f(h_2) where f is the activation function(ex: sigmoid)
# -3rd layer: Create net input of 3rd layer h_3: a_2 matrix_mult W_2

#SAMPLE COMPUTATION
# X = [.8,1]
# W_1 = [1.2, .7, 1] 
#       [2, .6 ,1.8]
#W_2 = [1,.9,1.5]

#h_2 = X matrix_mult W_1 = [2.96,1.56,1.8]
#a_2 = sigmoid(h_2) -> 1/(1 + e^(-h_2)) = [.95,.83,.86]
#h_3 = a_2 matrix_mult W_2 = 2.99
#output y = sigmoid(h_3) = .95

import numpy as np

class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # Each element represents the neurons in each layer
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        
        # initiate random weights
        self.weights =  []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1]) #Creating matrix rows = neurons in current layer, columns = neurons in next layer
            self.weights.append(w) #Will contain layers-1 weight matrices(in between layers)
    
    def forward_propagate(self, inputs):   
        activations = inputs #layer 1
        
        for w in self.weights:
            # Net inputs
            net_inputs = np.dot(activations, w)
            # Activations
            activations = self._sigmoid(net_inputs)
            
        return activations
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

if __name__ == "__main__":
    # create an MLP
    mlp = MLP()
    
    # create inputs
    inputs = np.random.rand(mlp.num_inputs)
    
    # peform forward propagation
    outputs = mlp.forward_propagate(inputs)
    
    # output
    print("Network input: {}".format(inputs))
    print("Network output: {}".format(outputs))    