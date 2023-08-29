# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:06:21 2023
NOTES https://www.youtube.com/watch?v=ScL18goxsSg&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=7
IMPLEMENTATION https://www.youtube.com/watch?v=Z97XGNUUx9o&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=8

@author: Evan
"""
#Tweak weights of the connections by feeding training data
#Step 1: pass sample input
# -To recieve the result of the forward propagation
# -Calculate an error from the result and return it back to the first layer(backward propagation)
# -^Gradient of error function over the weights to update the parameters
#Step 2: Error function / Loss function: E(p,y) -> Quadratic error function (1/2)(p-y)^2
#Step 3: Gradient of error function: dE/dW_(n)
# - Think of network as a function F = F(X,W)
# - Error function is then E = E( F(X,W), y)
# - Therefore using the chain rule we can represent the gradient as:
    # (dE / da_n) * (da_n / dh_n) * (dh_n / dW_n) [n being the max element for the corresponding idea (W_n where n will be 1 less than the n for a & h)]
    #Sigmoid derivative = sigmoid(x)(1-sigmoid(x))
    # REWRITTEN EXAMPLE: dE/dW_2 = (a_3 - y) * sig'(h_3) * a_2
    # Continue the gradient error, back propagate: dE/dW_1 = FROM PREV[(a_3 - y) * sig'(h_3) ] * W_2 * sig'(h_2) * X
    
#Step 4: Update Params
#-Gradient descent - take a step in opposite direction to gradient where Step = Learning rate


import numpy as np
from random import random

# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement training
# train with a data set to make some predictions

class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
        
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # Each element represents the neurons in each layer
        layers = [num_inputs] + num_hidden + [num_outputs]
        
        # initiate random weights
        weights =  []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1]) #Creating matrix rows = neurons in current layer, columns = neurons in next layer
            weights.append(w) #Will contain layers-1 weight matrices(in between layers)
            
        self.weights = weights
        
        #BACK PROPAGATION MATERIAL
        
        # Prepare activations matrix for each layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
            
        self.activations = activations
        
        # Prepare derivatives matrix for each layer
        derivatives = []
        for i in range(len(layers)-1): #-1 bc Error func is w/ respect to Weights(so same size as weights)
            d = np.zeros((layers[i], layers[i+1])) #Propagated along weights
            derivatives.append(d)
            
        self.derivatives = derivatives
    
    #Tweaked to save information for back propagation
    def forward_propagate(self, inputs):   
        activations = inputs #layer 1
        self.activations[0] = inputs 
        
        for i,w in enumerate(self.weights):
            # Net inputs
            net_inputs = np.dot(activations, w)
            # Activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
            #i+1? -> a_3 = s(h_3)
            #h_3 = a_2 dot W_2
            
        return activations
        
    
    def back_propagate(self,error,verbose=False):
        
        #dE/dW_i = (y - a_[i+1]) s'(h_[i+1])) a_i
        #s'(h_[i+1]) = s(h_[i+1])(1-s(h_[i+1]))
        #s(h_[i+1]) = a_[i+1]
        #NEXT 
        #dE/dW_[i-1] = (y - a_[i+1]) s'(h_[i+1])) W_i s'(h_i) a_[i-1]
        
        
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            
            delta = error * self._sigmoid_derivative(activations) # --> need to move from ndarray([0.1,0.2]) --> ndarray([[0.1,0.2]])
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            
            current_activations = self.activations[i] # --> need to move from ndarray([0.1,0.2]) --> ndarray([[0.1],[0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)            
            
            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)
            
            error = np.dot(delta, self.weights[i].T)
            
            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return error #Error propagated all the way to the input layer
    
    def gradient_descent(self, learning_rate, verbose=False):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            if(verbose):
                print("Original W{} {}".format(i,weights))
            
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            if(verbose):
                print("Post Gradient Descent W{} {}".format(i,weights))
        
    # epochs -> # of times the entire data set is fed to the network    
    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for inp, target in zip(inputs,targets):
                # forward propagation
                output = self.forward_propagate(inp)
                
                # calculate error
                error = target - output
                
                # back propagation
                self.back_propagate(error)
            
                # apply gradient descent
                self.gradient_descent(learning_rate)
                
                sum_error += self._mse(target,output)
            # report error in this iteration
            print("Error: {} at epoch {}".format(sum_error / len(inputs) ,i))
        
    #Minimum Square Error
    def _mse(self, target, output):
        return np.average((target - output)**2)
                   
    def _sigmoid_derivative(self,x):
        return x * (1.0 - x)
            
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
 

if __name__ == "__main__":
    
    #creating a dataset to train the network to perform the sum operation
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)]) #array of [[0.1,0.2], [0.3,0.4]]
    targets = np.array([[i[0] + i[1]] for i in inputs]) #                                   #array of sums [[0.3],      [0.7]]  
    
    
    # create an MLP
    mlp = MLP(2,[5],1)
    
    # train our mlp
    mlp.train(inputs, targets, 100, 0.15)
    
    # Create dummy data
    inp = np.array([0.3,0.1])
    target = np.array([0.4])
    
    output = mlp.forward_propagate(inp)
    print("\n\n")
    print("Network prediction: {} +  {} = {}".format(inp[0],inp[1],output[0]))
    

    

  