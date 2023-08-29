# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:05:34 2023
https://www.youtube.com/watch?v=qxIaW-WvLDU&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=3
@author: Evan

Neuron example using the sigmoid activation function
ArtificialNeuron.png visualizes example
"""
import math

def sigmoid(x):
    y = 1.0 / (1 + math.exp(-x))
    return y

def activate(inputs,weights):
    # perform net input
    h = 0
    #Essentially a dot product!
    for x, w in zip(inputs,weights):
        h += x*w
    
    # perform activation
    return sigmoid(h)
    
    
if __name__ == "__main__":
    inputs = [.5,.3,.2]
    weights = [.4,.7,.2]
    output = activate(inputs,weights)
    print(output)
    