# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:53:02 2023
IMPLEMENTATION https://www.youtube.com/watch?v=JdXxaZcQer8&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=9
@author: Evan
"""
import numpy as np #USING 1.22.1 !!
import tensorflow as tf #tensorflow-cpu
from random import random
from sklearn.model_selection import train_test_split

#inputs -> array([[0.1,0.2], [0.3,0.4]])
#outputs -> array([[0.3],[0.7]])

def generate_dataset(num_samples, test_size):
    #creating a dataset to train the network to perform the sum operation
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])                                   
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 1)
    #print("x_test: \n {}".format(x_test))
    #print("y_test: \n {}".format(y_test))
    
    
    
    # build model: 2(input layer) -> 5(hidden layers) -> 1(output layer)
    model = tf.keras.Sequential([
        #Fully connected, all neurons connected to each in next layer
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        #Output layer
        tf.keras.layers.Dense(1, activation="sigmoid") 
        ])
    
    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1) #Gradient descent
    model.compile(optimizer=optimizer, loss="MSE") #min square error function
    
    # train model
    model.fit(x_train, y_train, epochs=100)
    
    # evaluate model
    print("\nModel evaluation:")
    model.evaluate(x_test, y_test, verbose=1)
    
    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)
    
    print("\nNetwork prediction:")
    for d, p in zip(data,predictions):
        print("{} + {} = {}".format(d[0],d[1],p[0]))


