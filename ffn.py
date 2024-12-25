import numpy as np
import pandas as pd

def oneHotEncode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot

def relu(x):
    return np.maximum(0, x)

def deriv_relu(Z):
    return Z > 0
    
def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def init_param(layer1, layer2):
    # Xavier initialization
    W1 = np.random.randn(784, layer1) * np.sqrt(2. / (784 + layer1))  
    b1 = np.zeros((1, layer1)) 
    W2 = np.random.randn(layer1, layer2) * np.sqrt(2. / (layer1 + layer2)) 
    b2 = np.zeros((1, layer2))
    W3 = np.random.randn(layer2, 10) * np.sqrt(2. / (layer2 + 10)) 
    b3 = np.zeros((1, 10))
    
    return W1, b1, W2, b2, W3, b3

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = oneHotEncode(Y, 10) 
    dZ3 = A3 - one_hot_Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m  
    
    dZ2 = np.dot(dZ3, W3.T) * deriv_relu(Z2)
    dW2 = np.dot(A1.T, dZ2) / m  
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  
    
    dZ1 = np.dot(dZ2, W2.T) * deriv_relu(Z1) 
    dW1 = np.dot(X.T, dZ1) / m 
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m 
    return dW1, db1, dW2, db2, dW3, db3

def update_param(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, alpha):
    W1 -= alpha * dW1
    W2 -= alpha * dW2
    W3 -= alpha * dW3
    b1 -= alpha * db1
    b2 -= alpha * db2
    b3 -= alpha * db3
    return W1, W2, W3, b1, b2, b3

def get_predictions(A3):
    return np.argmax(A3, 1)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def train(X, Y, layer1, layer2, alpha, epochs):
    W1, b1, W2, b2, W3, b3 = init_param(layer1, layer2)
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
        
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
    
        W1, W2, W3, b1, b2, b3 = update_param(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, alpha)
        
        if epoch % 10 == 0:
            predictions = get_predictions(A3)
            print(f"Iteration: {epoch}, Accuracy: {get_accuracy(predictions, Y)}")
    
    # Save the trained parameters
    np.save("parameters/W1.npy", W1)
    np.save("parameters/b1.npy", b1)
    np.save("parameters/W2.npy", W2)
    np.save("parameters/b2.npy", b2)
    np.save("parameters/W3.npy", W3)
    np.save("parameters/b3.npy", b3)
    
def continue_train(X,Y,layer1,layer2,alpha,epochs):
    W1 = np.load("parameters/W1.npy")
    b1 = np.load("parameters/b1.npy")
    W2 = np.load("parameters/W2.npy")
    b2 = np.load("parameters/b2.npy")
    W3 = np.load("parameters/W3.npy")
    b3 = np.load("parameters/b3.npy")
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
        
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
    
        W1, W2, W3, b1, b2, b3 = update_param(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, alpha)
        
        if epoch % 10 == 0:
            predictions = get_predictions(A3)
            print(f"Iteration: {epoch}, Accuracy: {get_accuracy(predictions, Y)}")
    
    # Save the trained parameters
    np.save("parameters/W1.npy", W1)
    np.save("parameters/b1.npy", b1)
    np.save("parameters/W2.npy", W2)
    np.save("parameters/b2.npy", b2)
    np.save("parameters/W3.npy", W3)
    np.save("parameters/b3.npy", b3)



