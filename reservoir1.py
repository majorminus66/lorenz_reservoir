#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:57:49 2020

@author: josephharvey
"""

from lorenzrungekutta import rungekutta
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt

np.random.seed(0)

class Reservoir:
    def __init__(self, rsvr_size = 300, spectral_radius = 0.9, input_weight = 1):
        self.rsvr_size = rsvr_size
        
        #get spectral radius < 1
        #gets row density = 0.03333
        unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
        for i in range(unnormalized_W[:,0].size):
            for j in range(unnormalized_W[0].size):
                if np.random.rand(1) > 0.0333333:
                    unnormalized_W[i][j] = 0
    
        max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False)
        
        self.W = spectral_radius/np.abs(max_eig)*unnormalized_W 
        self.Win = (np.random.rand(rsvr_size, 4)*2 - 1)*input_weight
        self.X = (np.random.rand(rsvr_size, 1)*2 - 1)*0.01
        self.Wout = np.array([])
        
class RungeKutta:
    def __init__(self,x0 = 1,y0 = 1,z0 = 1, h = 0.01, T = 100):
        self.u_arr = normalize(rungekutta(x0,y0,z0,h,T)[:, ::10], axis = 1)
        
    
#takes a reservoir object res along with initial conditions
def getX(res, rk,x0 = 1,y0 = 1,z0 = 1):
    
    #loops through every timestep
    for i in range(0, rk.u_arr[0].size):
        u = np.append(1, rk.u_arr[:,i]).reshape(4,1)
        
        x = res.X[:,i].reshape(res.rsvr_size,1)
        x_update = np.tanh(np.add(np.matmul(res.Win, u), np.matmul(res.W, x)))
        
        res.X = np.concatenate((res.X,x_update), axis = 1)
    
    return res.X
    
def trainRRM(res, rk):
    #only train 200-1000
    #listen 100-200 (wash out initial conditions)
    
    rrm = Ridge(alpha = 10**-4, solver = 'cholesky')
    
    Y_train = rk.u_arr[:, 201:]
    #print("Y shape: " + str(Y.shape))
    
    X = getX(res, rk)[:, 201:(res.X[0].size - 1)]
    
    #concat X with u, 1
    X_train = np.concatenate((np.ones((1, 4800)), X, rk.u_arr[:, 200:(rk.u_arr[0].size - 1)]), axis = 0)
    #print("X shape: " + str(X.shape))
    
    #print(rk.u_arr.shape)
    
    Y_train = Y_train.transpose()
    X_train = X_train.transpose()
    
    #tweak regression param? use 10^-4, 10^-6
    #test Ridge() in simpler context
    rrm.fit(X_train,Y_train)
    res.Wout = rrm.coef_
    return
    
def predict(res, x0 = 0, y0 = 0, z0 = 0, steps = 500):
    Y = np.array([x0,y0,z0]).reshape(3,1)
    X = res.X[:, res.X[0].size-1].reshape(res.rsvr_size, 1)

    
    for i in range(0, steps):
        y_in = np.append(1, Y[:,i]).reshape(4,1)
        x_prev = X[:,i].reshape(res.rsvr_size,1)
        
        x_current = np.tanh(np.add(np.matmul(res.Win, y_in), np.matmul(res.W, x_prev)))
        X = np.concatenate((X, x_current), axis = 1)
        
        y_out = np.matmul(res.Wout, np.concatenate((np.array([[1]]), x_current, Y[:,i].reshape(3,1)), axis = 0))
        Y = np.concatenate((Y, y_out), axis = 1)

    return Y

res = Reservoir()
rk = RungeKutta(T = 500)
trainRRM(res, rk)
predictions = predict(res, x0 = rk.u_arr[0,rk.u_arr[0].size-1], y0 = rk.u_arr[1,rk.u_arr[0].size-1], z0 = rk.u_arr[2,rk.u_arr[0].size-1], steps = 500)

plt.plot(np.array(range(predictions[0].size)), predictions[0])

one_arr = np.ones((1,4800))
X_values = res.X[:, 201:(res.X[0].size - 1)]
u_values = rk.u_arr[:, 200:(rk.u_arr[0].size - 1)]
X = np.concatenate((one_arr, X_values, u_values), axis = 0)
result = np.matmul(res.Wout, X)

#plt.plot(result[0])
#plt.plot(rk.u_arr[0, 201:(rk.u_arr[0].size-1)])
#plt.plot(predictions[2], predictions[0])
#plt.plot(np.array(range(rk.u_arr[0].size)), rk.u_arr[0])


#from troubleshooting:
    
#u_arr = rungekutta(1, 1, 1)[:, 200::10]
#plt.plot(np.array(range(u_arr[0].size)), u_arr[0])

#initialize X, internal weights, input weights
#X = np.random.rand(500,1)*2 - 1 
#W = np.random.rand(500,500)*2 - 1
#print("W: \n" + str(W[:10,:10]))
#Win = np.random.rand(500, 4)*2 - 1

#loops through every timestep
#for i in range(0, 3):
#    u = np.append(1, u_arr[:,i]).reshape(4,1)
    #print("u at " + str(i) + " is " + str(u))
    
#    x = X[:,i].reshape(500,1)
#    print("x at " + str(i) + " is \n" + str(x[0:10,:]))
    
#    x_update = np.tanh(np.add(np.matmul(Win, u), np.matmul(W, x)))
#    print("W*x:\n" + str(np.matmul(W, x)[0:5,:]))
    #print("Win*u:\n" + str(np.matmul(Win, u)[0:5,:]))
    #print("x update at " + str(i) + " is \n" + str(x_update[0:10,:]))
    
#    X = np.concatenate((X,x_update), axis = 1)

