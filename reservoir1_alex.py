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
from scipy.linalg import solve
from matplotlib import pyplot as plt

np.random.seed(0)

class Reservoir:
    def __init__(self, rsvr_size = 300, spectral_radius = 0.9, input_weight = 1e-1):
        self.rsvr_size = rsvr_size
        
        #get spectral radius < 1
        #gets row density = 0.03333
        unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
        for i in range(unnormalized_W[:,0].size):
            for j in range(unnormalized_W[0].size):
                if np.random.rand(1) > 0.0333333:
                    unnormalized_W[i,j] = 0
    
        max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False)
        
        self.W = spectral_radius/np.abs(max_eig)*unnormalized_W 
        self.Win = (np.random.rand(rsvr_size, 3)*2 - 1)*input_weight
        self.X = np.random.rand(rsvr_size, 1)*2 - 1
        self.Wout = np.array([])
        
class RungeKutta:
    def __init__(self,x0 = 1,y0 = 1,z0 = 1, h = 0.01, T = 100):
        self.u_arr = rungekutta(x0,y0,z0,h,T)[:, ::5]
        
        

    
#takes a reservoir object res along with initial conditions
def getX(res, rk,x0 = 1,y0 = 1,z0 = 1):
    
    #loops through every timestep
    for i in range(0, rk.u_arr[0].size):
        u = rk.u_arr[:,i].reshape(3,1)
        
        x = res.X[:,i].reshape(res.rsvr_size,1)
        x_update = np.tanh(np.add(np.matmul(res.Win, u), np.matmul(res.W, x)))
        
        res.X = np.concatenate((res.X,x_update), axis = 1)
    print(res.X.shape)
    return res.X
    
def trainRRM(res, rk):
    #only train 200-1000
    #listen 100-200 (wash out initial conditions)
    alph = 1e-4
    rrm = Ridge(alpha = alph)
    
    Y_train = rk.u_arr[:, 201:]
    #print("Y shape: " + str(Y.shape))
    
    X = getX(res, rk)[:, 201:(res.X[0].size - 1)]
    
    #concat X with u, 1
    X_train = np.copy(X)
    #print("X shape: " + str(X.shape))
    
    #print(rk.u_arr.shape)
    idenmat = np.identity(res.rsvr_size)*alph
    data_trstates = np.matmul(Y_train, np.transpose(X_train))
    states_trstates = np.matmul(X_train,np.transpose(X_train))
    res.Wout = np.transpose(solve(np.transpose(states_trstates + idenmat),np.transpose(data_trstates)))
    Y_train = Y_train.transpose()
    X_train = X_train.transpose()
    
    #tweak regression param? use 10^-4, 10^-6
    #test Ridge() in simpler context
    rrm.fit(X_train,Y_train)
    Wout_sklearn = rrm.coef_
    print(np.linalg.norm(Wout_sklearn-res.Wout)/res.Wout.size)
    return
    
def predict(res, x0 = 0, y0 = 0, z0 = 0, steps = 1000):
    Y = np.array([x0,y0,z0]).reshape(3,1)
    X = res.X[:,-1].reshape(res.rsvr_size,1)

    
    for i in range(0, steps):
        y_in = Y[:,i].reshape(3,1)
        x_prev = X[:,i].reshape(res.rsvr_size,1)
        
        x_current = np.tanh(np.add(np.matmul(res.Win, y_in), np.matmul(res.W, x_prev)))
        X = np.concatenate((X, x_current), axis = 1)
        
        y_out = np.matmul(res.Wout, x_current)
        Y = np.concatenate((Y, y_out), axis = 1)

    return Y

res = Reservoir()
rk = RungeKutta(h = 0.01,T = 500)
trainRRM(res, rk)
predictions = predict(res, x0 = rk.u_arr[0,-1], y0 = rk.u_arr[1,-1], z0 = rk.u_arr[2,-1])

plt.plot(np.array(range(predictions[0].size)), predictions[0])
#plt.plot(predictions[2], predictions[0])
#plt.plot(np.array(range(rk.u_arr[0].size)), rk.u_arr[0])
plt.show()
