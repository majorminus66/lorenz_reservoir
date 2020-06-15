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
from scipy.linalg import solve
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt

np.random.seed(0)

#add normalization again
#then add ones
#kill concat
#spectral radius, input scaling

class Reservoir:
    def __init__(self, rsvr_size = 300, spectral_radius = 0.8, input_weight = 0.8):
        self.rsvr_size = rsvr_size
        
        #get spectral radius < 1
        #gets row density = 0.03333
        unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
        for i in range(unnormalized_W[:,0].size):
            for j in range(unnormalized_W[0].size):
                if np.random.rand(1) > 10/rsvr_size:
                    unnormalized_W[i][j] = 0
    
        max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False)
        
        self.W = spectral_radius/np.abs(max_eig)*unnormalized_W 
        self.Win = (np.random.rand(rsvr_size, 4)*2 - 1)*input_weight
        self.X = (np.random.rand(rsvr_size, 5002)*2 - 1)
        self.Wout = np.array([])
        
class RungeKutta:
    def __init__(self, x0 = 1,y0 = 1,z0 = 1, h = 0.01, T = 100, ttsplit = 5000):
        u_arr = rungekutta(x0,y0,z0,h,T)[:, ::5]
        
        for i in range(u_arr[:,0].size):
            u_arr[i] = (u_arr[i] - np.mean(u_arr[i]))/np.std(u_arr[i])
        
        self.u_arr_train = u_arr[:, :ttsplit+1]
        #size 5001
        
        #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]
        #size 1001
    
#takes a reservoir object res along with initial conditions
def getX(res, rk,x0 = 1,y0 = 1,z0 = 1):
    
    #loops through every timestep
    for i in range(0, rk.u_arr_train[0].size):
        u = np.append(1, rk.u_arr_train[:,i]).reshape(4,1)
        
        x = res.X[:,i].reshape(res.rsvr_size,1)
        x_update = np.tanh(np.add(np.matmul(res.Win, u), np.matmul(res.W, x)))
        
        res.X[:,i+1] = x_update.reshape(1,res.rsvr_size)    
    
    return res.X
    
def trainRRM(res, rk):

    alph = 10**-4
    rrm = Ridge(alpha = alph, solver = 'cholesky')
    
    Y_train = rk.u_arr_train[:, 201:]

    
    X = getX(res, rk)[:, 201:(res.X[0].size - 1)]
    X_train = np.copy(X)
        
    idenmat = np.identity(res.rsvr_size)*alph
    data_trstates = np.matmul(Y_train, np.transpose(X_train))
    states_trstates = np.matmul(X_train,np.transpose(X_train))
    res.Wout = np.transpose(solve(np.transpose(states_trstates + idenmat),np.transpose(data_trstates)))
    
    
    Y_train = Y_train.transpose()
    X_train = X.transpose()
    
    #tweak regression param? use 10^-4, 10^-6
    #test Ridge() in simpler context
    #rrm.fit(X_train,Y_train)
    #res.Wout = rrm.coef_
    return
    
def predict(res, x0 = 0, y0 = 0, z0 = 0, steps = 1000):
    Y = np.empty((3, steps + 1))
    X = np.empty((res.rsvr_size, steps + 1))
    
    Y[:,0] = np.array([x0,y0,z0]).reshape(1,3) 
    X[:,0] = res.X[:,-2]

    
    for i in range(0, steps):
        y_in = np.append(1, Y[:,i]).reshape(4,1)
        x_prev = X[:,i].reshape(res.rsvr_size,1)
        
        x_current = np.tanh(np.add(np.matmul(res.Win, y_in), np.matmul(res.W, x_prev)))
        X[:,i+1] = x_current.reshape(1,res.rsvr_size)
        #X = np.concatenate((X, x_current), axis = 1)
        
        y_out = np.matmul(res.Wout, x_current)
        Y[:,i+1] = y_out.reshape(1, 3)
        #Y = np.concatenate((Y, y_out), axis = 1)

    return Y

def test(res, num_tests = 10, rkTime = 105, split = 2000):
    for i in range(num_tests):
        ic = np.random.rand(3)*2
        rktest = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = ic[2], T = rkTime, ttsplit = split)
        res.X = (np.zeros((res.rsvr_size, split+2))*2 - 1)
        
        #sets res.X
        getX(res, rktest)
        
        pred = predict(res, x0 = rktest.u_arr_test[0,0], y0 = rktest.u_arr_test[1,0], z0 = rktest.u_arr_test[2,0], steps = (rkTime*20-split))
        
        plt.figure()
        plt.plot(pred[0])
        plt.plot(rktest.u_arr_test[0])
    
    plt.show()
    return

res = Reservoir()
rk = RungeKutta(T = 300)
trainRRM(res, rk)

#plot predictions immediately after training 
#predictions = predict(res, x0 = rk.u_arr_test[0,0], y0 = rk.u_arr_test[1,0], z0 = rk.u_arr_test[2,0])
#plt.plot(predictions[0])
#plt.plot(rk.u_arr_test[0])

np.random.seed()
test(res, 10)