#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:20:05 2020

@author: josephharvey
"""

import numpy as np

arr = np.random.rand(3, 23)
arr1 = np.random.rand(1, 23)

fullarr = np.concatenate((arr, arr1), axis = 0)

x0 = 0
y0 = 0
z0 = 0

arr = np.array([[x0],[y0],[z0]])

arr = np.append(arr, [[1],[2],[3]], axis = 1)

a = np.random.rand(500, 4)
b = np.random.rand(4, 1)

d = np.random.rand(500, 4)
e = np.random.rand(4, 1)

c = np.matmul(a,b)
f = np.matmul(d,e)
array = np.concatenate((c,f), axis = 1)
#print(array[:, 0].reshape(500,1))


fake_x = np.random.rand(500,1)
W = np.random.rand(500,500)
#print(np.matmul(W,fake_x).shape)

X = np.random.rand(500,1)
#print(X)

Y = np.array([x0,y0,z0]).reshape(3,1)


from scipy.sparse.linalg import eigs
rsvr_size = 300
spectral_radius = 0.9
unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
for i in range(unnormalized_W[:,0].size):
    for j in range(unnormalized_W[0].size):
        if np.random.rand(1) > 0.0333333:
            unnormalized_W[i][j] = 0
#print(np.count_nonzero(unnormalized_W[1])/(300))



max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False)
W = spectral_radius/np.abs(max_eig)*unnormalized_W


from sklearn.preprocessing import normalize

arr1 = np.array(np.random.rand(3,800))
arr2 = np.array(np.random.rand(8,800))
arr11 = np.ones((1, 800))
arr = np.concatenate((arr11, arr1, arr2), axis = 0)
print(arr.shape)
