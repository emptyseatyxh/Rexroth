# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:24:18 2016

@author: Xinghao

test local Rbf
"""

import numpy as np
from scipy import spatial
from scipy.interpolate import Rbf
from scipy.interpolate import griddata

def _h_multiquadric(dist, epsilon):
    return np.sqrt((1.0/epsilon*dist)**2 + 1)
    
np.random.seed(1)
known = np.random.rand(50, 3)
data = np.random.rand(len(known))
unknown = np.random.rand(2, 3)

#known = np.linspace(1, 5, 80).reshape()

mytree = spatial.cKDTree(known)
dist, indexes = mytree.query(unknown, 50)   
nearest = data[indexes]     
points = known[indexes].transpose((0,2,1))

x1 = points[:, ..., :, np.newaxis]
x2 = points[:, ..., np.newaxis, :]
diff = (x1 - x2)**2
sum_diff = diff.sum(axis=1)
r = np.sqrt(sum_diff)

epsilon = np.average(dist, axis=1)[:,np.newaxis,np.newaxis]
A = _h_multiquadric(r, epsilon)
w = np.linalg.solve(A, nearest)
f_unknown = _h_multiquadric(dist, np.squeeze(epsilon, axis=2))
res = np.sum(f_unknown*w, axis=1)
'''
for i in range(8):
    r = dist - dist[:,i,np.newaxis]
    lr.append(r[:,np.newaxis])
    Ai = _h_multiquadric(dist - dist[:,i,np.newaxis], epsilon[:,np.newaxis])
    l.append(Ai[:,np.newaxis,:])
r = np.concatenate(lr, axis=1)
l = np.concatenate(l, axis=1)
w = np.linalg.solve(l,nearest)

f_unknown = _h_multiquadric(dist, epsilon[:,np.newaxis])

res = np.sum(f_unknown*w, axis=1)
'''
rbf = Rbf(known[:,0], known[:,1], known[:,2], data)
res_rbf = rbf(unknown[:,0],unknown[:,1],unknown[:,2])
res_grd = griddata(known, data, unknown)
