# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:01:26 2016

@author: Xinghao
"""
from __future__ import division
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.special import xlogy
from scipy import spatial

import numpy as np


def IDW(known, data, unknown):
    mytree = spatial.cKDTree(known)
    dist, indexes = mytree.query(unknown, 8)
    nearest = data[indexes]
    denomiator = np.zeros(unknown.shape[0])
    nominator = np.zeros(unknown.shape[0])
    for i in range(8):
        ele = np.prod(np.hstack((dist[:,:i],dist[:,i+1:])), axis=1)
        denomiator += ele
        nominator += ele*nearest[:,i]
    res = nominator/denomiator
    return res

def local_RBF(known, data, unknown, local_n, fchoice):
    
    def _1_multiquadric(r, epsilon):
        return np.sqrt((1.0/epsilon*r)**2 + 1)
        
    def _2_inverse_multiquadric(r, epsilon):
        return 1.0/np.sqrt((1.0/epsilon*r)**2 + 1)

    def _3_gaussian(r, epsilon):
        return np.exp(-(1.0/epsilon*r)**2)

    def _4_linear(r):
        return r

    def _5_cubic(r):
        return r**3

    def _6_quintic(r):
        return r**5

    def _7_thin_plate(r):
        return xlogy(r**2, r)
    
    def _euclidean_norm(points):
        x1 = points[:, ..., :, np.newaxis]
        x2 = points[:, ..., np.newaxis, :]
        r = np.sqrt(((x1 - x2)**2).sum(axis=1))
        return r
        
    func_7 = {1:_1_multiquadric, 2:_2_inverse_multiquadric, 3:_3_gaussian,\
              4:_4_linear, 5:_5_cubic, 6:_6_quintic, 7:_7_thin_plate}
    func = func_7[fchoice]
    mytree = spatial.cKDTree(known)
    if len(unknown) < 1000:
        dist, indexes = mytree.query(unknown, local_n)   
        nearest = data[indexes]     
        r = _euclidean_norm(known[indexes].transpose((0,2,1)))
        if fchoice==1 or fchoice==2 or fchoice==3:
            epsilon = np.average(dist, axis=1)[:,np.newaxis,np.newaxis]
            A = func(r, epsilon)
            w = np.linalg.solve(A, nearest)
            f_unknown = func(dist, np.squeeze(epsilon, axis=2))
            res = np.sum(f_unknown*w, axis=1)
        else:
            A = func(r)
            w = np.linalg.solve(A, nearest)
            f_unknown = func(dist)
            res = np.sum(f_unknown*w, axis=1)               
    else:
        unknown = np.array_split(unknown, 10, axis=0)
        res_lst = []
        for i in unknown:
            dist, indexes = mytree.query(i, local_n)   
            nearest = data[indexes]     
            r = _euclidean_norm(known[indexes].transpose((0,2,1)))
            if fchoice==1 or fchoice==2 or fchoice==3:
                epsilon = np.average(dist, axis=1)[:,np.newaxis,np.newaxis]
                A = func(r, epsilon)
                w = np.linalg.solve(A, nearest)
                f_unknown = func(dist, np.squeeze(epsilon, axis=2))
                resi = np.sum(f_unknown*w, axis=1)
            else:
                A = func(r)
                w = np.linalg.solve(A, nearest)
                f_unknown = func(dist)
                resi = np.sum(f_unknown*w, axis=1)      
            res_lst.append(resi)
        res = res_lst[0]
        for i in range(len(res_lst)-1):
            res = np.hstack((res, res_lst[i+1]))
    return res


def testf(x,y,z):
    return x*y+x*z+y*z
    
#np.random.seed(1)
known = np.random.rand(100000, 3)
data = testf(known[:,0], known[:,1], known[:,2])

unknown = np.random.rand(5000, 3)
undata = testf(unknown[:,0], unknown[:,1], unknown[:,2])

res_grd = griddata(known, data, unknown)
res_IDW = IDW(known, data, unknown)
#rbf = Rbf(known[:,0], known[:,1], known[:,2], data)
#res_RBF = rbf(unknown[:,0],unknown[:,1],unknown[:,2])
res_lRBF = local_RBF(known, data, unknown, 50, 4)

error_grd = np.nansum(np.abs(undata-res_grd))
error_IDW = np.sum(np.abs(undata-res_IDW))
error_lRBF = np.sum(np.abs(undata-res_lRBF))

def clearall():
    all = [var for var in globals() if "__" not in (var[:2], var[-2:])]
    for var in all:
        del globals()[var]