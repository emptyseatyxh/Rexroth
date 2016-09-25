# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:58:24 2016

@author: Xinghao
"""

import numpy as np
from numpy import zeros, dot, sum, newaxis, exp, empty
from scipy import stats
from numba import jit, double
#from test_numba_kde import gaussian_kde


randm = np.random.rand(5000, 3).T
#inv_cov = np.random.rand(3, 3)
#kde_old = stats.gaussian_kde(randm)(randm)
#kde_new = gaussian_kde(randm)(randm)

@jit(double[:](double[:,:,:],double[:,:,:]))
def test_loop(data, posit):
    inv_cov = np.array([[0.32,0.12,0.44],[0.23,0.78,0.11],[0.91,0.4,0.75]])
    #result = empty((5000.0,), dtype=np.float64)
    for i in range(5000):
        a = data[0,i]
        b = data[1,i]
        c = data[2,i]
        diff = np.array([[a],[b],[c]]) - posit
        tdiff = dot(inv_cov, diff)
        energy = sum(diff*tdiff,axis=0) / 2.0
        energy += exp(-energy)
    return energy
    
#upgrade = jit(double[:](double[:,:,:],double[:,:,:]))(test_loop)

'''
import numpy

def filter2d(image, filt):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    result = numpy.zeros_like(image)
    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            num = 0.0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
            result[i, j] = num
    return result

# This kind of quadruply-nested for-loop is going to be quite slow.
# Using Numba we can compile this code to LLVM which then gets
# compiled to machine code:

from numba import double, jit

fastfilter_2d = jit(double[:,:](double[:,:], double[:,:]))(filter2d)

# Now fastfilter_2d runs at speeds as if you had first translated
# it to C, compiled the code and wrapped it with Python
image = numpy.random.random((1000, 1000))
filt = numpy.random.random((100, 100))
res = fastfilter_2d(image, filt)
'''