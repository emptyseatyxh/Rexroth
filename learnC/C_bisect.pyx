# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:46:25 2016

@author: apple
"""
import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def root_func(double x,double aa,double kk):
    return x**4 + kk*x**2 + aa*x - 3

def python_bisect(double a, double b, double aa, 
                  double kk, double tol, int mxiter):
    cdef int its = 0
    cdef double c, fa, fb, fc
    fa = root_func(a, aa, kk)
    fb = root_func(b, aa, kk)
    if abs(fa) < tol:
        return a
    elif abs(fb) < tol:
        return b
    c = (a+b)/2.
    fc = root_func(c, aa, kk)
    while abs(fc)>tol and its<mxiter:
        its = its + 1
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        c = (a+b)/2.
        fc = root_func(c, aa, kk)
    return c

def main(np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] kk):
    cdef int l_arr = a.shape[0]
    cdef int i
    cdef double c
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(l_arr, dtype=DTYPE)
    for i in range(l_arr):
        res[i] = python_bisect(0., 2., a[i], kk[i], 2e-12, 100)
    return res