# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:46:25 2016

@author: apple
"""
import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t
TTYPE = np.int64
ctypedef np.int64_t TTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def root_func(double x,double aa,double kk):
    return x**4 + kk*x**2 + aa*x - 3

def python_bisect(double a, double b, double aa, 
                  double kk, int mxiter=10):
    cdef int its = 0
    cdef double c, c2, fa, fb, fc
    fa = root_func(a, aa, kk)
    fb = root_func(b, aa, kk)
    c = (a+b)/2.
    fc = root_func(c, aa, kk)
    while its<mxiter:
        its += 1
        if its == mxiter:
            c2 = c
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        c = (a+b)/2.
        fc = root_func(c, aa, kk)
    return c, c2
    
def secant(double x0, double x1, double aa, double kk, 
           double tol=2e-12, double ftol=2e-12, int max_iter=90):
    cdef double x_old_old, x_old, x_new, f_old, f_old_old
    x_old = x0
    f_old = root_func(x_old, aa, kk)
    x_new = x1
    i = 0
    while i < max_iter:
        i += 1
        x_old_old = x_old
        x_old = x_new
        f_old_old = f_old
        f_old = root_func(x_old, aa, kk)
        if abs(f_old) <= ftol:
            return x_old, i
        if f_old - f_old_old == 0.0:
            return -9999., i
        x_new = x_old - f_old * (x_old - x_old_old) / (f_old - f_old_old)
        if abs(x_new - x_old) <= max(tol * abs(x_old), ftol):
            return x_new, i
    return x_new, max_iter
    
def main(np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] kk):
    cdef int l_arr = a.shape[0]
    cdef int i
    cdef double c, c2
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(l_arr, dtype=DTYPE)
    cdef np.ndarray[TTYPE_t, ndim=1] iii = np.zeros(l_arr, dtype=TTYPE)
    for i in range(l_arr):
        c, c2 = python_bisect(0., 2., a[i], kk[i], 10)
        res[i], iii[i] = secant(c, c2, a[i], kk[i])
    return res, iii
    
def main2(double [:] a, double [:] kk):
    cdef int l_arr = a.shape[0]
    cdef int i
    cdef double c, c2
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(l_arr, dtype=DTYPE)
    cdef np.ndarray[TTYPE_t, ndim=1] iii = np.zeros(l_arr, dtype=TTYPE)
    for i in range(l_arr):
        c, c2 = python_bisect(0., 2., a[i], kk[i], 10)
        res[i], iii[i] = secant(c, c2, a[i], kk[i])
    return res, iii
    
def main3(double [:] a, double [:] kk):
    cdef int l_arr = a.shape[0]
    cdef int i
    cdef double c, c2
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(l_arr, dtype=DTYPE)
    cdef np.ndarray[TTYPE_t, ndim=1] iii = np.zeros(l_arr, dtype=TTYPE)
    for i from 0 <= i < l_arr:
        c, c2 = python_bisect(0., 2., a[i], kk[i], 10)
        res[i], iii[i] = secant(c, c2, a[i], kk[i])
    return res, iii