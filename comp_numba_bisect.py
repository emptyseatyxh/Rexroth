# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:36:01 2016

@author: Xinghao
"""

import numpy as np
from scipy.optimize import bisect
from numba import f8, jit, njit
import time


def f(x, a):
    return x**4 - 2*x**2 - a*x - 3
a = np.random.rand(50000)
res = np.empty(a.size)
res1 = np.empty(a.size)

t1 = time.time()

for i in range(len(a)):
    res[i] = bisect(f, -.5, 50, args=(a[i]))

t2 = time.time()

def root_func(x,aa):
    return x**4 - 2*x**2 - aa*x - 3

def compile_specialized_bisect(f):
    def python_bisect(a, b, aa, tol, mxiter):
        its = 0
        fa = jit_root_func(a, aa)
        fb = jit_root_func(b, aa)

        if abs(fa) < tol:
            return a
        elif abs(fb) < tol:
            return b

        c = (a+b)/2.
        fc = jit_root_func(c, aa)

        while abs(fc)>tol and its<mxiter:
            its = its + 1
            if fa*fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            c = (a+b)/2.
            fc = jit_root_func(c, aa)
        return c
    jit_root_func = jit('float64(float64,float64)', nopython=True)(f)
    return jit(nopython=True)(python_bisect)

jit_bisect_root_func = compile_specialized_bisect(root_func)

for i in range(len(a)):
    res1[i] = jit_bisect_root_func(-.5, 50., a[i], 2e-12, 100)
    
t3 = time.time()

tscipy = t2-t1
tjit = t3-t2
error_0 = np.sum(f(res, a))
error_1 = np.sum(f(res1, a))