# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:38:13 2016

@author: apple
"""
import numpy as np
import time
from numba import jit, f8, njit
import C_bisect as Cbi

def root_func(x,aa,kk):
    return x**4 + kk*x**2 + aa*x - 3
    
a  = np.random.rand(50000)
kk = np.random.randint(-2,50,size=a.shape).astype(np.float64)
res = np.empty(a.size)
res1 = np.empty(a.size)
res2 = np.empty(a.size)


# [0] python bisection
t1 = time.time()
def python_bisect(a, b, aa, kk, tol, mxiter):
    its = 0
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

for i in range(len(a)):
    res[i] = python_bisect(0, 2., a[i], kk[i], 2e-12, 100)
t2 = time.time()


# [1] numba bisection 1
def compile_specialized_bisect(f):
    def python_bisect(a, b, aa, kk, tol, mxiter):
        its = 0
        fa = jit_root_func(a, aa, kk)
        fb = jit_root_func(b, aa, kk)

        if abs(fa) < tol:
            return a
        elif abs(fb) < tol:
            return b

        c = (a+b)/2.
        fc = jit_root_func(c, aa, kk)

        while abs(fc)>tol and its<mxiter:
            its = its + 1
            if fa*fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            c = (a+b)/2.
            fc = jit_root_func(c, aa, kk)
        return c
    jit_root_func = jit('float64(float64,float64,float64)', nopython=True)(f)
    return jit(nopython=True)(python_bisect)

jit_bisect1 = compile_specialized_bisect(root_func)

for i in range(len(a)):
    res1[i] = jit_bisect1(0, 2., a[i], kk[i], 2e-12, 100)
t3 = time.time()

'''
for i in range(len(a)):
    res2[i] = Cbi(0, 2., a[i], kk[i], 2e-12, 100)
'''
res2 = Cbi.main(a, kk)
t4 =time.time()

t_0 = t2-t1
t_1 = t3-t2
t_2 = t4-t3
error_0 = np.sum(root_func(res, a, kk))
error_1 = np.sum(root_func(res1, a, kk))
error_2 = np.sum(root_func(res2, a, kk))
print(t_0)
print(t_1)
print(t_2)