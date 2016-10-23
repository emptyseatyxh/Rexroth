# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:36:59 2016

@author: apple
"""

import numpy as np
import time
from numba import jit, f8, njit
import C_combine as Ccb

def root_func(x,aa,kk):
    return x**4 + kk*x**2 + aa*x - 3
    
a  = np.random.rand(50000)
kk = np.random.randint(-2,50,size=a.shape).astype(np.float64)
res = np.empty(a.size)
res1 = np.empty(a.size)
res2 = np.empty(a.size)

iii1 = np.empty(a.size)
iii2 = np.empty(a.size)

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


# [1] combine bisect and secant
def combine_bisect(f):
    def python_bisect(a, b, aa, kk, tol, mxiter):
        its = 0
        fa = jit_root_func(a, aa, kk)
        fb = jit_root_func(b, aa, kk)
        if abs(fa) < tol:
            return a, 0
        elif abs(fb) < tol:
            return b, 0
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
        return c, a
    jit_root_func = jit('float64(float64,float64,float64)', nopython=True)(f)
    return jit(nopython=True)(python_bisect)
    
def complie_secant2(f):
    def secant(x0, x1, aa, kk, tol=2e-12, ftol=2e-12, max_iter=100):
        x_old = x0
        f_old = jit_root_func(x_old, aa, kk)
        x_new = x1
        i = 0
        while i < max_iter:
            x_old_old = x_old
            x_old = x_new
            f_old_old = f_old
            f_old = jit_root_func(x_old, aa, kk)
            if abs(f_old) <= ftol:
                return x_old, i
            if f_old - f_old_old == 0.0:
                return x_old, i
            x_new = x_old - f_old * (x_old - x_old_old) / (f_old - f_old_old)
            if abs(x_new - x_old) <= max(tol * abs(x_old), ftol):
                return x_new, i
            i += 1
        return x_new, max_iter
    jit_root_func = jit('float64(float64,float64,float64)', nopython=True)(f)
    return jit(nopython=True)(secant)
    
jit_comb_bisect = combine_bisect(root_func)
jit_comb_secant = complie_secant2(root_func)

for i in range(len(a)):
    res1[i], iii1[i] = jit_comb_bisect(0, 2., a[i], kk[i], 2e-12, 10)
    if iii1[i] != 0:
        res1[i], iii1[i] = jit_comb_secant(res1[i], iii1[i], a[i], 
                                           kk[i], 2e-12, 2e-12, 90)
t3 = time.time()
res2, iii2 = Ccb.main(a, kk)
t4 =time.time()

t_0 = t2-t1
t_1 = t3-t2
t_2 = t4-t3
error_0 = np.sum(root_func(res, a, kk))
error_1 = np.sum(root_func(res1, a, kk))
error_2 = np.sum(root_func(res2, a, kk))
average_i1 = np.mean(iii1)
average_i2 = np.mean(iii2)
print(t_0)
print(t_1)
print('err: %s, i: %s'%(error_1, average_i1))
print(t_2)
print('err: %s, i: %s'%(error_2, average_i2))
