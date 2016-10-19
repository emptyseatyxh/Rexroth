# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:36:01 2016

@author: Xinghao
"""

import numpy as np
from scipy.optimize import bisect
from numba import f8, jit, njit
import time

# [0] scipy bisection
# [1] numba bisection 1
# [2] numba bisection 2
# [3] numba false_position
# [4] numba secant
# [5] numba combine bisect and secant

def root_func(x,aa,kk):
    return x**4 + kk*x**2 + aa*x - 3
    
a  = np.random.rand(50000)
kk = np.random.randint(-2,50,size=a.shape).astype(np.float64)
res = np.empty(a.size)
res1 = np.empty(a.size)
res2 = np.empty(a.size)
res3 = np.empty(a.size)
res4 = np.empty(a.size)
res5 = np.empty(a.size)

iii2 = np.empty(a.size)
iii3 = np.empty(a.size)
iii4 = np.empty(a.size)
iii5 = np.empty(a.size)

# [0] scipy bisection
t1 = time.time()
for i in range(len(a)):
    res[i] = bisect(root_func, 0, 2, args=(a[i], kk[i]))
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


# [2] numba bisection 2
def compile_specialized_bisect2(f):
    def bisection2(a0, b0, aa, kk, ftol=2e-12, max_iter=100):
        a = a0
        b = b0
        if a > b:
            a, b = b, a
        fa = jit_root_func(a, aa, kk)
        fb = jit_root_func(b, aa, kk)
        if fa * fb > 0.0:
            return 9999999.0, 0
        i = 0
        while i < max_iter:
            c = 0.5 * (a + b)
            fc = jit_root_func(c, aa, kk)
            if abs(fc) <= ftol:
                return c, i
            if fc * fa < 0.0:
                b = c
            elif fc * fa > 0.0:
                a = c
                fa = fc
            else:
                return c, i
            i += 1
        return 0.5 * (a + b), max_iter
    jit_root_func = jit('float64(float64,float64,float64)', nopython=True)(f)
    return jit(nopython=True)(bisection2)

jit_bisect2 = compile_specialized_bisect2(root_func)
for i in range(len(a)):
    res2[i], iii2[i]= jit_bisect2(0, 2., a[i], kk[i], 2e-12, 100)
t4 = time.time()


# [3] numba false_position
def compile_false_position(f):
    def false_position(a0, b0, aa, kk, ftol=2e-12, max_iter=100):
        a = a0
        b = b0
        if a > b:
            a, b = b, a
        fa = jit_root_func(a, aa, kk)
        fb = jit_root_func(b, aa, kk)
        if fa * fb > 0.0:
            return 9999999.0, 0
        i = 0
        while i < max_iter:
            if fb - fa == 0.0:
                return 0.5 * (a + b), i
            c = (a * fb - b * fa) / (fb - fa)
            fc = jit_root_func(c, aa, kk)
            if abs(fc) <= ftol:
                return c, i
            if fc * fa < 0.0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            i += 1
        return (a * fb - b * fa) / (fb - fa), max_iter
    jit_root_func = jit('float64(float64,float64,float64)', nopython=True)(f)
    return jit(nopython=True)(false_position)

jit_fp = compile_false_position(root_func)
for i in range(len(a)):
    res3[i], iii3[i] = jit_fp(0, 2., a[i], kk[i], 2e-12, 1000)
t5 = time.time()


# [4] numba secant
def complie_secant(f):
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

jit_secant = complie_secant(root_func)
for i in range(len(a)):
    res4[i], iii4[i] = jit_secant(0, 2., a[i], kk[i], 2e-12, 2e-12, 100)
t6 = time.time()


# [5] combine bisect and secant
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
    res5[i], iii5[i] = jit_comb_bisect(0, 2., a[i], kk[i], 2e-12, 10)
    if iii5[i] != 0:
        res5[i], iii5[i] = jit_comb_secant(res5[i], iii5[i], a[i], kk[i], 2e-12, 2e-12, 90)
t7 = time.time()

# compare results and performance
tscipy = t2-t1
tbisec1 = t3-t2
tbisec2 = t4-t3
tfalpos = t5-t4
tsecant = t6-t5
tcombin = t7-t6
error_0 = np.sum(root_func(res, a, kk))
error_1 = np.sum(root_func(res1, a, kk))
error_2 = np.sum(root_func(res2, a, kk))
error_3 = np.sum(root_func(res3, a, kk))
error_4 = np.sum(root_func(res4, a, kk))
error_5 = np.sum(root_func(res5, a, kk))
average_i2 = np.mean(iii2)
average_i3 = np.mean(iii3)
average_i4 = np.mean(iii4)
average_i5 = np.mean(iii5)