# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:35:48 2016

@author: Xinghao
"""

#
# Nonlinear equation solvers
#
# Functions:
#   - Fixed-point iteration method: fixed_point(f, x0)
#   - Bisection method: bisection(f, a0, b0)
#   - False position method: false_position(f, a0, b0)
#   - Newton method: newton(f, df, x0)
#   - Secant method: secant(f, x0, x1)
#
# Usage:
#   % python
#   >>> import nonlineq
#   >>> nonlineq.secant(lambda x: 2 ** x - 1.5, 0.0, 0.1)
#
# Solver Author: Yasunori Yusa
# Numba optimiz: Xinghao Yang

DEFAULT_TOLERANCE = 1.0E-9
DEFAULT_MAX_ITER = 100


def fixed_point(f, x0,
                tol=DEFAULT_TOLERANCE, xtol=0.0, ftol=0.0,
                max_iter=DEFAULT_MAX_ITER):
    """
    Solve nonlinear equation by fixed-point iteration method.
    """
    x_new = x0
    i = 0
    while i < max_iter:
        x_old = x_new
        f_old = f(x_old)
        if abs(f_old) <= ftol:
            return x_old, i, True
        x_new = x_old + f_old
        if abs(x_new - x_old) <= max(tol * abs(x_old), xtol):
            return x_new, i, True
        i += 1
    return x_new, max_iter, False


def bisection(f, a0, b0,
              xtol=DEFAULT_TOLERANCE, ftol=0.0,
              max_iter=DEFAULT_MAX_ITER):
    """
    Solve nonlinear equation by bisection method.
    """
    a = a0
    b = b0
    if a > b:
        a, b = b, a
    fa = f(a)
    if fa * f(b) > 0.0:
        return float("inf"), 0, False
    i = 0
    while i < max_iter:
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) <= ftol:
            return c, i, True
        if fc * fa < 0.0:
            b = c
        elif fc * fa > 0.0:
            a = c
            fa = fc
        else:
            return c, i, False
        if b - a <= xtol:
            return 0.5 * (a + b), i, True
        i += 1
    return 0.5 * (a + b), max_iter, False


def false_position(f, a0, b0,
                   ftol=DEFAULT_TOLERANCE,
                   max_iter=DEFAULT_MAX_ITER):
    """
    Solve nonlinear equation by false position method.
    """
    a = a0
    b = b0
    if a > b:
        a, b = b, a
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        return None, 0, False
    i = 0
    while i < max_iter:
        if fb - fa == 0.0:
            return 0.5 * (a + b), i, False
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if abs(fc) <= ftol:
            return c, i, True
        if fc * fa < 0.0:
            b = c
            fb = fc
        elif fc * fa > 0.0:
            a = c
            fa = fc
        else:
            return c, i, False
        i += 1
    return (a * fb - b * fa) / (fb - fa), max_iter, False


def newton(f, df, x0,
           tol=DEFAULT_TOLERANCE, xtol=0.0, ftol=0.0,
           max_iter=DEFAULT_MAX_ITER):
    """
    Solve nonlinear equation by Newton method.
    """
    x_new = x0
    i = 0
    while i < max_iter:
        x_old = x_new
        f_old = f(x_old)
        if abs(f_old) <= ftol:
            return x_old, i, True
        df_old = df(x_old)
        if df_old == 0.0:
            return x_old, i, False
        x_new = x_old - f_old / df_old
        if abs(x_new - x_old) <= max(tol * abs(x_old), xtol):
            return x_new, i, True
        i += 1
    return x_new, max_iter, False


def secant(f, x0, x1,
           tol=DEFAULT_TOLERANCE, xtol=0.0, ftol=0.0,
           max_iter=DEFAULT_MAX_ITER):
    """
    Solve nonlinear equation by secant method.
    """
    x_old = x0
    f_old = f(x_old)
    x_new = x1
    i = 0
    while i < max_iter:
        x_old_old = x_old
        x_old = x_new
        f_old_old = f_old
        f_old = f(x_old)
        if abs(f_old) <= ftol:
            return x_old, i, True
        if f_old - f_old_old == 0.0:
            return x_old, i, False
        x_new = x_old - f_old * (x_old - x_old_old) / (f_old - f_old_old)
        if abs(x_new - x_old) <= max(tol * abs(x_old), ftol):
            return x_new, i, True
        i += 1
    return x_new, max_iter, False