# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:34:57 2016

@author: apple
"""

import numpy as np
from matplotlib import pyplot

nx = 161
dx = 2./(nx-1)
nt = 200
dt = 0.004

u = np.ones(nx)
u[0.5/dx:1/dx+1] = 2

un = np.ones((nx,nt))

for n in range(nt):
    un[:,n] = u
    for i in range(1, nx):
        u[i] = un[i,n] - un[i,n]*dt/dx*(un[i,n]-un[i-1,n])

pyplot.plot(np.linspace(0,2,nx),un)