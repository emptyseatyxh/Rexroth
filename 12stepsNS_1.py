# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:44:07 2016

@author: apple
"""

import numpy as np
from matplotlib import pyplot
import time, sys

nx = 41
dx = 2./(nx-1)
nt = 25
dt = 0.005

c = 1
u = np.ones(nx)
u[0.5/dx : 1/dx+1] = 2

un = np.ones((nx,nt))

for n in range(nt):
    un[:,n] = u.copy()
    for i in range(1,nx):
        u[i] = u[i] - c*dt/dx*(u[i]-u[i-1])
        
pyplot.plot(np.linspace(0,2,nx),un)
        
