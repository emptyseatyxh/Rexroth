# -*- coding: utf-8 -*-

from scipy.interpolate import griddata
import numpy as np
from numpy import std

plane1 = np.random.rand(80000, 2)
long1  = np.linspace(0, 10, 20)
heigt1 = np.repeat(long1, len(plane1)/len(long1))
known  = np.hstack((plane1, heigt1[:,np.newaxis]))

data = np.random.rand(len(known))

plane2 = np.random.rand(80000, 2)
long2  = np.linspace(0, 10, 25)
heigt2 = np.repeat(long2, len(plane2)/len(long2))
unknown = np.hstack((plane2, heigt2[:,np.newaxis]))


#%%

# %timeit griddata(known, data, unknown)
# %timeit griddata(np.random.rand(len(known), 3), data, np.random.rand(len(unknown), 3))

known = known/std(known, axis=0)
unknown = unknown/std(unknown, axis=0)

test_github = 0