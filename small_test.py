# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 17:15:49 2016

@author: Xinghao
"""

import numpy as np
from scipy import linalg
'''test linear solver for 3 D
np.random.seed(1)
a = np.random.rand(6,3).reshape(2,3,3)
d1 = np.random.rand(1, 3)
d2 = np.random.rand(1, 3)
d  = np.vstack((d1,d2))
t1 = np.linalg.solve(a, d1)
t2 = np.linalg.solve(a, d2)
t  = np.linalg.solve(a, d)
'''

a = np.arange(1, 10)
l = [a,a,a,a,a,a]
r = l[0]
for i in range(len(l)-1):
     r = np.hstack((r, l[i]))


    