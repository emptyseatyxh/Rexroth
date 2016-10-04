# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:41:22 2016

@author: apple
"""
import re
import numpy as np
from numpy.lib.stride_tricks import as_strided

a = np.arange(90, dtype=np.float32).reshape(-1,5)
b = as_strided(a, shape=(10, 9), strides=(40,4))

