# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 11:15:17 2016

@author: apple
"""
# python setup.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("C_combine.pyx"),
    include_dirs=[numpy.get_include()]
    )