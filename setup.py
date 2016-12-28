#! -*- coding: utf-8 -*-
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

import numpy as np

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "bbox",
        ["bbox.pyx"],
        include_dirs=[numpy_include]
    ),
    Extension(
        "ssd",
        ["ssd.pyx"],
        include_dirs=[numpy_include]
    ),
    Extension(
        "prior",
        ["prior.pyx"],
        include_dirs=[numpy_include]
    ),
]
setup(
    name='bbox',
    ext_modules=cythonize(ext_modules),
)
