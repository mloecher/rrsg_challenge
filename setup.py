from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension("pygrid.c_grid", ["./pygrid/src/c_grid.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions),
)