from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules=cythonize("bbox.pyx"),
# )
# setup(
#     ext_modules=[
#         Extension("bbox", ["bbox.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("bbox.pyx"),
    include_dirs=[numpy.get_include()]
)
