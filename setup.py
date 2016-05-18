from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

gco_directory = "gco_src"

files = ['GCoptimization.cpp', 'graph.cpp',
         'LinkedBlockList.cpp', 'maxflow.cpp']

files = [os.path.join(gco_directory, f) for f in files]

files.append('GCoptMultiSmooth.cpp')

files.insert(0, "gco_python.pyx")

setup(name='gco_python',
      version='0.01',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pygco", files, language="c++",
                             include_dirs=['.', gco_directory, numpy.get_include()],
                             library_dirs=[gco_directory],
                             undef_macros = [ "NDEBUG" ],
                             extra_compile_args=["-fpermissive",
                                                 "-DGCO_ENERGYTERMTYPE=float",
                                                 "-DGCO_MAX_ENERGYTERM=(static_cast<GCO_ENERGYTERMTYPE>(1e10))"])])
