from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

#setup(ext_modules = cythonize('example_cy.pyx'))
#setup(ext_modules = cythonize('createHogFeatures36C.pyx'))
#setup(ext_modules = cythonize('nmsHOGC.pyx'))
extensions = [Extension("nmsHOGC", ["nmsHOGC.pyx"], include_dirs=[numpy.get_include()]),Extension('detect36C', ['detect36C.pyx', 'hog.c'], include_dirs=[numpy.get_include()]), Extension('flattenImageC', ['flattenImage.pyx'], include_dirs=[numpy.get_include()])]
setup(ext_modules = cythonize(extensions))
