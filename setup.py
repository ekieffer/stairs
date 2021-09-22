from setuptools import Extension, setup
from setuptools import find_packages
from distutils.command.build import build as build_orig
from Cython.Build import cythonize
import numpy
import subprocess
from subprocess import CalledProcessError
import sys


ext_modules = [
    Extension(
        "stairs.portfolios",
        ["stairs/portfolios.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]


setup(name='stairs_env',
      version='0.0.1',
      packages=find_packages(),
      #install_requires=install_requires,
      #install_requires=['pandas','gym','numpy','scipy','h5py'],
      ext_modules=cythonize(ext_modules),
      package_data={"stairs":["portfolios.pyx"]}
)

