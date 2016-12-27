from __future__ import print_function
from distutils.core import setup
import numpy as np
import os
import sys


setup(
    name='kozaipy-package',
    version="0.0.1",
    author='Diego J. Munoz',
    author_email = 'diego.munoz.anguita@gmail.com',
    url='https://github.com/',
    packages=['kozaipy'],
    description='Calculation of secular dynamics of stars and planets',
    install_requires = ['numpy','scipy'],
    package_data={'kozaipy':['data/*.txt']}
    include_package_data=True
)
