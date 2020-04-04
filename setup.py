#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['siamese-mask-rcnn'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 
	    'scikit_image',
	    'keras',
        'opencv-python',
        'h5py',
        'imgaug',
	    'cython',
        'tensorflow-gpu'
    ],
)

setup(**d)
