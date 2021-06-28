#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='uai-anonymizer',
      version='latest',
      packages=find_packages(exclude=['test', 'test.*']),

      install_requires=[
          'pytest~=6.2.4',
          'flake8~=3.9.2',
          'numpy~=1.18.5',
          'tensorflow-gpu~=1.15.0',
          'scipy~=1.7.0',
          'Pillow~=8.2.0',
          'requests~=2.25.1',
          'googledrivedownloader~=0.4',
          'tqdm~=4.61.1',
      ],

      dependency_links=[
      ],
      )
