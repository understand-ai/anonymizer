#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='uai-anonymizer',
      version='latest',
      packages=find_packages(exclude=['test', 'test.*']),

      install_requires=[
          'pytest>=3.9.1',
          'flake8>=3.5.0',
          'numpy>=1.15.2',
          'tensorflow-gpu>=1.11.0',
          'scipy>=1.1.0',
          'Pillow>=5.3.0',
          'requests>=2.20.0',
          'googledrivedownloader>=0.3',
          'tqdm>=4.28.0',
      ],

      dependency_links=[
      ],
      )
