"""Vendored python-glmnet package for knockoff-filter.

This is a vendored copy of https://github.com/civisanalytics/python-glmnet
with modifications for use as an internal package.
"""

from .logistic import LogitNet
from .linear import ElasticNet

__all__ = ['LogitNet', 'ElasticNet']

__version__ = "2.2.1-vendored"
