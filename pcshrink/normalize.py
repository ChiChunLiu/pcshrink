from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip


class Normalizer(object):
    """
    """
    def __init__(self, Y, eps, scale_type):
        
        # p x n data matrix
        self.Y = Y
    
        # allele frequency cutoff
        self.eps

        self.scale_type = scale_type

        # number of features x number of samples
        self.p, self.n = self.Y.shape

        # Estimate frequencies and filter out 
        # too rare or too common variants
        self._estimate_frequencies()
        self.Y = self.Y[self.snp_idx, :]
        self.p_fil = self.Y.shape[0]

        # compute the mean genpotype
        self.mu = np.nanmean(self.Y, axis=1).reshape(self.p_fil, 1)
    
        # center
        self.Y = self.Y - self.mu 
        self.Y[np.isnan(self.Y)] = 0.0

        # scale
        if self.scale_type == "emp":
            self.std = np.nanstd(self.Y, axis=1).reshape(self.p_fil, 1)
            self.Y = self.Y / self.std
        elif self.scale_type == "patterson":
            self.het = np.sqrt(2. * self.f[self.snp_idx * (1. - self.f[self.snp_idx])).reshape(self.p_fil, 1)
            self.Y = self.Y / self.het
        else:
            raise ValueError

    def _estimate_frequencies(self):
        """
        """
        # use allele frequency estimator from Price et al. 2006
        self.f = (1. + np.nansum(self.Y, axis=1)) / (2 + (2. * self.n))
        self.snp_idx = np.where((self.f > self.eps) & (self.f < (1. - self.eps)))[0]
