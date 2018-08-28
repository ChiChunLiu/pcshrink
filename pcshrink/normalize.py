from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Normalizer(object):
    """Normalizes the data matrix to have 0 mean accross each feature and
    scaled variance

    Arguments
    ---------
    Y : np.array
        p x n normalized genotype matrix
    eps : float
        min allele frequency cutoff
    scale_type : str
        method used for scaling each feature

    Attributes
    ----------
    Y : np.array
        p x n normalized genotype matrix
    eps : float
        min allele frequency cutoff
    scale_type : str
        method used for scaling each feature
    p : int
        number of features (snps)
    n : int
        number of samples (individuals)
    f : np.array
        frequency of each feature
    snp_idx : np.array
        indicies of features to keep
    p_fil : int
        number of features (snps) after
        filtering
    mu : np.array
        mean of each feature
    s : np.array
        scaling factor for each feature
    """
    def __init__(self, Y, eps, scale_type):

        # p x n data matrix
        self.Y = Y

        # allele frequency cutoff
        self.eps = eps

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

        # scale
        if self.scale_type == "emp":
            self.s = np.nanstd(self.Y, axis=1).reshape(self.p_fil, 1)
        elif self.scale_type == "patterson":
            het = 2. * self.f[self.snp_idx] * (1. - self.f[self.snp_idx])
            self.s = np.sqrt(het).reshape(self.p_fil, 1)
        else:
            raise ValueError

        # scale the data
        self.Y = self.Y / self.s

        # impute the missing genotypes with the mean
        self.Y[np.isnan(self.Y)] = 0.0

    def _estimate_frequencies(self):
        """estimates allele frequencies and creates the indicies for features
        to keep
        """
        # use allele frequency estimator from Price et al. 2006
        self.f = (1. + np.nansum(self.Y, axis=1)) / (2 + (2. * self.n))
        self.snp_idx = np.where((self.f > self.eps) &
                                (self.f < (1. - self.eps)))[0]
