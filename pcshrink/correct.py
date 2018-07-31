from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip
from sklearn import linear_model


class ShrinkageCorrector(object):
    """Corrects for regression towards the mean effect 
    when predicting PC scores for out of sample individuals. 
    We essentially implement the ideas outlined in 

    https://projecteuclid.org/download/pdfview_1/euclid.aos/1291126967

    Arugments
    ---------
    Y : np.array
        p x n normalized genotype matrix
    k : int
        rank used for truncated svd

    Attributes
    ----------
    Y : np.array
        p x n normalized genotype matrix
    k : int
        rank used for truncated svd
    p : int
        number of features (snps)
    n : int
        number of samples (individuals)
    L : np.array
        loadings matrix from running PCA 
        on the original matrix
    F : np.array
        factor matrix from running PCA
        on the original dataset
    Sigma : np.array
        matrix of singular values
    sample_idx : np.array
        boolean array of randomly sampled indicies for 
        fast jackknife approach
    L_shrunk : np.array
        loadings matrix of projected heldout individuals
    shrinkage_factors : np.array
        shrinkage correction factors for each pc
    """
    def __init__(self, Y, k):
        
        # p x n data matrix
        self.Y = Y
       
        # use rank k approximation
        self.k = k

        # number of features x number of samples
        self.p, self.n = self.Y.shape

        # loadings, factors, and singular values
        self.L, self.F, self.Sigma = self._pca(self.Y, self.k)

    def _pca(self, Y, k):
        """PCA using a fast truncated svd implementation 
        in scipy
        
        Arguments
        ---------
        Y : np.array
            p x n normalized genotype matrix
        k : int
            rank used for truncated svd

        Returns
        -------
        A tuple with ...

        L : np.array
            loadings matrix from running PCA 
            on the original matrix
        F : np.array
            factor matrix from running PCA
            on the original dataset
        Sigma : np.array
            matrix of singular values
        """
        # compute truncated svd of data matrix
        U, Sigma, VT = svds(Y.T, k)

        # singular values
        Sigma = np.diag(Sigma[::-1])

        # left and right eigenvectors
        U, VT = svd_flip(U[:, ::-1], VT[::-1])

        # assign to matricies for matrix 
        # factorization interpretation
        F = (Sigma @ VT).T
        
        # normalize vectors to be unit length
        F = F / np.linalg.norm(F, axis=0, ord=2)

        # project on to factors
        L = (F.T @ Y).T

        return((L, F, Sigma))

    def jackknife(self, q, s=None, o=10):
        """Jackknife estimate of shrinkage correction factor outlined in ...

        https://projecteuclid.org/download/pdfview_1/euclid.aos/1291126967

        We holdout each sample, run PCA on the remaining and subsequently project the 
        heldout sample on the training set pcs to obtain a predicted pc score. The predicted 
        pc scores alongside the original PCA run on all the samples can be used to estimate a 
        shrinkage correction factor that can be applied to new samples.

        Arguments
        ---------
        q : int
            number of pcs to estimate shrinkage factors on
        s : int
            number of random samples (without replacement) to draw
            for fast approximate jackknife estimate
        o : int
            interval of jackknife iterations to print update
        """
        if s != None:
            # shrink only a random subset of samples
            self.samp_idx = np.random.choice(self.Y.shape[1], s, replace=False)
            r = s
        else: 
            # shrink all the samples
            r = self.n

        # loadings matrix storing shrunk coordinates
        self.L_shrunk = np.empty((r, q))

        # jackknife 
        for i in range(r):
                
            if i % o == 0:
                sys.stdout.write("holding out sample {}\n".format(i))
                
            idx = np.ones(self.n, dtype="bool")
            idx[i] = False
                
            # PCA on the dataset holding the ith sample out 
            L, F, Sigma = self._pca(self.Y[:, idx], q)
                
            # project the ith sample back onto the dataset 
            self.L_shrunk[i, :] = F.T @ self.Y[:, i]

        # mean pc score from PCA on the full dataset
        mean_pc_scores = np.mean(self.L[self.samp_idx, :q], axis=0)

        # mean pc scores using the projected samples
        mean_pred_pc_scores = np.mean(self.L_shrunk, axis=0)

        # jackknife estimate of the shrinkage factor
        self.shrinkage_factors = np.sqrt((mean_pc_scores**2) / (mean_pred_pc_scores**2))
