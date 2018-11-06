from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip
from scipy import linalg

class ShrinkageCorrector(object):
    """Corrects for regression towards the mean effect when predicting PC
    scores for out of sample individuals. We essentially implement the the
    jackknife procedure briefly outlined in ...

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
    tau : np.array
        shrinkage correction factors for each PC
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
        """PCA using a fast truncated svd implementation in scipy

        Arguments
        ---------
        Y : np.array
            p x n normalized genotype matrix
        k : int
            rank used for truncated svd

        Returns
        -------
        A tuple with elements ...

        L : np.array
            loadings matrix from running PCA
            on the original matrix
        F : np.array
            factor matrix from running PCA
            on the original dataset
        Lamb : np.array
            matrix of eigen values
        """
        # compute truncated svd of data matrix
        V, lamb, VT = svds(Y.T @ Y, k)

        # singular values
        sigma = np.sqrt(lamb[::-1])
        sigma_inv = 1. / sigma

        Sigma = np.diag(sigma)
        Sigma_inv = np.diag(sigma_inv)

        # flip signs of right eigenvectors
        V, VT = svd_flip(V[:, ::-1], VT[::-1])

        F = (Y @ V @ Sigma_inv)

        # project on to factors
        L = (F.T @ Y).T

        return((L, F, Sigma))

    def jackknife(self, q, downdate=False, s=None, o=10):
        """Jackknife estimate of shrinkage correction factor outlined in ...

        https://projecteuclid.org/download/pdfview_1/euclid.aos/1291126967

        We holdout each sample, run PCA on the remaining and subsequently
        project the heldout sample on the training set pcs to obtain a
        predicted pc score. The predicted pc scores alongside the original PCA
        run on all the samples can be used to estimate a shrinkage correction
        factor that can be applied to new samples.

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
            self.sample_idx = np.random.choice(self.Y.shape[1],
                                               s, replace=False)
            r = s
        else:
            # shrink all the samples
            self.sample_idx = np.arange(self.n)
            r = self.n

        # loadings matrix storing shrunk coordinates
        self.L_shrunk = np.empty((r, q))

        # jackknife
        if downdate:
            self._U, self._s, self._Vt = self._full_svd(self.Y)
            for i in range(r):

                if i % o == 0:
                    sys.stdout.write("holding out sample {}\n".format(i))
                
                F = self._downdate(self._U, self._s, self._Vt, k=q, i=i)   
                F = self._orient_sign(F, self.F)
                # project the ith sample back onto the dataset
                self.L_shrunk[i, :] = F.T @ self.Y[:, i]        
        else:
            for i in range(r):

                if i % o == 0:
                    sys.stdout.write("holding out sample {}\n".format(i))

                idx = np.ones(self.n, dtype="bool")
                idx[i] = False

                # PCA on the dataset holding the ith sample out
                L, F, Sigma = self._pca(self.Y[:, idx], q)
                F = self._orient_sign(F, self.F)
                self.L_shrunk[i, :] = F.T @ self.Y[:, i]
        
        # mean pc score from PCA on the full dataset
        mean_pc_scores = np.mean(self.L[self.sample_idx, :q]**2, axis=0)

        # mean pc scores using the projected samples
        mean_pred_pc_scores = np.mean(self.L_shrunk**2, axis=0)

        # jackknife estimate of the shrinkage factor
        self.tau = 1. / np.sqrt(mean_pred_pc_scores / mean_pc_scores)
    
    
    def _full_svd(self, Y):
        U, s, Vt = linalg.svd(Y, full_matrices = False)
        
    def _downdate(U, s, Vt, k, i, sparse = True):
        '''
        Arguments:
        ----------
        U : ndarray
            Unitary matrix having left singular vectors as columns. 
        s : ndarray (r,)
            The singular values sorted in non-increasing order.
        Vt : ndarray
            Unitary matrix having right singular vectors as rows.
        k: int
            number of pcs to compute
        sparese: boolean
            use sparse algorithm for SVD
        Returns:
        --------
        F: new principal component leaving i-th sample out

        see Fast low-rank modifications of the thin singular value
        decomposition by Matthew Brand (2006)
        '''

        l = len(s)
        n = Vt[:, i].reshape(-1, 1)
        S = np.diag(s)
        # building matrix K
        K1 = np.block([[S,               np.zeros((l, 1))],
                      [np.zeros((1, l)), np.zeros((1, 1))]])
        t = np.sqrt(np.max([1 - n.T @ n, 0]))     # take max to prevent negative value
        K2 = np.eye(l+1) - np.vstack((n, 0)) @ np.vstack((n, t)).T
        K = K1 @ K2
        # truncated SVD on K
        if sparse:
            U2, _, _ = svds(K, k = k)
            newF = np.flip(np.hstack((U, np.zeros((U.shape[0],1)))) @ U2[:, :k], axis = 1)
        else:
            U2, _, _ = linalg.svd(K)
            newF = np.hstack((U, np.zeros((U.shape[0],1)))) @ U2[:, :k]

        return newF
    

    def _orient_sign(self, F, F_ref):
        """Orients the sign of the factors matrix to a reference
        factors matrix

        Arguments
        ---------
        F : np.array
            factor matrix whose sign is to be flipped
        F_ref : np.array
            factor matrix whose sign is to be oriented
            towards

        Returns
        -------
        F : np.array
            factor matrix whose with flipped signs
        """
        for k in range(F.shape[1]):

            # compute the sign of correlation
            s_k = np.sign(np.corrcoef(F[:, k], F_ref[:, k])[0, 1])

            # if negatively correlated
            if s_k == -1:
                F[:, k] = -F[:, k]

        return(F)

    def lstsq_project(self, y, k):
        """Projects an individual on pcs using non-missing features

        Arguments
        ---------
        y : np.array
            centered and scaled sample to projected
        k : int
            number of pcs to use for the projection

        Returns
        -------
        l : np.array
            uncorrected loadings for the focal individual
        """
        non_missing_idx = np.where(~np.isnan(y))[0]
        F = self.F[non_missing_idx, :k]
        y = y[non_missing_idx]
        l = np.linalg.lstsq(F, y)[0]

        return(l)
