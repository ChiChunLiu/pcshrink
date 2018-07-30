from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip


class PCShrinker(object):
    """
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

        Returns
        -------

        """
        # compute truncated svd of data matrix
        U, Sigma, VT = svds(Y.T, k)

        # singular values
        Sigma = np.diag(Sigma[::-1])

        # left and right eigenvectors
        U, VT = svd_flip(U[:, ::-1], VT[::-1])

        # assign to matricies for matrix 
        # factorization interpretation
        L = U 
        F = (Sigma @ VT).T

        # normalize vectors to be unit length
        L = L / np.linalg.norm(L, axis=0, ord=2)
        F = F / np.linalg.norm(F, axis=0, ord=2)

        return((L, F, Sigma))

    def shrink_coords(self, k, s=None, o=10):
        """Shrink PC coordinates by holding out an 
        individual, running PCA on the remaining, and 
        re-projecting the heldout individual back on to 
        the principal axes learned in the training dataset

        Arguments
        ---------

        Returns
        -------

        """
        if s != None:
            self.samp_idx = np.random.choice(self.Y.shape[1], s, replace=False)
            Y = self.Y[:, self.samp_idx]
            n = s
        else: 
            Y = np.copy(self.Y)
            n = self.n

        # loadings matrix storing shrunk coordinates
        L_shrunk = np.empty((n, k))
        for i in range(n):
                
            if i % o == 0:
                sys.stdout.write("holding out sample {}\n".format(i))
                
            idx = np.ones(n, dtype="bool")
            idx[i] = False
                
            # PCA on the dataset holding the ith sample out 
            L, F, Sigma = self._pca(Y[:, idx], k)
                
            # project the ith sample back onto the dataset 
            L_shrunk[i, :] = F.T @ Y[:, i]

        return(L_shrunk)
