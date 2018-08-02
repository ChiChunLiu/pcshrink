from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip


class ShrinkageCorrector(object):
    """Corrects for regression towards the mean effect 
    when predicting PC scores for out of sample individuals. 
    We essentially implement the the jackknife procedure briefly 
    outlined in ... 

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
    S : np.array
        matrix of singular values
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

        # compute truncated svd of data matrix
        self.U, self.S, self.VT = svds(self.Y, self.k)

        # singular values
        self.S = np.diag(self.S[::-1])

        # left and right eigenvectors
        self.U, self.VT = svd_flip(self.U[:, ::-1], self.VT[::-1])
        self.V = self.VT.T 
        
        self.F = self.U @ self.S
        self.F = self.F / np.linalg.norm(self.F, axis=0, ord=2)
        self.L = (self.F.T @ self.Y).T

    def _svd_rank1(self, i):
        """
        """
        a = -self.Y[:, i].reshape(self.p, 1)
        b = np.zeros(self.n).reshape(self.n, 1)
        b[i] = 1
        
        # swap the ith and last row of the 
        # right eigenvectos
        #v_last = self.V[-1, :]
        #v_i = self.V[i, :]
        #V = self.V
        #V[-1, :] = v_i
        #V[i, :] =  v_last

        Up, Sp, Vp = svdUpdate(self.U, self.S, self.V, a, b)

        return(Up, Sp)

    def jackknife(self, q, o=10):
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
        o : int
            interval of jackknife iterations to print update
        """
        # loadings matrix storing shrunk coordinates
        self.L_shrunk = np.empty((self.n, q))

        # jackknife 
        for i in range(self.n):
                
            if i % o == 0:
                sys.stdout.write("holding out sample {}\n".format(i))
                
            Up, Sp = self._svd_rank1(i)
            F = Up @ Sp
            F = F / np.linalg.norm(F, axis=0, ord=2)
                
            # project the ith sample back onto the dataset 
            self.L_shrunk[i, :] = F.T @ self.Y[:, i]

        # mean pc score from PCA on the full dataset
        mean_pc_scores = np.mean(self.L[:, :q]**2, axis=0)

        # mean pc scores using the projected samples
        mean_pred_pc_scores = np.mean(self.L_shrunk**2, axis=0)

        # jackknife estimate of the shrinkage factor
        self.tau = 1. / np.sqrt(mean_pred_pc_scores / mean_pc_scores)


def svdUpdate(U, S, V, a, b):
    S = np.asmatrix(S)
    U = np.asmatrix(U)
    if V is not None:
        V = np.asmatrix(V)
    a = np.asmatrix(a).reshape(a.size, 1)
    b = np.asmatrix(b).reshape(b.size, 1)
    
    rank = S.shape[0]
    
    # eq (6)
    m = U.T * a
    p = a - U * m
    Ra = np.sqrt(p.T * p)
    P = (1.0 / float(Ra)) * p
    
    if V is not None:
        # eq (7)
        n = V.T * b
        q = b - V * n
        Rb = np.sqrt(q.T * q)
        Q = (1.0 / float(Rb)) * q
    else:
        n = np.matrix(np.zeros((rank, 1)))
        Rb = np.matrix([[1.0]])    
    
    # eq (8)
    K = np.matrix(np.diag(list(np.diag(S)) + [0.0])) + np.bmat('m ; Ra') * np.bmat('n ; Rb').T
    
    # eq (5)
    #u, s, vt = np.linalg.svd(K, full_matrices = False)
    u, s, vt = svds(K, rank)
    u, vt = svd_flip(u[:, ::-1], vt[::-1])

    tUp = np.matrix(u[:, :rank])
    tVp = np.matrix(vt.T[:, :rank])
    tSp = np.matrix(np.diag(s[: rank]))
    Up = np.bmat('U P') * tUp
    if V is not None:
        Vp = np.bmat('V Q') * tVp
    else:
        Vp = None
    Sp = tSp
    
    return(Up, Sp, Vp)
