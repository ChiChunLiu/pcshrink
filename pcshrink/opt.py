from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats


class PCEM(object):

    def __init__(self, Y, F_hat):
        """Expectation Maximazation (EM) for maximum likelihood estimation of 
        the probalistic formulation of principal components analysis (PCA). In the 
        zero noise limit this results in an efficent alternating least squares 
        algorithim see Murphy 2012 for a nice review.

        Arguments
        ---------
        Y : np.array
            p x n normalized genotype matrix
        F : np.array
            p x k factors matrix

        Attributes
        ----------
        Y : np.array
            p x n normalized genotype matrix
        F : np.array
            p x k factors matrix
        L : np.array
            n x k loadings matrix
        liks : np.array
            t x 1 array of likelihood values for each iteration
            of the EM
        """
        self.Y = Y
        self.p, self.n = Y.shape

        self.F = F_hat
        self.K = F_hat.shape[1]
    
    def _e_step(self):
        """Compute the posterior expectation of the latent variable
        which can be see as obtaining an estimate of the loadings matrix
        """
        self.L = np.linalg.solve(self.F.T @ self.F, self.F.T @ self.Y).T

    def _m_step(self):
        """Maximize the expected complete data log-likelihood with respect
        to the parameters which can be seen as obtaining an estimate 
        of the factors matrix
        """
        self.F = np.linalg.solve(self.L.T @ self.L, self.L.T @ self.Y.T).T 

    def _comp_lik(self):
        """Compute the likelihood given the current parameters

        Returns
        -------
        lik : float
            the evaluted likelihood given the current estimate
        """
        lik = np.linalg.norm(self.Y - self.F @ self.L.T, "fro")
        
        return(lik)

    def run(self, eps, max_iter):
        """Run the EM algorithim for problastic pca 

        Arguments
        ---------
        eps : float
            small difference in likelihood (between iterations) to 
            terminate the algorithim
        max_iter : int
            maximum number of iterations to teriminate the algorithim
        sigma_e : float
            noise variance which is assumed to be very small to approximate the 
            zero noise limit
        """
        t = 0
        self.liks = [np.Infinity]

        not_converged = True

        while(not_converged):
            
            # updates
            self._e_step()
            self._m_step()
            
            t += 1

            # compute likelihood
            lik = self._comp_lik()
            self.liks.append(lik)

            # check convergence
            delta_t = self.liks[t-1] - self.liks[t]
            print(t, lik, delta_t)
            if (delta_t < eps) or (t > max_iter):
                not_converged = False

        # convert likelihoods to array
        self.liks = np.array(self.liks)
        
        # orthoganlize the factors with QR decomposition
        self.F, R = np.linalg.qr(self.F)