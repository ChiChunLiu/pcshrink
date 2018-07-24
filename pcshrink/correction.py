from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import linear_model

from .data import *
from .opt import *


class ShrinkageCorrector(object):
    
    def __init__(self, prefix, proj_ind_path, snp_weight_path, scale_method):
        """Fit a regression model to correct for shinkage effect of projecting
        out of sample data points onto trained PCs. The approach taken here is

        Step 1: Run PCA on the entire dataset to obtain 
        
        Step 2: Run PCA for n datasets where each dataset holds-out a unique
        sample. Project this held-out sample onto the principal axes learned
        in the remaining samples. 
        
        Step 3: Fit a regression model between the coordinates fit in 1 and 2 to 
        learn a correction factor

        Step 2 could be computationally intensive if n is large as it requires
        running n different PCAs (via truncated svd). Here we take advatage of the 
        fact that the feature weights should be similar accross all runs. We thus use 
        an EM algorithim for each run of step 2 intialized with feature weights
        learned in step 1 with the hope that the EM should converge quickly
        after a few iterations. 

        Arguments
        ---------
        prefix : str
            prefix of path to data files in eigenstrat format
        proj_ind_path : str
            path to pop list file which is a single column of individual
            ids whom are to be projected on the the referece principal axes
        snp_weight_path : str
            path to file with snp weights learned from eigenstrat
        scale_method : str
            either patterson or emprical which scales the gentoype matrix
            using snp heterozygosity or the emprical std deviation

        Attributes
        ----------
        prefix : str
            prefix of path to data files in eigenstrat format
        proj_ind_path : str
            path to pop list file which is a single column of individual
            ids whom are to be projected on the the referece principal axes
        snp_weight_path : str
            path to file with snp weights learned from eigenstrat
        scale_method : str
            either patterson or emprical which scales the gentoype matrix
            using snp heterozygosity or the emprical std deviation
        data : Genotpyes
            genotypes object of passed in eigenstrat data
        proj_ind_idx : np.array
            array of indicies for individuals to be projected
        snp_weight_df : pd.DataFrame
            DataFrame of snp weights output by eigenstrat
        snp_idx : np.array
            array of snp indicies defined by the snp weights data
        Y_train : np.array
            array of individuals genotype data used to train the correction
            model at the snps defined by the snp weight data
        mu : np.array
            empirical mean genotype for each snp
        sigma : np.array
            scaling factor for each snp
        L_proj : np.array
            coordinates for each heldout individual obtained by projecting the
            heldout datapoint onto principal axes learned from everyone else
        """
        # file paths
        self.prefix = prefix
        self.proj_ind_path = proj_ind_path
        self.snp_weight_path = snp_weight_path

        # method used to scale genotype matrix
        self.scale_method = scale_method

        # read genotype data
        self.data = UnpackedAncestryMap(prefix)
       
        # individuals ids who are to be projected 
        self._get_proj_ind_idx()
       
        if self.snp_weight_path == None:
            # run PCA on the training dataset using sklearn
            raise NotImplementedError
        else:
            # read in snp weights DataFrame 
            self._read_snp_weights()
            self._get_snp_idx()

            # create the factors matrix from the snp weights
            # TODO: this should be orthonormal but its not need 
            # to check ... also eigenstrat only outputs low number 
            # of digits so we might run into numerical issues here
            self.F_hat = self.snp_weight_df.iloc[:, 4:].as_matrix()
            
            # number of PCs used
            self.K = F_hat.shape[1]
       
        # dataset to train the correction model on 
        self.Y_train = data.Y[self.snp_idx, ~self.proj_ind_idx]

        # scale and center the training dataset
        self._normalize_genotypes() 

        # number of snps and individuals used in training dataset
        self.p_train, self.n_train = self.Y_train.shape
    
    def _get_proj_ind_idx(self):
        """gets the indicies of individuals whom are to be projected
        """
        proj_ind_df = pd.read_table(self.proj_ind_path, header=None, names=["iid"])
        ind_df = pd.DataFrame({"iid": data.ind})
        mrg_df = ind_df.merge(proj_ind_df, how="inner")
        self.proj_ind_idx = mrg_df.index()

    def _read_snp_weights(self):
        """gets a DataFrame of snp weights output from eigenstrat 
        """
        self.snp_weight_df = pd.read_table(snp_weight_path, header=None, sep="\t")[0].str.lstrip().str.split(expand=True)
        self.snp_weight_df.columns = ["IDX", "CHROM", "POS"] + ["PC{}".format(i) for i in range(1, self.snp_weight_df.shape[1] - 3 + 1)]

    def _get_snp_idx(self):
        """gets the snp indicies of the snps defined in the eigenstrat weights
        """
        mrg_df = self.data.snp_df.merge(self.snp_weight_df, how="inner")
        self.snp_idx = mrg_df.index()

    def _normalize_genotypes(self):
        """centers and scales the training gentoypes
        """
        # compute the emprical mean for each SNP ignoring NAs
        self.mu = np.nanmean(self.Y_train, axis=0)
        
        # compute the scaling factor using het or empircal sd
        if self.scale_method == "patterson":
            f = np.nansum(self.Y_train, axis=0) / (2. * self.data.n)
            # TODO: check sqrt 
            self.sigma = np.sqrt(2 * f * (1. - f))
        elif self.scale_method == "empirical": 
            self.sigma = np.nanstd(self.Y_train, axis=0)
        else:
            raise ValueError("scale method is not found please use patterson or empirical as arguments")
    
        # do the normalization and impute missing values to 0
        self.Y_train = (self.Y_train - self.mu) / self.sigma
        self.Y_train[np.isnan(Y_train)] = 0.0

    def train(self):
        """obtains the projection coordinates for each heldout individual
        """
        # loadings matrix of heldout individuals
        self.L_proj = np.empty((self.n_train, self.K))
        
        # run pca for each dataset with a single individual heldout
        # and then project them back on to obtain the shrunk coordinates
        for i in range(self.n_train):
            
            # keep everyone except ith individual
            idx = np.ones(self.data.n, dtype=bool)
            idx[i] = False

            # run the EM algorithim to estimate the factors for 
            # the heldout dataset
            pcem = PCEM(self.Y_train[:, idx], self.F_hat)
            pcem.run()

            # project the heldout person onto those factors 
            # TODO: implement projection
            self.L_proj[i, :] = l_proj

    def correct(self):
        """
        """
        raise NotImplementedError

