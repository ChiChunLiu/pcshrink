from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools as it
import pandas as pd


class Genotypes(object):
    """A class for storing genotype data which at its core 
    consists of a dense n x p genotype matrix, a dataframe 
    of metadata associated which each snp, and a list of 
    individual ids. 

    Arguments
    ---------
    prefix : str
        prefix of  path to data files

    Attributes
    ----------
    prefix : str
        prefix of  path to data files
    Y : np.array
        p x n genotype matrix 
    snp_df : pd.DataFrame
        Dataframe storing snp level meta data
    inds : list
        list of individual ids
    n : int
        number of individuals (samples)
    p : int 
        number of snps (features)
    """
    def __init__(self, prefix):
        # prefix of genotype files
        self.prefix = prefix

        # n x p genotype matrix
        self.Y = None

        # DataFrame of SNP level information
        self.snp_df = None

        # list of individual ids
        self.inds = None

        # number of individuals
        self.n = None

        # number of snps
        self.p = None


class AncestryMap(Genotypes):
    """A class for the eigenstrat genotype format
    which consists of a geno, snp, and ind file

    Arguments
    ---------
    prefix : str
        prefix of  path to data files

    Attributes
    ----------
    prefix : str
        prefix of  path to data files
    Y : np.array
        n x p genotype matrix
    snp_df : pd.DataFrame
        Dataframe storing snp level meta data
    inds : list
        list of individual ids
    n : int
        number of individuals (samples)
    p : int
        number of snps (features)
    geno_path : str
        path to eigenstrat geno file
    ind_path : str
        path to eigenstrat ind file
    snp_path : str
        path to eigenstart snp file
    """
    def __init__(self, prefix):
        # inherit attributes from GenotypeFormat
        super().__init__(prefix)

        # path to eigenstart geno file
        self.geno_path = "{}.geno".format(self.prefix)

        # path to eigenstrat ind file
        self.ind_path = "{}.ind".format(self.prefix)

        # path to eigenstrat snp file
        self.snp_path = "{}.snp".format(self.prefix)

    def _get_snp_dataframe(self):
        """Gets snp level dataframe stored as 
        pandas DataFrame
        """
        # read into pandas df
        self.snp_df = pd.read_table(self.snp_path, header=None,
                                    delim_whitespace=True)

        # check if the dataframe has 6 columns
        if self.snp_df.shape[1] < 6:
            raise ValueError("{}.snp must have 6 cols".format(self.snp_path))

        # name columns
        self.snp_df.columns = ["SNP", "CHROM", "CM_POS", "POS", "A1", "A2"]

        # number of snps
        self.p = self.snp_df.shape[0]

        # add rownumbers
        self.snp_df["idx"] = np.arange(0, self.p, 1)

    def _get_inds(self):
        """Get list of individual ids
        """
        ind_df = pd.read_table(self.ind_path, header=None,
                               delim_whitespace=True, delimiter="\t")
        
        # check if the dataframe has 6 columns
        if ind_df.shape[1] < 3:
            raise ValueError("{}.ind must have 3 cols".format(self.ind_path))

        ind_df.columns = ["IND", "SEX", "CLST"]
        self.inds = ind_df["IND"].tolist()

        # number of inds
        self.n = len(self.inds)


class PackedAncestryMap(AncestryMap):
    """Class for packed ancestry map eigenstrat format. The packed format's
    geno is binary so it requires a couple steps processing before being loaded 
    into a numpy array. We modify some code from
    
    https://github.com/mathii/pyEigenstrat
    
    Arguments
    ---------
    prefix : str
        prefix of  path to data files

    Attributes
    ----------
    prefix : str
        prefix of  path to data files

    Y : np.array
        n x p genotype matrix 
    snp_df : pd.DataFrame
        Dataframe storing snp level meta data
    inds : list
        list of individual ids
    n : int
        number of individuals (samples)
    p : int
        number of snps (features)
    geno_path : str
        path to eigenstrat geno file
    ind_path : str
        path to eigenstrat ind file
    snp_path : str
        path to eigenstart snp file
    """

    def __init__(self, prefix):
        # inhert attributes from AncestryMap
        super().__init__(prefix)

        # get the snp dataframe
        super()._get_snp_dataframe()

        # get list of individuals
        super()._get_inds()

        # get the genotype matrix
        self._get_genotype_matrix()

    def _get_genotype_matrix(self):
        """Gets the genotype matrix stored as a
        numpy array
        """
        rlen = max(48, int(np.ceil(self.n * 2 / 8)))

        # read in binary 
        self.Y = np.fromfile(self.geno_path, dtype='uint8')[rlen:]
        self.Y.shape = (self.p, rlen)

        # unpack 
        self.Y = np.unpackbits(self.Y, axis=1)[:, :(2 * self.n)]
        self.Y = 2 * self.Y[:, ::2] + self.Y[:, 1::2]

        # convert to float to allow missing data stored as nan
        self.Y = self.Y.astype(np.float32)
        
        # set missing data to nan
        self.Y[self.Y == 3] = np.nan


class UnpackedAncestryMap(AncestryMap):
    """Class for unpacked ancestry map eigenstrat format. The unpacked format's
    geno is not binary so it doesn't require the same processing before 
    being loaded into a numpy array. 
    
    Arguments
    ---------
    prefix : str
        prefix of  path to data files

    Attributes
    ----------
    prefix : str
        prefix of  path to data files
    Y : np.array
        n x p genotype matrix 
    snp_df : pd.DataFrame
        Dataframe storing snp level meta data
    inds : list
        list of individual ids
    n : int
        number of individuals (samples)
    p : int
        number of snps (features)
    geno_path : str
        path to eigenstrat geno file
    ind_path : str
        path to eigenstrat ind file
    snp_path : str
        path to eigenstart snp file
    """

    def __init__(self, prefix):
        # inhert attributes from AncestryMap
        super().__init__(prefix)

        # get the snp dataframe
        super()._get_snp_dataframe()

        # get list of individuals
        super()._get_inds()

        # get the genotype matrix
        self._get_genotype_matrix()

    def _get_genotype_matrix(self):
        """Gets the genotype matrix stored as a
        numpy array
        """
        # read the geno file
        with open(self.geno_path, "r") as f:
            matstr = f.read().replace("\n", "")
        
        # convert to long array
        ys = np.fromstring(matstr, dtype=np.int8) - 48

        # reshape to p x n matrix
        self.Y = ys.reshape(self.p, self.n).astype(np.float32)

        # replace 9s with nans
        self.Y[self.Y == 9] = np.nan
