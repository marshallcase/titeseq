import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class ProteinOneHotEncoder(OneHotEncoder):
    '''
    ProteinOneHotEncoder: custom OneHotEncoder from scikit-learn that handles protein sequences
    '''
    def __init__(self, *args, **kwargs):
        # Initialize the superclass (OneHotEncoder) with any arguments passed in
        super().__init__(*args, **kwargs)
        self.all_AA = list(set('ACDEFGHIKLMNPQRSTVWY'))
        self.AA_to_index = {aa: i for i, aa in enumerate(self.all_AA)}
        self.index_to_AA = {v: k for k, v in self.AA_to_index.items()}

        
    def fit(self,X,y=None):
        '''
        fit: given a list of protein sequences (strings), fit the ProteinOneHotEncoder
        
        Parameters
        ----------
        X: array-like of protein sequences
        y: target value. not processed. only here for pipeline integration
        
        Outputs
        -------
        None
        
        '''
        # transform the sequence to an integer array
        X = [np.array([self.AA_to_index[aa] for aa in sequence]) for sequence in X]
        super().fit(X)
        
    def transform(self,X,y=None):
        '''
        transform: given a list of protein sequences (strings) and a fit ProteinOneHotEncoder, transform sequences into a one hot encoded version
        
        Parameters
        ----------
        X: array-like of protein sequences
        y: target value. not processed. only here for pipeline integration
        
        Outputs
        -------
        X_modified: one-hot encoded protein sequences
        
        '''
        # transform the sequence to an integer array
        X = [np.array([self.AA_to_index[aa] for aa in sequence]) for sequence in X]
        # fit the OneHotEncoder using the fit_transform method of the superclass
        X_modified = super().transform(X)
        return X_modified
    
    def fit_transform(self, X, y=None):
        '''
        fit_transform: given a list of protein sequences (strings), return a one-hot encoded version
        
        Parameters
        ----------
        X: array-like of protein sequences
        y: target value. not processed. only here for pipeline integration
        
        Outputs
        -------
        X_modified: one-hot encoded protein sequences
        
        '''
        # fit onehotencoder using the fit method
        self.fit(X)
        # transform the sequences using the transform method
        X_modified = self.transform(X)
        return X_modified
    
    def inverse_transform(self,X_modified):
        '''
        inverse_transform: given a one-hot encoded protein sequence, return the protein sequence
        
        Parameters
        ----------
        X_modified: one hot encoded protein sequence
        
        Outputs
        -------
        X: protein sequence
        
        '''
        # get the integer array from the inverse_transform method of the superclass
        X = super().inverse_transform(X_modified)
        # get the sequence from the integer array
        X = [[self.index_to_AA[i] for i in j] for j in X]
        # turn the list of chars back into a string
        X = ["".join(i) for i in X]
        return X
    
    @staticmethod
    def get_varying_sites(X,threshold=1,min_count=1):
        '''
        get_varying_sites: given a list of aligned protein sequences, find the positions where the protein has been modified in the library mutations and return it as a list of indices
        
        Parameters
        ----------
        X: protein sequence
        threshold: minimum number of mutants needed to include a site as varying, default 1
        min_count: number of appearances necessary to include
        
        Outputs
        -------
        varying_sites: list of indices
        
        '''
        # get positions that have more than (threshold) unique mutations at a given position
        differing_positions_threshold = [i for i in range(len(X[0])) if len(set([s[i] for s in X])) >= threshold]
        
        #invert min_count
        min_count = len(X) - min_count
        
        #get positions that have more than (num_count) appearances
        differing_positions_min = [i for i in range(len(X[0])) if len([s[i] for s in X if s[i] == X[0][i]]) < min_count]
        return np.intersect1d(differing_positions_threshold,differing_positions_min)
    
    @staticmethod
    def transform_varying_sites(X,threshold=1,min_count=1):
        '''
        get_varying_sites: given a list of aligned protein sequences, find the positions where the protein has been modified in the library and return the protein sequences that vary only
        
        Parameters
        ----------
        X: protein sequence
        
        Outputs
        -------
        X_modified: protein sequence with only mutated positions (for subsequent one hot encoding)
        
        '''
        #get varied sites
        varied_sites = ProteinOneHotEncoder.get_varying_sites(X,threshold,min_count)
        
        X_modified = [getCharsFromString(i,varied_sites) for i in X]
        
        return X_modified
        
        
def getCharsFromString(string,indices):
    '''
    getCharsFromString: given a sequence, return characters in string according to indices
    Parameters
    ----------
    string: string
    indices: list of indices
    Outputs
    -------
    output: characters in string according to indices
    '''
    
    try:
        output = [string[i] for i in indices]
    except IndexError: 
        print ('string index out of range')
        output = []
    return ''.join(output)