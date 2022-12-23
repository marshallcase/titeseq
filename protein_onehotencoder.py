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