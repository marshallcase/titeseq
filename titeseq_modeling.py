# -*- coding: utf-8 -*-
"""
Created on Tue Dec  10 15:10:00 2022

@author: marsh
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import KBinsDiscretizer

class ModelTester:
    def __init__(self, X, y_binary, y_continuous):
        '''
        initialize ModelTester instance
        
        Parameters
        ----------
        X: np.array of shape (number of samples, number of features)
        y_binary: np.array of shape (number of samples, ) populated with binary labels
        y_continuous: np.array of shape (number of samples, )populated with scaled continuous labels
        
        Outputs
        -------
        None
        
        '''
        
        self.X = X
        self.y_binary = y_binary
        self.y_continuous = y_continuous
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train_binary, self.y_test_binary = train_test_split(self.X, self.y_binary, test_size=0.2,random_state=42)
        _, _, self.y_train_continuous, self.y_test_continuous = train_test_split(self.X, self.y_continuous, test_size=0.2,random_state=42)
        
    def test_models(self, models):
        '''
        test_models: given a list of models (implemented with a model.fit(X,y) method and a model.score(X,y) method), test them on the dataset
        
        Parameters
        ----------
        models: list of models (i.e. [LinearRegression(), linear_model.Ridge(alpha=0.5),DecisionTreeRegressor()])
        
        Outputs
        -------
        None
        
        '''
        
        # Test each model
        for model in models: 
            #TODO: classification and regression need different types models
            
            for i,modeltype in enumerate(['Binary','Continuous']):
                if i == 0: #binary mode
                    #fit model
                    model.fit(self.X_train, self.y_train_binary)
                    #test model
                    score = model.score(self.X_test, self.y_test_binary)
                elif i == 1: #continuous mode
                    #fit model
                    model.fit(self.X_train, self.y_train_continuous)
                    #test model
                    score = model.score(self.X_test, self.y_test_continuous)
            
                # Print the score
                print(f"{modeltype} mode : {model.__class__.__name__}: {score:.2f}")
                
    def test_models_paired(self,classifier_models,regressor_models):
        '''
        test_models_paired: given two lists of classification and regression models, train them on binary and continuous datasets. each model
        needs to implement the following methods: fit(X,y), score(X,y), and __name__()
        
        Parameters:
        -----------
        classifier_models: list of models (i.e. [ linear_model.RidgeClassifier(alpha=0.5),DecisionTreeClassifier(),RandomForestClassifier()])
        regressor_models: list of models (i.e. [linear_model.Ridge(alpha=0.5),DecisionTreeRegressor(),RandomForestRegressor()])
        
        Outputs:
        --------
        none
        
        '''
        # Test each model
        for classifier,regressor in zip(classifier_models,regressor_models): 
            ##CLASSIFICATION
            classifier.fit(self.X_train, self.y_train_binary)
            score = model.score(self.X_test, self.y_test_binary)
            print(f"{type(classifier).__name__} mode : {model.__class__.__name__}: {score:.2f}")
            
            ##REGRESSION
            regressor.fit(self.X_train, self.y_train_continuous)
            score = model.score(self.X_test, self.y_test_continuous)
            print(f"{type(regressor).__name__} mode : {model.__class__.__name__}: {score:.2f}")


            
def main():
    X, y = load_diabetes(return_X_y=True)
    y = StandardScaler().fit_transform(y.reshape(-1,1))
    y_continuous = y
    y_binary = KBinsDiscretizer(n_bins=2,encode='onehot-dense').fit_transform(y_continuous)[:,0]
    print(f'continuous labels are {y_continuous[:5]}')
    print(f'binary labels are {y_binary[:5]}')
    tester = ModelTester(X,y_binary,y_continuous)
    models = [KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
    tester.test_models(models)
    

if __name__ == "__main__":
    main()
