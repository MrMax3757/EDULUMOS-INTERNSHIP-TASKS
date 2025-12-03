"""
Feature Engineering Module

This module contains the FeatureEngineering transformer class that creates
new meaningful features from existing ones.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating new features from existing data.
    
    Creates features such as:
    - TotalSF: Total square footage
    - TotalBath: Total number of bathrooms
    - HasPool: Binary indicator for pool presence
    - HasGarage: Binary indicator for garage presence
    - HasBsmt: Binary indicator for basement presence
    - HouseAge: Age of the house
    - RemodAge: Age since last remodeling
    """
    
    def __init__(self):
        """Initialize the FeatureEngineering transformer."""
        pass
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no operation needed).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like, default=None
            Target values (unused)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Transform input data by creating new features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe with house features
            
        Returns:
        --------
        X : pd.DataFrame
            Dataframe with new features added
        """
        X = X.copy()
        
        # Total square feet
        X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'].fillna(0) + X['2ndFlrSF'].fillna(0)
        
        # Total bathrooms (count half bathrooms as 0.5)
        X['TotalBath'] = (
            X['FullBath'].fillna(0) + 0.5 * X['HalfBath'].fillna(0) + 
            X['BsmtFullBath'].fillna(0) + 0.5 * X['BsmtHalfBath'].fillna(0)
        )
        
        # Has pool, Has garage, Has basement etc
        X['HasPool'] = (X['PoolArea'].fillna(0) > 0).astype(int)
        X['HasGarage'] = (~X['GarageType'].isna()).astype(int)
        X['HasBsmt'] = (~X['BsmtQual'].isna()).astype(int)
        
        # Age of House
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
        
        return X

