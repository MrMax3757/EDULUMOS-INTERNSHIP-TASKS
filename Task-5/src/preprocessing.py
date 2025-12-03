"""
Preprocessing Module

This module contains preprocessing transformers and utilities for feature
identification and transformation pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer for mapping ordinal categorical features to numeric values.
    
    Maps quality ratings (Ex, Gd, TA, Fa, Po) to numeric values.
    """
    
    def __init__(self, cols, mapping):
        """
        Initialize the OrdinalMapper.
        
        Parameters:
        -----------
        cols : list
            List of column names to apply ordinal mapping to
        mapping : dict
            Dictionary mapping categorical values to numeric values
        """
        self.cols = cols
        self.mapping = mapping
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no operation needed).
        
        Parameters:
        -----------
        X : array-like
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
        Transform input data by mapping ordinal categories to numbers.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        X : pd.DataFrame
            Dataframe with ordinal columns mapped to numeric values
        """
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = X[c].fillna('Missing').map(self.mapping).astype(float)
        return X


# Ordinal mapping configuration
ORDINAL_MAPPINGS = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Missing': 0}

# Ordinal columns to apply mapping to
ORDINAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
    'KitchenQual', 'GarageQual', 'GarageCond'
]

# Numeric columns to treat as categorical
NUMERIC_AS_CAT = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']


def identify_features(df, ordinal_cols=ORDINAL_COLS, numeric_as_cat=NUMERIC_AS_CAT):
    """
    Identify numeric and categorical features from the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    ordinal_cols : list
        List of ordinal column names (will be removed from numeric features)
    numeric_as_cat : list
        List of numeric columns to treat as categorical
        
    Returns:
    --------
    numeric_feat : list
        List of numeric feature column names
    cat_feat : list
        List of categorical feature column names
    """
    # Start with all numeric columns
    numeric_feat = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID column if present
    if 'Id' in numeric_feat:
        numeric_feat.remove('Id')
    
    # Remove numeric columns that should be treated as categorical
    for c in numeric_as_cat:
        if c in numeric_feat:
            numeric_feat.remove(c)
    
    # Get categorical columns
    cat_feat = df.select_dtypes(include=['object']).columns.tolist()
    
    # Add numeric-as-categorical columns
    cat_feat = cat_feat + [c for c in numeric_as_cat if c in df.columns]
    
    return numeric_feat, cat_feat


def create_preprocessor(numeric_feat, cat_feat, ordinal_cols=ORDINAL_COLS, 
                       ordinal_mappings=ORDINAL_MAPPINGS):
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    Parameters:
    -----------
    numeric_feat : list
        List of numeric feature column names
    cat_feat : list
        List of categorical feature column names
    ordinal_cols : list
        List of ordinal column names
    ordinal_mappings : dict
        Dictionary for ordinal mapping
        
    Returns:
    --------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    """
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('ordinal', Pipeline(steps=[
            ('mapper', OrdinalMapper(ordinal_cols, ordinal_mappings)),
            ('impute_num', SimpleImputer(strategy='median'))
        ]), ordinal_cols),
        ('num', numeric_transformer, [c for c in numeric_feat if c not in ordinal_cols]),
        ('cat', categorical_transformer, cat_feat)
    ], remainder='drop', sparse_threshold=0)
    
    return preprocessor

