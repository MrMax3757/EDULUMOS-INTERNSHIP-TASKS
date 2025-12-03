"""
Utility Functions Module

This module contains utility functions for data loading, configuration,
and evaluation metrics.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score


# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
N_JOBS = -1


def load_data(train_path, test_path):
    """
    Load training and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
        
    Returns:
    --------
    train : pd.DataFrame
        Training dataframe
    test : pd.DataFrame
        Test dataframe
    test_ids : pd.Series
        Test IDs for submission
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_ids = test['Id']
    
    return train, test, test_ids


def prepare_target(train_df, target_col='SalePrice', log_transform=True):
    """
    Extract and optionally transform the target variable.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe with target column
    target_col : str, default='SalePrice'
        Name of the target column
    log_transform : bool, default=True
        Whether to apply log(1+x) transformation
        
    Returns:
    --------
    y : np.ndarray
        Target values (log-transformed if log_transform=True)
    train_df : pd.DataFrame
        Training dataframe without target column
    """
    train_df = train_df.copy()
    
    if log_transform:
        y = np.log1p(train_df[target_col])
    else:
        y = train_df[target_col].values
    
    train_df = train_df.drop(columns=[target_col])
    
    return y, train_df


def combine_train_test(train_df, test_df):
    """
    Combine training and test dataframes for consistent preprocessing.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
        
    Returns:
    --------
    all_data : pd.DataFrame
        Combined dataframe
    """
    all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    return all_data


def split_train_test(all_data, n_train):
    """
    Split combined data back into training and test sets.
    
    Parameters:
    -----------
    all_data : pd.DataFrame
        Combined dataframe
    n_train : int
        Number of training samples
        
    Returns:
    --------
    X : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    """
    X = all_data.iloc[:n_train, :].copy()
    X_test = all_data.iloc[n_train:, :].copy().reset_index(drop=True)
    
    return X, X_test


def cv_rmse(model, X, y, folds=5, random_state=RANDOM_STATE, n_jobs=N_JOBS):
    """
    Calculate cross-validation RMSE score.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    folds : int, default=5
        Number of CV folds
    random_state : int, default=RANDOM_STATE
        Random state for reproducibility
    n_jobs : int, default=N_JOBS
        Number of parallel jobs
        
    Returns:
    --------
    rmse : float
        Mean RMSE across CV folds
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        model, X, y, 
        scoring='neg_root_mean_squared_error', 
        cv=kf, 
        n_jobs=n_jobs
    )
    return -scores.mean()

