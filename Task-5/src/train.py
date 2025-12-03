"""
Training Module

This module contains functions for training models, hyperparameter tuning,
and model evaluation.
"""

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb

from .preprocessing import create_preprocessor
from .utils import cv_rmse, RANDOM_STATE, N_JOBS


def create_model_pipelines(preprocessor, random_state=RANDOM_STATE, n_jobs=N_JOBS):
    """
    Create baseline model pipelines for Random Forest and XGBoost.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    random_state : int, default=RANDOM_STATE
        Random state for reproducibility
    n_jobs : int, default=N_JOBS
        Number of parallel jobs
        
    Returns:
    --------
    rf_pipeline : Pipeline
        Random Forest pipeline
    xgb_pipeline : Pipeline
        XGBoost pipeline
    """
    # Random Forest pipeline
    rf_pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=300, 
            random_state=random_state, 
            n_jobs=n_jobs
        ))
    ])
    
    # XGBoost pipeline
    xgb_pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('model', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=8
        ))
    ])
    
    return rf_pipeline, xgb_pipeline


def tune_xgboost(xgb_pipeline, X, y, n_iter=25, cv=3, 
                 random_state=RANDOM_STATE, verbose=1):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.
    
    Parameters:
    -----------
    xgb_pipeline : Pipeline
        XGBoost pipeline to tune
    X : array-like
        Training features
    y : array-like
        Training target
    n_iter : int, default=25
        Number of parameter settings sampled
    cv : int, default=3
        Number of CV folds
    random_state : int, default=RANDOM_STATE
        Random state for reproducibility
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    best_xgb : Pipeline
        Best XGBoost pipeline after tuning
    best_params : dict
        Best hyperparameters found
    best_score : float
        Best CV score
    """
    xgb_param_dist = {
        'model__n_estimators': [300, 500, 800],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'model__reg_alpha': [0, 0.5, 1],
        'model__reg_lambda': [1, 2, 5]
    }
    
    rs = RandomizedSearchCV(
        xgb_pipeline, 
        xgb_param_dist, 
        n_iter=n_iter, 
        cv=cv, 
        scoring='neg_root_mean_squared_error',
        random_state=random_state, 
        n_jobs=1, 
        verbose=verbose
    )
    
    print('Starting randomized search for XGBoost...\n')
    rs.fit(X, y)
    
    best_xgb = rs.best_estimator_
    best_params = rs.best_params_
    best_score = -rs.best_score_
    
    print(f'Best XGB params: {best_params}')
    print(f'Best XGB CV RMSE: {best_score:.5f}\n')
    
    return best_xgb, best_params, best_score


def tune_random_forest(rf_pipeline, X, y, cv=3):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    
    Parameters:
    -----------
    rf_pipeline : Pipeline
        Random Forest pipeline to tune
    X : array-like
        Training features
    y : array-like
        Training target
    cv : int, default=3
        Number of CV folds
        
    Returns:
    --------
    best_rf : Pipeline
        Best Random Forest pipeline after tuning
    best_params : dict
        Best hyperparameters found
    best_score : float
        Best CV score
    """
    rf_grid = {
        'model__n_estimators': [200, 400],
        'model__max_features': ['sqrt', 0.3]
    }
    
    rg = GridSearchCV(
        rf_pipeline, 
        rf_grid, 
        cv=cv, 
        scoring='neg_root_mean_squared_error'
    )
    
    rg.fit(X, y)
    
    best_rf = rg.best_estimator_
    best_params = rg.best_params_
    best_score = -rg.best_score_
    
    print(f'Best RF params: {best_params}')
    print(f'Best RF CV RMSE: {best_score:.5f}\n')
    
    return best_rf, best_params, best_score


def train_models(rf_pipeline, xgb_pipeline, X, y, tune=True, 
                save_dir='models', save_models=True):
    """
    Train and optionally tune Random Forest and XGBoost models.
    
    Parameters:
    -----------
    rf_pipeline : Pipeline
        Random Forest pipeline
    xgb_pipeline : Pipeline
        XGBoost pipeline
    X : array-like
        Training features
    y : array-like
        Training target
    tune : bool, default=True
        Whether to perform hyperparameter tuning
    save_dir : str, default='models'
        Directory to save models
    save_models : bool, default=True
        Whether to save trained models
        
    Returns:
    --------
    best_rf : Pipeline
        Trained Random Forest model
    best_xgb : Pipeline
        Trained XGBoost model
    """
    if tune:
        print("=" * 60)
        print("TUNING XGBOOST MODEL")
        print("=" * 60)
        best_xgb, xgb_params, xgb_score = tune_xgboost(xgb_pipeline, X, y)
        
        print("=" * 60)
        print("TUNING RANDOM FOREST MODEL")
        print("=" * 60)
        best_rf, rf_params, rf_score = tune_random_forest(rf_pipeline, X, y)
    else:
        print("Training baseline models (no tuning)...")
        best_xgb = xgb_pipeline.fit(X, y)
        best_rf = rf_pipeline.fit(X, y)
    
    # Final training on full dataset
    print("\nTraining final models on full dataset...")
    best_xgb.fit(X, y)
    best_rf.fit(X, y)
    
    if save_models:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(best_xgb, f'{save_dir}/best_xgb__pipeline.joblib')
        joblib.dump(best_rf, f'{save_dir}/best_rf__pipeline.joblib')
        print(f'Saved models to {save_dir}/')
    
    return best_rf, best_xgb

