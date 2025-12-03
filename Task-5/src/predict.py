"""
Prediction Module

This module contains functions for making predictions, creating ensembles,
and generating submission files.
"""

import numpy as np
import pandas as pd
import joblib
from .preprocessing import ORDINAL_COLS


def predict_ensemble(best_xgb, best_rf, X_test, weights=(0.5, 0.5)):
    """
    Create ensemble predictions from XGBoost and Random Forest models.
    
    Parameters:
    -----------
    best_xgb : Pipeline
        Trained XGBoost model
    best_rf : Pipeline
        Trained Random Forest model
    X_test : array-like
        Test features
    weights : tuple, default=(0.5, 0.5)
        Weights for ensemble (xgb_weight, rf_weight)
        
    Returns:
    --------
    preds_ensemble : np.ndarray
        Ensemble predictions (in original scale, not log)
    preds_xgb_log : np.ndarray
        XGBoost predictions (log scale)
    preds_rf_log : np.ndarray
        Random Forest predictions (log scale)
    """
    print('Creating Ensemble Predictions...')
    
    # Predict in log space
    preds_xgb_log = best_xgb.predict(X_test)
    preds_rf_log = best_rf.predict(X_test)
    
    # Weighted ensemble in log space
    preds_ensemble_log = weights[0] * preds_xgb_log + weights[1] * preds_rf_log
    
    # Transform back to original scale
    preds_ensemble = np.expm1(preds_ensemble_log)
    
    return preds_ensemble, preds_xgb_log, preds_rf_log


def create_submission(test_ids, predictions, output_path='output/submission_ensemble.csv'):
    """
    Create submission file in the required format.
    
    Parameters:
    -----------
    test_ids : pd.Series or array-like
        Test sample IDs
    predictions : array-like
        Predicted values
    output_path : str, default='output/submission_ensemble.csv'
        Path to save submission file
        
    Returns:
    --------
    submission : pd.DataFrame
        Submission dataframe
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission written to '{output_path}'")
    
    return submission


def extract_feature_importances(best_xgb, numeric_feat, cat_feat, 
                                ordinal_cols=ORDINAL_COLS, 
                                output_path='models/feature_importances.csv',
                                top_n=30):
    """
    Extract and save feature importances from XGBoost model.
    
    Parameters:
    -----------
    best_xgb : Pipeline
        Trained XGBoost model
    numeric_feat : list
        List of numeric feature names
    cat_feat : list
        List of categorical feature names
    ordinal_cols : list
        List of ordinal column names
    output_path : str, default='models/feature_importances.csv'
        Path to save feature importances
    top_n : int, default=30
        Number of top features to save
        
    Returns:
    --------
    feature_importances : pd.Series
        Top feature importances
    """
    import os
    
    pre = best_xgb.named_steps["pre"]
    
    # 1. Ordinal block
    ordinal_feature_names = ordinal_cols
    
    # 2. Numeric block
    numeric_feature_names = [c for c in numeric_feat if c not in ordinal_cols]
    
    # 3. Categorical block (OneHotEncoder)
    cat_pipe = pre.named_transformers_["cat"]
    onehot = cat_pipe.named_steps["onehot"]
    
    if hasattr(onehot, "get_feature_names_out"):
        categorical_feature_names = list(onehot.get_feature_names_out(cat_feat))
    else:
        categorical_feature_names = []
    
    # Combine in EXACT ColumnTransformer order
    feature_names = (
        ordinal_feature_names
        + numeric_feature_names
        + categorical_feature_names
    )
    
    # Extract feature importances
    booster = best_xgb.named_steps["model"]
    
    if hasattr(booster, "feature_importances_"):
        importances = booster.feature_importances_
        
        if len(importances) == len(feature_names):
            fi = pd.Series(importances, index=feature_names)
            fi = fi.sort_values(ascending=False).head(top_n)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fi.to_csv(output_path)
            print(f"Saved top {top_n} feature importances to {output_path}")
            
            return fi
        else:
            print(f"Mismatch: {len(importances)} importances vs {len(feature_names)} features")
            return None
    else:
        print("Model does not provide feature_importances_.")
        return None


def load_models(models_dir='models'):
    """
    Load trained models from disk.
    
    Parameters:
    -----------
    models_dir : str, default='models'
        Directory containing saved models
        
    Returns:
    --------
    best_rf : Pipeline
        Loaded Random Forest model
    best_xgb : Pipeline
        Loaded XGBoost model
    """
    best_xgb = joblib.load(f'{models_dir}/best_xgb__pipeline.joblib')
    best_rf = joblib.load(f'{models_dir}/best_rf__pipeline.joblib')
    
    print(f"Loaded models from {models_dir}/")
    
    return best_rf, best_xgb

