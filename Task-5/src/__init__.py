"""
Smart Home Valuator Package

A machine learning package for predicting house prices using ensemble methods.
"""

__version__ = "1.0.0"

from .feature_engineering import FeatureEngineering
from .preprocessing import (
    OrdinalMapper,
    identify_features,
    create_preprocessor,
    ORDINAL_MAPPINGS,
    ORDINAL_COLS,
    NUMERIC_AS_CAT
)
from .utils import (
    load_data,
    prepare_target,
    combine_train_test,
    split_train_test,
    cv_rmse,
    RANDOM_STATE,
    N_JOBS
)
from .train import (
    create_model_pipelines,
    tune_xgboost,
    tune_random_forest,
    train_models
)
from .predict import (
    predict_ensemble,
    create_submission,
    extract_feature_importances,
    load_models
)

__all__ = [
    'FeatureEngineering',
    'OrdinalMapper',
    'identify_features',
    'create_preprocessor',
    'ORDINAL_MAPPINGS',
    'ORDINAL_COLS',
    'NUMERIC_AS_CAT',
    'load_data',
    'prepare_target',
    'combine_train_test',
    'split_train_test',
    'cv_rmse',
    'RANDOM_STATE',
    'N_JOBS',
    'create_model_pipelines',
    'tune_xgboost',
    'tune_random_forest',
    'train_models',
    'predict_ensemble',
    'create_submission',
    'extract_feature_importances',
    'load_models',
]

