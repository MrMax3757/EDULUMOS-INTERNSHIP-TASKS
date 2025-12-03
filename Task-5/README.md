# Smart Home Valuator

A machine learning project for predicting house prices using ensemble methods combining Random Forest and XGBoost regressors.

## Project Overview

This project implements a comprehensive machine learning pipeline for house price prediction using the Ames Housing dataset. The solution employs feature engineering, preprocessing, hyperparameter tuning, and ensemble methods to achieve optimal prediction accuracy.

## Features

- **Feature Engineering**: Creates meaningful features like total square footage, total bathrooms, house age, and binary indicators (pool, garage, basement)
- **Comprehensive Preprocessing**: Handles missing values, categorical encoding, ordinal mapping, and feature scaling
- **Hyperparameter Tuning**: Uses RandomizedSearchCV for XGBoost and GridSearchCV for Random Forest
- **Ensemble Method**: Combines predictions from both models with equal weighting
- **Model Persistence**: Saves trained models for future use

## Dataset

The project uses the Ames Housing dataset containing:
- **Training set**: 1,460 houses with 81 features (including target variable)
- **Test set**: 1,459 houses with 80 features
- Features include property characteristics such as lot size, building type, quality ratings, year built, square footage, and many more

## Requirements

See `requirements.txt` for the complete list of dependencies. Main packages include:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Task-5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

Open and run the `notebooks/Smart_Home_Valuator.ipynb` Jupyter notebook. The notebook contains all the code organized in cells demonstrating the complete pipeline:

1. **Data Loading**: Loads training and test datasets
2. **Feature Engineering**: Creates new features from existing ones
3. **Preprocessing**: Sets up transformers for numeric, categorical, and ordinal features
4. **Model Training**: Trains Random Forest and XGBoost models
5. **Hyperparameter Tuning**: Optimizes model parameters
6. **Prediction**: Generates ensemble predictions on test set
7. **Model Saving**: Saves trained models and feature importances

### Using the Source Code

The project is organized into modular Python scripts in the `src/` directory:

- **`feature_engineering.py`**: Contains the `FeatureEngineering` transformer class
- **`preprocessing.py`**: Contains preprocessing transformers and utilities (`OrdinalMapper`, `identify_features`, `create_preprocessor`)
- **`utils.py`**: Contains utility functions for data loading, configuration, and evaluation
- **`train.py`**: Contains model training and hyperparameter tuning functions
- **`predict.py`**: Contains prediction, ensemble creation, and submission generation functions

You can import and use these modules in your own scripts or notebooks:

```python
from src import (
    FeatureEngineering,
    load_data,
    prepare_target,
    combine_train_test,
    create_preprocessor,
    train_models,
    predict_ensemble,
    create_submission
)
```

### Expected Outputs

After running the notebook or source code, you'll get:
- `models/best_rf__pipeline.joblib`: Saved Random Forest model pipeline
- `models/best_xgb__pipeline.joblib`: Saved XGBoost model pipeline
- `models/feature_importances.csv`: Top 30 most important features from XGBoost model
- `output/submission_ensemble.csv`: Final predictions for the test set

## Model Performance

- **Random Forest CV RMSE (log-target)**: ~0.142
- **XGBoost Baseline CV RMSE (log-target)**: ~0.139
- **XGBoost Tuned CV RMSE (log-target)**: ~0.125
- **Ensemble**: Weighted average of both models (50/50)

## Project Structure

```
Task-5/
├── data/                          # Dataset directory
│   └── house-prices-dataset/
│       ├── train.csv              # Training data
│       ├── test.csv               # Test data
│       ├── data_description.txt   # Feature descriptions
│       └── sample_submission.csv  # Submission format
├── models/                        # Saved models and analysis
│   ├── best_rf__pipeline.joblib   # Saved RF model
│   ├── best_xgb__pipeline.joblib  # Saved XGBoost model
│   └── feature_importances.csv    # Feature importance analysis
├── notebooks/                     # Jupyter notebooks
│   └── Smart_Home_Valuator.ipynb  # Main notebook
├── output/                        # Output files
│   └── submission_ensemble.csv    # Final predictions
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── feature_engineering.py    # Feature engineering classes
│   ├── preprocessing.py          # Preprocessing transformers
│   ├── train.py                  # Training functions
│   ├── predict.py                # Prediction functions
│   └── utils.py                  # Utility functions
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Key Techniques

1. **Log Transformation**: Applied log(1+x) transformation to the target variable (SalePrice) to handle skewness
2. **Ordinal Encoding**: Quality ratings (Ex, Gd, TA, Fa, Po) mapped to numeric values
3. **One-Hot Encoding**: Categorical variables encoded using OneHotEncoder
4. **Feature Engineering**: Domain knowledge used to create composite features
5. **Cross-Validation**: 5-fold CV for model evaluation
6. **Ensemble Learning**: Combines strengths of tree-based models

## Top Features

Based on feature importance analysis, the most important features include:
- TotalSF (Total Square Footage)
- FireplaceQu (Fireplace Quality)
- ExterQual (Exterior Quality)
- KitchenQual (Kitchen Quality)
- GarageCars (Garage Capacity)
- TotalBath (Total Bathrooms)

## License

This project is part of the EduLumos Internship Tasks.

## Author

Created as part of Task 5 in the EduLumos Internship program.

