# Smart Study Score Predictor

A machine learning project that predicts student average performance scores based on various demographic and educational factors. This project implements multiple regression models and selects the best performing one for student performance prediction.

## 📋 Project Overview

The Smart Study Score Predictor analyzes student performance data to predict average test scores based on factors such as:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course Completion

The project compares multiple machine learning algorithms and selects the best model for accurate predictions.

## 🎯 Features

- **Data Analysis**: Comprehensive exploratory data analysis with correlation heatmaps
- **Multiple Models**: Comparison of Linear Regression, Ridge Regression, Lasso Regression, and Random Forest
- **Model Selection**: Automatic selection of the best performing model based on R² score
- **Performance Metrics**: Evaluation using MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² score
- **Model Persistence**: Saved trained model for future predictions

## 📊 Dataset

The project uses the `StudentsPerformance.csv` dataset located in the `data/` directory. The dataset contains:
- **1000 student records**
- **8 features**: gender, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score, writing score
- **Target Variable**: Average score (calculated from math, reading, and writing scores)

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Task-1
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning library
- `joblib` - Model serialization

## 📁 Project Structure

```
Task-1/
│
├── data/
│   └── StudentsPerformance.csv          # Student performance dataset
│
├── model/
│   └── Linear Regression_SmartStudyPredictor.pkl  # Trained model
│
├── SmartStudyScorePredictor.ipynb       # Main Jupyter notebook
├── requirements.txt                      # Python dependencies
└── README.md                            # Project documentation
```

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook SmartStudyScorePredictor.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the data
   - Encode categorical variables
   - Visualize data correlations
   - Train multiple models
   - Evaluate model performance
   - Save the best model

### Using the Trained Model

The best performing model is saved as `Linear Regression_SmartStudyPredictor.pkl` in the `model/` directory. You can load and use it for predictions:

```python
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model
model = joblib.load('model/Linear Regression_SmartStudyPredictor.pkl')

# Prepare your data (ensure it's preprocessed the same way as training data)
# Make predictions
predictions = model.predict(preprocessed_data)
```

## 📈 Model Performance

The project compares four different regression models:

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** | 1.49e-14 | 1.79e-14 | **1.000000** |
| Ridge Regression | 1.72e-02 | 2.20e-02 | 0.999998 |
| Lasso Regression | 8.24e-02 | 1.05e-01 | 0.999949 |
| Random Forest | 1.04e-01 | 8.76e-01 | 0.996420 |

**Best Model**: Linear Regression (selected automatically)

## 🔍 Data Preprocessing

1. **Label Encoding**: Categorical variables (gender, race/ethnicity, parental education, lunch, test preparation) are encoded using LabelEncoder
2. **Feature Engineering**: Average score is calculated from math, reading, and writing scores
3. **Feature Scaling**: Features are standardized using StandardScaler
4. **Train-Test Split**: Data is split into 80% training and 20% testing sets

## 📝 Key Steps

1. **Data Loading**: Load the StudentsPerformance.csv dataset
2. **Data Exploration**: Analyze data structure and check for missing values
3. **Feature Encoding**: Encode categorical variables to numerical values
4. **Visualization**: Create correlation heatmap to understand feature relationships
5. **Target Creation**: Calculate average score from math, reading, and writing scores
6. **Model Training**: Train multiple regression models
7. **Model Evaluation**: Compare models using MAE, RMSE, and R² metrics
8. **Model Selection**: Select and save the best performing model
9. **Visualization**: Plot actual vs predicted scores

## 🤝 Contributing

This is a project for EduLumos Internship Task-1. Contributions and improvements are welcome!

## 📄 License

This project is part of the EduLumos Internship program.

## 👤 Author

EduLumos Internship - Task 1

---

**Note**: The perfect R² score (1.000000) achieved by Linear Regression suggests that the features may be highly correlated or the dataset may have specific characteristics that allow for perfect linear separation. In real-world scenarios, such perfect scores are rare and may indicate overfitting or data leakage.

