# Weather Prediction Project - Australia Weather Dataset

This project performs comprehensive data preprocessing and machine learning on the Australian Weather Dataset to predict whether it will rain tomorrow.

## Overview

The project implements a complete machine learning pipeline including:
- Data cleaning and preprocessing
- Feature engineering
- Outlier detection and removal
- Class balancing using SMOTE
- Dimensionality reduction with PCA
- Neural network classification

## Dataset

The project uses the `weatherAUS.csv` dataset containing Australian weather observations with features like:
- Temperature (Min/Max, 9am/3pm)
- Rainfall, Evaporation, Sunshine
- Wind direction and speed
- Humidity and pressure readings
- Cloud cover
- Target variable: `RainTomorrow` (Yes/No)

## Key Features

### Data Preprocessing
- **Missing Value Handling**: Replaces zeros with NaN and imputes missing values
  - Numerical columns: Filled with median values
  - Categorical columns: Filled with mode values
- **Encoding**: 
  - Binary variables (`RainToday`, `RainTomorrow`): Yes/No → 1/0
  - Categorical variables (`WindDir9am`, `WindDir3pm`, `WindGustDir`, `Location`): Label encoded
  - Date formatting: Removes slashes and converts to float

### Data Balancing
- Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the target classes
- Ensures equal representation of rainy and non-rainy days

### Outlier Detection
- Implements IQR-based outlier removal for the `Rainfall` feature
- Uses 10th and 90th percentiles with 3×IQR bounds

### Feature Engineering
- **Correlation Analysis**: Visualizes feature correlations with heatmap
- **PCA (Principal Component Analysis)**: Reduces dimensionality to 4 components
- **Standardization**: Applies StandardScaler before PCA

### Machine Learning
- **Model**: Multi-Layer Perceptron (Neural Network)
- **Architecture**: 3 hidden layers with 16 neurons each
- **Activation**: ReLU
- **Solver**: Adam optimizer
- **Train/Test Split**: 70/30

## Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
```

## Installation

1. Install required packages:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
```

2. Ensure you have the `weatherAUS.csv` dataset in your working directory

## Usage

1. Run the notebook cells sequentially in [Preprocessing WeatherDS-checkpoint11.ipynb](Preprocessing WeatherDS-checkpoint11.ipynb)

2. The pipeline will:
   - Load and explore the dataset
   - Clean and preprocess the data
   - Save cleaned data to `cleaned_weatherAUS_zero_replaced.csv`
   - Apply SMOTE balancing
   - Perform PCA transformation
   - Train the neural network model
   - Output final accuracy score

## Output Files

- `cleaned_weatherAUS_zero_replaced.csv`: Preprocessed dataset with missing values handled
- `cleaned_weatherAUS_final.csv`: Final processed dataset

## Visualizations

The project includes several visualizations:
- Class distribution bar plots (before/after SMOTE)
- Correlation heatmap
- Boxplots for outlier detection
- Feature correlation with target variable

## Model Performance

The final neural network model achieves classification accuracy on the test set, with detailed performance metrics available through the classification report.

## Project Structure

```
├── Preprocessing WeatherDS-checkpoint11.ipynb    # Main notebook
├── weatherAUS.csv                               # Input dataset
├── cleaned_weatherAUS_zero_replaced.csv         # Preprocessed data
├── cleaned_weatherAUS_final.csv                 # Final processed data
└── README.md                                    # This file
```

## Future Improvements

- Experiment with different neural network architectures
- Try other classification algorithms (Random Forest, SVM, etc.)
- Implement cross-validation for more robust evaluation
- Add hyperparameter tuning
- Include more sophisticated feature selection techniques
