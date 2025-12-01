# Cardiovascular Disease EDA & Dashboard

This repository contains a complete exploratory data analysis (EDA) workflow for cardiovascular disease risk prediction using a Kaggle dataset.

## Project Overview

Two main components work together to provide a comprehensive analysis:

1. **`EDA_Cardio.ipynb`** — Jupyter notebook with step-by-step EDA analysis
2. **`streamlit_app.py`** — Interactive Streamlit dashboard for visualizations and modeling

## Files Description

### `EDA_Cardio.ipynb`
A comprehensive Jupyter notebook that covers:
- **Data Loading & Exploration**: Load data, check shape, info, describe statistics
- **Data Quality**: Missing values, duplicates, outlier detection (IQR method)
- **EDA Analysis**:
  - Target distribution and class balance
  - Demographics (sex, age category)
  - BMI, weight, height distributions
  - Lifestyle factors (exercise, smoking, alcohol, diet)
  - Comorbidities (diabetes, arthritis, depression, cancer)
  - Correlation heatmap for numeric features
- **Statistical Tests**: Chi-square tests for categorical variables
- **Machine Learning**:
  - Train-test split with stratification
  - SMOTE resampling for class imbalance
  - Logistic Regression & Random Forest models
  - Feature importance analysis
  - Advanced metrics (Precision, Recall, F1, AUC-ROC)
  - PCA visualization
  - Confusion matrix and ROC curves

### `streamlit_app.py`
An interactive Streamlit dashboard with 7 tabs:
1. **Target Distribution** — Heart disease class distribution (bar & pie charts)
2. **Demographics** — Disease patterns by sex and age category
3. **BMI & Lifestyle** — BMI analysis and lifestyle factor distributions
4. **Comorbidities** — Relationship between comorbidities and heart disease
5. **Correlation** — Heatmap of numeric feature correlations
6. **Modeling** — Train Logistic Regression & Random Forest with SMOTE
7. **PCA & Feature Importance** — PCA visualization and feature importance plots

**Features:**
- Upload or use local dataset
- Interactive filters (sex, age range)
- Train models with configurable test size
- View classification reports and confusion matrices
- Explore feature importances from Random Forest

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get the dataset:
   - Download `CVD_cleaned.csv` from [Kaggle](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset)
   - Place it in the `data/` directory
   
   Or use the Kaggle CLI:
```bash
kaggle datasets download -d alphiree/cardiovascular-diseases-risk-prediction-dataset -p data --unzip
```

## Usage

### Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
The app will open at `http://localhost:8502`

### Run the Jupyter Notebook
Open and run cells in `EDA_Cardio.ipynb` to execute the full EDA pipeline step-by-step.

## Dataset
- **Source**: [Kaggle - Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset)
- **File**: `CVD_cleaned.csv`
- **Target**: `Heart_Disease` (binary: 0 = No, 1 = Yes)