
# Risk Prediction on Prudential Life Insurance

## Overview
This project focuses on predicting risk in the life insurance sector using machine learning techniques. By leveraging historical data, the aim is to provide precise and efficient risk classification, enabling better underwriting decisions. The project compares models like Decision Tree, Random Forest, and XGBoost to determine the most effective method for risk evaluation.

## Repository Structure
```
.
├── notebooks/
│   └── Project_Final.ipynb
├── data/
│   └── train.csv
├── docs/
│   └── IE_7275_Project.md
├── .gitignore
├── LICENSE
└── README.md
```

## Instructions
1. Clone the repository.
    - git clone https://github.com/Vicky1240/Risk-Prediction.git
3. Run the Jupyter Notebook in the `notebooks/` directory to follow the workflow.

## Dependencies
- Python 3.10
- Jupyter Notebook
- Pandas, NumPy, scikit-learn, etc.

## Problem Setting
The insurance industry has traditionally relied on manual underwriting using statistical equations and life expectancy tables. While functional, these approaches are often slow and lack precision. The advent of data science enables faster and more accurate risk evaluation using machine learning models, which this project explores within the context of life insurance.

## Objective
To enhance risk evaluation in life insurance by:
- Categorizing risks based on historical data.
- Identifying the most effective predictive model.
- Balancing model sophistication with interpretability to comply with industry and regulatory requirements.

## Data Sources
- **Dataset**: Prudential Life Insurance Assessment
- **Source**: Kaggle
- **Size**: 59,381 rows and 128 columns
- **Types of Data**:
    - **Nominal:** Product_Info_1, Product_Info_2
    - **Continuous:** Wt, BMI, Employment_Info_6
    - **Binary:** Medical_Keyword_40 to Medical_Keyword_48
    - **Target Variable:** Response (risk levels from 1 to 8)

## Data Exploration
- Comprehensive analysis of dataset structure and feature types.
- Identified key features such as:
    - **Continuous Variables:** Ht, Wt, BMI
    - **Categorical Variables:** Product_Info_1 to Product_Info_7
    - **Binary Variables:** Medical_Keyword_40 to Medical_Keyword_48
- Found skewed distribution in the Response variable, with Class 8 being the most frequent.

## Data Pre-processing
1. **Handling Missing Values:**
    - Removed features with >50% missing values.
    - Used median imputation for numerical features.
2. **Outlier Treatment:**
    - Detected and capped outliers in Ht, Wt, and BMI using the IQR method.
3. **Encoding:**
    - Applied one-hot encoding to categorical variables (e.g Product_Info_2).
4. **Dimensionality Reduction:**
    - Analyzed correlations and implemented PCA to retain 95% variance.

## Modeling Approach
1. **Decision Tree (DT):**
    - Simple and interpretable, effective for quick insights.
2. **Random Forest (RF):**
    - Reduced overfitting with ensemble techniques, offering improved accuracy.
3. **XGBoost:**
    - Advanced gradient boosting model with strong regularization to prevent overfitting.
4. **Other Models:**
    - Logistic Regression, Multiple Linear Regression, ANN, and Gradient Boosting were also evaluated.

## Performance Evaluation
- **Metrics Used:** ROC AUC, Accuracy, F-Score, Precision, Recall
- **Key Results:**
    1. **Random Forest:**
        - **ROC AUC:** 0.819
        - **Accuracy:** 64.8%
        - **F-Score:** 0.642
    2. **XGBoost:**
        - **ROC AUC:** 0.840
        - **Accuracy:** 66.4%
        - **F-Score:** 0.623
    3. **Logistic Regression:**
        - **ROC AUC:** 0.623
        - **Accuracy:** 43.9%
        - **F-Score:** 0.440

## Conclusion and Future Work

### Conclusion:

- XGBoost performed best in terms of ROC AUC and precision, making it the most suitable model for the task.
- Random Forest offered a balanced performance across all metrics.
- Logistic Regression underperformed, likely due to the non-linear nature of the data.

### Future Work:

- Enhance feature engineering to capture more nuanced relationships.
- Address class imbalance using methods like SMOTE or class weighting.
- Experiment with ensemble methods and hyperparameter tuning.
- Explore real-time risk prediction capabilities.

## Authors
- Vikramadithya Pabba
- Dheeraj Goli
