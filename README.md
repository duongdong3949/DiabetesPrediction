# Diabetes Prediction using Machine Learning

## Team Members (Group 14)
- Võ Văn Khả - 2110264
- Trần Mậu Thật - 2112342
- Dương Thuận Đông - 2210762

## Project Overview

This project focuses on predicting diabetes in patients using machine learning techniques. Diabetes is a group of metabolic disorders characterized by prolonged high blood sugar levels. If left untreated, diabetes can lead to serious complications including cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and eye damage.

## Objective

The main goal of this project is to predict whether a patient has diabetes or not based on various diagnostic measurements included in the dataset.

## Dataset Description

The diabetes prediction dataset contains medical and demographic information from patients, along with their diabetes status (positive or negative). The dataset includes features such as age, gender, body mass index (BMI), hypertension status, heart disease, smoking history, HbA1c levels, and blood glucose levels.

### Dataset Features

- **age**: Age factor (0-80 years) - important because diabetes is more commonly diagnosed in older adults
- **gender**: Gender categories - Male, Female, Other (Other category was removed during preprocessing)
- **hypertension**: Hypertension status (0 or 1) - 0 indicates no hypertension, 1 indicates hypertension
- **heart_disease**: Heart disease status (0 or 1) - 0 indicates no heart disease, 1 indicates heart disease
- **smoking_history**: Smoking history categories:
  - `never`: Never smoked
  - `former`: Former smoker
  - `No Info`: No information available
  - `current`: Current smoker
  - `not current`: Not currently smoking
  - `ever`: Ever smoked
- **bmi**: Body Mass Index (10.16-71.55) - higher BMI is associated with higher diabetes risk
  - Under 18.5: Underweight
  - 18.5-24.9: Normal
  - 25-29.9: Overweight
  - 30+: Obese
- **HbA1c_level**: Average blood sugar level over 2-3 months (3.5-9.0%)
  - Levels above 6.5% are considered indicators of diabetes
- **blood_glucose_level**: Blood glucose level at a specific time (80-300 mg/dL)
  - High blood glucose is a primary sign of diabetes
- **diabetes**: Target variable (0 or 1) - 1 indicates presence of diabetes, 0 indicates absence

### Dataset Statistics
- **Total samples**: 100,000 (after removing duplicates: 96,146)
- **Features**: 9 columns
- **Missing values**: None
- **Class distribution**: 
  - No diabetes: 87,664 samples (91.2%)
  - Diabetes: 8,482 samples (8.8%)

## Methodology

### Data Preprocessing
1. **Data Cleaning**:
   - Removed duplicate entries (3,854 duplicates found)
   - Removed 'Other' gender category (18 samples)
   - No missing values detected

2. **Feature Engineering**:
   - Recategorized smoking history into 3 groups:
     - `non-smoker`: never, No Info
     - `current`: current
     - `past-smoker`: ever, former, not current
   - Applied one-hot encoding for categorical variables

3. **Data Visualization**:
   - Distribution analysis for all features
   - Correlation analysis
   - Box plots for feature relationships
   - Pair plots for quantitative variables

### Machine Learning Models

Three different machine learning algorithms were implemented and compared:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

### Model Training Strategy

- **Data Split**: 80% training, 20% testing
- **Cross-validation**: 5-fold cross-validation
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Class Imbalance Handling**: 
  - SMOTE oversampling (sampling_strategy=0.1)
  - RandomUnderSampler (sampling_strategy=0.5)
  - Class weight balancing

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Logistic Regression | 87.71% | 40.99% | 88.54% | 56.04% | 95.86% | 40.99s |
| Decision Tree | 88.70% | 43.36% | 90.53% | 58.63% | 97.18% | 53.08s |
| Random Forest | 95.20% | 70.30% | 79.19% | 74.48% | 97.33% | 1340.33s |

### Best Model: Random Forest

The Random Forest model achieved the best overall performance with:
- **Highest Accuracy**: 95.20%
- **Best Precision**: 70.30%
- **Good Recall**: 79.19%
- **Best F1 Score**: 74.48%
- **Highest ROC-AUC**: 97.33%

### Key Findings

1. **Feature Importance**: Quantitative features (HbA1c_level, blood_glucose_level, age, BMI) have stronger influence on diabetes prediction than categorical features.

2. **Critical Thresholds**:
   - HbA1c levels ≥ 7%: All samples had diabetes
   - Blood glucose levels ≥ 205 mg/dL: All samples had diabetes
   - HbA1c levels ≥ 5.5%: Higher diabetes risk
   - Blood glucose levels ≥ 125 mg/dL: Higher diabetes risk

3. **Demographic Patterns**:
   - Males have slightly higher diabetes rates than females
   - Hypertension and heart disease are associated with higher diabetes risk
   - Smoking history shows unclear correlation with diabetes

## Files Structure

```
DiabetesPrediction-main/
├── DiabetePrediction.ipynb          # Main Jupyter notebook with complete analysis
├── diabetes_prediction_dataset.csv  # Dataset file
└── README.md                        # This documentation file
```

## Requirements

### Python Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Usage

1. **Clone or download** the project files
2. **Install required dependencies** using pip
3. **Open** `DiabetePrediction.ipynb` in Jupyter Notebook
4. **Run all cells** to reproduce the analysis

## Key Insights

1. **Early Detection**: The model can help identify patients at risk of diabetes based on routine health metrics.

2. **Feature Importance**: HbA1c and blood glucose levels are the most predictive features, followed by age and BMI.

3. **Clinical Relevance**: The model's high recall (79.19%) means it correctly identifies most diabetic patients, which is crucial for medical applications.

4. **Prevention Focus**: The model can be used for preventive healthcare by identifying high-risk patients before they develop diabetes.

## Future Improvements

1. **Feature Engineering**: Include additional medical features like family history, physical activity levels, and diet information.

2. **Model Enhancement**: Experiment with ensemble methods and deep learning approaches.

3. **Real-time Prediction**: Develop a web application for real-time diabetes risk assessment.

4. **Clinical Validation**: Validate the model with real clinical data and medical expert feedback.

## Conclusion

This project successfully demonstrates the application of machine learning in healthcare, specifically for diabetes prediction. The Random Forest model achieved excellent performance with 95.20% accuracy and 97.33% ROC-AUC score, making it suitable for clinical decision support systems. The analysis provides valuable insights into the key factors contributing to diabetes risk, which can inform both medical professionals and patients about preventive measures.

---

*This project was developed as part of a machine learning course assignment by Group 14.*
