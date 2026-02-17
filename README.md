# Titanic Survival Prediction

Machine learning classification model predicting passenger survival on the Titanic.

This end-to-end ML project implements a Random Forest classifier to predict survival outcomes using the Kaggle Titanic dataset. The pipeline includes custom transformers for data preprocessing, hyperparameter tuning with GridSearchCV, and complete model evaluation.

## Features

- **Custom Transformers**: AgeImputer, FeatureEncoder, FeatureDropper
- **Preprocessing Pipeline**: ColumnTransformer with StandardScaler
- **Model Optimization**: GridSearchCV for hyperparameter tuning (500 estimators, max_depth=5)
- **Complete ML Pipeline**: Data cleaning → Feature engineering → Training → Evaluation

## Tools & Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Acknowledgments

- Kaggle Titanic Dataset: https://www.kaggle.com/competitions/titanic/overview
- Result: Achieved 78% accuracy on Kaggle Competition test set.
