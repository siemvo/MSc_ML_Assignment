import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  # For imputing numeric data
from sklearn.ensemble import RandomForestClassifier  # For imputing categorical data

'''
This script contains the class of functions used for the imputation.
For imputation of numeric data, IterativeImputer is used, which is a multivariate imputer resembling MICE.
For imputation of categorical data, RandomForestClassifier is used.
'''

# Define a custom class for imputation
class assignment_impute:
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
        self.numeric_imputer = None  # Will hold the IterativeImputer for numeric columns
        self.categorical_classifiers = {}  # Will store models for each categorical column
        self.numeric_cols = None  # Initializing numeric column names list
        self.categorical_cols = None  # Initializing categorical column names list

    def fit(self, data):
        # Separate column names by type (in lists created in __init__)
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = data.select_dtypes(exclude=[np.number]).columns

        # Predict missing numbers using IterativeImputer
        self.numeric_imputer = IterativeImputer(max_iter=self.max_iter, random_state=0)
        self.numeric_imputer.fit(data[self.numeric_cols])

        # Classifiers for each categorical column
        for col in self.categorical_cols:
            if data[col].isnull().sum() == 0: # No "NaN" or "None" returns false == 0
                continue  # Skip columns that don't have missing values

            # Only use rows where this column is not missing
            not_null_mask = data[col].notnull()
            X_train = data.loc[not_null_mask].drop(columns=[col])  # Features
            y_train = data.loc[not_null_mask, col]  # Target values

            # Fill missing values in other categorical columns with placeholder "Missing"
            X_train = X_train.copy()
            for other_col in self.categorical_cols:
                if other_col != col: # Don't fill the target column
                    X_train[other_col] = X_train[other_col].fillna("Missing")

            # Convert categorical variables into dummy variables
            X_train_encoded = pd.get_dummies(X_train, drop_first=True)

            # Train a RandomForestClassifier to predict missing values in this column
            clf = RandomForestClassifier(random_state=0)
            clf.fit(X_train_encoded, y_train)

            # Save the model and the feature names it expects
            self.categorical_classifiers[col] = {
                "model": clf,
                "features": X_train_encoded.columns
            }

    def transform(self, data):
        # Impute missing numeric values using the trained numeric imputer
        numeric = pd.DataFrame(
            self.numeric_imputer.transform(data[self.numeric_cols]),  # Fill in missing numbers
            columns=self.numeric_cols,
            index=data.index
        ).round(2)  # Round to 2 decimals

        # Impute missing categorical values using trained classifiers
        categorical = data[self.categorical_cols].copy()

        for col in self.categorical_cols:
            # If we didn't train a model for this column, just fill missing with the mode
            if col not in self.categorical_classifiers:
                categorical[col] = categorical[col].fillna(categorical[col].mode()[0])
                continue

            # Use the classifier we trained earlier
            clf_info = self.categorical_classifiers[col]
            model = clf_info["model"]
            feature_cols = clf_info["features"]

            # Find rows where this column is missing
            missing_mask = categorical[col].isnull()
            if missing_mask.sum() == 0:
                continue  # Skip if nothing to impute

            # Prepare input data for prediction
            X_missing = data.loc[missing_mask].drop(columns=[col])
            for other_col in self.categorical_cols:
                if other_col != col:
                    X_missing[other_col] = X_missing[other_col].fillna("Missing")

            # One-hot encode the input
            X_missing_encoded = pd.get_dummies(X_missing, drop_first=True)

            # Make sure all expected columns are present (fill missing ones with 0s)
            missing_cols = [f for f in feature_cols if f not in X_missing_encoded.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=X_missing_encoded.index, columns=missing_cols)
                X_missing_encoded = pd.concat([X_missing_encoded, missing_df], axis=1)

            # Ensure correct order of columns
            X_missing_encoded = X_missing_encoded[feature_cols]

            # Predict missing values and fill them in
            predicted = model.predict(X_missing_encoded)
            categorical.loc[missing_mask, col] = predicted

        # Combine numeric and categorical data back together
        imputed = pd.concat([numeric, categorical], axis=1)[data.columns]  # Preserve column order
        return imputed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
