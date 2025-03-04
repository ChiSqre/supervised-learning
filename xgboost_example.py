#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost Regressor Example
-------------------------
This script demonstrates how to use XGBRegressor from the xgboost package
for regression tasks with a simple example.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create sample data
print("Creating sample regression data...")
X = np.random.rand(100, 5) * 10  # 100 samples, 5 features
# Create a target variable with some relationship to the features plus some noise
y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] - X[:, 4] + np.random.normal(0, 1, 100)

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# 3. Initialize the XGBRegressor model with some common parameters
print("\nInitializing XGBoost Regressor model...")
xgb_model = XGBRegressor(
    n_estimators=100,       # Number of gradient boosted trees
    learning_rate=0.1,      # Learning rate (step size shrinkage)
    max_depth=5,            # Maximum tree depth for base learners
    min_child_weight=1,     # Minimum sum of instance weight needed in a child
    subsample=0.8,          # Subsample ratio of the training instances
    colsample_bytree=0.8,   # Subsample ratio of columns when constructing each tree
    objective='reg:squarederror',  # Regression with squared error
    random_state=42         # Random seed for reproducibility
)

# 4. Train the model
print("Training the model...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # Evaluation sets for monitoring training
    eval_metric='rmse',       # Evaluation metric
    verbose=True,             # Print training progress
    early_stopping_rounds=10  # Stop if no improvement after 10 rounds
)

# 5. Make predictions
print("\nMaking predictions...")
y_pred = xgb_model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 7. Plot feature importance
print("\nPlotting feature importance...")
plt.figure(figsize=(10, 6))
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')
print("Feature importance plot saved as 'xgboost_feature_importance.png'")

# 8. Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('xgboost_actual_vs_predicted.png')
print("Actual vs Predicted plot saved as 'xgboost_actual_vs_predicted.png'")

# 9. Advanced: Cross-validation example
print("\nDemonstrating cross-validation...")
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print("Cross-validation results:")
print(f"RMSE scores: {cv_rmse}")
print(f"Mean RMSE: {cv_rmse.mean():.4f}")
print(f"Standard deviation: {cv_rmse.std():.4f}")

# 10. Advanced: Simple hyperparameter tuning example
print("\nDemonstrating a simple hyperparameter tuning approach...")
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

grid_search = GridSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=0
)

print("Note: Hyperparameter tuning can take time. This is a simple example.")
print("For real-world tasks, consider using RandomizedSearchCV for efficiency.")
print("Running a small grid search...")

grid_search.fit(X, y)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

print("\nXGBoost regression example completed!")

