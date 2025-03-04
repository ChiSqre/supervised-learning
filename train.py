import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from xgboost import XGBRegressor

os.makedirs("outputs", exist_ok=True)

df = pd.read_excel("ENB2012_data.xlsx")
X = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]]
y1 = df["Y1"]
y2 = df["Y2"]

X_train, X_test, y1_train, y1_test = train_test_split(
    X, y1, test_size=0.2, random_state=42
)
_, _, y2_train, y2_test = train_test_split(
    X, y2, test_size=0.2, random_state=42
)

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
ridge_params = {
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
}

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(objective="reg:squarederror"))
])
xgb_params = {
    "xgb__learning_rate": [0.01, 0.1],
    "xgb__max_depth": [3, 5],
    "xgb__n_estimators": [100, 200]
}

ridge_results_df = pd.DataFrame()
xgb_results_df = pd.DataFrame()

def full_grid_search(pipeline, param_grid, X_tr, y_tr, X_te, y_te, model_name, target_name):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=0,
        n_jobs=-1
    )
    grid.fit(X_tr, y_tr)

    cv_res = grid.cv_results_

    rows = []
    for i, param_combo in enumerate(cv_res["params"]):
        row_dict = {}
        row_dict["model"] = model_name
        row_dict["target"] = target_name

        for k, v in param_combo.items():
            row_dict[k] = v

        mean_cv_neg_mse = cv_res["mean_test_score"][i]
        std_cv_neg_mse = cv_res["std_test_score"][i]
        rank = cv_res["rank_test_score"][i]

        row_dict["mean_cv_mse"] = -mean_cv_neg_mse
        row_dict["std_cv_mse"] = std_cv_neg_mse
        row_dict["rank_test_score"] = rank

        pipe_copy = clone(pipeline)
        pipe_copy.set_params(**param_combo)
        pipe_copy.fit(X_tr, y_tr)
        preds = pipe_copy.predict(X_te)
        test_mse = np.mean((preds - y_te) ** 2)
        row_dict["test_mse"] = test_mse

        rows.append(row_dict)

    return pd.DataFrame(rows)

ridge_y1_df = full_grid_search(
    ridge_pipeline, ridge_params,
    X_train, y1_train, X_test, y1_test,
    "Ridge", "Y1"
)
ridge_results_df = pd.concat([ridge_results_df, ridge_y1_df], ignore_index=True)

ridge_y2_df = full_grid_search(
    ridge_pipeline, ridge_params,
    X_train, y2_train, X_test, y2_test,
    "Ridge", "Y2"
)
ridge_results_df = pd.concat([ridge_results_df, ridge_y2_df], ignore_index=True)

xgb_y1_df = full_grid_search(
    xgb_pipeline, xgb_params,
    X_train, y1_train, X_test, y1_test,
    "XGBoost", "Y1"
)
xgb_results_df = pd.concat([xgb_results_df, xgb_y1_df], ignore_index=True)

xgb_y2_df = full_grid_search(
    xgb_pipeline, xgb_params,
    X_train, y2_train, X_test, y2_test,
    "XGBoost", "Y2"
)
xgb_results_df = pd.concat([xgb_results_df, xgb_y2_df], ignore_index=True)

ridge_results_path = os.path.join("outputs", "ridge_results.csv")
xgb_results_path = os.path.join("outputs", "xgb_results.csv")

ridge_results_df.to_csv(ridge_results_path, index=False)
xgb_results_df.to_csv(xgb_results_path, index=False)

from sklearn.model_selection import learning_curve

def get_best_params(df_results):
    best_idx = df_results["mean_cv_mse"].idxmin()
    best_row = df_results.loc[best_idx].to_dict()
    param_dict = {}
    for k, v in best_row.items():
        if k.startswith("ridge__") or k.startswith("xgb__"):
            param_dict[k] = v
    return param_dict

def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True, random_state=42,
        n_jobs=-1
    )

    train_mse = -train_scores
    val_mse = -val_scores

    train_mse_mean = np.mean(train_mse, axis=1)
    train_mse_std = np.std(train_mse, axis=1)
    val_mse_mean = np.mean(val_mse, axis=1)
    val_mse_std = np.std(val_mse, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mse_mean, marker='o', label='Training MSE')
    plt.fill_between(train_sizes,
                        train_mse_mean - train_mse_std,
                        train_mse_mean + train_mse_std,
                        alpha=0.2)
    plt.plot(train_sizes, val_mse_mean, marker='s', label='Validation MSE')
    plt.fill_between(train_sizes,
                        val_mse_mean - val_mse_std,
                        val_mse_mean + val_mse_std,
                        alpha=0.2)

    plt.title(f"Learning Curves: {title}")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()

best_ridge_y1 = get_best_params(ridge_results_df[ridge_results_df["target"]=="Y1"])
ridge_pipeline_best_y1 = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
ridge_pipeline_best_y1.set_params(**best_ridge_y1)

plot_learning_curves(ridge_pipeline_best_y1, X_train, y1_train, "Ridge (Best) - Y1")
plt.savefig(os.path.join("outputs", "learning_curve_ridge_y1.png"), dpi=300)
plt.close()

best_ridge_y2 = get_best_params(ridge_results_df[ridge_results_df["target"]=="Y2"])
ridge_pipeline_best_y2 = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
ridge_pipeline_best_y2.set_params(**best_ridge_y2)

plot_learning_curves(ridge_pipeline_best_y2, X_train, y2_train, "Ridge (Best) - Y2")
plt.savefig(os.path.join("outputs", "learning_curve_ridge_y2.png"), dpi=300)
plt.close()

best_xgb_y1 = get_best_params(xgb_results_df[xgb_results_df["target"]=="Y1"])
xgb_pipeline_best_y1 = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(objective="reg:squarederror"))
])
xgb_pipeline_best_y1.set_params(**best_xgb_y1)

plot_learning_curves(xgb_pipeline_best_y1, X_train, y1_train, "XGBoost (Best) - Y1")
plt.savefig(os.path.join("outputs", "learning_curve_xgb_y1.png"), dpi=300)
plt.close()

best_xgb_y2 = get_best_params(xgb_results_df[xgb_results_df["target"]=="Y2"])
xgb_pipeline_best_y2 = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(objective="reg:squarederror"))
])
xgb_pipeline_best_y2.set_params(**best_xgb_y2)

plot_learning_curves(xgb_pipeline_best_y2, X_train, y2_train, "XGBoost (Best) - Y2")
plt.savefig(os.path.join("outputs", "learning_curve_xgb_y2.png"), dpi=300)
plt.close()

print("All model results saved to 'ridge_results.csv' and 'xgb_results.csv'.")
print("Learning curves saved to 'outputs/'.")