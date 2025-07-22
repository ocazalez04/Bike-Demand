# ...existing code...
###Working with Kaggle API to gather the data


# =============================================================================
# 0. Case Study: Bike Sharing Demand Prediction
# Objective: Predict the total number of bike rentals in a given hour.
# Business Value: This forecast helps the company with resource management.
# Accurate predictions mean they can optimize the number of available bikes at each station
# , preventing shortages during peak hours and avoiding oversupply during lulls.
# This improves customer satisfaction and operational efficiency.

# Analyze the Evaluation Metric: RMSLE
# The Root Mean Squared Logarithmic Error (RMSLE) is a suitable metric for this problem
# because it penalizes underestimations more than overestimations. This is important in
# the context of bike rentals, where failing to predict high demand can lead to lost
# revenue and customer dissatisfaction.

# =============================================================================
# =============================================================================
# 1. Library Imports
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error  # Removed unused make_scorer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set plot style
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =============================================================================
# 2. Data Loading & Initial Setup
# =============================================================================
# Load the datasets
train_df = pd.read_csv("./bike-sharing-demand/train.csv", parse_dates=["datetime"])
test_df = pd.read_csv("./bike-sharing-demand/test.csv", parse_dates=["datetime"])


# =============================================================================
# 3. Feature Engineering
# =============================================================================
def extract_time_features(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    return df


train_df = extract_time_features(train_df)
test_df = extract_time_features(test_df)

# Apply log transformation to the target variable 'count'
train_df["count"] = np.log1p(train_df["count"])

# Drop unnecessary or leaky columns
train_df = train_df.drop(["datetime", "casual", "registered"], axis=1)
test_df_datetime = test_df["datetime"]
test_df = test_df.drop(["datetime"], axis=1)

# =============================================================================
# 4. Outlier Removal
# =============================================================================
print(f"Original train data size: {len(train_df)}")
train_df = train_df[
    np.abs(train_df["count"] - train_df["count"].mean())
    <= (3 * train_df["count"].std())
]
print(f"Train data size after outlier removal: {len(train_df)}")

# =============================================================================
# 5. Data Splitting and Preprocessing (for Validation)
# =============================================================================
# Separate features (X) and target (y) from the final, cleaned training data
X = train_df.drop("count", axis=1)
y = train_df["count"]

# Split for validation purposes
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale numerical features - Fit ONLY on the training set
scaler = MinMaxScaler()
numerical_features = ["temp", "atemp", "humidity", "windspeed"]

# Use .loc for safer assignment and to avoid SettingWithCopyWarning
X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val.loc[:, numerical_features] = scaler.transform(X_val[numerical_features])


# =============================================================================
# 6. Model Training and Evaluation
# =============================================================================
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_val)

    # Revert transformations for evaluation
    y_val_orig = np.expm1(y_val)
    y_pred_orig = np.expm1(y_pred_log)

    score = rmsle(y_val_orig, y_pred_orig)
    results[name] = score
    print(f"{name}: RMSLE = {score:.4f}")

# =============================================================================
# 7. Hyperparameter Tuning (Example with LightGBM)
# =============================================================================
print("\n--- Hyperparameter Tuning for LightGBM ---")
param_grid = {
    "n_estimators": [500, 600],
    "learning_rate": [0.05],
    "num_leaves": [40, 50],
    "max_depth": [10, 15],
}

grid_search = GridSearchCV(
    estimator=LGBMRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)
best_lgbm = grid_search.best_estimator_


# =============================================================================
# 8. Feature Importance
# =============================================================================
def plot_feature_importance(model, feature_names, model_name=""):
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=importances)
    plt.title(f"Feature Importance for {model_name}")
    plt.tight_layout()
    plt.show()


plot_feature_importance(best_lgbm, X_train.columns, "Tuned LightGBM")

# =============================================================================
# 9. Final Prediction and Submission (Corrected Workflow)
# =============================================================================
print("\nTraining final model on all available training data...")

# CRITICAL FIX: Scale the full dataset before final training
final_scaler = MinMaxScaler()
X_final_scaled = X.copy()
X_final_scaled.loc[:, numerical_features] = final_scaler.fit_transform(
    X[numerical_features]
)

test_df_final_scaled = test_df.copy()
test_df_final_scaled.loc[:, numerical_features] = final_scaler.transform(
    test_df[numerical_features]
)

# Train the best model on the consistently scaled, full training dataset
final_model = best_lgbm
final_model.fit(X_final_scaled, y)

# Predict on the consistently scaled Kaggle test set
final_predictions_log = final_model.predict(test_df_final_scaled)

# Reverse the log transformation and ensure non-negativity
final_predictions = np.expm1(final_predictions_log)
final_predictions[final_predictions < 0] = 0

# Create the submission file
submission_df = pd.DataFrame({"datetime": test_df_datetime, "count": final_predictions})


submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())
