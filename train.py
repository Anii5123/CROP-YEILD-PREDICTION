# train.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

RND = 42
DATA_PATH = "yield_df.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1) Load
df = pd.read_csv(DATA_PATH)

# Drop unnamed index columns if present
unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

TARGET_COL = "hg/ha_yield"

# Try to ensure target column is available
target_col = TARGET_COL
if target_col not in df.columns:
    # try to auto-detect a yield-like column
    possible = [c for c in df.columns if "yield" in c.lower()]
    if possible:
        target_col = possible[0]
        print(f"Using detected target column: {target_col}")
        df = df.rename(columns={target_col: TARGET_COL})
        target_col = TARGET_COL
    else:
        raise ValueError("Couldn't find target yield column. Edit train.py to set the target column name.")  # noqa E501

# Select/re-order expected feature columns (edit if dataset differs)
expected_features = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area','Item']  # noqa E501
for c in expected_features:
    if c not in df.columns:
        raise ValueError(f"Expected column '{c}' not found in dataset. Columns present: {df.columns.tolist()}")   # noqa E501

df = df[expected_features + [TARGET_COL]]

# Quick cleaning: drop rows with missing target
df = df.dropna(subset=['hg/ha_yield']).reset_index(drop=True)

# 2) Train/test split
X = df[expected_features]
y = df['hg/ha_yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RND, shuffle=True)   # noqa E501

# 3) Build preprocessing pipelines
numeric_features = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp']   # noqa E501
categorical_features = ['Area', 'Item']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
], remainder='drop')

# Fit preprocessor separately so we can inspect generated feature names later
preprocessor.fit(X_train)


# helper to get transformed feature names (useful for feature importance)
def get_feature_names(preproc, numeric_feats, categorical_feats):
    names = []
    # numeric names
    names.extend(numeric_feats)
    # categorical one-hot names
    ohe = None
    # ColumnTransformer stores fitted transformers in named_transformers_
    if 'cat' in preproc.named_transformers_:
        cat_pipe = preproc.named_transformers_['cat']
        if hasattr(cat_pipe, 'named_steps') and 'onehot' in cat_pipe.named_steps:   # noqa E501
            ohe = cat_pipe.named_steps['onehot']
    if ohe is not None:
        cat_names = ohe.get_feature_names_out(categorical_feats)
        names.extend(list(cat_names))
    return names

feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)   # noqa E501

# 4) Models to train
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, random_state=RND,
        min_samples_leaf=1, max_features="sqrt"
    ),
    'DecisionTree': DecisionTreeRegressor(
        random_state=RND, ccp_alpha=0.0
    ),
    'GradientBoosting': GradientBoostingRegressor(
        random_state=RND, learning_rate=0.1
    )
}


results = []

for name, estimator in models.items():
    print(f"\nTraining: {name}")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', estimator)
        ], memory=None)

    # fit
    pipe.fit(X_train, y_train)

    # predict
    y_pred = pipe.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    results.append({
        'model': name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    })

    # save pipeline (preprocessor + model) so at inference time we can call pipeline.predict(raw_df)  # noqa E501
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(pipe, model_path)
    print(f"Saved pipeline to {model_path}")

    # If RandomForest, print top feature importances
    if name == 'RandomForest':
        # get regressor and feature importances
        reg = pipe.named_steps['regressor']
        if hasattr(reg, 'feature_importances_'):
            importances = reg.feature_importances_
            # align with feature_names
            if len(importances) == len(feature_names):
                feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]  # noqa E501
                print("Top features (RandomForest):")
                for fn, imp in feat_imp:
                    print(f"  {fn}: {imp:.4f}")

# Save metrics
metrics_df = pd.DataFrame(results).sort_values('r2', ascending=False)
metrics_df.to_csv("outputs/metrics.csv", index=False)
print("\nAll done. Metrics saved to outputs/metrics.csv")
print(metrics_df)
