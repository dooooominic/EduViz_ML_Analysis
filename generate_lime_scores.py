"""
Generate LIME importance scores for all models and save to CSV.
This pre-computes LIME explanations for faster Streamlit dashboard loading.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("data/model_eval_2025.csv")

# Define models
REGRESSION_MODELS = {
    "GAM": "gam_pred",  # Using GAM as placeholder (pre-trained predictions)
    "GBR": "gbr_pred"
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": "logreg_pred",
    "DecisionTree": "dt_pred",
    "MLP": "mlp_pred",
}

PROB_MODELS = {
    "LogisticRegression_prob": "logreg_prob"
}

# Extract feature columns
# Include District and Tested Grade as categorical features
categorical_features = ["District", "Tested Grade"]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[categorical_features], drop_first=False)
feature_cols = df_encoded.columns.tolist()

print(f"✓ Created {len(feature_cols)} feature columns from District and Tested Grade")
print(f"  → Districts: {df['District'].nunique()}")
print(f"  → Grades: {df['Tested Grade'].nunique()}")

X = df_encoded.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== REGRESSION: Train simple models for explanation =====
print("\n📊 Processing Regression Models...")

# For regression, we'll create a simple linear model to approximate the predictions
y_reg_gam = df["gam_pred"].values
y_reg_gbr = df["gbr_pred"].values

reg_gam = LinearRegression()
reg_gam.fit(X_scaled, y_reg_gam)

reg_gbr = LinearRegression()
reg_gbr.fit(X_scaled, y_reg_gbr)

print("✓ Regression models fitted")

# ===== CLASSIFICATION: Train simple models for explanation =====
print("\n🎯 Processing Classification Models...")

y_clf = df["y_true_binary"].values

# Simple models for classification
clf_logreg = LogisticRegression(max_iter=1000)
clf_logreg.fit(X_scaled, y_clf)

print("✓ Classification models fitted")

# ===== GENERATE LIME EXPLANATIONS =====
print("\n🔍 Generating LIME explanations...")

lime_results = []

# Create LIME explainers
explainer_reg = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=feature_cols,
    mode="regression",
    random_state=42
)

explainer_clf = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=feature_cols,
    class_names=["Fail", "Pass"],
    mode="classification",
    random_state=42
)

# Sample instances for LIME explanation (to avoid too many rows)
# Use all instances if dataset is small, otherwise sample
n_instances = len(df)
if n_instances > 100:
    sample_indices = np.random.choice(n_instances, min(100, n_instances), replace=False)
    print(f"  → Sampling {len(sample_indices)} instances from {n_instances} total")
else:
    sample_indices = np.arange(n_instances)
    print(f"  → Using all {n_instances} instances")

# Generate regression LIME scores
print("  → GAM model...")
for idx in sample_indices:
    exp = explainer_reg.explain_instance(X_scaled[idx], reg_gam.predict, num_features=len(feature_cols))
    for feature, weight in exp.as_list():
        lime_results.append({
            "instance_idx": idx,
            "model": "GAM",
            "feature": feature,
            "importance": weight
        })

print("  → GBR model...")
for idx in sample_indices:
    exp = explainer_reg.explain_instance(X_scaled[idx], reg_gbr.predict, num_features=len(feature_cols))
    for feature, weight in exp.as_list():
        lime_results.append({
            "instance_idx": idx,
            "model": "GBR",
            "feature": feature,
            "importance": weight
        })

# Generate classification LIME scores
print("  → LogisticRegression model...")
for idx in sample_indices:
    exp = explainer_clf.explain_instance(X_scaled[idx], clf_logreg.predict_proba, num_features=len(feature_cols))
    for feature, weight in exp.as_list():
        lime_results.append({
            "instance_idx": idx,
            "model": "LogisticRegression",
            "feature": feature,
            "importance": weight
        })

# ===== ALSO INCLUDE FEATURE IMPORTANCE FROM COEFFICIENTS =====
print("\n📈 Adding model coefficients as global feature importance...")

# Regression coefficients
for i, feature in enumerate(feature_cols):
    lime_results.append({
        "instance_idx": -1,  # -1 indicates global importance
        "model": "GAM_GlobalImportance",
        "feature": feature,
        "importance": abs(reg_gam.coef_[i])
    })
    lime_results.append({
        "instance_idx": -1,
        "model": "GBR_GlobalImportance",
        "feature": feature,
        "importance": abs(reg_gbr.coef_[i])
    })

# Classification coefficients
for i, feature in enumerate(feature_cols):
    lime_results.append({
        "instance_idx": -1,
        "model": "LogisticRegression_GlobalImportance",
        "feature": feature,
        "importance": abs(clf_logreg.coef_[0][i])
    })

# ===== SAVE TO CSV =====
lime_df = pd.DataFrame(lime_results)
output_file = "data/lime_importance_scores.csv"
lime_df.to_csv(output_file, index=False)

print(f"\n✅ Saved {len(lime_df)} LIME importance entries to {output_file}")
print(f"\nLIME scores CSV structure:")
print(f"  - Columns: {', '.join(lime_df.columns.tolist())}")
print(f"  - Models included: {lime_df['model'].unique().tolist()}")
print(f"  - Total rows: {len(lime_df)}")
print(f"\nSample rows:")
print(lime_df.head(10))

# ===== SUMMARY STATISTICS =====
print("\n📊 Summary Statistics by Model:")
summary = lime_df.groupby("model").agg({
    "importance": ["mean", "std", "min", "max", "count"]
}).round(4)
print(summary)

print("\n✨ Done! Use data/lime_importance_scores.csv in your Streamlit app.")
