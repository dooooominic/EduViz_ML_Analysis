import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import IsolationForest
import lime, shap
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(page_title="Educational Data Visualization", layout="wide")
st.title("Texas Statewide Assessment Data Visualization Dashboard")

st.write("""
This dashboard visualizes and analyzes Texas Statewide Assessment data from 2022–2025.
Machine learning models were trained on 2022–2024 data and predict 2025 outcomes across all Texas districts.
Compare multiple models across **regression** (continuous score prediction) and **classification** (pass/fail) paradigms.
""")

@st.cache_data
def load_eval_data():
    return pd.read_csv("data/model_eval_2025.csv")

eval_df = load_eval_data()

# ============= Model Definitions =============
REGRESSION_MODELS = {"GAM (Generalized Additive Model)": "gam_pred", "Gradient Boosting Regressor": "gbr_pred"}
CLASSIFICATION_MODELS = {
    "Logistic Regression": "logreg_pred",
    "Decision Tree": "dt_pred",
    "Neural Network (MLP)": "mlp_pred",
}
PROB_MODELS = {"Logistic Regression (w/ probability)": "logreg_prob"}

# ============= Sidebar Filters =============
st.sidebar.header("Dashboard Filters & Navigation")

tab_choice = st.sidebar.radio(
    "Select View",
    ["Overview & Model Comparison", "Regression Deep-Dive", "Classification Deep-Dive", "Trend Analysis", "Explainability (SHAP/LIME)", "Anomaly Detection", "Testing Tab"],
)

# Filters
districts = ["All"] + sorted(eval_df["District"].unique().tolist())
selected_district = st.sidebar.selectbox("Filter by District", districts)

grades = ["All"] + sorted(eval_df["Tested Grade"].unique().tolist())
selected_grade = st.sidebar.selectbox("Filter by Grade", grades)

# Apply filters
filtered_df = eval_df.copy()
if selected_district != "All":
    filtered_df = filtered_df[filtered_df["District"] == selected_district]
if selected_grade != "All":
    filtered_df = filtered_df[filtered_df["Tested Grade"] == selected_grade]

st.sidebar.write(f"**Rows after filtering:** {len(filtered_df)} / {len(eval_df)}")

# ============= TAB 1: OVERVIEW & MODEL COMPARISON =============
if tab_choice == "Overview & Model Comparison":
    st.header("Model Performance Overview")
    st.write("Compare all models side-by-side across regression and classification metrics.")
    
    # Compute metrics for all regression models
    reg_metrics = {}
    for label, col in REGRESSION_MODELS.items():
        y_true = filtered_df["y_true_cont"]
        y_pred = filtered_df[col]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        reg_metrics[label] = {"MAE": mae, "RMSE": np.sqrt(mse), "R²": r2}
    
    # Compute metrics for all classification models
    clf_metrics = {}
    for label, col in CLASSIFICATION_MODELS.items():
        y_true = filtered_df["y_true_binary"]
        y_pred = filtered_df[col]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        clf_metrics[label] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    
    # Compute metrics for probability models
    for label, col in PROB_MODELS.items():
        y_true = filtered_df["y_true_binary"]
        y_prob = filtered_df[col]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = np.nan
        clf_metrics[label] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression Models")
        reg_df = pd.DataFrame(reg_metrics).T
        st.dataframe(reg_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))
        
    with col2:
        st.subheader("Classification Models")
        clf_df = pd.DataFrame(clf_metrics).T
        st.dataframe(clf_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))
    
    # Heatmap comparison
    st.subheader("Metric Heatmap: Regression Models")
    fig, ax = plt.subplots(figsize=(2, 1))
    sns.heatmap(reg_df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "Score"})
    st.pyplot(fig)
    
    st.subheader("Metric Heatmap: Classification Models")
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.heatmap(clf_df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, cbar_kws={"label": "Score"})
    st.pyplot(fig)
    
    # Insights
    st.subheader("Key Insights")
    best_reg = reg_df["R²"].idxmax()
    best_clf = clf_df["F1"].idxmax()
    st.write(f"""
    - **Best Regression Model (by R²):** {best_reg} ({reg_df.loc[best_reg, "R²"]:.3f})
    - **Best Classification Model (by F1):** {best_clf} ({clf_df.loc[best_clf, "F1"]:.3f})
    - **Data filtered to:** {len(filtered_df)} districts/grades
    """)


# ============= TAB 2: REGRESSION DEEP-DIVE =============
elif tab_choice == "Regression Deep-Dive":
    st.header("🔍 Regression Model Deep-Dive")
    
    selected_reg_model_label = st.selectbox("Choose regression model", list(REGRESSION_MODELS.keys()))
    selected_reg_model_col = REGRESSION_MODELS[selected_reg_model_label]
    
    y_true = filtered_df["y_true_cont"]
    y_pred = filtered_df[selected_reg_model_col]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    col1.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error (lower is better)")
    col2.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error")
    col3.metric("R²", f"{r2:.3f}", help="R² Score (0-1, higher is better)")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect Prediction")
        ax.set_xlabel("Actual STAAR %")
        ax.set_ylabel("Predicted STAAR %")
        ax.set_title(f"{selected_reg_model_label}")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Residual Distribution")
        residuals = y_pred - y_true
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel("Residual (Predicted - Actual)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        ax.legend()
        st.pyplot(fig)
    
    st.write(f"**Mean Residual:** {residuals.mean():.2f} {'(underprediction)' if residuals.mean() < 0 else '(overprediction)'}")
    
    # Error by group
    st.subheader("📊 Error Analysis by Group")
    group_col = st.radio("Group by:", ["Tested Grade", "District"], key="reg_group")
    
    residuals_by_group = filtered_df.groupby(group_col).apply(
        lambda df: pd.Series({
            "Mean Residual": (df[selected_reg_model_col] - df["y_true_cont"]).mean(),
            "Std Dev": (df[selected_reg_model_col] - df["y_true_cont"]).std(),
            "Count": len(df),
        })
    ).sort_values("Mean Residual")
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    residuals_by_group["Mean Residual"].plot(kind="barh", ax=ax, color=plt.cm.RdYlGn((residuals_by_group["Mean Residual"] + 10) / 20))
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("Mean Residual")
    ax.set_title(f"Mean Prediction Error by {group_col}")
    st.pyplot(fig)
    
    st.dataframe(residuals_by_group)


# ============= TAB 3: CLASSIFICATION DEEP-DIVE =============
elif tab_choice == "Classification Deep-Dive":
    st.header("Classification Model Deep-Dive")
    
    clf_model_options = {**CLASSIFICATION_MODELS, **PROB_MODELS}
    selected_clf_model_label = st.selectbox("Choose classification model", list(clf_model_options.keys()))
    selected_clf_model_col = clf_model_options[selected_clf_model_label]
    
    y_true = filtered_df["y_true_binary"]
    y_prob_or_pred = filtered_df[selected_clf_model_col]
    
    # Determine if it's a probability column
    is_prob = "prob" in selected_clf_model_col.lower() or selected_clf_model_label.endswith("(w/ probability)")
    if is_prob:
        y_pred = (y_prob_or_pred >= 0.5).astype(int)
    else:
        y_pred = y_prob_or_pred.astype(int)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    col1.metric("Accuracy", f"{acc:.3f}", help="% of correct predictions")
    col2.metric("Precision", f"{prec:.3f}", help="Of predicted positives, % correct")
    col3.metric("Recall", f"{rec:.3f}", help="Of actual positives, % caught")
    col4.metric("F1-Score", f"{f1:.3f}", help="Harmonic mean of precision & recall")
    
    if is_prob:
        try:
            auc = roc_auc_score(y_true, y_prob_or_pred)
            st.metric("AUC-ROC", f"{auc:.3f}", help="Area Under ROC Curve (0.5-1.0)")
        except:
            pass
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Prediction Distribution")
        if is_prob:
            fig, ax = plt.subplots(figsize=(3, 2.5))
            ax.hist(y_prob_or_pred[y_true == 0], bins=30, alpha=0.6, label="Actual Fail", color='red')
            ax.hist(y_prob_or_pred[y_true == 1], bins=30, alpha=0.6, label="Actual Pass", color='green')
            ax.axvline(0.5, color='k', linestyle='--', linewidth=2, label='Decision Threshold')
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
        else:
            counts = pd.Series(y_pred).value_counts()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["Fail", "Pass"], [counts.get(0, 0), counts.get(1, 0)], color=['red', 'green'])
            ax.set_ylabel("Count")
            st.pyplot(fig)
    
    # ROC curve (if probability)
    if is_prob:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_or_pred)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
    
    # Error by group
    st.subheader("Classification Performance by Group")
    group_col = st.radio("Group by:", ["Tested Grade", "District"], key="clf_group")
    
    perf_by_group = filtered_df.groupby(group_col).apply(
        lambda df: pd.Series({
            "Accuracy": accuracy_score(df["y_true_binary"], (df[selected_clf_model_col] >= 0.5).astype(int) if is_prob else df[selected_clf_model_col].astype(int)),
            "Count": len(df),
        })
    ).sort_values("Accuracy")
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    perf_by_group["Accuracy"].plot(kind="barh", ax=ax, color=plt.cm.RdYlGn(perf_by_group["Accuracy"]))
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Accuracy by {group_col}")
    st.pyplot(fig)
    
    st.dataframe(perf_by_group)


# ============= TAB 4: TREND ANALYSIS =============
elif tab_choice == "Trend Analysis":
    st.header("Trend Analysis & Model Agreement")
    
    st.subheader("Model Agreement (Do predictions align?)")
    st.write("""
    Compare regression and classification models to identify where they agree or disagree.
    This can reveal systematic biases or data quality issues.
    """)
    
    # Regression consensus
    gam_pred = filtered_df["gam_pred"]
    gbr_pred = filtered_df["gbr_pred"]
    reg_consensus = np.abs(gam_pred - gbr_pred).mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Regression Disagreement", f"{reg_consensus:.2f}", 
                  help="Mean absolute difference between GAM and GBR predictions")
    
    # Classification consensus
    logreg = filtered_df["logreg_pred"].astype(int)
    dt = filtered_df["dt_pred"].astype(int)
    mlp = filtered_df["mlp_pred"].astype(int)
    clf_agreement = ((logreg == dt) & (dt == mlp)).sum() / len(filtered_df) * 100
    
    with col2:
        st.metric("Classification Agreement %", f"{clf_agreement:.1f}%", 
                  help="% of rows where Logistic Reg, Decision Tree, and MLP all agree")
    
    # Scatter: GAM vs GBR
    st.subheader("Regression Model Predictions")
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(gam_pred, gbr_pred, alpha=0.5, s=20)
    ax.plot([gam_pred.min(), gam_pred.max()], [gam_pred.min(), gam_pred.max()], "r--")
    ax.set_xlabel("GAM Prediction")
    ax.set_ylabel("GBR Prediction")
    ax.set_title("Regression Model Predictions Comparison")
    st.pyplot(fig)
    
    # Classification agreement heatmap
    st.subheader("Classification Model Agreement Matrix")
    clf_models = {"LogReg": logreg, "DecisionTree": dt, "MLP": mlp}
    agreement_matrix = pd.DataFrame({
        name: [100 * (clf_models[name] == clf_models[other]).sum() / len(filtered_df) 
               for other in clf_models.keys()]
        for name in clf_models.keys()
    })
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax, vmin=0, vmax=100)
    ax.set_title("Classification Model Agreement (% same prediction)")
    st.pyplot(fig)
    
    # Regression residuals by grade
    st.subheader("Residual Trends by Grade & Model")
    
    grade_residuals = []
    for grade in sorted(filtered_df["Tested Grade"].unique()):
        grade_data = filtered_df[filtered_df["Tested Grade"] == grade]
        for model_label, model_col in REGRESSION_MODELS.items():
            residuals = grade_data[model_col] - grade_data["y_true_cont"]
            grade_residuals.append({
                "Grade": grade,
                "Model": model_label.split("(")[0].strip(),
                "Mean Residual": residuals.mean(),
                "Std": residuals.std(),
            })
    
    residuals_df = pd.DataFrame(grade_residuals)
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    for model in residuals_df["Model"].unique():
        data = residuals_df[residuals_df["Model"] == model]
        ax.plot(data["Grade"], data["Mean Residual"], marker="o", label=model)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("Tested Grade")
    ax.set_ylabel("Mean Residual")
    ax.set_title("Residual Trends by Grade")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # Scatter: True vs predictions for both regression models
    st.subheader("Prediction Landscape: True vs Both Regression Models")
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
    
    for idx, (label, col) in enumerate(REGRESSION_MODELS.items()):
        ax = axes[idx]
        ax.scatter(filtered_df["y_true_cont"], filtered_df[col], alpha=0.5, s=20)
        lims = [min(filtered_df["y_true_cont"].min(), filtered_df[col].min()),
                max(filtered_df["y_true_cont"].max(), filtered_df[col].max())]
        ax.plot(lims, lims, "r--")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(label)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # District-level performance heatmap
    st.subheader("Top/Bottom Districts by Residual")
    
    district_mae = filtered_df.groupby("District").apply(
        lambda df: pd.Series({
            "GAM_MAE": mean_absolute_error(df["y_true_cont"], df["gam_pred"]),
            "GBR_MAE": mean_absolute_error(df["y_true_cont"], df["gbr_pred"]),
        })
    ).mean(axis=1).sort_values()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Best Performing Districts (Lowest Error)**")
        st.dataframe(district_mae.head(10))
    
    with col2:
        st.write("**Worst Performing Districts (Highest Error)**")
        st.dataframe(district_mae.tail(10))


# ============= TAB 5: EXPLAINABILITY (SHAP & LIME) =============
elif tab_choice == "Explainability (SHAP/LIME)":
    st.header("Model Explainability with SHAP & LIME")
    
    st.write("""
    Understand **why** models make specific predictions. Use LIME for local instance-level explanations.
    """)
    
    expl_paradigm = st.radio("Choose paradigm:", ["Regression", "Classification"])
    
    if expl_paradigm == "Regression":
        selected_reg_model_label = st.selectbox("Choose regression model", list(REGRESSION_MODELS.keys()), key="expl_reg")
        selected_reg_model_col = REGRESSION_MODELS[selected_reg_model_label]
        
        # Prepare data for explainability (feature columns)
        feature_cols = [c for c in filtered_df.columns if c not in ["District", "Year", "y_true_binary", "y_true_cont", 
                                                                       "logreg_pred", "logreg_prob", "dt_pred", "mlp_pred", 
                                                                       "gam_pred", "gbr_pred"]]
        
        if len(feature_cols) == 0:
            st.warning("⚠️ No feature columns available for explainability (only predictions and targets in dataset).")
            st.info("The current dataset only contains model predictions and target values. To enable explainability, add raw feature data (e.g., student demographics, school resources, etc.).")
        else:
            from sklearn.linear_model import LinearRegression
            
            X = filtered_df[feature_cols].fillna(0)
            y = filtered_df[selected_reg_model_col]
            
            st.subheader("Feature Importance Overview")
            
            # Create simple model approximation for explanation
            model = LinearRegression()
            model.fit(X, y)
            
            # Feature importance based on coefficients
            feature_importance = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": np.abs(model.coef_),
            }).sort_values("Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance["Feature"], feature_importance["Importance"], color='steelblue')
            ax.set_xlabel("Absolute Coefficient Value")
            ax.set_title(f"Feature Importance: {selected_reg_model_label}")
            st.pyplot(fig)
            
            st.dataframe(feature_importance)
            
            # LIME for instance-level explanation
            st.subheader("Instance-Level Explanation (LIME)")
            
            instance_idx = st.slider("Select instance (row) to explain", 0, len(filtered_df) - 1, 0)
            
            try:
                import lime.tabular
                explainer = lime.tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=feature_cols,
                    class_names=["Score"],
                    mode="regression",
                    random_state=42
                )
                
                exp = explainer.explain_instance(X.iloc[instance_idx].values, model.predict, num_features=10)
                
                # Extract LIME explanation
                lime_exp_data = []
                for feature, weight in exp.as_list():
                    lime_exp_data.append({"Feature": feature, "Weight": weight})
                
                lime_df = pd.DataFrame(lime_exp_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Instance #{instance_idx}**")
                    st.write(f"- District: {filtered_df.iloc[instance_idx]['District']}")
                    st.write(f"- Grade: {filtered_df.iloc[instance_idx]['Tested Grade']}")
                    st.write(f"- Actual: {filtered_df.iloc[instance_idx]['y_true_cont']:.1f}")
                    st.write(f"- Predicted: {filtered_df.iloc[instance_idx][selected_reg_model_col]:.1f}")
                
                with col2:
                    st.write("**Top Features Contributing to Prediction**")
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    colors = ['green' if w > 0 else 'red' for w in lime_df["Weight"]]
                    ax.barh(lime_df["Feature"], lime_df["Weight"], color=colors, alpha=0.7)
                    ax.set_xlabel("Contribution")
                    ax.set_title(f"LIME Explanation (Instance {instance_idx})")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
    
    else:  # Classification
        clf_model_options = {**CLASSIFICATION_MODELS, **PROB_MODELS}
        selected_clf_model_label = st.selectbox("Choose classification model", list(clf_model_options.keys()), key="expl_clf")
        selected_clf_model_col = clf_model_options[selected_clf_model_label]
        
        feature_cols = [c for c in filtered_df.columns if c not in ["District", "Year", "y_true_binary", "y_true_cont", 
                                                                       "logreg_pred", "logreg_prob", "dt_pred", "mlp_pred", 
                                                                       "gam_pred", "gbr_pred"]]
        
        if len(feature_cols) == 0:
            st.warning("⚠️ No feature columns available for explainability.")
        else:
            from sklearn.linear_model import LogisticRegression
            
            X = filtered_df[feature_cols].fillna(0)
            y = filtered_df["y_true_binary"]
            
            st.subheader("Feature Importance Overview")
            
            # Train a simple classifier for explanation
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": np.abs(clf.coef_[0]),
            }).sort_values("Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(feature_importance["Feature"], feature_importance["Importance"], color='coral')
            ax.set_xlabel("Absolute Coefficient Value")
            ax.set_title(f"Feature Importance: {selected_clf_model_label}")
            st.pyplot(fig)
            
            st.dataframe(feature_importance)
            
            # LIME for instance-level
            st.subheader("Instance-Level Explanation (LIME)")
            
            instance_idx = st.slider("Select instance (row) to explain", 0, len(filtered_df) - 1, 0, key="clf_instance")
            
            try:
                import lime.tabular
                explainer = lime.tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=feature_cols,
                    class_names=["Fail", "Pass"],
                    mode="classification",
                    random_state=42
                )
                
                is_prob_col = "prob" in selected_clf_model_col.lower()
                if is_prob_col:
                    pred_fn = lambda x: np.column_stack([1 - clf.predict_proba(x)[:, 1], clf.predict_proba(x)[:, 1]])
                else:
                    pred_fn = lambda x: np.column_stack([1 - clf.predict_proba(x)[:, 1], clf.predict_proba(x)[:, 1]])
                
                exp = explainer.explain_instance(X.iloc[instance_idx].values, pred_fn, num_features=10)
                
                lime_exp_data = []
                for feature, weight in exp.as_list():
                    lime_exp_data.append({"Feature": feature, "Weight": weight})
                
                lime_df = pd.DataFrame(lime_exp_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Instance #{instance_idx}**")
                    st.write(f"- District: {filtered_df.iloc[instance_idx]['District']}")
                    st.write(f"- Grade: {filtered_df.iloc[instance_idx]['Tested Grade']}")
                    actual_label = "Pass" if filtered_df.iloc[instance_idx]['y_true_binary'] == 1 else "Fail"
                    st.write(f"- Actual: {actual_label}")
                    pred_label = "Pass" if filtered_df.iloc[instance_idx][selected_clf_model_col] >= 0.5 else "Fail"
                    st.write(f"- Predicted: {pred_label}")
                
                with col2:
                    st.write("**Top Features Contributing to Prediction**")
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    colors = ['green' if w > 0 else 'red' for w in lime_df["Weight"]]
                    ax.barh(lime_df["Feature"], lime_df["Weight"], color=colors, alpha=0.7)
                    ax.set_xlabel("Contribution")
                    ax.set_title(f"LIME Explanation (Instance {instance_idx})")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")


# ============= TAB 6: ANOMALY DETECTION =============
elif tab_choice == "Anomaly Detection":
    st.header("Anomaly Detection & Outlier Analysis")
    
    st.write("""
    Identify unusual districts, grades, or predictions that deviate from expected patterns.
    Uses Isolation Forest for unsupervised anomaly detection on prediction errors and input features.
    """)
    
    # Prepare features for anomaly detection
    feature_cols = [c for c in filtered_df.columns if c not in ["District", "Year", "y_true_binary", "y_true_cont", 
                                                                   "logreg_pred", "logreg_prob", "dt_pred", "mlp_pred", 
                                                                   "gam_pred", "gbr_pred"]]
    
    if len(feature_cols) > 0:
        X = filtered_df[feature_cols].fillna(0)
    else:
        # If no explicit features, use prediction residuals
        X = pd.DataFrame({
            "gam_residual": filtered_df["gam_pred"] - filtered_df["y_true_cont"],
            "gbr_residual": filtered_df["gbr_pred"] - filtered_df["y_true_cont"],
        })
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_scores = iso_forest.fit_predict(X)
    anomaly_probs = -iso_forest.score_samples(X)  # Negative scores = anomaly probability
    
    filtered_df_with_anomalies = filtered_df.copy()
    filtered_df_with_anomalies["anomaly"] = anomaly_scores
    filtered_df_with_anomalies["anomaly_score"] = anomaly_probs
    
    n_anomalies = (anomaly_scores == -1).sum()
    
    st.subheader(f"📊 Anomaly Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Anomalies Detected", n_anomalies, f"{n_anomalies / len(filtered_df) * 100:.1f}%")
    col2.metric("Normal Instances", (anomaly_scores == 1).sum())
    col3.metric("Avg Anomaly Score", f"{anomaly_probs.mean():.3f}")
    
    # Visualize anomaly scores
    st.subheader("Anomaly Score Distribution")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    normal_scores = anomaly_probs[anomaly_scores == 1]
    anomaly_scores_arr = anomaly_probs[anomaly_scores == -1]
    
    ax.hist(normal_scores, bins=30, alpha=0.6, label="Normal", color='blue')
    if len(anomaly_scores_arr) > 0:
        ax.hist(anomaly_scores_arr, bins=20, alpha=0.6, label="Anomaly", color='red')
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Anomaly Scores")
    ax.legend()
    st.pyplot(fig)
    
    # Show anomalies
    st.subheader("Detected Anomalies")
    
    anomalies_df = filtered_df_with_anomalies[filtered_df_with_anomalies["anomaly"] == -1][
        ["District", "Tested Grade", "y_true_cont", "gam_pred", "gbr_pred", "y_true_binary", "logreg_pred", "anomaly_score"]
    ].sort_values("anomaly_score", ascending=False).head(20)
    
    if len(anomalies_df) > 0:
        st.dataframe(anomalies_df)
        
        st.write("**Interpretation:** Instances with high anomaly scores are unusual and warrant investigation.")
    else:
        st.info("No anomalies detected with current settings.")
    
    # Anomalies by district
    st.subheader("Anomaly Concentration by District")
    
    district_anomalies = filtered_df_with_anomalies.groupby("District").agg({
        "anomaly": lambda x: (x == -1).sum(),
    }).rename(columns={"anomaly": "anomaly_count"})
    
    district_anomalies["total"] = filtered_df_with_anomalies.groupby("District").size()
    district_anomalies["anomaly_pct"] = (district_anomalies["anomaly_count"] / district_anomalies["total"] * 100).round(1)
    district_anomalies = district_anomalies.sort_values("anomaly_count", ascending=False)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    top_districts = district_anomalies.head(15)
    ax.barh(top_districts.index, top_districts["anomaly_count"], color='crimson', alpha=0.7)
    ax.set_xlabel("Number of Anomalies")
    ax.set_title("Top 15 Districts with Most Anomalies")
    st.pyplot(fig)
    
    st.dataframe(district_anomalies.head(20))
    
    # Anomalies by grade
    st.subheader("Anomaly Concentration by Grade")
    
    grade_anomalies = filtered_df_with_anomalies.groupby("Tested Grade").agg({
        "anomaly": lambda x: (x == -1).sum(),
    }).rename(columns={"anomaly": "anomaly_count"})
    
    grade_anomalies["total"] = filtered_df_with_anomalies.groupby("Tested Grade").size()
    grade_anomalies["anomaly_pct"] = (grade_anomalies["anomaly_count"] / grade_anomalies["total"] * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(grade_anomalies.index, grade_anomalies["anomaly_count"], color='orange', alpha=0.7)
    ax.set_xlabel("Tested Grade")
    ax.set_ylabel("Number of Anomalies")
    ax.set_title("Anomalies by Grade")
    st.pyplot(fig)
    
    st.dataframe(grade_anomalies)
    
    # Feature-level anomaly analysis
    st.subheader("🔍 Feature-Level Anomaly Insights")
    
    st.write("For anomalies detected, which features are most unusual?")
    
    if len(feature_cols) > 0 and len(anomalies_df) > 0:
        anomaly_indices = filtered_df_with_anomalies[filtered_df_with_anomalies["anomaly"] == -1].index
        
        feature_stats = pd.DataFrame({
            "Normal_Mean": X.iloc[anomaly_scores == 1].mean(),
            "Normal_Std": X.iloc[anomaly_scores == 1].std(),
            "Anomaly_Mean": X.iloc[anomaly_scores == -1].mean(),
        })
        
        feature_stats["Deviation"] = np.abs(feature_stats["Anomaly_Mean"] - feature_stats["Normal_Mean"]) / (feature_stats["Normal_Std"] + 1e-6)
        feature_stats = feature_stats.sort_values("Deviation", ascending=False)
        
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(feature_stats.index, feature_stats["Deviation"], color='purple', alpha=0.7)
        ax.set_xlabel("Std Dev from Normal")
        ax.set_title("Features Most Deviated in Anomalies")
        st.pyplot(fig)
        
        st.dataframe(feature_stats)
elif tab_choice == "Testing Tab":
    st.header("This is a testing tab.")