import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import IsolationForest
import lime
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import warnings
import plotly.express as px
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

@st.cache_data
def load_lime_importances(path: str = "data/lime_importance_scores.csv"):
    """Load precomputed LIME importance scores and normalize types.

    Expects CSV with columns: `instance_idx, model, feature, importance`.
    Returns DataFrame with a numeric `importance` column and an additional
    `importance_abs` column for aggregation.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["instance_idx", "model", "feature", "importance"])
    if "importance" not in df.columns:
        return pd.DataFrame(columns=["instance_idx", "model", "feature", "importance"])
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    df["importance_abs"] = df["importance"].abs()
    return df


@st.cache_data
def load_lime_top5(path: str = "data/lime_top5_per_model.csv"):
    """Load precomputed top-5 LIME features per model.

    Expects CSV with columns: `model, feature, importance`.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    if "importance" not in df.columns:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    return df

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
    ["Overview & Model Comparison", "Regression Deep-Dive", "Classification Deep-Dive", "Trend Analysis", "Explainability (Lime)"],
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
    
        clf_metrics[label] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression Models")
        reg_df = pd.DataFrame(reg_metrics).T
        st.dataframe(reg_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightyellow'))
        
    with col2:
        st.subheader("Classification Models")
        clf_df = pd.DataFrame(clf_metrics).T
        st.dataframe(clf_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightyellow'))
    
    # Heatmap comparison
    st.subheader("Metric Heatmap: Regression Models")
    fig = px.imshow(reg_df, labels=dict(color="Score"), color_continuous_scale="YlGn", aspect="auto", text_auto=".3f")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Metric Heatmap: Classification Models")
    fig = px.imshow(clf_df, labels=dict(color="Score"), color_continuous_scale="YlGn", aspect="auto", text_auto=".3f")
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("Key Insights")
    best_reg = reg_df["R²"].idxmax()
    best_clf = clf_df["F1"].idxmax()
    st.write(f"""
    - **Best Regression Model (by R²):** {best_reg} ({reg_df.loc[best_reg, "R²"]:.3f})
    - **Best Classification Model (by F1):** {best_clf} ({clf_df.loc[best_clf, "F1"]:.3f})
    - **Data filtered to:** {len(filtered_df)} districts/grades
    """)

    # Geographic Performance Disparities
    st.subheader("📍 Geographic Performance Disparities")
    st.write("Examine performance variation across Texas school districts.")
    
    district_stats = filtered_df.groupby("District").agg({
        "y_true_cont": ["mean", "std"],
        "y_true_binary": "mean"
    }).round(2)
    district_stats.columns = ["Avg Score", "Std Dev", "Pass Rate"]
    district_stats["Pass Rate"] = (district_stats["Pass Rate"] * 100).round(1)
    district_stats = district_stats.sort_values("Avg Score", ascending=False)

    # Prepare display copy that excludes districts with 0 or missing Std Dev
    display_stats = district_stats.copy()
    display_stats["Std Dev"] = pd.to_numeric(display_stats["Std Dev"], errors="coerce")
    filtered_out = display_stats[(display_stats["Std Dev"].isna()) | (display_stats["Std Dev"] == 0)].shape[0]
    display_stats = display_stats[~(display_stats["Std Dev"].isna()) & (display_stats["Std Dev"] != 0)]
    # convert Pass Rate to formatted string for display (do this after filtering)
    district_stats["Pass Rate"] = district_stats["Pass Rate"].astype(str)+ "%"
    display_stats["Pass Rate"] = display_stats["Pass Rate"].astype(str)+ "%"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Highest-Performing Districts**")
        # prefer display_stats (filtered) for charts; fall back to full stats if empty
        if display_stats.empty:
            if filtered_out > 0:
                st.info(f"No districts with non-zero Std Dev available; {filtered_out} districts were excluded from charts.")
            top_districts = district_stats.head(10).reset_index().sort_values("Avg Score")
        else:
            #if filtered_out > 0:
                #st.info(f"Dropped {filtered_out} districts with 0 or missing Std Dev from charts.")
            top_districts = display_stats.head(10).reset_index().sort_values("Avg Score")
        top_chart = px.bar(top_districts, x="Avg Score", y="District", orientation="h", 
                          color="Avg Score", color_continuous_scale="Greens",
                          labels={"Avg Score": "Average Score", "District": ""})
        top_chart.update_layout(height=350, showlegend=False, xaxis=dict(range=[0, 100]))
        st.plotly_chart(top_chart, use_container_width=True)
        st.dataframe(top_districts.set_index('District')[['Avg Score','Std Dev','Pass Rate']], use_container_width=True)
    
    with col2:
        st.write("**Bottom 10 Lowest-Performing Districts**")
        if display_stats.empty:
            bottom_districts = district_stats.tail(10).reset_index()
        else:
            bottom_districts = display_stats.tail(10).reset_index()
        bottom_chart = px.bar(bottom_districts, x="Avg Score", y="District", orientation="h",
                             color="Avg Score", color_continuous_scale="Reds",
                             labels={"Avg Score": "Average Score", "District": ""})
        bottom_chart.update_layout(height=350, showlegend=False, xaxis=dict(range=[0, 100]))
        st.plotly_chart(bottom_chart, use_container_width=True)
        st.dataframe(bottom_districts.set_index('District')[['Avg Score','Std Dev','Pass Rate']], use_container_width=True)
    
    st.write("**Key Observations:**")
    obs_stats = display_stats if not display_stats.empty else district_stats
    if not obs_stats.empty:
        obs_sorted = obs_stats.sort_values("Avg Score", ascending=False)
        top_avg = obs_sorted.iloc[0]["Avg Score"]
        bottom_avg = obs_sorted.iloc[-1]["Avg Score"]
        gap = top_avg - bottom_avg
        st.write(f"- **Performance Gap:** {gap:.1f} points between top and bottom performers")
        st.write(f"- **Range:** {bottom_avg:.1f} to {top_avg:.1f}")
        st.write(f"- **Total Districts Analyzed (used in charts):** {len(obs_stats)}")
        if filtered_out > 0:
            st.write(f"- **Note:** {filtered_out} districts were excluded from charts because Std Dev was 0 or missing.")
    else:
        st.write("No district statistics available to compute key observations after filtering.")


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
        scatter_fig = px.scatter(pd.DataFrame({"Actual": y_true, "Predicted": y_pred}), 
                                 x="Actual", y="Predicted", opacity=0.6, trendline="ols")
        scatter_fig.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), 
                             x1=y_true.max(), y1=y_true.max(), line=dict(color="red", dash="dash"))
        scatter_fig.update_layout(height=300, width=400)
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    with col2:
        st.subheader("Residual Distribution")
        residuals = y_pred - y_true
        hist_fig = px.histogram(residuals, nbins=30, labels={"value": "Residual", "count": "Frequency"})
        hist_fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
        hist_fig.update_layout(height=300, width=400)
        st.plotly_chart(hist_fig, use_container_width=True)
    
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
    
    bar_fig = px.bar(residuals_by_group.reset_index(), x="Mean Residual", y=group_col, 
                     orientation="h", color="Mean Residual", color_continuous_scale="RdYlGn")
    bar_fig.update_layout(height=400, coloraxis_colorbar=dict(thickness=20))
    st.plotly_chart(bar_fig, use_container_width=True)
    
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
        cm_df = pd.DataFrame(cm, index=["Fail", "Pass"], columns=["Fail", "Pass"])
        heatmap_fig = px.imshow(cm_df, labels=dict(color="Count"), text_auto=True, color_continuous_scale="Blues", aspect="auto")
        heatmap_fig.update_layout(height=300, width=300)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Distribution")
        if is_prob:
            dist_data = pd.DataFrame({
                "Probability": list(y_prob_or_pred[y_true == 0]) + list(y_prob_or_pred[y_true == 1]),
                "Actual": ["Fail"]*len(y_prob_or_pred[y_true == 0]) + ["Pass"]*len(y_prob_or_pred[y_true == 1])
            })
            hist_fig = px.histogram(dist_data, x="Probability", color="Actual", nbins=30, barmode="overlay")
            hist_fig.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Decision Threshold")
            hist_fig.update_layout(height=300, width=300)
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            counts = pd.Series(y_pred).value_counts()
            bar_data = pd.DataFrame({"Label": ["Fail", "Pass"], "Count": [counts.get(0, 0), counts.get(1, 0)]})
            bar_fig = px.bar(bar_data, x="Label", y="Count", color="Label", color_discrete_map={"Fail": "red", "Pass": "green"})
            bar_fig.update_layout(height=300, width=300)
            st.plotly_chart(bar_fig, use_container_width=True)
    
    # ROC curve (if probability)
    if is_prob:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_or_pred)
        auc_score = roc_auc_score(y_true, y_prob_or_pred)
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        roc_fig = px.line(roc_data, x="FPR", y="TPR", 
                         labels={"FPR": "False Positive Rate", "TPR": "True Positive Rate"},
                         title=f"ROC Curve (AUC={auc_score:.3f})")
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
        roc_fig.update_layout(height=350, width=350)
        st.plotly_chart(roc_fig, use_container_width=True)
    
    # Error by group
    st.subheader("Classification Performance by Group")
    group_col = st.radio("Group by:", ["Tested Grade", "District"], key="clf_group")
    
    perf_by_group = filtered_df.groupby(group_col).apply(
        lambda df: pd.Series({
            "Accuracy": accuracy_score(df["y_true_binary"], (df[selected_clf_model_col] >= 0.5).astype(int) if is_prob else df[selected_clf_model_col].astype(int)),
            "Count": len(df),
        })
    ).sort_values("Accuracy")
    
    perf_fig = px.bar(perf_by_group.reset_index(), x="Accuracy", y=group_col, 
                     orientation="h", color="Accuracy", color_continuous_scale="RdYlGn",
                     range_x=[0, 1])
    perf_fig.update_layout(height=400, coloraxis_colorbar=dict(thickness=20))
    st.plotly_chart(perf_fig, use_container_width=True)
    
    #st.dataframe(perf_by_group)


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
    scatter_data = pd.DataFrame({"GAM": gam_pred, "GBR": gbr_pred})
    scatter_fig = px.scatter(scatter_data, x="GAM", y="GBR", opacity=0.6, trendline="ols")
    scatter_fig.add_shape(type="line", x0=gam_pred.min(), y0=gam_pred.min(), 
                         x1=gam_pred.max(), y1=gam_pred.max(), line=dict(color="red", dash="dash"))
    scatter_fig.update_layout(height=500, width=500)
    
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Classification agreement heatmap
    st.subheader("Classification Model Agreement Matrix")
    clf_models = {"LogReg": logreg, "DecisionTree": dt, "MLP": mlp}
    agreement_matrix = pd.DataFrame({
        name: [100 * (clf_models[name] == clf_models[other]).sum() / len(filtered_df) 
               for other in clf_models.keys()]
        for name in clf_models.keys()
    })
    
    agree_fig = px.imshow(agreement_matrix, labels=dict(color="%"), color_continuous_scale="RdYlGn", 
                         text_auto=".1f", aspect="auto", color_continuous_midpoint=50)
    agree_fig.update_layout(height=350, width=350)
    st.plotly_chart(agree_fig, use_container_width=True)
    
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
    
    line_fig = px.line(residuals_df, x="Grade", y="Mean Residual", color="Model", markers=True)
    line_fig.add_hline(y=0, line_dash="dash", line_color="black")
    line_fig.update_layout(height=350, hovermode="x unified")
    st.plotly_chart(line_fig, use_container_width=True)
    
    # Scatter: True vs predictions for both regression models
    st.subheader("Prediction Landscape: True vs Both Regression Models")
    pred_data = pd.DataFrame({
        "True": list(filtered_df["y_true_cont"]) * 2,
        "Predicted": list(filtered_df["gam_pred"]) + list(filtered_df["gbr_pred"]),
        "Model": ["GAM"]*len(filtered_df) + ["GBR"]*len(filtered_df)
    })
    pred_fig = px.scatter(pred_data, x="True", y="Predicted", color="Model", opacity=0.6, trendline="ols")
    pred_fig.update_layout(height=400, hovermode="closest")
    st.plotly_chart(pred_fig, use_container_width=True)
    
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
elif tab_choice == "Explainability (Lime)":
    st.header("Model Explainability LIME")
    
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
            
            st.subheader("Feature Importance Overview (Precomputed Top-5 LIME)")
            top5_df = load_lime_top5()
            
            if top5_df.empty:
                st.warning("No precomputed top-5 LIME file found at `data/lime_top5_per_model.csv`. Run `compute_lime_top5.py`.")
            else:
                # Allow user to choose how to view importances: per-model or combined local/global
                view_choice = st.selectbox(
                    "View top-5 importances:",
                    ["Per model", "Combined Local (LIME)", "Combined Global (surrogate)"],
                    key="top5_view_reg",
                )

                if view_choice == "Per model":
                    # Show model comparison by sum of top-5 importances
                    model_comp = (
                        top5_df.groupby("model", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
                    )
                    fig_model_comp = px.bar(model_comp, x="model", y="importance", color="model")
                    fig_model_comp.update_layout(showlegend=False, height=300, xaxis_title="Model", yaxis_title="Sum of Top-5 Importances")
                    st.plotly_chart(fig_model_comp, use_container_width=True)

                    st.write("**Top 5 Features Per Model**")
                    models = top5_df["model"].unique().tolist()
                    # Render models in rows of up to 3 columns to improve label wrapping
                    for row_start in range(0, len(models), 3):
                        row_models = models[row_start: row_start + 3]
                        cols = st.columns(len(row_models))
                        for col_idx, m in enumerate(row_models):
                            with cols[col_idx]:
                                st.markdown(f"**{m}**")
                                top5 = top5_df[top5_df["model"] == m].sort_values("importance", ascending=True)
                                bar = px.bar(top5, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
                                bar.update_layout(height=280, showlegend=False, margin=dict(l=20, r=10, t=20, b=10))
                                st.plotly_chart(bar, use_container_width=True)

                    # Allow user to pick a model to view its top-5 features (avoid showing all models at once)
                    selected_top5_model = st.selectbox("Choose model to view top-5 features", models, key="top5_choice_reg")
                    sel_df = top5_df[top5_df["model"] == selected_top5_model].sort_values("importance", ascending=False)
                    st.write(f"**Top-5 LIME features for {selected_top5_model}**")
                    st.dataframe(sel_df)
                else:
                    # Combined view: aggregate across either local LIME entries or global surrogate importance entries
                    if view_choice == "Combined Local (LIME)":
                        combined_df = top5_df[~top5_df["model"].str.contains("GlobalImportance", na=False)].copy()
                        title = "Combined Local LIME importances"
                    else:
                        combined_df = top5_df[top5_df["model"].str.contains("GlobalImportance", na=False)].copy()
                        title = "Combined Global (surrogate) importances"

                    if combined_df.empty:
                        st.warning("No data available for the selected combined view.")
                    else:
                        agg = (
                            combined_df.groupby("feature", as_index=False)["importance"].sum()
                            .sort_values("importance", ascending=False)
                            .head(30)
                        )
                        st.subheader(title)
                        agg_chart = px.bar(agg, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
                        agg_chart.update_layout(height=500, showlegend=False, margin=dict(l=20, r=10, t=20, b=10))
                        st.plotly_chart(agg_chart, use_container_width=True)
                        st.dataframe(agg)
    
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
            
            st.subheader("Feature Importance Overview (Precomputed Top-5 LIME)")
            top5_df = load_lime_top5()
            
            if top5_df.empty:
                st.warning("No precomputed top-5 LIME file found at `data/lime_top5_per_model.csv`. Run `compute_lime_top5.py`.")
            else:
                view_choice = st.selectbox(
                    "View top-5 importances:",
                    ["Per model", "Combined Local (LIME)", "Combined Global (surrogate)"],
                    key="top5_view_clf",
                )

                if view_choice == "Per model":
                    model_comp = (
                        top5_df.groupby("model", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
                    )
                    fig_model_comp = px.bar(model_comp, x="model", y="importance", color="model")
                    fig_model_comp.update_layout(showlegend=False, height=300, xaxis_title="Model", yaxis_title="Sum of Top-5 Importances")
                    st.plotly_chart(fig_model_comp, use_container_width=True)

                    st.write("**Top 5 Features Per Model**")
                    models = top5_df["model"].unique().tolist()
                    # Render models in rows of up to 3 columns to improve label wrapping
                    for row_start in range(0, len(models), 3):
                        row_models = models[row_start: row_start + 3]
                        cols = st.columns(len(row_models))
                        for col_idx, m in enumerate(row_models):
                            with cols[col_idx]:
                                st.markdown(f"**{m}**")
                                top5 = top5_df[top5_df["model"] == m].sort_values("importance", ascending=True)
                                bar = px.bar(top5, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
                                bar.update_layout(height=280, showlegend=False, margin=dict(l=20, r=10, t=20, b=10))
                                st.plotly_chart(bar, use_container_width=True)

                    # Allow user to pick a model to view its top-5 features (avoid showing all models at once)
                    selected_top5_model = st.selectbox("Choose model to view top-5 features", models, key="top5_choice_clf")
                    sel_df = top5_df[top5_df["model"] == selected_top5_model].sort_values("importance", ascending=False)
                    st.write(f"**Top-5 LIME features for {selected_top5_model}**")
                    st.dataframe(sel_df)
                else:
                    if view_choice == "Combined Local (LIME)":
                        combined_df = top5_df[~top5_df["model"].str.contains("GlobalImportance", na=False)].copy()
                        title = "Combined Local LIME importances"
                    else:
                        combined_df = top5_df[top5_df["model"].str.contains("GlobalImportance", na=False)].copy()
                        title = "Combined Global (surrogate) importances"

                    if combined_df.empty:
                        st.warning("No data available for the selected combined view.")
                    else:
                        agg = (
                            combined_df.groupby("feature", as_index=False)["importance"].sum()
                            .sort_values("importance", ascending=False)
                            .head(30)
                        )
                        st.subheader(title)
                        agg_chart = px.bar(agg, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
                        agg_chart.update_layout(height=500, showlegend=False, margin=dict(l=20, r=10, t=20, b=10))
                        st.plotly_chart(agg_chart, use_container_width=True)
                        st.dataframe(agg)
