import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.calibration import calibration_curve

DATA_PATH = "data/model_eval_2025.csv"
OUT_DIR = "analysis_outputs"

os.makedirs(OUT_DIR, exist_ok=True)


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def classification_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = None
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def plot_pred_vs_true(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=40, alpha=0.7)
    mins = min(y_true.min(), y_pred.min())
    maxs = max(y_true.max(), y_pred.max())
    plt.plot([mins, maxs], [mins, maxs], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Pred vs True: {model_name}")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"pred_vs_true_{model_name}.png")
    plt.savefig(p)
    plt.close()


def plot_residuals(y_true, y_pred, model_name):
    res = y_pred - y_true
    plt.figure(figsize=(6, 4))
    sns.histplot(res, bins=40, kde=True)
    plt.xlabel("Residual (pred - true)")
    plt.title(f"Residuals: {model_name}")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"residuals_{model_name}.png")
    plt.savefig(p)
    plt.close()


def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"confusion_{model_name}.png")
    plt.savefig(p)
    plt.close()


def plot_calibration(y_true, y_prob, model_name):
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(figsize=(5, 5))
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration: {model_name}")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f"calibration_{model_name}.png")
        plt.savefig(p)
        plt.close()
    except Exception as e:
        print(f"Could not plot calibration for {model_name}: {e}")


def summarize_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        else:
            print(f"{k}: {v}")


def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded", len(df), "rows from", DATA_PATH)

    # Identify columns
    # Ground truth
    if "y_true_cont" not in df.columns or "y_true_binary" not in df.columns:
        raise ValueError("CSV must contain 'y_true_cont' and 'y_true_binary' columns")

    y_true_cont = df["y_true_cont"].astype(float)
    y_true_bin = df["y_true_binary"].astype(int)

    # Model outputs observed in the CSV
    # binary preds: logreg_pred, dt_pred, mlp_pred
    bin_models = [c for c in ["logreg_pred", "dt_pred", "mlp_pred"] if c in df.columns]
    # regression preds: gam_pred, gbr_pred
    reg_models = [c for c in ["gam_pred", "gbr_pred"] if c in df.columns]
    # probabilities: logreg_prob maybe
    prob_models = [c for c in ["logreg_prob"] if c in df.columns]

    results = {"regression": {}, "classification": {}}

    # Regression evaluation
    for m in reg_models:
        preds = pd.to_numeric(df[m], errors="coerce")
        mask = ~preds.isna()
        if mask.sum() == 0:
            continue
        metrics = regression_metrics(y_true_cont[mask], preds[mask])
        results["regression"][m] = metrics
        plot_pred_vs_true(y_true_cont[mask], preds[mask], m)
        plot_residuals(y_true_cont[mask], preds[mask], m)

    # Classification evaluation
    # For models with binary predictions
    for m in bin_models:
        preds = pd.to_numeric(df[m], errors="coerce").fillna(0).astype(int)
        metrics = classification_metrics(y_true_bin, preds)
        results["classification"][m] = metrics
        plot_confusion(y_true_bin, preds, m)

    # For models with probability scores, compute metrics using thresholds and AUC & calibration
    for m in prob_models:
        probs = pd.to_numeric(df[m], errors="coerce")
        # default threshold 0.5
        preds = (probs >= 0.5).astype(int)
        metrics = classification_metrics(y_true_bin, preds, y_score=probs)
        results["classification"][m] = metrics
        plot_confusion(y_true_bin, preds, m)
        plot_calibration(y_true_bin, probs, m)
        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC {m} (AUC={metrics['auc']:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve: {m}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"roc_{m}.png"))
            plt.close()
        except Exception:
            pass

    # Print summary
    summarize_metrics("Regression Results", results["regression"]) 
    summarize_metrics("Classification Results", results["classification"]) 

    # Additional grouping: metrics by grade for regression models
    grades = df["Tested Grade"].unique() if "Tested Grade" in df.columns else []
    if len(reg_models) > 0 and len(grades) > 0:
        grade_summary = {}
        for m in reg_models:
            grade_summary[m] = {}
            for g in sorted(grades):
                mask = df["Tested Grade"] == g
                preds = pd.to_numeric(df.loc[mask, m], errors="coerce")
                if preds.dropna().empty:
                    continue
                grade_summary[m][f"grade_{g}"] = regression_metrics(df.loc[mask, "y_true_cont"].astype(float), preds)
        # save grade summary to CSV
        outpath = os.path.join(OUT_DIR, "grade_regression_summary.csv")
        rows = []
        for m, gdict in grade_summary.items():
            for g, met in gdict.items():
                rows.append({"model": m, "grade": g, **met})
        if rows:
            pd.DataFrame(rows).to_csv(outpath, index=False)
            print("Saved grade-level regression summary to", outpath)

    print("\nAll plots and outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
