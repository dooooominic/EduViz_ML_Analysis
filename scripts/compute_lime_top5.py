import pandas as pd

"""Compute and save top-5 LIME features per model.
Reads: data/lime_importance_scores.csv
Writes: data/lime_top5_per_model.csv with columns: model, feature, importance
"""

def main():
    src = "data/lime_importance_scores.csv"
    dst = "data/lime_top5_per_model.csv"
    df = pd.read_csv(src)
    if df.empty:
        print(f"Source {src} is empty or missing.")
        return
    # ensure numeric
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    df["importance_abs"] = df["importance"].abs()

    agg = df.groupby(["model", "feature"], as_index=False)["importance_abs"].mean()
    agg = agg.rename(columns={"importance_abs": "importance"})
    # For each model, pick top 5 features
    top5_list = []
    for model, g in agg.groupby("model"):
        top5 = g.sort_values("importance", ascending=False).head(5).copy()
        top5["model"] = model
        top5_list.append(top5)
    if len(top5_list) == 0:
        print("No models found in LIME file.")
        return
    top5_df = pd.concat(top5_list, ignore_index=True)
    top5_df.to_csv(dst, index=False)
    print(f"Saved top-5 per model to {dst}")

if __name__ == '__main__':
    main()
