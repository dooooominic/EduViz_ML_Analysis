import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 50)

eval_df = pd.read_csv('data/model_eval_2025.csv')
print('\nRows in eval_df:', len(eval_df))

# Regression
reg_models = {'GAM':'gam_pred','GBR':'gbr_pred'}
print('\nRegression metrics:')
for name,col in reg_models.items():
    y_true = eval_df['y_true_cont']
    y_pred = eval_df[col]
    mae = mean_absolute_error(y_true,y_pred)
    rmse = (mean_squared_error(y_true,y_pred))**0.5
    r2 = r2_score(y_true,y_pred)
    print(f"  {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# Classification
clf_models = {'LogReg':'logreg_pred','DecisionTree':'dt_pred','MLP':'mlp_pred'}
print('\nClassification metrics:')
for name,col in clf_models.items():
    y_true = eval_df['y_true_binary']
    y_pred = eval_df[col].astype(int)
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred, zero_division=0)
    rec = recall_score(y_true,y_pred, zero_division=0)
    f1 = f1_score(y_true,y_pred, zero_division=0)
    print(f"  {name}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

# Prob model AUC
if 'logreg_prob' in eval_df.columns:
    try:
        auc = roc_auc_score(eval_df['y_true_binary'], eval_df['logreg_prob'])
        print(f"\nLogistic Regression (prob) AUC: {auc:.3f}")
    except Exception as e:
        print('\nLogReg prob AUC: error computing AUC:', e)

# Confusion matrix for Logistic Regression (threshold 0.5 if prob exists, else pred column)
if 'logreg_prob' in eval_df.columns:
    y_pred = (eval_df['logreg_prob']>=0.5).astype(int)
else:
    y_pred = eval_df['logreg_pred'].astype(int)
cm = confusion_matrix(eval_df['y_true_binary'], y_pred)
print('\nLogReg confusion matrix (rows=true: [0,1], cols=pred [0,1]):')
print(cm)

# Model agreement among logreg/dt/mlp
agree = ((eval_df['logreg_pred'].astype(int)==eval_df['dt_pred'].astype(int)) & (eval_df['dt_pred'].astype(int)==eval_df['mlp_pred'].astype(int))).sum()
print(f"\nAll three classifiers agree on {agree} / {len(eval_df)} rows ({agree/len(eval_df)*100:.1f}%)")

# District stats
district_stats = eval_df.groupby('District').agg({'y_true_cont':['mean','std','count'], 'y_true_binary':'mean'})
district_stats.columns=['Avg Score','Std Dev','Count','Pass Mean']
district_stats = district_stats.reset_index()
# Count districts with Std Dev 0 or NaN
zero_std = district_stats['Std Dev'].isna().sum() + (district_stats['Std Dev']==0).sum()
print(f"\nDistricts: {district_stats['District'].nunique()}, districts with Std Dev 0 or NaN: {zero_std}")

# Top/bottom districts by Avg Score
top5 = district_stats.sort_values('Avg Score', ascending=False).head(5)
bot5 = district_stats.sort_values('Avg Score', ascending=True).head(5)
print('\nTop 5 districts by Avg Score:')
print(top5[['District','Avg Score','Std Dev','Count','Pass Mean']].to_string(index=False))
print('\nBottom 5 districts by Avg Score:')
print(bot5[['District','Avg Score','Std Dev','Count','Pass Mean']].to_string(index=False))

# District-level MAE for GAM and GBR
mae_per_district = eval_df.groupby('District').apply(lambda df: pd.Series({'GAM_MAE': mean_absolute_error(df['y_true_cont'], df['gam_pred']), 'GBR_MAE': mean_absolute_error(df['y_true_cont'], df['gbr_pred']), 'Count': len(df)}))
mae_per_district = mae_per_district.reset_index()
worst_gam = mae_per_district.sort_values('GAM_MAE', ascending=False).head(5)
worst_gbr = mae_per_district.sort_values('GBR_MAE', ascending=False).head(5)
print('\nTop 5 districts by GAM MAE:')
print(worst_gam.to_string(index=False))
print('\nTop 5 districts by GBR MAE:')
print(worst_gbr.to_string(index=False))

# LIME top5 analysis
try:
    lime = pd.read_csv('data/lime_top5_per_model.csv')
    print('\nLIME top5 rows:', len(lime))
    # Which features dominate? proportion that are district indicators
    lime['is_district'] = lime['feature'].str.startswith('District_')
    pct_district = lime['is_district'].mean()*100
    global_count = lime['model'].str.contains('GlobalImportance', na=False).sum()
    print(f"Features that are district indicators: {pct_district:.1f}% of top entries")
    print(f"GlobalImportance rows present: {global_count}")
    # Top global features if any
    if global_count>0:
        g = (lime[lime['model'].str.contains('GlobalImportance', na=False)].groupby('feature', as_index=False)['importance'].sum().sort_values('importance', ascending=False).head(10))
        print('\nTop global importance features:')
        print(g.to_string(index=False))
    # Top aggregated local features
    local = (lime[~lime['model'].str.contains('GlobalImportance', na=False)].groupby('feature', as_index=False)['importance'].sum().sort_values('importance', ascending=False).head(10))
    print('\nTop aggregated local LIME features:')
    print(local.to_string(index=False))
except Exception as e:
    print('Could not read lime_top5_per_model.csv:', e)

print('\nDone.')
