"""
================================================================================
  STROKE PREDICTION — COMPLETE TRAINING SCRIPT (No TensorFlow)
  Run: python train.py
  Author: Aqsa Siddiqui
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, joblib, time, os
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.metrics            import (accuracy_score, precision_score, recall_score,
                                        f1_score, roc_auc_score, confusion_matrix,
                                        classification_report, roc_curve,
                                        precision_recall_curve, average_precision_score,
                                        brier_score_loss)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection  import mutual_info_classif
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import (RandomForestClassifier, GradientBoostingClassifier,
                                        StackingClassifier)
from xgboost   import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

SEED = 42
np.random.seed(SEED)

sns.set_style("whitegrid")
PALETTE = {"stroke": "#e74c3c", "no_stroke": "#2ecc71", "accent": "#3498db"}
plt.rcParams.update({"figure.figsize": (14, 6)})

print("=" * 70)
print("  STROKE PREDICTION — ML TRAINING SCRIPT")
print("=" * 70)
print("✅ All libraries loaded!\n")


# ==============================================================================
# STEP 1 — LOAD DATA
# ==============================================================================
print("=" * 70)
print("STEP 1 — LOADING DATA")
print("=" * 70)

CSV_FILE = "healthcare-dataset-stroke-data.csv"

if not os.path.exists(CSV_FILE):
    print(f"\n❌ ERROR: '{CSV_FILE}' not found!")
    print("   Place the CSV in the same folder as train.py")
    print("   Download: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
    exit()

df = pd.read_csv(CSV_FILE)
print(f"\n✅ Dataset loaded!")
print(f"   Shape      : {df.shape}")
print(f"   Duplicates : {df.duplicated().sum()}")
print(f"\n   Missing values:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())
print(f"\n   Target distribution:")
print(df['stroke'].value_counts().to_string())
print(f"\n   Stroke Rate    : {df['stroke'].mean()*100:.2f}%")
print(f"   Imbalance Ratio: {df['stroke'].value_counts()[0]/df['stroke'].value_counts()[1]:.1f} : 1")


# ==============================================================================
# STEP 2 — EDA
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 3, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
vc  = df['stroke'].value_counts()
bars = ax1.bar(['No Stroke','Stroke'], vc.values,
               color=[PALETTE['no_stroke'], PALETTE['stroke']], edgecolor='black')
for bar, cnt in zip(bars, vc.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
             f'{cnt:,}\n({cnt/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
ax1.set_title('Target Distribution', fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
for lbl, grp in df.groupby('stroke'):
    grp['age'].plot.kde(ax=ax2, label=['No Stroke','Stroke'][lbl],
                        color=[PALETTE['no_stroke'],PALETTE['stroke']][lbl], linewidth=2)
ax2.set_title('Age Distribution (KDE)', fontweight='bold')
ax2.legend(); ax2.set_xlabel('Age')

ax3 = fig.add_subplot(gs[0, 2])
df.boxplot(column='bmi', by='stroke', ax=ax3)
ax3.set_title('BMI by Stroke', fontweight='bold')
ax3.set_xticklabels(['No Stroke','Stroke']); plt.sca(ax3); plt.suptitle('')

ax4 = fig.add_subplot(gs[1, 0])
for lbl, grp in df.groupby('stroke'):
    grp['avg_glucose_level'].plot.kde(ax=ax4, label=['No Stroke','Stroke'][lbl],
                                      color=[PALETTE['no_stroke'],PALETTE['stroke']][lbl], linewidth=2)
ax4.set_title('Glucose Distribution (KDE)', fontweight='bold'); ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
work_stroke = pd.crosstab(df['work_type'], df['stroke'], normalize='index') * 100
work_stroke.plot(kind='bar', stacked=True, ax=ax5,
                 color=[PALETTE['no_stroke'],PALETTE['stroke']], edgecolor='black')
ax5.set_title('Stroke Rate by Work Type', fontweight='bold')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')

ax6 = fig.add_subplot(gs[1, 2])
pd.DataFrame({
    'Hypertension' : df.groupby('hypertension')['stroke'].mean() * 100,
    'Heart Disease': df.groupby('heart_disease')['stroke'].mean() * 100,
}, index=['No','Yes']).plot(kind='bar', ax=ax6,
                             color=[PALETTE['accent'],'#9b59b6'], edgecolor='black')
ax6.set_title('Stroke Rate: Comorbidities', fontweight='bold')
ax6.set_xticklabels(['No','Yes'], rotation=0)

ax7 = fig.add_subplot(gs[2, 0:2])
for lbl, color, label in [(0, PALETTE['no_stroke'],'No Stroke'),(1, PALETTE['stroke'],'Stroke')]:
    sub = df[df['stroke']==lbl]
    ax7.scatter(sub['age'], sub['avg_glucose_level'], c=color, alpha=0.4, s=20, label=label)
ax7.set_title('Age vs Glucose Level', fontweight='bold'); ax7.legend()

ax8 = fig.add_subplot(gs[2, 2])
sns.heatmap(df[['age','hypertension','heart_disease','avg_glucose_level','bmi','stroke']].corr(),
            annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax8)
ax8.set_title('Correlation Matrix', fontweight='bold')

plt.suptitle('EDA — Stroke Prediction', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
print("✅ EDA saved → eda_plots.png")
plt.show()


# ==============================================================================
# STEP 3 — PREPROCESSING
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3 — PREPROCESSING")
print("=" * 70)

df_proc = df.copy()
df_proc.drop(columns=['id'], inplace=True, errors='ignore')
df_proc = df_proc[df_proc['gender'] != 'Other'].copy()

# Group-based BMI imputation
df_proc['age_decade'] = (df_proc['age'] // 10 * 10).astype(int)
bmi_map = df_proc.groupby(['gender','age_decade'])['bmi'].median()

def impute_bmi(row):
    if pd.isna(row['bmi']):
        return bmi_map.get((row['gender'], row['age_decade']), df_proc['bmi'].median())
    return row['bmi']

before = df_proc['bmi'].isna().sum()
df_proc['bmi'] = df_proc.apply(impute_bmi, axis=1)
df_proc.drop(columns=['age_decade'], inplace=True)
print(f"✅ BMI: {before} missing values imputed (group-based median)")

# Outlier capping
for col in ['bmi','avg_glucose_level','age']:
    lo, hi = df_proc[col].quantile(0.01), df_proc[col].quantile(0.99)
    df_proc[col] = df_proc[col].clip(lo, hi)
print("✅ Outliers capped at 1st/99th percentile")


# ==============================================================================
# STEP 4 — FEATURE ENGINEERING
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4 — FEATURE ENGINEERING")
print("=" * 70)

df_proc['age_group']       = pd.cut(df_proc['age'], bins=[0,18,30,45,60,100],
                                     labels=['Teen','Young_Adult','Adult','Middle_Aged','Senior'])
df_proc['bmi_category']    = pd.cut(df_proc['bmi'], bins=[0,18.5,25,30,35,100],
                                     labels=['Underweight','Normal','Overweight','Obese','Severely_Obese'])
df_proc['glucose_category']= pd.cut(df_proc['avg_glucose_level'], bins=[0,100,125,200,500],
                                     labels=['Normal','Prediabetic','Diabetic','High_Risk'])

df_proc['cardiovascular_risk']     = df_proc['hypertension'] + df_proc['heart_disease']
df_proc['age_risk']                = (df_proc['age'] > 60).astype(int)
df_proc['glucose_risk']            = (df_proc['avg_glucose_level'] > 125).astype(int)
df_proc['bmi_risk']                = (df_proc['bmi'] > 30).astype(int)
df_proc['health_risk_score']       = (df_proc['cardiovascular_risk'] + df_proc['age_risk'] +
                                       df_proc['glucose_risk'] + df_proc['bmi_risk'])
df_proc['age_glucose_interaction'] = df_proc['age'] * df_proc['avg_glucose_level'] / 1000
df_proc['age_bmi_interaction']     = df_proc['age'] * df_proc['bmi'] / 100
df_proc['glucose_bmi_interaction'] = df_proc['avg_glucose_level'] * df_proc['bmi'] / 1000
df_proc['age_squared']             = df_proc['age'] ** 2 / 1000
df_proc['high_risk_combo']         = ((df_proc['hypertension']==1) &
                                       (df_proc['heart_disease']==1) &
                                       (df_proc['age_risk']==1)).astype(int)

smoking_risk = {'never smoked':0,'Unknown':1,'formerly smoked':2,'smokes':3}
df_proc['smoking_risk_score'] = df_proc['smoking_status'].map(smoking_risk).fillna(1)

categorical_cols = ['gender','ever_married','work_type','Residence_type',
                    'smoking_status','age_group','bmi_category','glucose_category']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    label_encoders[col] = le

print(f"✅ Feature engineering done — Final shape: {df_proc.shape}")


# ==============================================================================
# STEP 5 — FEATURE SELECTION
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5 — MUTUAL INFORMATION FEATURE SELECTION")
print("=" * 70)

X_all = df_proc.drop('stroke', axis=1)
y_all = df_proc['stroke']

mi_scores = mutual_info_classif(X_all, y_all, random_state=SEED)
mi_df = pd.DataFrame({'Feature': X_all.columns, 'MI Score': mi_scores})\
          .sort_values('MI Score', ascending=False)

MI_THRESHOLD   = 0.005
selected_features = mi_df[mi_df['MI Score'] > MI_THRESHOLD]['Feature'].tolist()

plt.figure(figsize=(14, 7))
colors = [PALETTE['stroke'] if s > MI_THRESHOLD else '#bdc3c7' for s in mi_df['MI Score']]
plt.barh(mi_df['Feature'][::-1], mi_df['MI Score'][::-1], color=colors[::-1])
plt.axvline(MI_THRESHOLD, color='black', linestyle='--', label=f'Threshold={MI_THRESHOLD}')
plt.title('Feature Relevance — Mutual Information', fontweight='bold')
plt.xlabel('MI Score'); plt.legend(); plt.tight_layout()
plt.savefig('feature_selection.png', dpi=150, bbox_inches='tight')
print("✅ Feature selection saved → feature_selection.png")
plt.show()
print(f"✅ Selected {len(selected_features)}/{len(X_all.columns)} features")


# ==============================================================================
# STEP 6 — SPLIT & SMOTE
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6 — TRAIN/TEST SPLIT & SMOTE")
print("=" * 70)

X = df_proc[selected_features]
y = df_proc['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

smote = SMOTE(random_state=SEED, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

print(f"   Train after SMOTE : {dict(pd.Series(y_train_bal).value_counts())}")
print(f"   Test  shape       : {X_test_sc.shape}")
print("✅ Data ready!")


# ==============================================================================
# STEP 7 — TRAIN MODELS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7 — TRAINING ML MODELS")
print("=" * 70)

def evaluate_model(model, X_tr, y_tr, X_te, y_te, name, fit=True):
    if fit:
        model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        'Model'    : name,
        'Accuracy' : accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred, zero_division=0),
        'Recall'   : recall_score(y_te, y_pred, zero_division=0),
        'F1'       : f1_score(y_te, y_pred, zero_division=0),
        'ROC-AUC'  : roc_auc_score(y_te, y_prob),
        'PR-AUC'   : average_precision_score(y_te, y_prob),
        'Brier'    : brier_score_loss(y_te, y_prob),
    }, y_pred, y_prob

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                   random_state=SEED, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    'XGBoost'            : XGBClassifier(n_estimators=200, scale_pos_weight=19,
                                          random_state=SEED, eval_metric='logloss', n_jobs=-1),
    'LightGBM'           : lgb.LGBMClassifier(n_estimators=200, class_weight='balanced',
                                               random_state=SEED, n_jobs=-1, verbose=-1),
}

results, preds = [], {}
for name, model in models.items():
    t0 = time.time()
    m, yp, ypr = evaluate_model(model, X_train_sc, y_train_bal, X_test_sc, y_test, name)
    m['Time(s)'] = round(time.time()-t0, 2)
    results.append(m); preds[name] = (yp, ypr)
    print(f"  ✅ {name:22s} | AUC={m['ROC-AUC']:.4f} | F1={m['F1']:.4f} | {m['Time(s)']}s")

results_df = pd.DataFrame(results).round(4)


# ==============================================================================
# STEP 8 — HYPERPARAMETER TUNING
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8 — HYPERPARAMETER TUNING")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("\n🔍 Tuning XGBoost...")
xgb_search = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=19, random_state=SEED, eval_metric='logloss', n_jobs=-1),
    {'n_estimators':[100,200,300],'max_depth':[3,4,5,6],
     'learning_rate':[0.01,0.05,0.1],'subsample':[0.6,0.8,1.0],
     'colsample_bytree':[0.6,0.8,1.0]},
    n_iter=20, scoring='roc_auc', cv=cv, random_state=SEED, n_jobs=-1)
xgb_search.fit(X_train_sc, y_train_bal)
tuned_xgb = xgb_search.best_estimator_
print(f"   Best CV AUC: {xgb_search.best_score_:.4f}")

print("\n🔍 Tuning LightGBM...")
lgb_search = RandomizedSearchCV(
    lgb.LGBMClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1),
    {'n_estimators':[100,200,300],'num_leaves':[20,31,50],
     'learning_rate':[0.01,0.05,0.1],'feature_fraction':[0.6,0.8,1.0],
     'bagging_fraction':[0.6,0.8,1.0]},
    n_iter=20, scoring='roc_auc', cv=cv, random_state=SEED, n_jobs=-1)
lgb_search.fit(X_train_sc, y_train_bal)
tuned_lgb = lgb_search.best_estimator_
print(f"   Best CV AUC: {lgb_search.best_score_:.4f}")

m_xgb, yp_xgb, ypr_xgb = evaluate_model(tuned_xgb, None, None, X_test_sc, y_test, 'XGBoost (Tuned)',  fit=False)
m_lgb, yp_lgb, ypr_lgb = evaluate_model(tuned_lgb, None, None, X_test_sc, y_test, 'LightGBM (Tuned)', fit=False)
results += [m_xgb, m_lgb]
preds['XGBoost (Tuned)']  = (yp_xgb, ypr_xgb)
preds['LightGBM (Tuned)'] = (yp_lgb, ypr_lgb)
print(f"\n   XGBoost (Tuned)  → AUC={m_xgb['ROC-AUC']:.4f} | F1={m_xgb['F1']:.4f}")
print(f"   LightGBM (Tuned) → AUC={m_lgb['ROC-AUC']:.4f} | F1={m_lgb['F1']:.4f}")


# ==============================================================================
# STEP 9 — STACKING ENSEMBLE
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9 — STACKING ENSEMBLE")
print("=" * 70)

stack = StackingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                        random_state=SEED, n_jobs=-1)),
        ('xgb', tuned_xgb),
        ('lgb', tuned_lgb),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=5, n_jobs=-1, passthrough=True
)
print("Training stacking ensemble...")
stack.fit(X_train_sc, y_train_bal)
m_stack, yp_stack, ypr_stack = evaluate_model(stack, None, None, X_test_sc, y_test,
                                               'Stacking Ensemble', fit=False)
results.append(m_stack)
preds['Stacking Ensemble'] = (yp_stack, ypr_stack)
results_df = pd.DataFrame(results).round(4)
print(f"✅ Stacking → AUC={m_stack['ROC-AUC']:.4f} | F1={m_stack['F1']:.4f}")


# ==============================================================================
# STEP 10 — EVALUATION & OPTIMAL THRESHOLD
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 10 — EVALUATION & OPTIMAL THRESHOLD")
print("=" * 70)

print("\n📊 FULL MODEL COMPARISON:")
print(results_df.sort_values('ROC-AUC', ascending=False).to_string(index=False))

best_name    = results_df.sort_values('ROC-AUC', ascending=False).iloc[0]['Model']
_, ypr_best  = preds[best_name]

prec, rec, thresholds = precision_recall_curve(y_test, ypr_best)
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
opt_thr   = thresholds[f1_scores.argmax()]
yp_opt    = (ypr_best >= opt_thr).astype(int)

print(f"\n✅ Best Model       : {best_name}")
print(f"   Optimal Threshold: {opt_thr:.4f}")
print(f"\n{classification_report(y_test, yp_opt, target_names=['No Stroke','Stroke'])}")

# ROC & PR curves
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for name, (_, ypr) in preds.items():
    fpr, tpr, _ = roc_curve(y_test, ypr)
    axes[0].plot(fpr, tpr, lw=1.5, label=f'{name} ({roc_auc_score(y_test,ypr):.3f})')
axes[0].plot([0,1],[0,1],'k--')
axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].legend(fontsize=8)

for name, (_, ypr) in preds.items():
    p, r, _ = precision_recall_curve(y_test, ypr)
    axes[1].plot(r, p, lw=1.5, label=f'{name} ({average_precision_score(y_test,ypr):.3f})')
axes[1].set_title('Precision-Recall Curves', fontweight='bold')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
print("✅ ROC/PR curves saved → roc_pr_curves.png")
plt.show()

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, yp_opt), annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Stroke','Stroke'], yticklabels=['No Stroke','Stroke'])
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Confusion matrix saved → confusion_matrix.png")
plt.show()


# ==============================================================================
# STEP 11 — SAVE MODELS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 11 — SAVING MODELS & ARTIFACTS")
print("=" * 70)

joblib.dump(tuned_lgb,      'best_ml_model.pkl')
joblib.dump(scaler,         'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump({
    'selected_features'  : selected_features,
    'categorical_columns': categorical_cols,
    'optimal_threshold'  : float(opt_thr),
    'smoking_risk_map'   : smoking_risk,
}, 'metadata.pkl')
results_df.to_csv('model_results.csv', index=False)

print("✅ best_ml_model.pkl  saved")
print("✅ scaler.pkl         saved")
print("✅ label_encoders.pkl saved")
print("✅ metadata.pkl       saved")
print("✅ model_results.csv  saved")
print("\n" + "=" * 70)
print("  ALL DONE! Now run:  streamlit run app.py")
print("=" * 70)