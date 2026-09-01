"""
DEEP ML PIPELINE AUDIT SCRIPT
Investigates the root cause of 100% model metrics.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

from app.config import settings
from app.ml.preprocessing import clean_data
from app.ml.feature_engineering import engineer_features

raw_path = os.path.join(settings.RAW_DATA_DIR, "placement_10k.csv")
df_raw = pd.read_csv(raw_path)

print("=" * 70)
print("1. RAW DATASET SCHEMA")
print("=" * 70)
print(f"Shape: {df_raw.shape}")
print(f"Columns: {df_raw.columns.tolist()}")

# ---- CHECK 1: Target distribution ----
print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION (RAW)")
print("=" * 70)
placed = (df_raw['PlacementStatus'] == 'Placed').sum()
not_placed = (df_raw['PlacementStatus'] == 'Not Placed').sum()
print(f"Placed:     {placed} ({placed / len(df_raw) * 100:.1f}%)")
print(f"Not Placed: {not_placed} ({not_placed / len(df_raw) * 100:.1f}%)")

# ---- CHECK 2: Duplicates ----
print("\n" + "=" * 70)
print("3. DUPLICATE ANALYSIS")
print("=" * 70)
print(f"Exact duplicate rows: {df_raw.duplicated().sum()}")
feature_cols = [c for c in df_raw.columns if c not in ['StudentID', 'PlacementStatus']]
print(f"Feature-only duplicates: {df_raw[feature_cols].duplicated().sum()}")

# ---- Clean + Engineer ----
df_clean, _ = clean_data(df_raw)
df_feat = engineer_features(df_clean)

target_col = "PlacementStatus"
excluded = ["StudentID", "Gender"]
y = (df_feat[target_col] == "Placed").astype(int)
X = df_feat.drop(columns=[target_col] + [c for c in excluded if c in df_feat.columns])

# ---- CHECK 3: Verify target NOT in features ----
print("\n" + "=" * 70)
print("4. TARGET LEAKAGE CHECK: PlacementStatus NOT in X")
print("=" * 70)
print(f"X columns: {X.columns.tolist()}")
print(f"Target in X? {'PlacementStatus' in X.columns}")

# ---- CHECK 4: Feature correlations with target ----
print("\n" + "=" * 70)
print("5. FEATURE CORRELATIONS WITH TARGET")
print("=" * 70)
X_with_y = X.copy()
X_with_y['_target'] = y.values
numeric_X = X_with_y.select_dtypes(include=[np.number])
corr = numeric_X.corr()['_target'].drop('_target').sort_values(key=abs, ascending=False)
for feat, val in corr.items():
    flag = " *** SUSPICIOUS ***" if abs(val) > 0.85 else (" ** HIGH **" if abs(val) > 0.5 else "")
    print(f"  {feat:35s}  r = {val:+.4f}{flag}")

# ---- CHECK 5: Univariate class separability ----
print("\n" + "=" * 70)
print("6. UNIVARIATE CLASS SEPARABILITY")
print("=" * 70)
for col in numeric_X.columns:
    if col == '_target':
        continue
    placed_vals = numeric_X.loc[y == 1, col]
    not_placed_vals = numeric_X.loc[y == 0, col]
    overlap = (placed_vals.min() <= not_placed_vals.max()) and (not_placed_vals.min() <= placed_vals.max())
    if not overlap:
        print(f"  {col}: ZERO OVERLAP => PERFECT SEPARATOR!")
        print(f"    Placed range:     [{placed_vals.min():.2f}, {placed_vals.max():.2f}]")
        print(f"    Not Placed range: [{not_placed_vals.min():.2f}, {not_placed_vals.max():.2f}]")
    else:
        # Check how much overlap
        threshold_accuracy = 0
        for t in np.linspace(numeric_X[col].min(), numeric_X[col].max(), 200):
            acc = max(((numeric_X[col] >= t) == (y == 1)).mean(),
                      ((numeric_X[col] < t) == (y == 1)).mean())
            threshold_accuracy = max(threshold_accuracy, acc)
        if threshold_accuracy > 0.95:
            print(f"  {col}: Best single-threshold accuracy = {threshold_accuracy:.4f}  ** NEAR-PERFECT **")
        elif threshold_accuracy > 0.85:
            print(f"  {col}: Best single-threshold accuracy = {threshold_accuracy:.4f}  * HIGH *")

# ---- CHECK 6: Train/test split + baseline ----
print("\n" + "=" * 70)
print("7. TRAIN/TEST SPLIT")
print("=" * 70)
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train target dist: Placed={y_train.sum()} ({y_train.mean()*100:.1f}%), Not Placed={len(y_train)-y_train.sum()}")
print(f"Test  target dist: Placed={y_test.sum()} ({y_test.mean()*100:.1f}%), Not Placed={len(y_test)-y_test.sum()}")

# Check train/test overlap (identical feature rows)
train_set = set(X_train.apply(lambda row: tuple(row), axis=1))
test_set = set(X_test.apply(lambda row: tuple(row), axis=1))
overlap = train_set.intersection(test_set)
print(f"Feature-vector overlap between train and test: {len(overlap)} rows")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

# ---- CHECK 7: DummyClassifier baseline ----
print("\n" + "=" * 70)
print("8. DUMMY CLASSIFIER BASELINE")
print("=" * 70)
for strategy in ['most_frequent', 'stratified']:
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_train[num_features], y_train)
    d_pred = dummy.predict(X_test[num_features])
    d_acc = accuracy_score(y_test, d_pred)
    print(f"  DummyClassifier(strategy={strategy}): accuracy = {d_acc:.4f}")

# ---- CHECK 8: Actual model eval ----
print("\n" + "=" * 70)
print("9. MODEL EVALUATION (CURRENT PIPELINE)")
print("=" * 70)
models = {
    "Logistic Regression": Pipeline([('pre', preprocessor), ('cls', LogisticRegression(random_state=42, class_weight='balanced', max_iter=500))]),
    "Random Forest": Pipeline([('pre', preprocessor), ('cls', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))]),
    "XGBoost": Pipeline([('pre', preprocessor), ('cls', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'))]),
}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_prob)
    print(f"  {name:25s}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

# ---- CHECK 9: Cross-validation ----
print("\n" + "=" * 70)
print("10. 5-FOLD STRATIFIED CROSS-VALIDATION (on training set only)")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"  {name:25s}: CV ROC-AUC = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  (folds: {[f'{s:.4f}' for s in cv_scores]})")

# ---- CHECK 10: Feature importances from RF ----
print("\n" + "=" * 70)
print("11. RANDOM FOREST FEATURE IMPORTANCES")
print("=" * 70)
rf_pipe = models["Random Forest"]
rf_model = rf_pipe.named_steps['cls']
pre = rf_pipe.named_steps['pre']
feature_names = num_features + list(pre.named_transformers_['cat'].get_feature_names_out(cat_features))
importances = rf_model.feature_importances_
for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    flag = " *** DOMINANT ***" if imp > 0.3 else (" ** HIGH **" if imp > 0.15 else "")
    print(f"  {fname:35s}  importance = {imp:.4f}{flag}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
