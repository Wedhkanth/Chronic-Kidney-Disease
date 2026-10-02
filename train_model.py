import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    f_classif,
    SelectKBest,
    RFE
)

# ==============================
# Create folders
# ==============================
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("data/kidney_disease.csv")

df.columns = df.columns.str.strip().str.lower()
df.replace('?', np.nan, inplace=True)

# Encode target BEFORE converting to numeric
# Handle string values like 'ckd', 'ckd\t', 'notckd', etc.
try:
    df['classification'] = df['classification'].astype(str).str.strip().map({'ckd': 1, 'notckd': 0})
except AttributeError:
    pass

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaN in target with 0, then ensure int
df['classification'] = df['classification'].fillna(0).astype(int)

# Fill NaN in features with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop ID
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Drop any remaining non-numeric columns
df = df.select_dtypes(include=[np.number])

# ==============================
# eGFR Calculation
# ==============================
def calculate_eGFR(age, sc):
    if sc <= 0:
        sc = 0.01  # avoid division by zero
    if age <= 0:
        age = 1
    return 175 * (sc ** -1.154) * (age ** -0.203)

df['egfr'] = df.apply(lambda x: calculate_eGFR(x['age'], x['sc']), axis=1)

# ==============================
# Feature Extraction (ALL numeric)
# ==============================
target_col = 'classification'
all_features = [col for col in df.columns if col != target_col]

X_all = df[all_features].copy()

# Drop columns that are entirely NaN (categorical columns that couldn't convert to numeric)
all_nan_cols = X_all.columns[X_all.isna().all()].tolist()
if all_nan_cols:
    print(f"   Dropping all-NaN columns: {all_nan_cols}")
    X_all.drop(columns=all_nan_cols, inplace=True)
    all_features = [f for f in all_features if f not in all_nan_cols]

# Fill remaining NaN with median
X_all = X_all.fillna(X_all.median())

# Replace any remaining inf values
X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
X_all = X_all.fillna(0)

y = df[target_col].fillna(0).astype(int)

print(f"Total features available: {len(all_features)}")
print(f"   Features: {all_features}")

# ==============================
# FEATURE SELECTION — 4 METHODS
# ==============================
TOP_K = min(10, len(all_features))  # Select top 10 (or fewer if not enough features)
CONSENSUS_THRESHOLD = 3  # Feature must be picked by >= 3 methods

# --- Helper: get top-K feature names from scores ---
def get_top_k(scores, feature_names, k):
    """Return top-k feature names sorted by score (descending)."""
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [name for name, score in ranked[:k]]


# ============================================================
# METHOD 1: Mutual Information
# ============================================================
print("\nMethod 1: Mutual Information...")
mi_scores = mutual_info_classif(X_all, y, random_state=42)
mi_selected = get_top_k(mi_scores, all_features, TOP_K)
print(f"   Selected: {mi_selected}")

# ============================================================
# METHOD 2: Chi-Squared (requires non-negative values)
# ============================================================
print("\nMethod 2: Chi-Squared (chi2)...")
# MinMaxScaler to ensure non-negative values for chi2
mm_scaler = MinMaxScaler()
X_nonneg = mm_scaler.fit_transform(X_all)
chi2_scores, chi2_pvalues = chi2(X_nonneg, y)
chi2_selected = get_top_k(chi2_scores, all_features, TOP_K)
print(f"   Selected: {chi2_selected}")

# ============================================================
# METHOD 3: ANOVA F-test
# ============================================================
print("\nMethod 3: ANOVA F-test...")
f_scores, f_pvalues = f_classif(X_all, y)
anova_selected = get_top_k(f_scores, all_features, TOP_K)
print(f"   Selected: {anova_selected}")

# ============================================================
# METHOD 4: Recursive Feature Elimination (RFE)
# ============================================================
print("\nMethod 4: Recursive Feature Elimination (RFE)...")
rfe_estimator = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=TOP_K, step=1)
rfe_selector.fit(X_all, y)
rfe_selected = [f for f, selected in zip(all_features, rfe_selector.support_) if selected]
print(f"   Selected: {rfe_selected}")

# ============================================================
# CONSENSUS: Features selected by >= CONSENSUS_THRESHOLD methods
# ============================================================
print(f"\nConsensus Selection (>= {CONSENSUS_THRESHOLD}/4 methods)...")

all_selections = {
    'Mutual Information': mi_selected,
    'Chi-Squared': chi2_selected,
    'ANOVA F-test': anova_selected,
    'RFE': rfe_selected
}

# Count how many methods selected each feature
feature_vote_count = {}
for method_name, selected_list in all_selections.items():
    for feat in selected_list:
        feature_vote_count[feat] = feature_vote_count.get(feat, 0) + 1

# Consensus features
consensus_features = [f for f, count in feature_vote_count.items() if count >= CONSENSUS_THRESHOLD]

# Sort by vote count (descending), then alphabetically
consensus_features.sort(key=lambda f: (-feature_vote_count[f], f))

# Fallback: if consensus is too small, use union of top features
if len(consensus_features) < 3:
    print("   Too few consensus features, falling back to top-voted features...")
    consensus_features = sorted(feature_vote_count.keys(),
                                key=lambda f: -feature_vote_count[f])[:TOP_K]

print(f"   Consensus features ({len(consensus_features)}): {consensus_features}")
print(f"   Vote counts: {[(f, feature_vote_count[f]) for f in consensus_features]}")

features = consensus_features

# ==============================
# Visualization: Feature Selection Comparison Heatmap
# ==============================
print("\nGenerating feature selection comparison heatmap...")

# Build a binary matrix: rows = all_features, cols = methods
method_names = list(all_selections.keys())
heatmap_data = pd.DataFrame(0, index=all_features, columns=method_names)

for method_name, selected_list in all_selections.items():
    for feat in selected_list:
        heatmap_data.loc[feat, method_name] = 1

# Add vote count column
heatmap_data['Votes'] = heatmap_data.sum(axis=1)

# Sort by total votes (descending)
heatmap_data = heatmap_data.sort_values('Votes', ascending=False)

# Plot heatmap (without Votes column, it's just for sorting)
fig, ax = plt.subplots(figsize=(10, max(8, len(all_features) * 0.4)))
sns.heatmap(
    heatmap_data[method_names],
    annot=True,
    fmt='d',
    cmap='YlGn',
    cbar_kws={'label': 'Selected (1) / Not Selected (0)'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)
ax.set_title('Feature Selection Comparison (4 Methods)', fontsize=14, fontweight='bold')
ax.set_xlabel('Selection Method', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)

# Highlight consensus features
for i, feat in enumerate(heatmap_data.index):
    if feat in consensus_features:
        ax.get_yticklabels()[i].set_fontweight('bold')
        ax.get_yticklabels()[i].set_color('darkgreen')

plt.tight_layout()
plt.savefig("static/feature_selection_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: static/feature_selection_comparison.png")

# ==============================
# Save feature selection rankings
# ==============================
rankings = {
    'mi_scores': dict(zip(all_features, mi_scores.tolist())),
    'chi2_scores': dict(zip(all_features, chi2_scores.tolist())),
    'anova_scores': dict(zip(all_features, f_scores.tolist())),
    'rfe_ranking': dict(zip(all_features, rfe_selector.ranking_.tolist())),
    'selections': {k: v for k, v in all_selections.items()},
    'consensus': consensus_features,
    'vote_counts': feature_vote_count,
    'top_k': TOP_K,
    'consensus_threshold': CONSENSUS_THRESHOLD
}
joblib.dump(rankings, "models/feature_rankings.pkl")

# ==============================
# Prepare Final Dataset
# ==============================
X = X_all[features]
X = X.fillna(X.median())

# ==============================
# Scaling
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Model Training
# ==============================
print("\nTraining models...")

lgb = LGBMClassifier(random_state=42, verbose=-1)
cb = CatBoostClassifier(iterations=100, random_state=42, verbose=0)

lgb.fit(X_train, y_train)
cb.fit(X_train, y_train)

def evaluate_model(model_name, y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0) 
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    print(f"--- {model_name} Evaluation ---")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

lgb_pred = lgb.predict(X_test)
cb_pred = cb.predict(X_test)

evaluate_model("LightGBM", y_test, lgb_pred)
evaluate_model("CatBoost", y_test, cb_pred)

# ==============================
# Save Models
# ==============================
joblib.dump(lgb, "models/lgb_model.pkl")
joblib.dump(cb, "models/cb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")

print(f"\nModels trained with {len(features)} consensus features: {features}")

# ==============================
# SHAP GLOBAL EXPLANATION
# ==============================
print("\nGenerating SHAP explanations...")

explainer = shap.TreeExplainer(lgb)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.savefig("static/shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print("   Saved: static/shap_summary.png")

# ==============================
# SHAP Bar Plot (feature importance)
# ==============================
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
plt.savefig("static/shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()

print("   Saved: static/shap_bar.png")

print("\n" + "=" * 50)
print("TRAINING PIPELINE COMPLETE!")
print("=" * 50)