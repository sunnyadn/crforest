"""TreeSHAP example for comprisk competing-risk forests.

Demonstrates:
1. Fitting a CompetingRiskForest
2. Computing cause-specific CIF SHAP values
3. Additivity verification
4. Feature ranking by mean |SHAP|
5. Per-subject waterfall-style attribution
"""

import numpy as np

from comprisk import CompetingRiskForest

# ---------------------------------------------------------------------------
# 1. Synthetic competing-risks data
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n = 500
p = 10
X = rng.normal(size=(n, p))

# Feature 0 drives time-to-event; feature 1 modulates cause distribution.
time = rng.exponential(2.0, size=n) + 0.5 * np.abs(X[:, 0]) + 0.1
event = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])

# ---------------------------------------------------------------------------
# 2. Fit forest
# ---------------------------------------------------------------------------

forest = CompetingRiskForest(n_estimators=100, random_state=42, max_depth=8, n_jobs=-1).fit(
    X, time, event
)

print(
    f"Fitted {forest.n_estimators} trees, {forest.n_causes_} causes, "
    f"{len(forest.unique_times_)} time grid points."
)

# ---------------------------------------------------------------------------
# 3. TreeSHAP: explain CIF for a held-out subset
# ---------------------------------------------------------------------------

X_explain = X[:20]
shap, base = forest.shap_values(X_explain)

print(f"SHAP shape: {shap.shape}")  # (n_samples, n_features, n_times, n_causes)
print(f"Base shape: {base.shape}")  # (n_times, n_causes)

# ---------------------------------------------------------------------------
# 4. Additivity check (the fundamental SHAP property)
# ---------------------------------------------------------------------------

cif_pred = forest.predict_cif(X_explain)
reconstructed = (shap.sum(axis=1) + base).transpose(0, 2, 1)

max_abs_err = np.max(np.abs(reconstructed - cif_pred))
print(f"Additivity max |error|: {max_abs_err:.2e}")
assert max_abs_err < 1e-6, "Additivity violated!"

# ---------------------------------------------------------------------------
# 5. Global feature importance: mean |SHAP| over subjects x times x causes
# ---------------------------------------------------------------------------

mean_abs_shap = np.abs(shap).mean(axis=(0, 2, 3))
top5_idx = np.argsort(mean_abs_shap)[::-1][:5]

print("\nTop 5 features by mean |SHAP|:")
for rank, idx in enumerate(top5_idx, 1):
    print(f"  {rank}. feature_{idx}: {mean_abs_shap[idx]:.4f}")

# ---------------------------------------------------------------------------
# 6. Per-subject attribution for cause 1 at the last timepoint
# ---------------------------------------------------------------------------

cause = 0  # cause index 0 = cause 1 in 1-based
time_idx = -1  # last timepoint

shap_slice = shap[:, :, time_idx, cause]  # (n_samples, n_features)
base_slice = base[time_idx, cause]

print(f"\nSubject 0 — cause 1 at t={forest.unique_times_[time_idx]:.2f}:")
print(f"  Baseline (expected CIF): {base_slice:.4f}")
print(f"  Predicted CIF: {cif_pred[0, cause, time_idx]:.4f}")
print(f"  Sum of attributions: {shap_slice[0].sum():.4f}")
print("  Attribution breakdown:")
for feat_idx in np.argsort(np.abs(shap_slice[0]))[::-1][:5]:
    print(f"    feature_{feat_idx}: {shap_slice[0, feat_idx]:+.4f}")
