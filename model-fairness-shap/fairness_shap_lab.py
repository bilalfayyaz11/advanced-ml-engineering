import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

os.makedirs("outputs", exist_ok=True)

print("Loading Adult Income dataset...")
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.copy()

df = df.replace("?", np.nan).dropna()

target_col = "class"
df[target_col] = df[target_col].astype(str).str.strip()
y = df[target_col].map({"<=50K": 0, ">50K": 1}).astype(int)

X = df.drop(columns=[target_col])

sensitive_feature = "sex"
sensitive_raw = X[sensitive_feature].astype(str)

categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

X_train_raw, X_test_raw, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_raw, test_size=0.2, random_state=42, stratify=y
)

print("Preprocessing data...")
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

feature_names = (
    numeric_cols
    + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
)

print("Training TensorFlow model...")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

print("Evaluating model...")
pred_probs = model.predict(X_test, verbose=0).ravel()
preds = (pred_probs >= 0.5).astype(int)

accuracy = accuracy_score(y_test, preds)
print(f"Overall Accuracy: {accuracy:.4f}")

def true_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def selection_rate(y_pred):
    return np.mean(y_pred)

fairness_rows = []
for group in sorted(sensitive_test.unique()):
    mask = sensitive_test == group
    group_acc = accuracy_score(y_test[mask], preds[mask])
    group_tpr = true_positive_rate(y_test[mask], preds[mask])
    group_sr = selection_rate(preds[mask])

    fairness_rows.append({
        "group": group,
        "sample_count": int(mask.sum()),
        "accuracy": round(group_acc, 4),
        "true_positive_rate_equal_opportunity": round(group_tpr, 4),
        "selection_rate_disparate_impact": round(group_sr, 4),
    })

fairness_df = pd.DataFrame(fairness_rows)
fairness_df.to_csv("outputs/fairness_metrics_by_sex.csv", index=False)
print("\nFairness metrics by sex:")
print(fairness_df)

print("\nGenerating SHAP explanations...")
background = X_train[:100]
sample = X_test[:100]

explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(sample, nsamples=100)

if isinstance(shap_values, list):
    shap_array = shap_values[0]
else:
    shap_array = shap_values

shap_array = np.array(shap_array)
if shap_array.ndim == 3:
    shap_array = shap_array[:, :, 0]

sample_df = pd.DataFrame(sample, columns=feature_names)

plt.figure()
shap.summary_plot(shap_array, sample_df, show=False, max_display=15)
plt.tight_layout()
plt.savefig("outputs/shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = np.abs(shap_array).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap_value": mean_abs_shap
}).sort_values("mean_absolute_shap_value", ascending=False)

importance_df.to_csv("outputs/shap_feature_importance.csv", index=False)

with open("outputs/model_audit_summary.txt", "w") as f:
    f.write("Model Fairness and Explainability Audit\n")
    f.write("======================================\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write("Fairness Metrics by Sex:\n")
    f.write(fairness_df.to_string(index=False))
    f.write("\n\nTop 10 SHAP Features:\n")
    f.write(importance_df.head(10).to_string(index=False))
    f.write("\n\nInterpretation:\n")
    f.write("- Accuracy shows general model performance.\n")
    f.write("- True positive rate supports equal opportunity analysis.\n")
    f.write("- Selection rate supports disparate impact analysis.\n")
    f.write("- SHAP values explain which features most influenced predictions.\n")

print("\nTop 10 SHAP features:")
print(importance_df.head(10))

print("\nFiles created:")
print("- outputs/fairness_metrics_by_sex.csv")
print("- outputs/shap_feature_importance.csv")
print("- outputs/shap_summary_plot.png")
print("- outputs/model_audit_summary.txt")
print("\nLab script completed successfully.")
