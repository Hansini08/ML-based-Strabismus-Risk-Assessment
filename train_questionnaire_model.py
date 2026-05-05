import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib.pyplot as plt

# -------------------------
# Plot Output Path
# -------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("final_questionnaire_dataset.csv")

# -------------------------
# Check Class Distribution
# -------------------------
print("\nClass Distribution:\n", data["Risk"].value_counts())

# -------------------------
# Class Distribution Bar Chart
# -------------------------
class_counts = data["Risk"].value_counts().sort_index()
plt.figure()
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Class Distribution")
plt.xlabel("Risk")
plt.ylabel("Count")
plt.show()

# -------------------------
# Split Features & Label
# -------------------------
X = data.drop("Risk", axis=1)
y = data["Risk"]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Feature Scaling
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Train Logistic Regression
# -------------------------
model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------
# Evaluate Model
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
class_labels = list(model.classes_)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# -------------------------
# Confusion Matrix Heatmap
# -------------------------
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.xticks(range(len(class_labels)), [str(c) for c in class_labels])
plt.yticks(range(len(class_labels)), [str(c) for c in class_labels])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "questionnaire_confusion_matrix.png"), dpi=300)
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------
# 📊 Accuracy Bar Graph
# -------------------------
plt.figure()
plt.bar(['Accuracy'], [accuracy])
plt.title("Model Accuracy")
plt.ylabel("Score")
plt.ylim(0,1)
plt.show()

# -------------------------
# 📊 Classification Metrics Graph
# -------------------------
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_df.iloc[:-1, :-1].plot(kind='bar')
plt.title("Classification Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.show()

# -------------------------
# ROC Curve (One-vs-Rest)
# -------------------------
y_test_bin = label_binarize(y_test, classes=class_labels)

plt.figure(figsize=(5, 4))
for i, cls in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} AUC = {roc_auc:.3f}")

fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, linestyle='--', color='black', label=f"Micro AUC = {roc_auc_micro:.3f}")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "questionnaire_roc_curve.png"), dpi=300)
plt.show()

# -------------------------
# Precision-Recall Curve (One-vs-Rest)
# -------------------------
plt.figure(figsize=(5, 4))
for i, cls in enumerate(class_labels):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
    plt.plot(recall, precision, label=f"Class {cls} AP = {ap:.3f}")

precision_micro, recall_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_prob.ravel())
ap_micro = average_precision_score(y_test_bin, y_prob, average='micro')
plt.plot(recall_micro, precision_micro, linestyle='--', color='black', label=f"Micro AP = {ap_micro:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()

# -------------------------
# Calibration Curve (One-vs-Rest)
# -------------------------
plt.figure(figsize=(5, 4))
for i, cls in enumerate(class_labels):
    prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_prob[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f"Class {cls}")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend(loc="upper left")
plt.show()

# -------------------------
# Feature Importance (Coefficients)
# -------------------------
feature_names = X.columns.tolist()
coefs = model.coef_
mean_abs = np.mean(np.abs(coefs), axis=0)
sorted_idx = np.argsort(mean_abs)[::-1]

plt.figure(figsize=(7, 6))
plt.barh(
    [feature_names[i] for i in sorted_idx],
    mean_abs[sorted_idx]
)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -------------------------
# Correlation Heatmap
# -------------------------
corr = data.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()

# -------------------------
# Save Model & Scaler
# -------------------------
joblib.dump(model, "questionnaire_model.pkl")
joblib.dump(scaler, "questionnaire_scaler.pkl")

print("\n✅ Model and scaler saved successfully!")