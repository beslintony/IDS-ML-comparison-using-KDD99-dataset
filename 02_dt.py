import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    jaccard_score,
    hamming_loss,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    matthews_corrcoef,
    zero_one_loss,
)
import time
from common import attack_mapping, col_names, num_features
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./dataset/kddcup.data_10_percent_corrected", names=col_names)

data["label"] = data["label"].map(attack_mapping)
data = pd.get_dummies(data, columns=["protocol_type", "service", "flag"])
X = data.drop("label", axis=1)

scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["label"])
labels = label_encoder.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier()

start_time = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start_time

start_time = time.time()
y_pred = clf.predict(X_test)
prediction_time = time.time() - start_time

mapped_labels = label_encoder.inverse_transform(y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred, average=None
)
accuracy = accuracy_score(y_test, y_pred)
jaccard_score_val = jaccard_score(y_test, y_pred, average="weighted")
hamming_loss_val = hamming_loss(y_test, y_pred)
matthews_corrcoef_val = matthews_corrcoef(y_test, y_pred)
zero_one_loss_val = zero_one_loss(y_test, y_pred)

metrics = ["Precision", "Recall", "F1-Score"]
scores = [precision, recall, f1_score]

plt.figure(figsize=(12, 6))
x = np.arange(len(labels))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, scores[i], width, label=metric)
    for j, score in enumerate(scores[i]):
        plt.text(
            x[j] + i * width - 0.06,
            score + 0.01,
            "{:.2f}".format(score),
            color="black",
            fontweight="bold",
        )

plt.xlabel("Labels")
plt.ylabel("Scores")
plt.title("Precision, Recall, and F1-Score for Each Label")
plt.xticks(x + (len(metrics) - 1) * width / 2, labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

metrics = ["Accuracy", "Matthews Corr. Coef."]
scores = [accuracy, matthews_corrcoef_val]

plt.figure(figsize=(6, 4))
plt.bar(metrics, scores)
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, "{:.2f}".format(score), color="black", fontweight="bold")
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Accuracy and Matthews Correlation Coefficient")
plt.show()

metrics = ["Jaccard Score", "Hamming Loss"]
scores = [jaccard_score_val, hamming_loss_val]

plt.figure(figsize=(6, 4))
plt.bar(metrics, scores)
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, "{:.6f}".format(score), color="black", fontweight="bold")
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Jaccard Score and Hamming Loss")
plt.show()

fpr = {}
tpr = {}
roc_auc = {}

for label in labels:
    binary_true_labels = np.where(y_test == label_encoder.transform([label])[0], 1, 0)
    binary_pred_labels = np.where(y_pred == label_encoder.transform([label])[0], 1, 0)
    fpr[label], tpr[label], _ = roc_curve(binary_true_labels, binary_pred_labels)
    roc_auc[label] = auc(fpr[label], tpr[label])

plt.figure(figsize=(8, 6))
for label in labels:
    plt.plot(
        fpr[label],
        tpr[label],
        label="ROC curve for {} (AUC = {:.2f})".format(label, roc_auc[label]),
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve for Each Label")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(["Prediction Time", "Training Time"], [prediction_time, training_time])
for i, time_value in enumerate([prediction_time, training_time]):
    plt.text(i, time_value + 0.001, "{:.4f}".format(time_value), color="black", fontweight="bold")
plt.xlabel("Metrics")
plt.ylabel("Time (seconds)")
plt.title("Prediction Time and Training Time")
plt.show()

