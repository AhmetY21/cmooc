---
title: "Evaluation Metrics: Precision, Recall, F1-Score"
date: "2026-02-12"
week: 7
lesson: 5
slug: "evaluation-metrics-precision-recall-f1-score"
---

# Topic: Evaluation Metrics: Precision, Recall, F1-Score

## 1) Formal definition (what is it, and how can we use it?)

Precision, Recall, and F1-Score are evaluation metrics commonly used to assess the performance of classification models, especially in tasks like information retrieval, machine learning, and natural language processing. They provide a more nuanced understanding of a model's performance than simple accuracy, especially when dealing with imbalanced datasets (where one class has significantly more instances than the other).

*   **Precision:** Precision measures the proportion of positive identifications that were actually correct. It answers the question: "Of all the instances the model predicted as positive, how many were actually positive?"  A high precision indicates that the model has a low rate of *false positives* (i.e., it's good at avoiding labeling negative instances as positive).

    **Formula:** Precision = True Positives / (True Positives + False Positives)

*   **Recall:** Recall measures the proportion of actual positives that were correctly identified by the model. It answers the question: "Of all the actual positive instances, how many did the model correctly identify?"  A high recall indicates that the model has a low rate of *false negatives* (i.e., it's good at finding all the positive instances).

    **Formula:** Recall = True Positives / (True Positives + False Negatives)

*   **F1-Score:** The F1-Score is the harmonic mean of precision and recall. It provides a single score that balances both concerns. It's particularly useful when you want to find a balance between precision and recall, as high performance on both is desired.

    **Formula:** F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

In summary:

*   Use these metrics when you want to understand *how* your classification model is performing in terms of correctly identifying positive instances and avoiding errors.
*   Consider precision when minimizing false positives is crucial (e.g., spam detection, where misclassifying a legitimate email as spam is highly undesirable).
*   Consider recall when minimizing false negatives is crucial (e.g., disease detection, where missing a positive case has severe consequences).
*   Use the F1-score when you want to balance both precision and recall or when the cost of false positives and false negatives are similar.

## 2) Application scenario

Let's consider a scenario where we are building a machine learning model to detect fraudulent transactions in online banking.

*   **Positive Class:** Fraudulent transaction
*   **Negative Class:** Non-fraudulent transaction

In this case:

*   **High Precision:**  Ensuring that when the model flags a transaction as fraudulent, it is highly likely to be genuinely fraudulent.  A low precision would mean many legitimate transactions are incorrectly flagged, causing customer inconvenience and distrust.
*   **High Recall:** Ensuring that the model detects as many fraudulent transactions as possible.  A low recall would mean that many fraudulent transactions slip through undetected, leading to financial losses.
*   **F1-Score:** A good balance between not annoying legitimate customers too frequently (precision) and catching as many fraudulent activities as possible (recall).

In this particular scenario, achieving a high recall might be more important initially, even at the cost of a slightly lower precision. This is because the cost of missing a fraudulent transaction is potentially much higher than the cost of temporarily flagging a legitimate transaction. A human reviewer can later verify the transactions flagged by the model.

## 3) Python method (if possible)

The `scikit-learn` library in Python provides convenient functions for calculating precision, recall, and F1-score.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Example:
y_true = [0, 1, 0, 1, 0, 0, 1, 1]  # Actual labels
y_pred = [0, 1, 1, 0, 0, 1, 1, 0]  # Predicted labels

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Weighted average F1
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print("Weighted F1-Score:", f1_weighted)

# Using classification_report for all metrics at once
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred)
print(report)
```

Explanation:

*   `precision_score(y_true, y_pred)` calculates the precision score.
*   `recall_score(y_true, y_pred)` calculates the recall score.
*   `f1_score(y_true, y_pred)` calculates the F1-score.
*   `average='weighted'` in `f1_score` calculates a weighted average of the F1-scores for each class, taking into account the number of instances in each class. Useful for imbalanced datasets. Other options for `average` include `'macro'` (unweighted average of F1-scores for each class) and `'micro'` (calculate metrics globally by counting the total true positives, false negatives and false positives).
*   `classification_report(y_true, y_pred)` generates a comprehensive report including precision, recall, F1-score, and support (number of instances) for each class, along with averages.

## 4) Follow-up question

How do these metrics relate to the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC), and when would I prefer to use ROC/AUC over Precision/Recall/F1-score, or vice versa?