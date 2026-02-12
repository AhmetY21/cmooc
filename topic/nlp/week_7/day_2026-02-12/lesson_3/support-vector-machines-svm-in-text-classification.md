---
title: "Support Vector Machines (SVM) in Text Classification"
date: "2026-02-12"
week: 7
lesson: 3
slug: "support-vector-machines-svm-in-text-classification"
---

# Topic: Support Vector Machines (SVM) in Text Classification

## 1) Formal definition (what is it, and how can we use it?)

Support Vector Machines (SVMs) are supervised machine learning models that can be used for both classification and regression tasks. In text classification, the goal is to assign a predefined category (or label) to a given text document. SVMs achieve this by finding an optimal hyperplane that separates data points belonging to different categories in a high-dimensional space, where each dimension corresponds to a feature extracted from the text (e.g., word frequencies, TF-IDF scores).

More formally:

*   **Goal:** To find the hyperplane that maximizes the margin between different classes. The margin is the distance between the hyperplane and the closest data points from each class (called support vectors).
*   **Features:** Text documents are converted into numerical feature vectors. Common techniques include:
    *   **Bag-of-Words (BoW):** Counts the frequency of each word in the document.
    *   **Term Frequency-Inverse Document Frequency (TF-IDF):** Weights words based on their frequency within a document and rarity across the entire corpus.
    *   **Word Embeddings (e.g., Word2Vec, GloVe):** Maps words to dense vector representations capturing semantic relationships.
*   **Kernel Trick:** SVMs can use kernel functions (e.g., linear, polynomial, radial basis function (RBF)) to implicitly map the input data into a higher-dimensional space where linear separation is possible, even if the data is not linearly separable in the original space.  This avoids explicitly computing transformations of the feature space, making the process computationally efficient. The kernel function computes the dot product of the data points in this higher-dimensional space.
*   **Classification:** Given a new text document (converted into a feature vector), the SVM classifies it by determining which side of the optimal hyperplane the feature vector falls on.

We use SVMs in text classification by first training the model on a labeled dataset of text documents. During training, the SVM learns the optimal hyperplane parameters.  Then, we can use the trained model to predict the category of new, unseen text documents.

## 2) Application scenario

A good application scenario is **Spam Email Detection**.  We can train an SVM classifier to distinguish between spam and non-spam emails (also called "ham").

*   **Input Data:** A dataset of emails labeled as either "spam" or "ham."
*   **Feature Extraction:** Use TF-IDF to convert the email text into numerical feature vectors.  Words common in spam emails (e.g., "discount," "urgent," "free") will likely have high TF-IDF scores in spam emails and low scores in ham emails.
*   **Training:** Train an SVM classifier (e.g., with a linear or RBF kernel) on the labeled dataset. The SVM learns the optimal hyperplane that separates spam emails from ham emails based on their TF-IDF feature vectors.
*   **Prediction:**  When a new email arrives, convert its text into a TF-IDF feature vector and feed it to the trained SVM model. The SVM will predict whether the email is spam or ham based on which side of the hyperplane the email's feature vector falls on.
*   **Benefit:**  SVMs can effectively capture the subtle differences in word usage and patterns that distinguish spam from legitimate emails, even when spammers try to obfuscate their messages.

## 3) Python method (if possible)

Here's how you can use scikit-learn to train an SVM classifier for text classification:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your actual text data)
data = datasets.load_files('text_data',encoding='utf-8', decode_error='ignore') #Example is data stored in a directory called text_data, with subdirs labelled the category
X, y = data.data, data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) #remove words present in > 70% documents.  Remove English stopwords
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier (Linear Kernel)
svm_classifier = SVC(kernel='linear', C=1.0)  # You can experiment with different kernels and C values
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

**Explanation:**

1.  **Data Loading and Splitting:**  Loads the text data and splits it into training and testing sets. Replace `'text_data'` with the directory containing subdirectories each labelled a category, and containing the text files for that category.
2.  **TF-IDF Vectorization:** `TfidfVectorizer` converts the text data into TF-IDF feature vectors. `stop_words='english'` removes common English stop words (e.g., "the," "a," "is"). `max_df=0.7` ignores terms that have a document frequency strictly higher than 70% of the documents. This helps to focus on more informative words. The `fit_transform` method is used on the training data to learn the vocabulary and IDF weights, and then `transform` is used on the testing data to convert it into the same feature space.
3.  **SVM Training:** `SVC` is the SVM classifier implementation in scikit-learn.  `kernel='linear'` specifies a linear kernel. `C` is the regularization parameter, which controls the trade-off between maximizing the margin and minimizing the training error.  A smaller `C` allows for more misclassifications in the training data to achieve a larger margin, which can improve generalization performance. The `fit` method trains the classifier on the training data.
4.  **Prediction and Evaluation:** The `predict` method makes predictions on the test set.  The code then calculates and prints the accuracy and classification report to assess the model's performance.  The classification report provides precision, recall, F1-score, and support for each class.

## 4) Follow-up question

How does the choice of the kernel function (e.g., linear, RBF, polynomial) affect the performance of an SVM in text classification, and how can you choose the best kernel for a specific task?  What are the computational trade-offs of using different kernel functions?