---
title: "Logistic Regression for NLP"
date: "2026-02-12"
week: 7
lesson: 2
slug: "logistic-regression-for-nlp"
---

# Topic: Logistic Regression for NLP

## 1) Formal definition (what is it, and how can we use it?)

Logistic Regression, in the context of NLP, is a statistical method used for binary or multi-class classification tasks where the goal is to predict the probability of a sample belonging to a particular class. Although named "regression," it's fundamentally a classification algorithm.

Here's a breakdown:

*   **The core idea:** Logistic regression models the probability of a binary outcome (e.g., spam/not spam, positive sentiment/negative sentiment) as a function of input features. It uses the sigmoid function (also known as the logistic function) to map the linear combination of input features to a probability between 0 and 1.

*   **Mathematical formulation:**
    *   Let *x* be a vector of input features (e.g., word counts, TF-IDF values).
    *   Let *w* be a vector of weights corresponding to each feature.
    *   Let *b* be a bias term (also called the intercept).
    *   The linear combination of features is:  *z = w<sup>T</sup>x + b*
    *   The sigmoid function, *σ(z)*, is defined as:  *σ(z) = 1 / (1 + e<sup>-z</sup>)*
    *   The predicted probability *P(y=1|x)* of belonging to class 1 is: *P(y=1|x) = σ(w<sup>T</sup>x + b)*
    *   The predicted probability *P(y=0|x)* of belonging to class 0 is: *P(y=0|x) = 1 - P(y=1|x)*

*   **Multi-class extension:**  For multi-class classification (e.g., sentiment analysis with positive, negative, and neutral classes), logistic regression can be extended using techniques like "one-vs-rest" (OvR) or "multinomial logistic regression" (also called softmax regression). In OvR, a separate logistic regression model is trained for each class, treating that class as the positive class and all other classes as the negative class.  Softmax regression directly models the probabilities of all classes, ensuring they sum to 1.

*   **How we use it in NLP:**
    1.  **Feature Extraction:**  We extract relevant features from the text data. Common features include:
        *   **Bag-of-Words (BoW):** Count the occurrences of each word in a document.
        *   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their frequency in the document and their rarity across the entire corpus.
        *   **N-grams:** Sequences of *n* words (e.g., "very good" as a 2-gram).
        *   **Word Embeddings (Word2Vec, GloVe, FastText):**  Represent words as dense vectors capturing semantic relationships.
    2.  **Training:**  We train the logistic regression model using labeled data (text documents with corresponding class labels). The model learns the optimal weights *w* and bias *b* that best separate the classes.
    3.  **Prediction:**  For new, unseen text data, we extract the same features and use the trained model to predict the probability of belonging to each class. The class with the highest probability is assigned as the predicted class.

## 2) Application scenario

A common application scenario is **Spam Detection**.

*   **Problem:**  Given an email, determine whether it is spam or not spam (ham).

*   **Features:**
    *   Word frequencies (e.g., the number of times words like "urgent," "free," "discount" appear).
    *   Presence of certain phrases (e.g., "click here," "limited time offer").
    *   Sender's email address.
    *   Subject line length.
    *   Use of excessive punctuation.

*   **Logistic Regression Model:**  A logistic regression model is trained on a dataset of labeled emails (spam or ham) using these features.  The model learns the weights associated with each feature, indicating how much each feature contributes to the probability of an email being spam.  For instance, a high frequency of the word "urgent" might have a high positive weight, increasing the probability of the email being classified as spam.

*   **Prediction:** When a new email arrives, the same features are extracted. The trained logistic regression model calculates the probability of the email being spam. If the probability exceeds a certain threshold (e.g., 0.5), the email is classified as spam; otherwise, it is classified as ham.

## 3) Python method (if possible)

The `sklearn` library in Python provides a straightforward way to implement Logistic Regression.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
documents = [
    "This is a great movie!",
    "I hated the movie, it was terrible.",
    "Free offer, click here!",
    "Important information regarding your account.",
    "Excellent service, highly recommended."
]
labels = [1, 0, 0, 1, 1]  # 1 = positive/ham, 0 = negative/spam

# 1. Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 3. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example of predicting on a new document
new_document = ["This is a scam!"]
new_X = vectorizer.transform(new_document) # Important: Use transform, not fit_transform
prediction = model.predict(new_X)[0]
print(f"Prediction for '{new_document[0]}': {prediction}")
```

**Explanation:**

1.  **`TfidfVectorizer`:**  This is used to convert the text documents into a matrix of TF-IDF features.  `fit_transform` is used on the training data to learn the vocabulary and transform it. `transform` is used on the test data and any new data to apply the learned vocabulary without learning new terms. Using `fit_transform` on the test data would lead to data leakage and incorrect evaluation.

2.  **`train_test_split`:**  The data is split into training and testing sets to evaluate the model's performance on unseen data.

3.  **`LogisticRegression`:**  A Logistic Regression model is initialized and trained on the training data using `model.fit(X_train, y_train)`.

4.  **`model.predict(X_test)`:**  The trained model is used to predict the labels for the test data.

5.  **`accuracy_score`:** The accuracy of the model is calculated by comparing the predicted labels to the true labels.

## 4) Follow-up question

How does the performance of Logistic Regression for NLP compare to more complex models like Recurrent Neural Networks (RNNs) or Transformers (e.g., BERT) for tasks like sentiment analysis or text classification, and when would you choose Logistic Regression over these more advanced models?  Consider factors such as dataset size, computational resources, interpretability, and desired accuracy.