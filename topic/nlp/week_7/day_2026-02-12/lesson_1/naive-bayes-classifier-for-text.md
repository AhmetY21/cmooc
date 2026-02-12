---
title: "Naive Bayes Classifier for Text"
date: "2026-02-12"
week: 7
lesson: 1
slug: "naive-bayes-classifier-for-text"
---

# Topic: Naive Bayes Classifier for Text

## 1) Formal definition (what is it, and how can we use it?)

The Naive Bayes Classifier is a probabilistic machine learning algorithm that's particularly well-suited for text classification tasks. It's based on Bayes' Theorem with a "naive" assumption of feature independence. In the context of text, this means the algorithm assumes that the occurrence of a particular word in a document is independent of the occurrence of other words, given the document's class.  While this assumption is rarely true in real-world text (words *do* tend to appear in related clusters), the Naive Bayes classifier often performs surprisingly well, especially for its simplicity and speed.

**Bayes' Theorem:**  The foundation of the classifier. It states:

P(c|d) = [P(d|c) * P(c)] / P(d)

Where:

*   P(c|d) is the *posterior probability* of the document `d` belonging to class `c` (the probability we want to calculate).
*   P(d|c) is the *likelihood* of observing document `d` given class `c`.  This is where the naive assumption comes in.  If `d` is a set of words {w1, w2, ..., wn}, then P(d|c) is approximated as P(w1|c) * P(w2|c) * ... * P(wn|c). Each P(wi|c) is the probability of seeing word *wi* in documents of class *c*.
*   P(c) is the *prior probability* of class `c` (the probability of a document belonging to class `c` regardless of its content).  This is usually estimated from the proportion of documents belonging to each class in the training data.
*   P(d) is the *evidence* (the probability of seeing document `d`).  Since we're only interested in comparing the probabilities for different classes for the *same* document, P(d) acts as a normalizing constant and can often be ignored in the classification process.  We only need to find the class `c` that maximizes P(c|d).

**How we can use it:**

1.  **Training:**  The classifier is trained on a labeled dataset of text documents, where each document is assigned to a specific class (e.g., "spam" or "not spam," "positive review" or "negative review"). During training, the algorithm estimates the probabilities P(wi|c) for each word *wi* in the vocabulary and for each class *c*, and it calculates the prior probabilities P(c).

2.  **Classification:** To classify a new, unseen document, the algorithm calculates P(c|d) for each possible class *c* using the learned probabilities.  The document is then assigned to the class with the highest posterior probability.

**Types of Naive Bayes Classifiers:**

Different variations of Naive Bayes are used depending on the nature of the features:

*   **Multinomial Naive Bayes:**  Best suited for discrete features, such as word counts or term frequencies (how many times a word appears in a document).  This is the most common type used for text classification.
*   **Bernoulli Naive Bayes:** Suitable for binary features (e.g., does a word appear in the document or not?).
*   **Gaussian Naive Bayes:**  Suitable for continuous features. This is generally not used for text directly, but could be used if features like average sentence length or other numerical attributes are used in addition to the text.

## 2) Application scenario

Here are some common application scenarios for Naive Bayes classifiers in text processing:

*   **Spam Filtering:** Identifying email messages as either spam or not spam. This was one of the early and very successful applications of Naive Bayes.
*   **Sentiment Analysis:** Determining the sentiment (positive, negative, neutral) expressed in a piece of text, such as a customer review or a social media post.
*   **Topic Classification:** Assigning documents to different categories or topics, such as sports, politics, or technology.  News categorization is a common use case.
*   **Language Detection:** Identifying the language of a text document.
*   **Author Attribution:** Determining the author of a text based on their writing style (though more sophisticated methods often outperform Naive Bayes in this area).
*   **Fake News Detection:** Identifying potentially false or misleading news articles.

Naive Bayes is often chosen for these applications due to its speed, simplicity, and reasonable accuracy, especially when dealing with large datasets.  It serves as a good baseline model.

## 3) Python method (if possible)

We can use the `scikit-learn` library in Python to implement a Naive Bayes classifier for text classification.  Here's an example using `MultinomialNB` after text vectorization using `TfidfVectorizer`:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your actual data)
data = [
    ("This is a positive movie review.", "positive"),
    ("This is an excellent film.", "positive"),
    ("I did not enjoy this movie at all.", "negative"),
    ("What a terrible film.", "negative"),
    ("This movie was okay.", "neutral"),
]

# Separate text and labels
texts = [text for text, label in data]
labels = [label for text, label in data]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize the text using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)  # Important: only transform the test data

# Create a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vectors, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Example of predicting new text
new_text = ["This is a great movie!"]
new_text_vectors = vectorizer.transform(new_text)
prediction = classifier.predict(new_text_vectors)
print(f"Prediction for new text: {prediction}")
```

**Explanation:**

1.  **Data Preparation:** We create sample data (texts and labels).
2.  **Train/Test Split:**  The data is split into training and testing sets.
3.  **Text Vectorization:** `TfidfVectorizer` converts the text into numerical features that the Naive Bayes classifier can understand.  TF-IDF (Term Frequency-Inverse Document Frequency) measures the importance of words in a document relative to a corpus. The vectorizer is fit on the *training* data and then *transformed* on both training and test data.  This is crucial to avoid data leakage.
4.  **Classifier Initialization:**  `MultinomialNB` is initialized.
5.  **Training:** The classifier is trained using the training data and corresponding labels.
6.  **Prediction:** The classifier predicts labels for the test set.
7.  **Evaluation:** The accuracy and classification report are used to evaluate the model's performance.
8. **New Text Prediction:**  Demonstrates how to predict the class of a new unseen piece of text, making sure to transform it using the *same* vectorizer.

## 4) Follow-up question

While Naive Bayes is computationally efficient and relatively easy to implement, its "naive" independence assumption is often violated in real-world text data. What are some strategies to mitigate the impact of this assumption, or alternative algorithms that address this issue more directly while still maintaining reasonable computational cost? Consider factors such as feature engineering techniques, parameter tuning, and algorithmic alternatives.