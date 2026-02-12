---
title: "Decision Trees and Random Forests for Text"
date: "2026-02-12"
week: 7
lesson: 4
slug: "decision-trees-and-random-forests-for-text"
---

# Topic: Decision Trees and Random Forests for Text

## 1) Formal definition (what is it, and how can we use it?)

**Decision Trees:** A decision tree is a supervised learning algorithm that uses a tree-like structure to model decisions and their possible consequences. In the context of text, a decision tree learns to classify text documents or predict a target variable based on features extracted from the text.  Each internal node in the tree represents a test on an attribute (e.g., the presence or absence of a specific word, or the frequency of a particular n-gram). Each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a predicted value (for regression). The algorithm recursively splits the data based on the most informative attribute, aiming to create homogeneous subsets with respect to the target variable.  Common splitting criteria include Information Gain, Gini Impurity, and Variance Reduction.

**Random Forests:** A random forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (for classification) or the mean/average prediction (for regression) of the individual trees.  Randomness is introduced in two key ways:

*   **Bagging (Bootstrap Aggregating):** Each tree is trained on a random subset of the training data, sampled *with replacement*. This creates diverse training sets for each tree.
*   **Random Subspace:** At each node in the tree, the algorithm considers only a random subset of the features when deciding how to split the node. This further decorrelates the trees.

**How we can use them for text:**

*   **Text Classification:** Categorizing documents into predefined classes (e.g., spam/not spam, positive/negative/neutral sentiment, news categories like sports/politics/technology).
*   **Topic Modeling:** While not their primary use, decision trees can be used as part of a more complex topic modeling pipeline. They can help identify key terms associated with different topics.
*   **Sentiment Analysis:** Determining the emotional tone or attitude expressed in a piece of text.
*   **Spam Detection:** Identifying unwanted or unsolicited messages.
*   **Information Retrieval:** Ranking documents based on their relevance to a search query (although more advanced methods like ranking SVMs and neural networks are more common here).
*   **Author Attribution:** Identifying the author of a text based on stylistic features.

**Feature extraction is critical**.  Before feeding text data to a decision tree or random forest, it must be transformed into numerical features. Common techniques include:

*   **Bag-of-Words (BoW):** Represents a document as a vector of word counts.
*   **Term Frequency-Inverse Document Frequency (TF-IDF):** Weights words based on their frequency in a document and their inverse document frequency across the corpus.
*   **Word Embeddings:**  Represents words as dense vectors, capturing semantic relationships (e.g., using pre-trained models like Word2Vec, GloVe, or FastText).
*   **N-grams:** Sequences of N consecutive words, providing more contextual information than individual words.

## 2) Application scenario

**Scenario:** Sentiment Analysis of Customer Reviews for an E-commerce Website

An e-commerce company wants to automatically analyze customer reviews to understand customer sentiment towards their products. They have a large dataset of customer reviews, each labeled with a sentiment score (e.g., positive, negative, neutral).

**Using Random Forest:**

1.  **Data Preprocessing:** The text reviews are preprocessed by removing punctuation, converting to lowercase, and potentially stemming or lemmatizing words.

2.  **Feature Extraction:** TF-IDF is used to convert the text reviews into numerical feature vectors. Each feature represents the TF-IDF weight of a particular word in the review.

3.  **Model Training:** A Random Forest classifier is trained on the labeled reviews, using the TF-IDF feature vectors as input and the sentiment score as the target variable.  Hyperparameter tuning (e.g., number of trees, maximum tree depth) is performed using cross-validation to optimize the model's performance.

4.  **Sentiment Prediction:** The trained Random Forest model is used to predict the sentiment of new, unseen customer reviews.

5.  **Analysis and Insights:** The company can analyze the predicted sentiment scores to identify products with predominantly positive or negative reviews. They can also examine the features (words) that the Random Forest model considers most important for predicting sentiment. This helps them understand what aspects of their products are driving customer satisfaction or dissatisfaction.

**Why Random Forest is suitable here:**

*   **Handles high-dimensional data:** TF-IDF typically results in a large number of features (one for each word), which Random Forests can handle effectively.
*   **Robust to overfitting:** The ensemble nature of Random Forests helps prevent overfitting, which is especially important when dealing with noisy text data.
*   **Relatively easy to interpret:**  Feature importance scores can be used to understand which words are most indicative of positive or negative sentiment.
*   **Good accuracy:**  Random Forests often achieve good performance on text classification tasks.

## 3) Python method (if possible)

```python
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources (run this once)
# nltk.download('stopwords')
# nltk.download('punkt')


def preprocess_text(text):
    """Preprocesses the text by removing punctuation, converting to lowercase, removing stopwords, and stemming."""
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.lower()  # Convert to lowercase

    # Tokenization and Stopword Removal
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]


    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(stemmed_tokens)


# Sample data (replace with your actual dataset)
data = [
    ("This is a great product!", "positive"),
    ("I am very happy with my purchase.", "positive"),
    ("This is terrible. I want a refund.", "negative"),
    ("The product is okay, nothing special.", "neutral"),
    ("Absolutely love it!", "positive"),
    ("Very disappointed with the quality.", "negative"),
    ("It's alright.", "neutral"),
    ("The best ever!", "positive"),
    ("A complete waste of money.", "negative"),
    ("Mediocre at best.", "neutral")
]

texts, labels = zip(*data)

# Preprocess the texts
processed_texts = [preprocess_text(text) for text in texts]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100) # Limit features to top 100 words
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) # Adjust hyperparameters as needed
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
```

## 4) Follow-up question

What are some limitations of using decision trees and random forests for text data, and what are some alternative or more advanced methods that can address these limitations?