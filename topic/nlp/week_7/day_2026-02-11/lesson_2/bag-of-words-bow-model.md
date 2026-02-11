---
title: "Bag of Words (BoW) Model"
date: "2026-02-11"
week: 7
lesson: 2
slug: "bag-of-words-bow-model"
---

# Topic: Bag of Words (BoW) Model

## 1) Formal definition (what is it, and how can we use it?)

The Bag of Words (BoW) model is a simplifying representation used in natural language processing and information retrieval.  It disregards grammar and word order, focusing solely on the frequency of words within a document. In essence, it treats a document as an unordered "bag" containing its words.

Formally, a BoW model represents a document as a multiset of its words, disregarding grammar and even word order but keeping multiplicity. This means each word's occurrence count is important. A vocabulary of all words across the corpus is first created. Each document is then represented as a vector, where each element corresponds to a word in the vocabulary and its value represents the frequency of that word in the document.

We can use the BoW model for:

*   **Text Classification:** Categorizing documents based on their word content.
*   **Sentiment Analysis:** Determining the sentiment (positive, negative, neutral) expressed in a text.
*   **Information Retrieval:** Finding documents relevant to a query based on overlapping word frequencies.
*   **Topic Modeling:** Discovering underlying themes or topics within a collection of documents.
*   **Document Similarity:** Calculating the similarity between documents based on the overlap of their word frequencies.

The simplicity of BoW makes it computationally efficient and relatively easy to implement. However, its primary limitation is the loss of semantic information due to the disregard for word order and grammar.  "This is good, not bad" and "This is bad, not good" would be represented similarly, despite having opposite meanings.

## 2) Application scenario

Let's consider a text classification task: spam email detection.

Suppose we have two emails:

*   Email 1 (Spam): "Free money! Win a prize! Claim now for immediate cash!"
*   Email 2 (Ham): "Hi, I'm writing to confirm our meeting for tomorrow."

We can apply the BoW model as follows:

1.  **Vocabulary Creation:** Identify all unique words across both emails: `{"free", "money", "win", "a", "prize", "claim", "now", "for", "immediate", "cash", "hi", "i'm", "writing", "to", "confirm", "our", "meeting", "tomorrow"}`.

2.  **Vector Representation:** Represent each email as a vector, counting the occurrences of each word in the vocabulary:

    *   Email 1: `[1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]`
    *   Email 2: `[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]`

3.  **Classification:**  A classifier (e.g., Naive Bayes, Logistic Regression) can then be trained on these vector representations along with their corresponding labels (spam/ham).  The classifier learns to associate certain word frequencies with spam or ham and can then predict the category of new, unseen emails based on their BoW representations. Higher frequencies of words like "free", "money", and "cash" would likely be associated with spam.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the documents
X = vectorizer.fit_transform(documents)

# Get the vocabulary
vocabulary = vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense array for easier viewing
dense_matrix = X.toarray()

# Print the vocabulary and the document-term matrix
print("Vocabulary:", vocabulary)
print("Document-Term Matrix:\n", dense_matrix)

# Example of how to transform new text
new_text = ["This is a new document."]
new_vector = vectorizer.transform(new_text)
print("\nVector for new text:", new_vector.toarray())
```

This code uses scikit-learn's `CountVectorizer` to create the BoW representation. `fit_transform` learns the vocabulary from the documents and transforms them into a sparse matrix representing the word counts. `get_feature_names_out` returns the learned vocabulary, and `transform` can be used to transform new, unseen text into a BoW vector using the existing vocabulary.  The `toarray()` method converts the sparse matrix to a dense numpy array, which is easier to read, but uses more memory.  For large datasets, working with sparse matrices is more efficient.

## 4) Follow-up question

What are some common techniques used to improve the performance of the BoW model, and how do they address the model's limitations? Consider issues like term frequency bias, the presence of very common words that don't contribute to meaning (stop words), and the lack of semantic understanding.