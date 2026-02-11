---
title: "TF-IDF (Term Frequency-Inverse Document Frequency)"
date: "2026-02-11"
week: 7
lesson: 3
slug: "tf-idf-term-frequency-inverse-document-frequency"
---

# Topic: TF-IDF (Term Frequency-Inverse Document Frequency)

## 1) Formal definition (what is it, and how can we use it?)

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval, text mining, and user modeling.

**Term Frequency (TF):** Measures how frequently a term occurs in a document.  The assumption is that the more often a term appears in a document, the more important it is *to that document*. A common formula is:

`TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

There are other variations of TF, such as raw count (just the number of times the term appears) or log-normalized TF.

**Inverse Document Frequency (IDF):** Measures how rare a term is across the entire corpus. The assumption is that terms that appear in many documents are less informative than terms that appear in only a few.  A common formula is:

`IDF(t, D) = log_e(Total number of documents in the corpus / Number of documents containing term t)`

Where:
*   `D` is the set of all documents (the corpus).

**TF-IDF Calculation:**  The TF-IDF score for a term in a document is simply the product of its TF and IDF scores:

`TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

**How can we use it?**

*   **Information Retrieval:** Ranking search results. Documents with higher TF-IDF scores for the query terms are considered more relevant.
*   **Text Summarization:** Identifying important words in a document for generating concise summaries.
*   **Keyword Extraction:**  Identifying keywords that best represent a document.
*   **Document Similarity:**  Comparing documents by calculating the cosine similarity between their TF-IDF vectors. Documents with high cosine similarity are considered more similar.
*   **Classification/Clustering:**  Using TF-IDF vectors as features for machine learning models.

## 2) Application scenario

Imagine you have a collection of news articles about different topics (sports, politics, technology). You want to build a search engine that allows users to find articles related to a specific query.

Let's say a user searches for "artificial intelligence".

1.  **TF-IDF Calculation:**  You would calculate the TF-IDF scores for the terms in each article in your corpus, including the terms "artificial" and "intelligence".  Terms like "the", "a", "is" will likely have low IDF scores because they appear in almost every document.  "Artificial" and "Intelligence" will likely have higher IDF scores since they are relatively less common across the entire corpus. An article specifically *about* artificial intelligence will have high TF values for "artificial" and "intelligence", resulting in a high TF-IDF score.
2.  **Ranking:** The articles are then ranked based on their TF-IDF scores for the query terms. Articles with higher TF-IDF scores are displayed higher in the search results because they are considered more relevant to the query "artificial intelligence".

In this scenario, TF-IDF helps to identify articles that are both about the topic ("artificial intelligence") *and* where the terms are relatively important within the context of the entire collection of news articles.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Example documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array (for easier printing)
dense_tfidf = tfidf_matrix.toarray()

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    for j, term in enumerate(feature_names):
        print(f"  {term}: {dense_tfidf[i][j]:.4f}", end="  ") # Print to 4 decimal places
    print()

# You can also access the TF-IDF scores for a specific term in a specific document:
# For example, the TF-IDF score for "document" in the first document:
term_index = feature_names.tolist().index("document")
print(f"\nTF-IDF for 'document' in document 1: {dense_tfidf[0][term_index]:.4f}")
```

**Explanation:**

*   `TfidfVectorizer` from `sklearn.feature_extraction.text` is a convenient tool for calculating TF-IDF.
*   `fit_transform(documents)` calculates the TF-IDF scores for all terms in all documents. The result is a sparse matrix where rows represent documents and columns represent terms.
*   `get_feature_names_out()` returns a list of the terms (features) in the same order as the columns in the TF-IDF matrix.
*   `toarray()` converts the sparse matrix to a dense array, which is easier to print and work with, but can consume a lot of memory for large datasets.  For very large datasets, it's better to work with the sparse matrix directly.
*   The printed output shows the TF-IDF score for each term in each document. A score of 0 means the term does not appear in the document.  Higher scores indicate higher importance.

## 4) Follow-up question

How can we deal with the problem of TF-IDF favoring longer documents, as longer documents tend to have higher TF values for many terms simply because they have more words in general?  Are there any modifications to the TF-IDF formula or techniques that can mitigate this issue?