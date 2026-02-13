---
title: "Topic Modeling: Latent Semantic Analysis (LSA)"
date: "2026-02-13"
week: 7
lesson: 2
slug: "topic-modeling-latent-semantic-analysis-lsa"
---

# Topic: Topic Modeling: Latent Semantic Analysis (LSA)

## 1) Formal definition (what is it, and how can we use it?)

Latent Semantic Analysis (LSA), also known as Latent Semantic Indexing (LSI), is an unsupervised natural language processing technique used to discover underlying topics in a collection of documents. It aims to find the hidden (latent) semantic relationships between words and documents based on their co-occurrence patterns.  Essentially, it's a way to reduce the dimensionality of the term-document matrix while preserving important semantic relationships.

**What it is:**

*   LSA starts with a term-document matrix, where each row represents a term (word) and each column represents a document. The value in each cell (i,j) represents the frequency (or a weighted version like TF-IDF) of term 'i' in document 'j'.
*   Then, it applies Singular Value Decomposition (SVD) to decompose this matrix into three matrices: U, Σ, and V<sup>T</sup>.  The SVD decomposes the original term-document matrix into three matrices, approximating it.
*   **U (Term-Topic Matrix):** Represents the relationship between terms and topics. Each row corresponds to a term, and each column corresponds to a topic. The values indicate the importance of each term to each topic.
*   **Σ (Singular Values Matrix):** A diagonal matrix containing singular values. These values represent the strength or importance of each topic. We typically select the top *k* singular values (and corresponding rows and columns from U and V<sup>T</sup>) to reduce dimensionality and retain the most important topics.
*   **V<sup>T</sup> (Topic-Document Matrix):** Represents the relationship between topics and documents. Each row corresponds to a topic, and each column corresponds to a document. The values indicate the importance of each topic to each document.

**How we can use it:**

*   **Topic discovery:** LSA helps identify the main topics present in a corpus of documents, even if those topics are not explicitly named.
*   **Document retrieval:** By representing documents and queries in the same latent semantic space, LSA allows for semantic matching, improving the accuracy of information retrieval systems. Documents that are semantically similar but do not share many words can be identified as relevant.
*   **Document similarity:** By comparing the vectors in the topic-document space (V<sup>T</sup>), we can measure the similarity between documents.
*   **Dimensionality reduction:** LSA reduces the dimensionality of the term-document matrix, making it more efficient to store and process text data.
*   **Text Summarization:** LSA can identify the most important sentences/passages related to identified topics, forming the basis for text summarization.

## 2) Application scenario

Imagine you have a large collection of news articles from different sources covering various topics. You want to automatically group these articles into different categories (e.g., "Politics," "Sports," "Technology," "Business") without manually reading and labeling each article.

LSA can be used to:

1.  **Build a term-document matrix:** Represent each article as a document, and each unique word as a term. The matrix entries contain TF-IDF scores for each word in each document.
2.  **Apply SVD:** Decompose the term-document matrix using SVD and reduce the dimensionality by selecting the top *k* singular values (and corresponding vectors).
3.  **Identify topics:** Analyze the term-topic matrix (U) to identify the most important words associated with each of the *k* topics. For example, topic 1 might have words like "election," "candidate," "vote," suggesting it's related to "Politics."
4.  **Assign articles to topics:** Analyze the topic-document matrix (V<sup>T</sup>) to determine the strength of each topic in each article.  Assign each article to the topic with the highest score.
5.  **Recommend related articles:** Once articles are grouped by topics, you can easily recommend similar articles to users based on their reading history or interests.

## 3) Python method (if possible)

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "Dogs are loyal pets.",
    "Cats are independent animals.",
    "Space exploration is fascinating.",
    "The moon is a satellite of Earth.",
    "Rockets are used for space travel."
]

# 1. Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english') # Remove common English words
X = vectorizer.fit_transform(documents)

# 2. Apply Truncated SVD (LSA)
n_components = 3  # Number of topics to extract
lsa = TruncatedSVD(n_components=n_components, algorithm = 'arpack') # Using arpack for sparse data.

lsa.fit(X)
topic_vectors = lsa.transform(X)  # Documents represented in the topic space

# 3. Print topics and their top words
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(lsa.components_):
    terms_with_score = zip(terms, comp)
    sorted_terms = sorted(terms_with_score, key=lambda x: x[1], reverse=True)[:10]  # Top 10 words
    print(f"Topic {i+1}:")
    for term, score in sorted_terms:
        print(f"\t{term}: {score:.3f}")

# 4. Document representation in topic space:
# topic_vectors[i] now represents the i-th document in the lower-dimensional topic space.
# You can then use these vectors for document similarity or clustering.

print("\nDocument representation in topic space:")
for i, doc_vector in enumerate(topic_vectors):
    print(f"Document {i+1}: {doc_vector}")
```

**Explanation:**

1.  **TF-IDF Vectorization:** We use `TfidfVectorizer` from scikit-learn to create the term-document matrix and apply TF-IDF weighting, which normalizes the word frequencies.  Stop words are removed to focus on more meaningful terms.
2.  **TruncatedSVD:**  `TruncatedSVD` is used for dimensionality reduction. It's computationally more efficient than full SVD when dealing with large sparse matrices, which is common in text data.  We set `n_components` to specify the number of topics to extract. We also specified `algorithm='arpack'` which is usually better for sparse data.
3.  **Topic Interpretation:**  We iterate through the components (topics) of the LSA model and print the top words associated with each topic based on their scores in the `lsa.components_` matrix. This helps us interpret the meaning of each topic.  `terms` variable contains the vocabulary.
4. **Document representation:** After `lsa.transform(X)` we obtained document vectors. These vectors represent how each document is related to each of the topic vectors.

## 4) Follow-up question

How does LSA handle polysemy (words with multiple meanings) and synonymy (multiple words with the same meaning), and what are the limitations of LSA in addressing these challenges? How can other topic modeling techniques, like LDA, overcome some of these limitations?