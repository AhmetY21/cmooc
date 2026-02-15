---
title: "Limitations of Static Word Embeddings"
date: "2026-02-15"
week: 7
lesson: 1
slug: "limitations-of-static-word-embeddings"
---

# Topic: Limitations of Static Word Embeddings

## 1) Formal definition (what is it, and how can we use it?)

Static word embeddings are pre-trained vector representations of words where each word is assigned a single, fixed vector in the embedding space. These embeddings are typically generated using methods like Word2Vec (CBOW and Skip-gram), GloVe (Global Vectors for Word Representation), and FastText. The core idea is that words appearing in similar contexts will have closer vector representations, capturing semantic relationships.

**How we use them:**

*   **Initialization for downstream tasks:**  Static word embeddings are often used to initialize the embedding layers of neural networks for NLP tasks like text classification, sentiment analysis, machine translation, and named entity recognition.  Instead of randomly initializing the embedding layer, we start with pre-trained embeddings, which provide a strong prior knowledge about word relationships.

*   **Feature engineering:** Static word embeddings can be used as features in traditional machine learning models (e.g., logistic regression, SVM) after some aggregation (e.g., averaging the embeddings of all words in a document).

*   **Semantic similarity analysis:** We can compute the cosine similarity between the embeddings of different words to assess their semantic similarity.  This is useful for tasks like synonym detection and identifying related concepts.

**Limitations:** The key limitation is that **each word has only one representation regardless of its context**. This means words with multiple meanings (polysemy) or words whose meaning shifts depending on context are poorly represented. For example, the word "bank" will have a single embedding, even though it can refer to a financial institution or the side of a river.  This inability to handle context-dependent meaning hinders performance in tasks requiring nuanced understanding of language. Furthermore, static embeddings are fixed after training; they cannot be updated during the training of downstream tasks, potentially limiting adaptability to specific task requirements. Finally, they often struggle with rare or out-of-vocabulary words.
## 2) Application scenario

Consider a sentiment analysis task on customer reviews for a new product. Let's say some reviews mention the word "apple."

*   **Scenario 1:** "The apple product is sleek and innovative." Here, "apple" refers to the company Apple and its products.

*   **Scenario 2:** "The apple I received was bruised and rotten." Here, "apple" refers to the fruit.

A static word embedding model will assign the *same* vector to "apple" in both sentences. This means the sentiment analyzer won't be able to distinguish between positive sentiment toward Apple products in the first sentence and negative sentiment toward a defective fruit in the second sentence. The single vector represents an average of *all* uses of "apple" seen during its training, leading to potentially incorrect sentiment classification. The model would struggle to correctly infer the user’s intended meaning in each case because it cannot leverage context.
## 3) Python method (if possible)

```python
import gensim.downloader as api
import numpy as np

# Load pre-trained Word2Vec model
wv = api.load('word2vec-google-news-300') #Download a common pre-trained model

# Example sentences
sentence1 = "The bank near the river is very calm."
sentence2 = "I need to deposit money at the bank."

# Get embeddings
embedding_bank_river = wv['bank']
embedding_bank_money = wv['bank'] #Same vector despite different contexts

# Calculate cosine similarity
similarity = np.dot(embedding_bank_river, embedding_bank_money) / (np.linalg.norm(embedding_bank_river) * np.linalg.norm(embedding_bank_money))

print(f"Cosine similarity between 'bank' in both sentences: {similarity}") #Output will be close to 1, indicating high similarity despite the difference in meaning.
```

This code demonstrates how static word embeddings assign the same vector to the word "bank" regardless of its context.  The cosine similarity, being near 1, shows the model doesn't differentiate the different meanings, highlighting the limitation of static embeddings in handling polysemy. Using contextual embeddings like those from BERT, RoBERTa, or other transformer-based models would provide significantly different embeddings for "bank" in each of these sentences.
## 4) Follow-up question

Given the limitations of static word embeddings, what are some alternative approaches that address the issue of context dependency and polysemy in word representations? Explain how these approaches improve upon static embeddings.