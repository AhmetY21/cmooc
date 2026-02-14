---
title: "GloVe (Global Vectors for Word Representation)"
date: "2026-02-14"
week: 7
lesson: 2
slug: "glove-global-vectors-for-word-representation"
---

# Topic: GloVe (Global Vectors for Word Representation)

## 1) Formal definition (what is it, and how can we use it?)

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations of words. It's based on aggregating global word-word co-occurrence statistics from a corpus. Unlike methods like Word2Vec, which learn embeddings based on local context windows, GloVe leverages the entire corpus to capture global statistics.

Specifically, GloVe aims to learn word vectors such that the dot product of two word vectors is equal to the logarithm of their frequency of co-occurrence.  Formally:

Let:

*   `X` be the word-word co-occurrence matrix. `X_ij` represents the number of times word `j` appears in the context of word `i`.
*   `X_i = sum_j X_ij` be the number of times any word appears in the context of word `i`.
*   `P_ij = P(j|i) = X_ij / X_i` be the probability that word `j` appears in the context of word `i`.
*   `v_i` and `v_j` be the word vectors for words `i` and `j`, respectively.
*   `b_i` and `b_j` are bias terms for words `i` and `j`, respectively.

GloVe's objective function is to minimize the following cost function:

```
J = sum_{i=1}^{V} sum_{j=1}^{V} f(X_{ij}) (v_i^T v_j + b_i + b_j - log(X_{ij}))^2
```

where:

*   `V` is the vocabulary size.
*   `f(X_{ij})` is a weighting function that helps prevent frequently co-occurring words from dominating the learning process.  A typical weighting function is:

    ```
    f(x) = (x/x_{max})^alpha  if x < x_{max}
           1                       otherwise
    ```

    where `x_{max}` is usually set to 100, and `alpha` is typically 0.75.

**How can we use it?**

GloVe word embeddings can be used in a variety of downstream NLP tasks, including:

*   **Word Similarity/Analogy tasks:** Measuring semantic similarity between words by calculating the cosine similarity between their GloVe vectors. GloVe shines in analogy tasks (e.g., "king - man + woman = queen").
*   **Text Classification:** Used as input features to train text classifiers. The word vectors can be averaged, summed, or concatenated to represent a document.
*   **Named Entity Recognition (NER):** Providing word embeddings as features for NER models.
*   **Machine Translation:** Contributing to better word alignment and semantic understanding in machine translation systems.
*   **Question Answering:**  Helping the model to understand the relationship between words in the question and the answer.

## 2) Application scenario

Imagine you are building a sentiment analysis model for movie reviews.  You want your model to understand the meaning of words like "fantastic," "terrible," "amazing," and "awful."  Instead of training word embeddings from scratch on your relatively small dataset of movie reviews, you can use pre-trained GloVe embeddings.

You download a pre-trained GloVe model trained on a much larger corpus (e.g., Wikipedia and Gigaword). These embeddings have already learned the semantic relationships between words based on their co-occurrence patterns in the massive corpus.  Therefore:

1.  You map each word in your movie reviews to its corresponding GloVe vector.  If a word is not in the GloVe vocabulary, you can handle it using techniques like randomly initializing a vector or using a special `<UNK>` token vector.
2.  You use these GloVe vectors as input features to your sentiment analysis model (e.g., a Recurrent Neural Network or a Convolutional Neural Network).

By using pre-trained GloVe embeddings, your sentiment analysis model can leverage the pre-existing knowledge of word meanings, leading to better performance, especially when you have limited training data.  The model can recognize that "fantastic" and "amazing" are semantically similar and have a positive sentiment, while "terrible" and "awful" are semantically similar and have a negative sentiment, even if those exact words don't appear frequently in your specific movie review dataset.

## 3) Python method (if possible)

While you won't train GloVe from scratch using Python in a typical application (due to computational complexity and the existence of pre-trained models), you *can* load and use pre-trained GloVe embeddings using libraries like Gensim or directly reading the embedding file.

Here's an example using Gensim to load and use pre-trained GloVe vectors:

```python
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np

# Download the GloVe embeddings (e.g., glove-wiki-gigaword-100)
# This only needs to be done once.  Uncomment the line below to download:
# glove_vectors = api.load('glove-wiki-gigaword-100') # Takes some time.

# Load pre-trained GloVe vectors
try:
    glove_vectors = api.load('glove-wiki-gigaword-100')
except ValueError:
    print("GloVe model not found. Downloading...")
    glove_vectors = api.load('glove-wiki-gigaword-100')
except OSError:
    print("Could not load pre-trained GloVe model.  Check your internet connection and try again.")
    exit()


# Example usage:
word1 = "king"
word2 = "queen"
word3 = "man"
word4 = "woman"

# Calculate cosine similarity between words
similarity = glove_vectors.similarity(word1, word2)
print(f"Similarity between '{word1}' and '{word2}': {similarity}")

# Solve an analogy:  king - man + woman = ?
result = glove_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Analogy result for '{word1} - {word3} + {word4}': {result}")

# Get the vector representation of a word
vector = glove_vectors[word1]
print(f"Vector representation of '{word1}': {vector[:10]}...") # Print first 10 elements

# Check if a word is in the vocabulary:
if "apple" in glove_vectors:
    print("The word 'apple' is in the GloVe vocabulary.")
else:
    print("The word 'apple' is not in the GloVe vocabulary.")


# Example for handling out-of-vocabulary words (simplified):
def get_embedding(word, model):
    try:
        return model[word]
    except KeyError:
        # Handle out-of-vocabulary words.  Here, returning a zero vector.
        # More sophisticated approaches exist (e.g., random initialization, character-level embeddings).
        print(f"Warning: Word '{word}' not found in vocabulary. Returning a zero vector.")
        return np.zeros(model.vector_size)


word_not_in_vocab = "unbelievablysupercalifragilisticexpialidocious"
embedding = get_embedding(word_not_in_vocab, glove_vectors)
print(f"Embedding for '{word_not_in_vocab}': {embedding[:10]}...") # Prints all zeros

```

This code downloads and loads a pre-trained GloVe model, demonstrates how to calculate word similarity, solve analogies, retrieve word vectors, and handle out-of-vocabulary words.  Remember to install Gensim if you haven't already (`pip install gensim`). Also, the first time you run the script, it will download the glove embeddings, which can take some time.

## 4) Follow-up question

What are some techniques for handling out-of-vocabulary (OOV) words when using pre-trained GloVe embeddings, and what are the trade-offs between these techniques?