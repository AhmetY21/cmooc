---
title: "Word2Vec: Negative Sampling and Optimization"
date: "2026-02-14"
week: 7
lesson: 1
slug: "word2vec-negative-sampling-and-optimization"
---

# Topic: Word2Vec: Negative Sampling and Optimization

## 1) Formal definition (what is it, and how can we use it?)

**Word2Vec** is a popular technique for creating word embeddings, which are dense, low-dimensional vector representations of words that capture semantic relationships. Word embeddings allow machine learning models to better understand the meaning of words in context and perform various natural language processing tasks more effectively.

**Negative Sampling** is an optimization technique used in Word2Vec, specifically in the Skip-gram model, to address the computational cost of the original softmax approach.  The original softmax requires calculating the probability of a word given its context for *all* words in the vocabulary, which becomes prohibitively expensive for large vocabularies.

Instead of updating all weights for every training example, negative sampling updates only a small sample of weights.  Specifically, for each target word/context word pair:

*   It updates the weights for the target word/context word pair (positive example).
*   It randomly samples a small number of "negative" words from the vocabulary (words that are unlikely to appear in the context of the target word).
*   It updates the weights for these negative examples.

The probability of a word being selected as a negative sample is proportional to its frequency in the corpus raised to the power of 3/4 (0.75). This helps to sample frequent words less often, and rare words more often, leading to better performance. The 3/4 power is empirically determined to work well.

Formally, the objective function for Negative Sampling is to maximize:

```
J = log σ(v_w^T u_c) + Σ_{i=1}^{k} log σ(-v_w^T u_{n_i})
```

where:

*   `w` is the target word.
*   `c` is the context word.
*   `v_w` is the vector representation of the target word.
*   `u_c` is the vector representation of the context word.
*   `n_i` are the `k` negative samples.
*   `σ(x) = 1 / (1 + exp(-x))` is the sigmoid function.

The first term maximizes the probability that the target word and context word are related. The second term minimizes the probability that the target word and the negative samples are related.

**Optimization** in this context primarily refers to the optimization algorithm used to update the word vectors based on the negative sampling loss function. Stochastic Gradient Descent (SGD) and its variants (e.g., Adam, Adagrad) are commonly used. The learning rate is a crucial hyperparameter to tune.

In summary, negative sampling makes Word2Vec scalable to large vocabularies by approximating the full softmax. We use it to efficiently train word embeddings that capture semantic relationships, which can then be used in downstream NLP tasks.

## 2) Application scenario

Consider a sentiment analysis task on a large dataset of customer reviews.  We want to train a model to predict whether a review is positive or negative.

Without word embeddings, we might represent words using one-hot encoding, which results in high-dimensional, sparse vectors that don't capture semantic relationships.  Using these features directly in a sentiment classifier would likely result in poor performance, especially if the dataset is limited or contains rare words.

Instead, we can use Word2Vec with negative sampling to train word embeddings on the customer review dataset (or a larger corpus of text). These embeddings would capture semantic relationships between words, such as "good" being similar to "excellent" and "bad" being similar to "terrible."

We can then use these pre-trained word embeddings as input features to the sentiment classifier. Each word in a review is replaced by its corresponding word vector, and these vectors are aggregated (e.g., averaged or summed) to create a feature vector for the entire review.  The classifier can then learn to associate these vector representations with positive or negative sentiment, leading to improved accuracy and generalization.  Even if the dataset contains rare words, the embeddings may capture enough similarity to existing words to make a good prediction.

## 3) Python method (if possible)

The `gensim` library provides a convenient implementation of Word2Vec with negative sampling.

```python
from gensim.models import Word2Vec

# Sample sentences (replace with your actual corpus)
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["dogs", "are", "loyal", "and", "friendly"],
    ["cats", "are", "independent", "and", "curious"],
    ["the", "cat", "sits", "on", "the", "mat"]
]

# Train Word2Vec model with negative sampling
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, negative=5, workers=4)

# vector_size: Dimensionality of the word vectors.
# window: Maximum distance between the current and predicted word within a sentence.
# min_count: Ignores all words with total frequency lower than this.
# sg: 1 for skip-gram; otherwise, CBOW.
# negative: Number of negative samples to draw for negative sampling.
# workers: Use these many worker threads to train the model.

# Access word vectors
vector_of_word = model.wv["dog"]  # Get the vector for the word "dog"
print(vector_of_word)

# Find similar words
similar_words = model.wv.most_similar("dog", topn=3) #topn specifies top n similar words
print(similar_words)

# Save the model
model.save("word2vec.model")

# Load the model
loaded_model = Word2Vec.load("word2vec.model")

```

## 4) Follow-up question

How does the choice of the number of negative samples (`negative` parameter in `gensim`) affect the performance and training time of Word2Vec? What are some guidelines for selecting an appropriate value for this parameter?