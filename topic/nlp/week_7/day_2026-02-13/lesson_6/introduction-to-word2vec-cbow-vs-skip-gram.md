---
title: "Introduction to Word2Vec (CBOW vs Skip-gram)"
date: "2026-02-13"
week: 7
lesson: 6
slug: "introduction-to-word2vec-cbow-vs-skip-gram"
---

# Topic: Introduction to Word2Vec (CBOW vs Skip-gram)

## 1) Formal definition (what is it, and how can we use it?)

Word2Vec is a group of related models used to produce word embeddings. Word embeddings are vector representations of words that capture semantic meaning and relationships between words in a given corpus. Unlike one-hot encoding which represents each word as a unique vector with all elements being 0 except for one element being 1, word embeddings are dense, low-dimensional vectors (typically 100-1000 dimensions) where similar words are closer together in the vector space.

Word2Vec primarily consists of two model architectures:

*   **Continuous Bag-of-Words (CBOW):** Predicts a target word given the context of surrounding words. The input to the model is the surrounding words, and the output is the target word. It aims to learn the probability of a word given its context.

*   **Skip-gram:** Predicts the surrounding words given a target word.  The input is the target word, and the output is the surrounding words. It aims to learn the probability of a context given a word.

How we can use it:

*   **Semantic similarity:** Determine the similarity between words based on the distance between their embeddings. Cosine similarity is often used.
*   **Analogy reasoning:**  Solve analogies like "man is to king as woman is to ____".  We can find the word whose embedding is closest to embedding("king") - embedding("man") + embedding("woman").
*   **Downstream tasks:** Use pre-trained word embeddings as input features for other NLP tasks like text classification, sentiment analysis, machine translation, and named entity recognition.
*   **Vocabulary expansion:** Suggest related words for a given word.
*   **Feature Engineering:** Generate more effective features for Machine Learning models in any task that utilizes text.

## 2) Application scenario

**Scenario:** Building a sentiment analysis model for movie reviews.

**Without Word2Vec:** We might use techniques like bag-of-words or TF-IDF to represent each review as a vector of word counts. These methods ignore the semantic relationships between words.  For example, "good" and "amazing" are considered completely unrelated.

**With Word2Vec:**

1.  **Pre-train Word2Vec:** Train a Word2Vec model (either CBOW or Skip-gram) on a large corpus of text (e.g., Wikipedia, movie review datasets).
2.  **Represent Reviews:** Represent each word in a review using its corresponding Word2Vec embedding.
3.  **Aggregate Word Embeddings:** Average the word embeddings in each review to obtain a single vector representing the entire review.  Alternatively, more sophisticated techniques like using recurrent neural networks (RNNs) or transformers can be employed, initialized with the Word2Vec embeddings.
4.  **Train Sentiment Classifier:** Train a sentiment classifier (e.g., logistic regression, support vector machine, neural network) using the aggregated review vectors as input features and the sentiment label (positive/negative) as the target variable.

By using Word2Vec, the sentiment analysis model can leverage the semantic relationships between words. For instance, the model can learn that "good" and "amazing" are similar and contribute positively to the sentiment score, leading to improved accuracy compared to methods that treat words as independent entities.  Word embeddings often capture subtle nuances of language better than simpler approaches.

## 3) Python method (if possible)

We can use the `gensim` library in Python to train Word2Vec models.

```python
from gensim.models import Word2Vec

# Sample sentences
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["dogs", "are", "loyal", "pets"],
    ["cats", "are", "independent", "animals"],
    ["foxes", "are", "clever"],
    ["the", "dog", "chases", "the", "cat"]
]

# Train CBOW model
model_cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)  # sg=0 for CBOW

# Train Skip-gram model
model_skipgram = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1) # sg=1 for Skip-gram

# Get vector for a word
vector_dog_cbow = model_cbow.wv['dog']
vector_dog_skipgram = model_skipgram.wv['dog']

# Find similar words
similar_words_cbow = model_cbow.wv.most_similar('dog', topn=3)
similar_words_skipgram = model_skipgram.wv.most_similar('dog', topn=3)

print("CBOW vector for 'dog':", vector_dog_cbow)
print("Skip-gram vector for 'dog':", vector_dog_skipgram)
print("CBOW similar to 'dog':", similar_words_cbow)
print("Skip-gram similar to 'dog':", similar_words_skipgram)

# Save and load the model
model_cbow.save("word2vec_cbow.model")
loaded_model_cbow = Word2Vec.load("word2vec_cbow.model")
```

Explanation:

*   `sentences`: A list of sentences, where each sentence is a list of words.  This is the training corpus.
*   `Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0/1)`: Creates and trains the Word2Vec model.
    *   `vector_size`: The dimensionality of the word vectors (e.g., 100).
    *   `window`: The maximum distance between the current and predicted word within a sentence.
    *   `min_count`: Ignores all words with total frequency lower than this.
    *   `workers`: Use these many worker threads to train the model (=faster training).
    *   `sg`: Training algorithm: 1 for skip-gram; otherwise CBOW.
*   `model.wv['dog']`: Returns the vector representation of the word "dog".
*   `model.wv.most_similar('dog', topn=3)`: Returns the top 3 most similar words to "dog".
*   `model.save()` and `Word2Vec.load()`: Used to save and load the trained model.

## 4) Follow-up question

Given the choice between CBOW and Skip-gram, how would you decide which model to use for a specific NLP task, considering factors like dataset size, desired accuracy, and computational resources? What are the typical trade-offs involved?