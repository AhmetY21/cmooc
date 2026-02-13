---
title: "Distributional Semantics and Word Vectors"
date: "2026-02-13"
week: 7
lesson: 5
slug: "distributional-semantics-and-word-vectors"
---

# Topic: Distributional Semantics and Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

Distributional semantics is a theory that states that words that occur in similar contexts tend to have similar meanings. This is often summarized as "You shall know a word by the company it keeps" (J.R. Firth). In simpler terms, the meaning of a word is determined by the words that frequently appear around it.

Word vectors, also known as word embeddings, are numerical representations of words in a multi-dimensional space. These vectors are learned from large corpora of text using techniques based on distributional semantics. The core idea is that words with similar contexts will have similar vector representations. The distance between word vectors (e.g., cosine similarity) reflects the semantic similarity between the corresponding words.

**How can we use it?**

*   **Semantic Similarity:** Determine how similar two words are in meaning.
*   **Word Analogy:** Solve analogies like "man is to woman as king is to ____" by performing vector arithmetic (king - man + woman ≈ queen).
*   **Text Classification:** Use word vectors as features in machine learning models for tasks like sentiment analysis or topic classification.
*   **Machine Translation:** Map words from one language to another by aligning their vector spaces.
*   **Information Retrieval:** Improve search relevance by matching queries to documents based on semantic similarity rather than just keyword matching.
*   **Question Answering:** Understand the meaning of questions and find relevant answers based on semantic relationships.
*   **Named Entity Recognition:** Improve accuracy by understanding the context of named entities.

## 2) Application scenario

**Scenario:** Building a recommendation system for movies.

**How distributional semantics and word vectors can help:**

Instead of relying solely on genre or keyword matching, we can analyze the plot synopses of movies. We can train a word embedding model on a large corpus of movie descriptions and reviews. This model will learn vector representations for words.

1.  When a user rates a movie highly (e.g., "The Matrix"), we can identify the keywords related to that movie using the trained word embeddings.
2.  We can then search for other movies whose plot synopses contain words with high cosine similarity to the keywords identified in step 1.
3.  This allows the recommendation system to suggest movies that are semantically similar to the user's preferred movie, even if they don't share the exact same genre or keywords. For example, the system might recommend "Inception" because its plot synopsis contains words that are semantically similar to those found in "The Matrix" (e.g., "simulation," "reality," "dreams," "conspiracy").

This approach captures more nuanced relationships between movies than simply matching keywords, leading to better and more relevant recommendations.

## 3) Python method (if possible)

Here's how to train word vectors using the `gensim` library in Python:

```python
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown

# Download Brown corpus if you haven't already
try:
    brown.words()
except LookupError:
    nltk.download('brown')


# Load the Brown corpus (a collection of text)
sentences = brown.sents()

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# `vector_size`: Dimensionality of the word vectors
# `window`: Maximum distance between the current and predicted word within a sentence
# `min_count`: Ignores all words with total frequency lower than this
# `workers`: Use these many worker threads to train the model (=faster training)

# Save the model (optional)
model.save("word2vec.model")

# Load a pre-trained model if you have one (instead of training)
#model = Word2Vec.load("word2vec.model")


# Access the vector for a word
vector = model.wv['king']
print(f"Vector for 'king': {vector[:10]}...")  # Print only the first 10 elements

# Find the most similar words to a given word
similar_words = model.wv.most_similar('king', topn=5)
print(f"Words most similar to 'king': {similar_words}")

# Solve an analogy
try:
    analogy = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(f"woman is to man as king is to: {analogy}")
except KeyError as e:
    print(f"Some words are not in the vocabulary. Error: {e}")

```

**Explanation:**

1.  **Import necessary libraries:** `gensim` for Word2Vec and `nltk` for accessing a text corpus.
2.  **Load a text corpus:**  The `brown` corpus is used as an example.  You can replace this with your own dataset (a list of sentences, where each sentence is a list of words).
3.  **Train the Word2Vec model:**  `Word2Vec()` takes the sentences as input and trains the model.  The parameters control the training process.
4.  **Save/Load the model (optional):**  This allows you to reuse the trained model later without retraining.
5.  **Access word vectors:** `model.wv['word']` retrieves the vector representation for a given word.
6.  **Find similar words:** `model.wv.most_similar()` returns a list of words that are most similar to the given word based on cosine similarity.
7.  **Solve analogies:** The `most_similar` function can also perform vector arithmetic to solve analogies (e.g., `woman - man + king`). It tries to find a word closest to the resulting vector.  A `try-except` block is used because the analogy might fail if one of the words is not in the model's vocabulary.

## 4) Follow-up question

How do different word embedding models (e.g., Word2Vec, GloVe, FastText) differ in their training objectives and how do these differences affect their performance on various NLP tasks? Specifically, when would you choose one model over another?