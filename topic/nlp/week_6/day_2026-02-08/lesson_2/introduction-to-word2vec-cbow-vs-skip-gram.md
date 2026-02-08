Topic: Introduction to Word2Vec (CBOW vs Skip-gram)

1- Provide formal definition, what is it and how can we use it?

Word2Vec is a group of models used to produce word embeddings. Word embeddings are vector representations of words that capture semantic relationships between them. These vectors are learned from large text corpora. The core idea is that words appearing in similar contexts should have similar vector representations. Essentially, it turns words into numbers (vectors) that a machine learning model can understand and that encode meaning.

There are two main architectures in Word2Vec:

*   **Continuous Bag-of-Words (CBOW):**  CBOW predicts a target word given the context words surrounding it. Formally, given context words *w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>c</sub>*, CBOW aims to predict the target word *w<sub>t</sub>*. The model averages the vector representations of the context words and then uses this average vector to predict the target word. The goal is to maximize the probability *P(w<sub>t</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>c</sub>)*.

*   **Skip-gram:** Skip-gram predicts the context words given a target word. Formally, given a target word *w<sub>t</sub>*, Skip-gram aims to predict the surrounding context words *w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>c</sub>*. The model tries to maximize the probability *P(w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>c</sub> | w<sub>t</sub>)*.

**How can we use it?**

*   **Semantic Similarity:** Calculate the similarity between words. Words with similar meanings will have vectors that are close to each other in vector space (e.g., using cosine similarity).
*   **Analogy Completion:** Solve analogy problems like "man is to king as woman is to ?" by using vector arithmetic (e.g.,  *vector('king') - vector('man') + vector('woman')* should be close to *vector('queen')*).
*   **Feature Engineering:** Use word embeddings as features for various NLP tasks like text classification, sentiment analysis, machine translation, and named entity recognition. This is especially useful for models that can't directly process text (e.g., certain types of neural networks).
*   **Recommendation Systems:** Represent items (e.g., products) using embeddings based on co-occurrence in user interactions, similar to how words co-occur in text.

2- Provide an application scenario

**Application Scenario: Sentiment Analysis of Customer Reviews**

Imagine you want to build a sentiment analysis system to automatically classify customer reviews as positive, negative, or neutral. Instead of using traditional bag-of-words approaches, you can use Word2Vec to create word embeddings for the words in the reviews.

1.  **Train a Word2Vec model:** Train a Word2Vec model (either CBOW or Skip-gram) on a large corpus of text data (e.g., Wikipedia, news articles, or even a collection of other reviews). This gives you vector representations for a large vocabulary of words.
2.  **Embed the reviews:** For each customer review, look up the word embeddings for each word in the review.
3.  **Aggregate the word embeddings:** Average the word embeddings of all words in a review to create a single vector representation for the entire review. Alternatively, more sophisticated methods like using RNNs or Transformers to combine the word embeddings can be employed.
4.  **Train a classifier:** Train a machine learning classifier (e.g., logistic regression, support vector machine, or a neural network) using the aggregated review embeddings as features and the sentiment labels (positive, negative, neutral) as the target variable.

By using Word2Vec embeddings, the sentiment analysis system can capture semantic relationships between words, allowing it to better understand the context and meaning of the reviews. For example, it can recognize that "great" and "fantastic" are semantically similar and contribute positively to the sentiment.

3- Provide a method to apply in python

python
import gensim
from gensim.models import Word2Vec
from nltk.corpus import brown # Example corpus
from nltk.tokenize import word_tokenize

# Download brown corpus if you haven't already
import nltk
try:
    brown.words()
except LookupError:
    nltk.download('brown')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# 1. Prepare the data: Tokenize sentences in the corpus
sentences = brown.sents() # Already tokenized into sentences

# 2. Train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=0)  # CBOW by default
# vector_size: Dimensionality of the word vectors
# window: Maximum distance between the current and predicted word within a sentence
# min_count: Ignores all words with total frequency lower than this
# workers: Use these many worker threads to train the model (=faster training with multicore machines).
# sg: Training algorithm: 1 for skip-gram; otherwise, CBOW.

# To train Skip-gram model, change sg=1
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1)

# 3. Access word vectors
vector = model.wv['king'] # Get the vector representation of the word 'king'
print(f"Vector for 'king': {vector[:10]}...") # Print the first 10 elements

# 4. Find similar words
similar_words = model.wv.most_similar('king', topn=5) # Find the 5 most similar words to 'king'
print(f"Similar words to 'king': {similar_words}")

# 5. Save and load the model
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")

# Example usage after loading the model
vector_loaded = loaded_model.wv['king']
print(f"Vector for 'king' from loaded model: {vector_loaded[:10]}...")

# Clean up the saved model file (optional)
import os
os.remove("word2vec.model")


**Explanation:**

1.  **Import Libraries:** Import the necessary libraries: `gensim` for Word2Vec and `nltk` for text processing (Brown corpus).
2.  **Prepare Data:** The Brown corpus is used as an example text corpus. The `brown.sents()` function provides sentences already tokenized into words.
3.  **Train the Model:**
    *   `Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=0)`:  This line creates and trains the Word2Vec model.
        *   `sentences`: The training data (list of sentences, where each sentence is a list of words).
        *   `vector_size`: The dimensionality of the word vectors (e.g., 100 dimensions).  A higher dimensionality can capture more complex semantic relationships, but it also increases the computational cost.
        *   `window`: The maximum distance between a target word and words around the target word.  A larger window captures more context.
        *   `min_count`:  Words that appear less than `min_count` times in the corpus are ignored.  This helps to remove rare words and improve the quality of the embeddings.
        *   `workers`:  The number of worker threads to use for training.  This can speed up training on multi-core machines.
        *   `sg`:  Determines the training algorithm: `0` for CBOW (default), `1` for Skip-gram.
4.  **Access Word Vectors:** `model.wv['king']` retrieves the vector representation of the word "king".
5.  **Find Similar Words:** `model.wv.most_similar('king', topn=5)` finds the 5 words most similar to "king" based on cosine similarity.
6.  **Save and Load the Model:**  The trained model is saved to disk using `model.save("word2vec.model")` and loaded using `Word2Vec.load("word2vec.model")`. Saving the model allows you to reuse it later without retraining.

4- Provide a follow up question about that topic

How does the choice of training corpus size, vector size, window size, and min_count affect the quality of the resulting word embeddings, and what are some strategies for tuning these hyperparameters to optimize performance for a specific downstream task?