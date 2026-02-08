Topic: Word2Vec: Negative Sampling and Optimization

1- Provide formal definition, what is it and how can we use it?

Word2Vec is a group of techniques to learn word embeddings, which are vector representations of words in a high-dimensional space. These embeddings capture semantic relationships between words, allowing algorithms to understand context and meaning beyond simple string matching. Word2Vec has two main architectures: Continuous Bag-of-Words (CBOW) and Skip-gram.

**Negative Sampling** is an optimization technique for training word embeddings, primarily used with Skip-gram and CBOW models. Instead of updating all word vectors for every training sample, negative sampling only updates a small sample of the weights. This significantly reduces the computational burden, making training feasible on large datasets.

*   **Formal Definition:**  For each observed word pair (center word `w_c`, context word `w_o`), treated as a *positive* example, we generate *k* random *negative* samples, denoted as `w_i`. These negative samples are words from the vocabulary that did *not* appear in the context of the center word.  The model is then trained to distinguish between the true pair (`w_c`, `w_o`) and the randomly sampled negative pairs (`w_c`, `w_i`).

*   **How it works:**
    *   **Positive Sample:** For a word pair (context, target) observed in the corpus, we aim to maximize the probability:  `P(D=1 | w_o, w_c; θ)` where D=1 indicates a positive example (real context).  `θ` represents the model's parameters (word vectors).
    *   **Negative Samples:** For each positive sample, we sample *k* words from a noise distribution `P_n(w)`.  These represent words that are unlikely to appear as context words for the target word. We aim to minimize the probability: `P(D=0 | w_i, w_c; θ)` for each negative sample `w_i`, where D=0 indicates a negative example.
    *   **Noise Distribution `P_n(w)`:**  A common choice for the noise distribution is the unigram distribution raised to the power of 3/4: `P_n(w) = U(w)^{3/4} / Z`, where `U(w)` is the unigram frequency of word *w* in the corpus, and Z is a normalization constant. This favors less frequent words for negative sampling, since frequent words are already likely to be related to many words.

*   **Loss Function:** The objective function is typically the negative log-likelihood of the training data. After negative sampling, the loss function becomes a sum over the positive and negative samples:

    `J(θ) = -log σ(v_o^T v_c) - Σ_{i=1}^k log σ(-v_i^T v_c)`

    where:
        *   `σ` is the sigmoid function ( `σ(x) = 1 / (1 + exp(-x))` )
        *   `v_o` is the vector representation of the context word.
        *   `v_c` is the vector representation of the center word.
        *   `v_i` is the vector representation of the i-th negative sample.
        *   `k` is the number of negative samples.

*   **How we can use it:** By training the Word2Vec model with negative sampling, we obtain high-quality word embeddings. These embeddings can be used in downstream NLP tasks like:
    *   **Semantic Similarity:** Finding words with similar meanings.
    *   **Word Analogies:**  Solving analogies like "king - man + woman = queen".
    *   **Text Classification:** Using word embeddings as features for classifiers.
    *   **Machine Translation:** Improving translation quality by aligning embedding spaces.
    *   **Recommendation Systems:** Recommending items based on semantic similarity of their descriptions.

2- Provide an application scenario

**Application Scenario: Building a Product Recommendation System**

Imagine an e-commerce platform wants to improve its product recommendations. They have a large dataset of product descriptions and user purchase histories.  Instead of relying solely on collaborative filtering (user-item interactions), they want to incorporate semantic information from the product descriptions.

Here's how Word2Vec with negative sampling can be applied:

1.  **Data Preparation:** The product descriptions are treated as text sequences. Each sentence can be considered as a training example.
2.  **Word Embedding Training:** A Word2Vec Skip-gram model is trained on the combined corpus of product descriptions using negative sampling. This produces word embeddings where words describing similar products are close in the embedding space.
3.  **Product Embedding Generation:** Each product is represented by the average of the embeddings of the words in its description. This creates a product embedding for each item in the catalog. (Other approaches include using Doc2Vec or similar paragraph embedding techniques for richer embeddings.)
4.  **Recommendation Engine:**  When a user views a particular product, the system calculates the cosine similarity between the viewed product's embedding and the embeddings of all other products. Products with high cosine similarity are recommended to the user, as they are semantically similar based on their descriptions.
5.  **Hybrid Approach:** This semantic similarity-based recommendation can be combined with traditional collaborative filtering to improve recommendation accuracy and provide more diverse suggestions.  For example, products similar to what other users who bought the viewed product also purchased could be added to the recommendations.

In this scenario, negative sampling is crucial because the vocabulary of product descriptions can be very large. Training without negative sampling would be computationally expensive and time-consuming.

3- Provide a method to apply in python

python
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

# Download required NLTK resources if not already present
try:
    brown.words()
except LookupError:
    nltk.download('brown')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# 1. Prepare the Training Data
# Use the Brown corpus as an example (replace with your own data)
sentences = brown.sents()

# Tokenize the sentences (if not already tokenized)
tokenized_sentences = [[word.lower() for word in sentence] for sentence in sentences]


# 2. Train the Word2Vec Model with Negative Sampling
model = Word2Vec(sentences=tokenized_sentences,
                 vector_size=100,       # Dimensionality of the word vectors
                 window=5,            # Context window size
                 min_count=5,         # Minimum word frequency
                 workers=4,           # Number of worker threads
                 sg=1,              # Use Skip-gram (1) or CBOW (0)
                 negative=10,          # Number of negative samples (key parameter)
                 epochs=10)           # Number of training epochs

# 3. Access Word Embeddings
# Get the vector for a specific word
word_vector = model.wv['king']  # Example: Get the vector for "king"
print(f"Vector for 'king': {word_vector}")

# Find similar words
similar_words = model.wv.most_similar('king', topn=5)
print(f"Words similar to 'king': {similar_words}")

# 4. Save and Load the Model
# Save the trained model
model.save("word2vec_brown_negative_sampling.model")

# Load the saved model
loaded_model = Word2Vec.load("word2vec_brown_negative_sampling.model")


# Example usage with the loaded model:
print(f"Vector for 'queen' (loaded model): {loaded_model.wv['queen']}")
print(f"Words similar to 'queen' (loaded model): {loaded_model.wv.most_similar('queen', topn=5)}")


**Explanation:**

1.  **Data Preparation:** The code loads sentences from the NLTK Brown corpus (you would replace this with your own corpus). It converts all words to lowercase and tokenizes the sentences.
2.  **Model Training:**  The `Word2Vec` class from `gensim` is used to train the model.
    *   `sentences`: The training data (list of lists of words).
    *   `vector_size`:  The dimensionality of the word vectors (e.g., 100, 300).  A higher dimension can capture more nuanced relationships but requires more data.
    *   `window`:  The maximum distance between the current and predicted word within a sentence.
    *   `min_count`: Ignores all words with total frequency lower than this.  This helps to remove rare words that may not contribute meaningfully to the embedding space.
    *   `workers`: Use these many worker threads to train the model (=faster training with multicore machines).
    *   `sg`: Training algorithm: 1 for skip-gram; otherwise CBOW.  Skip-gram tends to perform better with smaller datasets and better captures rare words.
    *   `negative`:  The key parameter! Specifies how many "noise words" should be drawn for each positive sample. A value between 5 and 20 is common for smaller datasets, and between 2 and 5 for large datasets.
    *   `epochs`: Number of iterations (epochs) over the corpus.
3.  **Accessing Embeddings:** The `model.wv` attribute provides access to the word vectors. `model.wv['word']` returns the vector for a given word. `model.wv.most_similar('word')` returns a list of words most similar to the given word, based on cosine similarity.
4.  **Saving/Loading:** The trained model can be saved to disk and loaded later for use.

4- Provide a follow up question about that topic

**Follow-up Question:**

How does the choice of the negative sampling distribution, `P_n(w)`, affect the quality and characteristics of the learned word embeddings, and are there scenarios where alternative distributions (other than the unigram distribution raised to the 3/4 power) might be more appropriate? Explain with some examples.