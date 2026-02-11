---
title: "N-Grams and Language Models"
date: "2026-02-11"
week: 7
lesson: 4
slug: "n-grams-and-language-models"
---

# Topic: N-Grams and Language Models

## 1) Formal definition (what is it, and how can we use it?)

**N-grams:** An N-gram is a contiguous sequence of *n* items from a given sequence of text or speech. The items can be characters, syllables, words, or phrases, depending on the application.

*   If *n* = 1, it's a unigram (e.g., "the")
*   If *n* = 2, it's a bigram (e.g., "the cat")
*   If *n* = 3, it's a trigram (e.g., "the cat sat")
*   And so on...

**Language Model (LM):** A language model assigns a probability to a sequence of words (or characters). It tries to predict the next word in a sequence, given the preceding words. The probability of a sequence *w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>m</sub>* is denoted as P(w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>m</sub>).

An N-gram language model calculates this probability based on the frequencies of N-grams in a training corpus. The basic idea is that the probability of a word depends only on the *n-1* preceding words (Markov assumption).  For example, in a bigram model:

P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>i-1</sub>) ≈ P(w<sub>i</sub> | w<sub>i-1</sub>)

Therefore, the probability of a sequence using a bigram model is:

P(w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>m</sub>) ≈ P(w<sub>1</sub>) * P(w<sub>2</sub> | w<sub>1</sub>) * P(w<sub>3</sub> | w<sub>2</sub>) * ... * P(w<sub>m</sub> | w<sub>m-1</sub>)

**Uses:**

*   **Text generation:**  Generate new text that resembles the training data.
*   **Speech recognition:**  Help determine the most likely sequence of words given the acoustic signal.
*   **Machine translation:**  Evaluate the fluency of translated text.
*   **Spelling correction:** Suggest corrections based on probable word sequences.
*   **Autocomplete:**  Predict the next word a user is likely to type.
*   **Sentiment analysis:** Can be used as features to enhance classification models.

## 2) Application scenario

**Scenario: Autocomplete Functionality**

Imagine you're building an autocomplete feature for a search engine.  A user types "the qu".  Using an N-gram language model, you can predict the next word.

1.  **Training Data:** You train a bigram language model on a large corpus of text (e.g., web pages, books). This corpus is used to estimate probabilities P(w<sub>i</sub> | w<sub>i-1</sub>).
2.  **Prediction:** Given the input "the qu", you look for bigrams that start with "qu". For example, you might find "qu ick", "qu ite", "qu estion", etc.
3.  **Probability Calculation:** You calculate the probabilities P("ick" | "qu"), P("ite" | "qu"), P("estion" | "qu"), etc., based on the trained model.
4.  **Ranking:** You rank the possible next words based on their probabilities.  If "qu ick" is the most frequent bigram starting with "qu" in your training data, then "ick" will have the highest probability.
5.  **Suggestion:** You suggest the most probable next word to the user (e.g., "quick").  You might also offer a few other suggestions with lower probabilities.

## 3) Python method (if possible)

```python
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

def create_ngrams(text, n):
    """
    Generates n-grams from a given text.
    """
    tokenized_text = word_tokenize(text.lower())
    ngrams = zip(*[tokenized_text[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def train_ngram_model(corpus, n):
    """
    Trains an n-gram language model from a corpus.
    """
    ngram_counts = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in corpus:
        ngrams = create_ngrams(sentence, n)
        for ngram in ngrams:
            history, word = ngram.rsplit(" ", 1)  # Split into history and target word
            ngram_counts[history][word] += 1

    ngram_probabilities = defaultdict(lambda: defaultdict(lambda: 0.0))
    for history, words in ngram_counts.items():
        total_count = float(sum(words.values()))
        for word, count in words.items():
            ngram_probabilities[history][word] = count / total_count

    return ngram_probabilities

# Example Usage
corpus = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat ate the fish."
]

# Train a bigram model (n=2)
bigram_model = train_ngram_model(corpus, 2)

# Predict the next word after "the"
if "the" in bigram_model:
    predictions = bigram_model["the"]
    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True) # Sort by probability
    print("Predictions after 'the':", sorted_predictions)
else:
    print("'the' not found in the trained model.")

# Example of calculating the probability of a sentence:
def calculate_sentence_probability(sentence, model, n):
    """Calculates the probability of a sentence given an n-gram model."""
    ngrams = create_ngrams(sentence, n)
    probability = 1.0
    for ngram in ngrams:
        history, word = ngram.rsplit(" ", 1)
        if history in model and word in model[history]:
            probability *= model[history][word]
        else:
            #Handle unseen n-grams. Can use smoothing here
            probability = 0  #Probability of 0 if ngram is not present in the model
            break #no need to continue, result is zero.
    return probability

sentence_to_check = "the cat sat"
sentence_probability = calculate_sentence_probability(sentence_to_check, bigram_model, 2)
print(f"Probability of '{sentence_to_check}': {sentence_probability}")
```

**Explanation:**

1.  **`create_ngrams(text, n)`:** This function takes text and `n` as input and returns a list of n-grams. It uses `nltk` for tokenization.
2.  **`train_ngram_model(corpus, n)`:** This function takes a corpus of text and `n` as input and trains an n-gram language model.  It calculates the conditional probabilities of words given their history (the preceding *n-1* words). The results are stored in a nested dictionary `ngram_probabilities`.
3.  **Example Usage:** The code provides an example of how to train a bigram model and use it to predict the next word after "the". The `calculate_sentence_probability` function calculates the probability of a given sentence using the trained model.
4. **Smoothing:** The code includes a very simple way to handle cases where the n-gram is not in the model (by setting the probability to 0). More sophisticated smoothing techniques (like Laplace smoothing, add-k smoothing, or Kneser-Ney smoothing) are typically used in practice to avoid zero probabilities and improve the model's generalization ability.  Smoothing is essential for handling unseen n-grams.

## 4) Follow-up question

How can we improve the performance of an N-gram language model, especially in dealing with unseen n-grams and the sparsity of data? What are some common smoothing techniques, and how do they work?