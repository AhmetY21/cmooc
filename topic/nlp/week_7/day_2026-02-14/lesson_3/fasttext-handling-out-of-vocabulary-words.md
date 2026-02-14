---
title: "FastText: Handling Out-of-Vocabulary Words"
date: "2026-02-14"
week: 7
lesson: 3
slug: "fasttext-handling-out-of-vocabulary-words"
---

# Topic: FastText: Handling Out-of-Vocabulary Words

## 1) Formal definition (what is it, and how can we use it?)

FastText is a word embedding and text classification library developed by Facebook AI Research. It addresses the challenge of **Out-of-Vocabulary (OOV)** words, which are words not seen during training, by representing words as bags of character n-grams. This approach allows FastText to derive meaningful representations even for unseen words, leveraging the subword information.

Here's how it works:

*   **Character n-grams:** Instead of treating a word as an atomic unit, FastText breaks it down into its constituent character n-grams. For example, the word "apple" might be represented by character n-grams like "ap", "pp", "pl", "le", "<ap", "app", "ple>", where the angle brackets denote the beginning and end of the word.
*   **Embedding learning:** Each n-gram is associated with a vector embedding. FastText learns these n-gram embeddings during training.
*   **Word representation:** The vector representation of a word is the sum of the vector embeddings of its constituent n-grams. This is done regardless of whether the word was present in the training vocabulary or not.
*   **OOV Handling:** When encountering an OOV word, FastText simply calculates its vector representation by summing the embeddings of its n-grams, effectively creating a representation based on its constituent parts. If the word is similar to known words (e.g., shares many common n-grams) its vector will be similar to the vectors of those known words.
*   **Use:** The resulting word vectors can be used for various NLP tasks, like word similarity, text classification, and information retrieval.

Essentially, FastText avoids a complete vocabulary breakdown by utilizing subword information to create embeddings for new words dynamically, instead of assigning them a random or unknown vector.

## 2) Application scenario

Consider a sentiment analysis task where the training dataset contains common words like "good," "bad," and "neutral." When deploying the model, it encounters the word "awesomeness," which wasn't in the training data (an OOV word).

*   **Traditional Word Embeddings (e.g., Word2Vec):** These models typically represent OOV words with a special `<UNK>` token or a random vector, meaning the model would not be able to understand the meaning of "awesomeness" based on its morphology.
*   **FastText:** FastText, on the other hand, would break "awesomeness" into character n-grams like "aw", "we", "es", "so", "om", "me", "en", "ness", "<aw", "awe", "wes", "eso", ...". Even if the model hasn't seen "awesomeness" before, it might have seen similar n-grams in words like "awesome," "amazing," "effectiveness," and can construct a meaningful representation for "awesomeness" based on these known parts.  This allows the sentiment analysis model to better predict the sentiment associated with sentences containing this unseen word, likely classifying it as positive.

Other scenarios include:

*   **Languages with rich morphology:** Languages like German or Finnish frequently create new words through compounding or inflection. FastText is particularly effective in such languages.
*   **Spelling variations and typos:** FastText is more robust to minor spelling errors or variations in word forms because it relies on subword information.
*   **Rare words in specialized domains:**  In fields like biomedicine, many specialized terms might be rare or completely absent from general-purpose corpora. FastText can create meaningful representations even for these rare terms.

## 3) Python method (if possible)
```python
import fasttext

# Create a text file named 'data.txt' with your training data.
# For example:
# This is a good example.
# This is another good example.
# This is a bad example.

# Skipgram model
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# Access word vector
word_vector = model.get_word_vector('example') # Get vector for existing word
print(f"Vector for 'example': {word_vector[:10]}...") # Print first 10 values to conserve space

# Access word vector for OOV word
oov_word_vector = model.get_word_vector('unseenword') # Get vector for an unseen word
print(f"Vector for 'unseenword': {oov_word_vector[:10]}...") # Print first 10 values to conserve space

# Save the model
model.save_model("model_filename.bin")

# Load a model previously trained.
model = fasttext.load_model("model_filename.bin")

# Get nearest neighbors
neighbors = model.get_nearest_neighbors("example")
print(f"Nearest neighbors of 'example': {neighbors}")

# Example usage for text classification (if you have labeled data)
# Create a text file named 'data_labeled.txt' with your labeled data.
# The file should be formatted as:
# __label__positive This is a good example.
# __label__negative This is a bad example.

# Train a supervised (classification) model
# model = fasttext.train_supervised(input='data_labeled.txt')
# label = model.predict("This is a fantastic example")
# print(f"Prediction for 'This is a fantastic example': {label}")
```

## 4) Follow-up question

How does the choice of n-gram size impact the performance of FastText, and what are the trade-offs involved in selecting different n-gram lengths?