---
title: "The History of NLP: From Rules to Statistics"
date: "2026-02-08"
week: 6
lesson: 6
slug: "the-history-of-nlp-from-rules-to-statistics"
---

# Topic: The History of NLP: From Rules to Statistics

## 1) Formal definition (what is it, and how can we use it?)

The history of NLP: From Rules to Statistics describes the evolution of Natural Language Processing (NLP) techniques, broadly categorized into two major paradigms:

*   **Rule-Based NLP (Pre-1990s):** This early approach relied on manually crafted linguistic rules to process text. These rules, typically based on grammar and vocabulary, were used for tasks like parsing, machine translation, and information extraction. Key components involved defining lexicons (dictionaries), morphological rules, syntactic rules (grammars), and semantic rules. These systems were deterministic and followed predefined pathways. The "knowledge" was explicitly programmed in.

*   **Statistical NLP (1990s onwards):** This approach leverages statistical models learned from large corpora (datasets of text and speech). Instead of manually defining rules, these models learn patterns and relationships from data. Common techniques include:
    *   **N-grams:** Probabilistic models that predict the next word based on the preceding *n* words.
    *   **Hidden Markov Models (HMMs):** Used for sequence labeling tasks like Part-of-Speech (POS) tagging and named entity recognition.
    *   **Probabilistic Context-Free Grammars (PCFGs):** Statistical versions of CFGs that assign probabilities to production rules.
    *   **Machine Learning (ML) and Deep Learning (DL):** Algorithms like Support Vector Machines (SVMs), Naive Bayes, and more recently, Recurrent Neural Networks (RNNs), LSTMs, Transformers, and large language models (LLMs) are used for a wide range of NLP tasks.

**How can we use this understanding?**

*   **Choosing the right approach:** Knowing the historical context helps you understand the strengths and weaknesses of different NLP techniques. Rule-based systems are useful when precision is crucial and the domain is well-defined, but they are brittle and difficult to scale. Statistical approaches are more robust and adaptable to new data but require large training datasets.
*   **Debugging and improving NLP systems:** Understanding the underlying principles of different approaches is crucial for identifying and resolving issues in NLP systems. If a rule-based system fails, you need to adjust the rules. If a statistical model performs poorly, you might need to improve the data, adjust the model architecture, or tune the hyperparameters.
*   **Appreciating the progress of NLP:** A historical perspective provides context for the current state of NLP and helps to appreciate the challenges overcome and the opportunities ahead. This can inspire further innovation and research.
*   **Interpreting the output of different systems:** Rule-based and statistical systems often produce different types of output. Knowing which type of system produced the output can help you better interpret its results.

## 2) Application scenario

**Scenario:** Building a spam filter.

*   **Rule-Based Approach:** We could create a set of rules based on keywords, phrases, and patterns commonly found in spam emails. For example:
    *   IF the email contains the word "Viagra" AND contains excessive exclamation marks, THEN classify as spam.
    *   IF the email's sender address is from a known spam domain, THEN classify as spam.
    *   IF the email body contains a high ratio of capitalized words, THEN classify as spam.

    **Pros:** Easy to understand and implement initially. Provides high precision if the rules are carefully crafted.

    **Cons:** Very brittle and easily bypassed by spammers who find new ways to avoid the rules. Requires constant updating and maintenance. Difficult to handle nuanced language and evolving spam techniques.

*   **Statistical Approach (using Naive Bayes):** We could train a Naive Bayes classifier on a dataset of labeled spam and ham (non-spam) emails. The classifier learns the probability of certain words and phrases appearing in spam emails versus ham emails.

    **Pros:** More robust and adaptable to new spam techniques. Requires less manual effort in the long run. Can handle more nuanced language.

    **Cons:** Requires a large labeled dataset. May occasionally misclassify legitimate emails as spam (false positives). Requires periodic retraining to stay up-to-date. Can be computationally expensive for very large datasets.

In a real-world spam filter, a combination of rule-based and statistical approaches is often used to maximize accuracy and robustness. The rule-based system handles obvious spam, while the statistical system handles more sophisticated and evolving spam techniques. The combination allows the strengths of both approaches to be leveraged.

## 3) Python method (if possible)

Here's a Python example demonstrating a simple statistical NLP technique: N-gram language modeling using the `nltk` library. We'll build a trigram (n=3) model to predict the next word given the previous two.

```python
import nltk
from nltk.corpus import gutenberg
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

# Load a text corpus (e.g., from Gutenberg)
sentences = gutenberg.sents('austen-emma.txt')

# Preprocess the data for N-gram modeling
n = 3  # Trigram model
train_data, padded_sents = padded_everygram_pipeline(n, sentences)

# Train the N-gram model (Maximum Likelihood Estimation)
model = MLE(n)
model.fit(train_data, padded_sents)

# Example prediction
print(f"Vocabulary size: {len(model.vocab)}")

context = ['to', 'be']
next_word = model.generate(1, context=context)
print(f"Given the context '{context}', the next predicted word is: {next_word}")

# Function to generate a sentence
def generate_sentence(model, num_words=10):
    text = [None] * 2 # start with two None context words as per padded_everygram_pipeline
    for i in range(num_words):
        text.append(model.generate(1, context=text[-2:]))
    return ' '.join([t for t in text[2:]]) # slice to remove None tokens

print("Generated sentence:")
print(generate_sentence(model, num_words=15))


```

**Explanation:**

1.  **Import Libraries:** `nltk` is a powerful library for NLP tasks. `gutenberg` is a corpus reader providing access to various texts.  `padded_everygram_pipeline` helps prepare sentences for N-gram modeling by adding padding symbols (start and end tokens). `MLE` is a class for Maximum Likelihood Estimation for N-gram models.
2.  **Load Data:** We load the text of "Emma" by Jane Austen from the Gutenberg corpus. `gutenberg.sents()` returns a list of sentences, each sentence being a list of words.
3.  **Preprocess Data:** `padded_everygram_pipeline` prepares the data for N-gram modeling by:
    *   Adding padding symbols (``<s>`` and ``</s>``) to the beginning and end of each sentence. This helps the model learn probabilities for the first and last words of sentences.
    *   Generating N-grams from the sentences.
4.  **Train Model:** We create an `MLE` (Maximum Likelihood Estimation) model with *n* = 3 (trigram model). Then, we train the model using the preprocessed data.
5.  **Prediction:** The `model.generate()` method predicts the next word given a context (the previous two words in this case).
6.  **Sentence Generation:** A `generate_sentence` function is defined to generate text based on the probabilities learned by the model, using the context of the previous two words to generate each new word.

This example showcases a simple statistical NLP technique for language modeling. Modern NLP methods, especially those leveraging deep learning, are far more complex, but this illustrates the shift from manually coded rules to data-driven models.

## 4) Follow-up question

Given the success of statistical NLP, particularly deep learning models, are rule-based systems completely obsolete? Are there still situations where they might be preferred or even necessary?