---
title: "POS Tagging Algorithms (HMM, Viterbi)"
date: "2026-02-10"
week: 7
lesson: 5
slug: "pos-tagging-algorithms-hmm-viterbi"
---

# Topic: POS Tagging Algorithms (HMM, Viterbi)

## 1) Formal definition (what is it, and how can we use it?)

**Part-of-Speech (POS) tagging** is the process of assigning a grammatical tag to each word in a sentence. These tags represent the word's role in the sentence, such as noun, verb, adjective, adverb, etc. POS tagging is a crucial step in many NLP tasks, including parsing, information extraction, machine translation, and question answering.

The **Hidden Markov Model (HMM)** is a probabilistic sequence model used to predict a sequence of hidden states (POS tags) based on a sequence of observed outputs (words). It makes two key assumptions:

*   **Markov Assumption:** The probability of a particular state depends only on the previous state.  In the context of POS tagging, this means the tag of a word depends only on the tag of the previous word.
*   **Observation Independence:** The probability of an observation (word) depends only on the current state (tag) that produced it.

The HMM is defined by the following parameters:

*   **States (S):** The set of possible POS tags (e.g., noun, verb, adjective).
*   **Observations (O):** The set of possible words in the vocabulary.
*   **Transition Probabilities (A):**  `P(state_t | state_t-1)`, the probability of transitioning from one tag to another.  For example, `P(verb | noun)` is the probability of a verb following a noun.
*   **Emission Probabilities (B):** `P(observation_t | state_t)`, the probability of a word being generated given a particular tag. For example, `P("run" | verb)` is the probability of the word "run" being a verb.
*   **Initial Probabilities (π):** `P(state_1)`, the probability of starting with a particular tag.

The **Viterbi algorithm** is a dynamic programming algorithm used to find the most likely sequence of hidden states (POS tags) given a sequence of observed outputs (words) and an HMM.  It efficiently calculates the probability of the most likely path through the HMM state space for a given sentence. The algorithm works by maintaining a table (Viterbi trellis) that stores the probability of the most likely path ending in each possible state at each time step (word position). It then traces back through the trellis to find the sequence of states that maximizes the overall probability.

In summary, we use an HMM to model the relationship between words and their tags and the Viterbi algorithm to efficiently find the optimal sequence of tags for a given sentence based on the HMM.

## 2) Application scenario

Consider the sentence: "The dog chased the cat."

We want to assign POS tags to each word:

*   "The": Determiner (DT)
*   "dog": Noun (NN)
*   "chased": Verb (VBD)
*   "the": Determiner (DT)
*   "cat": Noun (NN)

An HMM, trained on a large corpus of tagged text, would have learned the transition and emission probabilities necessary to perform this tagging. For example, it might have learned that:

*   P(NN | DT) is high (a noun is likely to follow a determiner).
*   P("dog" | NN) is high (the word "dog" is likely to be a noun).
*   P(DT | VBD) is low (a determiner is unlikely to follow a verb).

The Viterbi algorithm would then use these probabilities to find the most likely sequence of tags (DT NN VBD DT NN) for the sentence.  If, without context, "chased" could also be a noun (unlikely but possible), the Viterbi algorithm would weigh the probabilities of different tag sequences (e.g., DT NN NN DT NN) and choose the one with the highest overall probability based on the HMM parameters.

Another application scenario includes automatically generating subtitles for videos, where accurate POS tagging is crucial for later stages of speech processing such as named entity recognition, which helps identify keywords and build better indices.

## 3) Python method (if possible)

Here's a simple example using the `hmmlearn` library to demonstrate the Viterbi algorithm with a pre-trained HMM.  Note that training the HMM on real data requires a substantial tagged corpus and is beyond the scope of this basic illustration.  Also, for real-world usage, consider more advanced and pre-trained models from libraries like `spaCy` or `NLTK`.

```python
import numpy as np
from hmmlearn import hmm

# Simplified example with 3 states (Noun, Verb, Determiner) and a small vocabulary
states = ["Noun", "Verb", "Determiner"]
n_states = len(states)
observations = ["dog", "chased", "the", "cat", "run"]
n_observations = len(observations)

# Example HMM parameters (replace with trained parameters)
start_probability = np.array([0.3, 0.1, 0.6]) # Initial probabilities
transition_probability = np.array([
    [0.4, 0.4, 0.2],  # Noun -> Noun, Verb, Determiner
    [0.6, 0.1, 0.3],  # Verb -> Noun, Verb, Determiner
    [0.5, 0.0, 0.5]   # Determiner -> Noun, Verb, Determiner
])
emission_probability = np.array([
    [0.4, 0.0, 0.0, 0.6, 0.0],  # Noun -> dog, chased, the, cat, run
    [0.0, 0.8, 0.0, 0.0, 0.2],  # Verb -> dog, chased, the, cat, run
    [0.0, 0.0, 1.0, 0.0, 0.0]   # Determiner -> dog, chased, the, cat, run
])

# Create the HMM model
model = hmm.CategoricalHMM(n_components=n_states, random_state=42)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability


def viterbi(model, observations_sequence, states):
    """Simple Viterbi implementation for demonstration."""
    logprob, sequence = model.decode(observations_sequence, lengths=[len(observations_sequence)])
    return [states[i] for i in sequence]


# Example sentence
sentence = ["the", "dog", "chased", "the", "cat"]
# Map words to observation indices
observations_indices = [observations.index(word) for word in sentence]

# Run Viterbi algorithm
predicted_tags = viterbi(model, observations_indices, states)
print(f"Sentence: {sentence}")
print(f"Predicted Tags: {predicted_tags}")

#Alternative: Using nltk
import nltk
from nltk.tag import hmm

# Sample training data (replace with a larger corpus)
training_data = [
    [("the", "DT"), ("dog", "NN"), ("barked", "VB")],
    [("the", "cat", "DT"), ("slept", "VB")],
]

# Train the HMM tagger
trainer = hmm.HiddenMarkovModelTrainer(states=states, symbols=observations) # states and symbols as defined above
trained_tagger = trainer.train(training_data)

# Example sentence
sentence = ["the", "cat", "slept"]

# Tag the sentence using the trained tagger
tagged_sentence = trained_tagger.tag(sentence)
print(f"Sentence: {sentence}")
print(f"Predicted Tags: {tagged_sentence}")
```

**Explanation:**

1.  **HMM Parameters:** We define the states (POS tags), observations (words), and the HMM parameters (initial, transition, and emission probabilities).  These are *crucially* derived from training on a large, tagged corpus.
2.  **`hmmlearn`:** The `hmmlearn` library provides tools for working with HMMs. We create a `CategoricalHMM` model.
3.  **Observation Indices:**  The Viterbi algorithm expects numerical input (indices of observations).  We map the words in the sentence to their corresponding indices in the `observations` list.
4.  **`model.decode`:** The `model.decode()` function applies the Viterbi algorithm, leveraging pre-trained parameters, and returns the log probability of the most likely sequence as well as that sequence.
5.  **Result:** The code prints the original sentence and the predicted POS tags.

The NLTK example is closer to real-world use since it involves actually training a tagger on a limited set of training data. Replace with a larger training dataset to generate a more meaningful result.

## 4) Follow-up question

How can we handle unseen words (words not present in the training data) when using an HMM for POS tagging?  What are some common techniques to address the problem of *Out-Of-Vocabulary* (OOV) words?