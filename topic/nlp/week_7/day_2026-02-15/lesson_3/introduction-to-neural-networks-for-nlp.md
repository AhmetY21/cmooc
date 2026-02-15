---
title: "Introduction to Neural Networks for NLP"
date: "2026-02-15"
week: 7
lesson: 3
slug: "introduction-to-neural-networks-for-nlp"
---

# Topic: Introduction to Neural Networks for NLP

## 1) Formal definition (what is it, and how can we use it?)

In the context of Natural Language Processing (NLP), neural networks are computational models inspired by the structure and function of biological neural networks. They are composed of interconnected nodes (neurons) organized in layers, each layer transforming the input to provide a progressively higher-level representation of the data.

More formally, a neural network can be defined as a function approximator. Given an input *x*, the network's goal is to learn a function *f(x)* that maps *x* to a desired output *y*. This is achieved through learning the weights and biases of the connections between neurons.

Here's a breakdown of key components:

*   **Neurons (Nodes):** The basic unit of a neural network. It receives inputs, applies a weight to each input, sums them up, adds a bias, and then applies an activation function.
*   **Weights:** Parameters that determine the strength of the connection between neurons. These are adjusted during the learning process.
*   **Biases:** An additional parameter added to the weighted sum of inputs. It allows the neuron to activate even when all inputs are zero.
*   **Activation Function:** A non-linear function applied to the weighted sum of inputs plus the bias. This introduces non-linearity, enabling the network to learn complex patterns. Common examples include ReLU, sigmoid, and tanh.
*   **Layers:**
    *   **Input Layer:** Receives the initial input data (e.g., word embeddings).
    *   **Hidden Layers:** Perform intermediate computations. A neural network can have multiple hidden layers.
    *   **Output Layer:** Produces the final output (e.g., predicted class label).

**How can we use it in NLP?**

Neural networks are used extensively in NLP for various tasks, including:

*   **Text Classification:** Sentiment analysis, spam detection, topic categorization. The input is a text document, and the output is a class label.
*   **Machine Translation:** Translating text from one language to another.  Sequence-to-sequence models are often used.
*   **Named Entity Recognition (NER):** Identifying and classifying named entities in text (e.g., person, organization, location).
*   **Language Modeling:** Predicting the next word in a sequence.
*   **Question Answering:** Answering questions based on a given text.
*   **Text Summarization:** Generating a shorter version of a text document.
*   **Part-of-Speech (POS) Tagging:** Assigning grammatical tags to words in a sentence.

## 2) Application scenario

**Scenario:** Sentiment Analysis of Customer Reviews

A company wants to automatically analyze customer reviews of their products to understand overall customer sentiment. Manually reading and categorizing thousands of reviews is time-consuming and expensive.

**Using Neural Networks:**

1.  **Data Preparation:** The customer reviews are collected and preprocessed. This includes cleaning the text (removing punctuation, converting to lowercase), tokenization (splitting the text into words or subwords), and creating numerical representations (e.g., word embeddings).
2.  **Model Building:** A neural network model, such as a Convolutional Neural Network (CNN) or a Recurrent Neural Network (RNN), is built to classify the reviews. CNNs are good at capturing local patterns in text, while RNNs are good at processing sequential data. A common architecture is to use an embedding layer to convert words to vectors, followed by CNN or RNN layers, and finally a dense (fully connected) layer to produce a sentiment score (e.g., positive, negative, neutral).
3.  **Training:** The model is trained on a labeled dataset of reviews (reviews that have been manually labeled with their sentiment).  The model learns to associate specific words and phrases with positive or negative sentiment.
4.  **Evaluation:** The trained model is evaluated on a held-out dataset to assess its performance. Metrics like accuracy, precision, recall, and F1-score are used.
5.  **Deployment:** The trained model is deployed to automatically analyze new customer reviews in real-time. The company can then use this information to identify areas for improvement in their products and services.

## 3) Python method (if possible)

Here's a basic example of a simple feedforward neural network for text classification using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Sample data (replace with your actual dataset)
sentences = [
    "This is a great movie",
    "I did not like the movie",
    "The movie was okay",
    "I really enjoyed the show",
    "The show was terrible"
]
labels = [1, 0, 0, 1, 0]  # 1: positive, 0: negative

# Tokenization and Vocabulary creation
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100) # only keep the 100 most frequent words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

# Padding sequences to have the same length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=10)

# Model Definition
model = Sequential([
    Embedding(len(word_index) + 1, 8, input_length=10),  # Embedding layer: vocabulary size, embedding dimension, input length
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer: sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10)

# Example prediction
new_sentence = ["I loved the movie"]
new_sequence = tokenizer.texts_to_sequences(new_sentence)
padded_new_sequence = tf.keras.preprocessing.sequence.pad_sequences(new_sequence, maxlen=10)
prediction = model.predict(padded_new_sequence)
print(f"Prediction: {prediction}")
```

**Explanation:**

*   **Tokenization:** Converts text into sequences of integers, where each integer represents a word in the vocabulary.
*   **Padding:** Makes all sequences the same length by adding padding tokens.
*   **Embedding Layer:** Converts integer word indices into dense vectors of fixed size. This layer learns word representations during training.
*   **Flatten Layer:** Flattens the 2D output of the embedding layer into a 1D vector.
*   **Dense Layers:** Fully connected layers that learn complex relationships between the input features.
*   **Activation Functions:** ReLU (Rectified Linear Unit) is used in the hidden layer, and sigmoid is used in the output layer for binary classification.
*   **Compilation:** Configures the model for training, specifying the optimizer (adam), loss function (binary cross-entropy), and metrics (accuracy).
*   **Training:** The model learns to map input sequences to the corresponding labels by adjusting its weights and biases.

**Important Note:** This is a very basic example for demonstration purposes. Real-world NLP tasks often require more complex models and more sophisticated preprocessing techniques.

## 4) Follow-up question

How do Recurrent Neural Networks (RNNs), specifically LSTMs and GRUs, improve upon traditional feedforward neural networks for NLP tasks involving sequential data, and what are the trade-offs involved in choosing between them?