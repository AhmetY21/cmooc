---
title: "Topic Modeling: Latent Dirichlet Allocation (LDA)"
date: "2026-02-13"
week: 7
lesson: 3
slug: "topic-modeling-latent-dirichlet-allocation-lda"
---

# Topic: Topic Modeling: Latent Dirichlet Allocation (LDA)

## 1) Formal definition (what is it, and how can we use it?)

Latent Dirichlet Allocation (LDA) is a generative probabilistic model for collections of discrete data such as text corpora.  It's a type of topic modeling algorithm used to discover the underlying thematic structure (topics) within a corpus of documents.  "Latent" means these topics are not explicitly observed but are inferred from the observed data (words in the documents).  "Dirichlet" refers to the Dirichlet distribution, which is used as a prior distribution for both the document-topic and topic-word distributions.

Here's a simplified explanation of how LDA works:

*   **Documents as Mixtures of Topics:** LDA assumes each document is a mixture of multiple topics. For example, a news article might be 60% about "Politics" and 40% about "Economics."
*   **Topics as Distributions over Words:**  Each topic is a distribution over words.  For example, the "Politics" topic might have a high probability of containing words like "election," "candidate," "government," "policy," etc.
*   **Generative Process:**  LDA imagines that each document is generated as follows:
    1.  Choose a distribution over topics for the document (using a Dirichlet prior).
    2.  For each word in the document:
        a. Choose a topic from the document's topic distribution.
        b. Choose a word from the chosen topic's word distribution.

**How we can use it:**

LDA can be used for various tasks:

*   **Topic Discovery:**  The primary use is to automatically discover the hidden topics present in a corpus.
*   **Document Clustering:** Group documents based on their dominant topics.
*   **Information Retrieval:**  Retrieve documents relevant to a specific topic.
*   **Text Summarization:**  Identify the main topics and use them to create summaries.
*   **Feature Engineering:** Use topic distributions as features for other machine learning models.

## 2) Application scenario

Consider a large collection of research papers in the field of computer science.  We want to understand the main areas of research being conducted. Using LDA, we can:

1.  **Input:** The corpus of research papers (represented as a bag of words).
2.  **LDA Model:** Train an LDA model on this corpus. The model will estimate:
    *   The topic distribution for each paper (e.g., paper 1 is 70% "Machine Learning," 20% "Computer Vision," 10% "Natural Language Processing").
    *   The word distribution for each topic (e.g., "Machine Learning" topic has a high probability of words like "algorithm," "model," "training," "neural network").
3.  **Output:** The LDA model will provide:
    *   A list of topics (e.g., "Machine Learning," "Computer Vision," "Natural Language Processing," "Databases," "Networking").
    *   The most relevant words for each topic (providing an interpretation of the topic).
    *   The topic distribution for each research paper (indicating which topics each paper primarily covers).

This allows researchers and librarians to efficiently browse, categorize, and retrieve relevant research papers based on automatically discovered research areas. For example, someone interested in "Deep Learning" can easily identify papers with a high probability of belonging to the "Machine Learning" topic and containing relevant keywords like "neural networks" and "backpropagation."

## 3) Python method (if possible)

The `gensim` library in Python is commonly used for topic modeling, including LDA.

```python
import gensim
from gensim import corpora

# Sample documents (replace with your actual data)
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# 1. Tokenize and clean the text
texts = [[word for word in document.lower().split()]
         for document in documents]

# 2. Create a dictionary (mapping words to IDs)
dictionary = corpora.Dictionary(texts)

# 3. Create a corpus (document-term matrix)
corpus = [dictionary.doc2bow(text) for text in texts]

# 4. Train the LDA model
num_topics = 3  # Specify the desired number of topics
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# 5. Print the topics and their associated words
for topic_id in range(num_topics):
    print(f"Topic {topic_id + 1}:")
    print(lda_model.print_topic(topic_id))

# 6. Get the topic distribution for a specific document
for i, doc in enumerate(documents):
    bow = dictionary.doc2bow(doc.lower().split())  # Convert document to bag-of-words format
    topic_distribution = lda_model.get_document_topics(bow)
    print(f"Document {i+1}: {doc}")
    print(f"Topic Distribution: {topic_distribution}\n")
```

**Explanation:**

1.  **Import libraries:** Imports `gensim` and `corpora`.
2.  **Sample Documents:** Replace the sample documents with your own corpus.
3.  **Tokenization and Cleaning:**  Converts documents to lowercase and splits them into words. This is a simplified example; you might want to add stemming, lemmatization, and stop word removal for better results.
4.  **Create Dictionary:** `corpora.Dictionary` creates a mapping between words and unique IDs.
5.  **Create Corpus:** `dictionary.doc2bow` converts each document into a "bag of words" representation, which is a list of (word ID, word count) tuples.  This is the input format required by `gensim`.
6.  **Train LDA Model:**  `gensim.models.LdaModel` trains the LDA model.
    *   `corpus`: The bag-of-words representation of the documents.
    *   `num_topics`: The number of topics you want to discover.  This is a crucial hyperparameter that often requires experimentation.
    *   `id2word`:  The dictionary mapping word IDs to words.
    *   `passes`: The number of times the model iterates through the entire corpus during training.  Higher values can lead to better results but also increase training time.
7.  **Print Topics:**  Prints the top words associated with each topic. `lda_model.print_topic` displays the words with the highest probabilities for each topic.
8.  **Document Topic Distribution:** Shows which topics a document has a higher chance of belonging to.

## 4) Follow-up question

How can you evaluate the quality of the topics generated by LDA? Are there metrics or techniques to determine whether the discovered topics are meaningful and coherent?