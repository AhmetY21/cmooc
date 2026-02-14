---
title: "Document Embeddings (Doc2Vec)"
date: "2026-02-14"
week: 7
lesson: 4
slug: "document-embeddings-doc2vec"
---

# Topic: Document Embeddings (Doc2Vec)

## 1) Formal definition (what is it, and how can we use it?)

Doc2Vec, also known as Paragraph Vector, is an unsupervised learning technique to generate fixed-length vector representations of entire documents.  Unlike Word2Vec, which focuses on learning vector representations of individual words based on their context within a corpus, Doc2Vec aims to learn vector representations of documents, taking into account the context of the words *within* that document *and* the document itself.

In essence, Doc2Vec extends Word2Vec to consider entire documents as another 'word' in the training process.  Two primary models exist:

*   **Distributed Memory Model of Paragraph Vectors (PV-DM):** Similar to the Continuous Bag-of-Words (CBOW) model in Word2Vec, PV-DM predicts a target word given the surrounding context words *and* the document ID.  The document ID acts as a "memory" that remembers what is missing from the current context – it represents the topic of the paragraph.  During training, both the word vectors and the document vectors are learned.

*   **Distributed Bag of Words version of Paragraph Vector (PV-DBOW):** Similar to the Skip-gram model in Word2Vec, PV-DBOW predicts words randomly sampled from the document, given only the document ID.  It ignores the word order.

How can we use it? Document embeddings capture semantic meaning and can be used for:

*   **Document Similarity:**  Comparing document embeddings using cosine similarity or other distance metrics can reveal how similar two documents are in terms of content and meaning.
*   **Document Classification:** Document embeddings can be used as features for training machine learning classifiers.
*   **Document Clustering:** Grouping documents with similar embeddings together.
*   **Information Retrieval:** Finding documents relevant to a search query, where the query is represented as a document embedding.
*   **Sentiment Analysis:** As input features for sentiment classification models, particularly helpful when sentiment depends on the overall document context.
*   **Recommendation Systems:** Recommending documents to users based on their past interactions.

## 2) Application scenario

Imagine you're building a news article aggregator and want to group articles based on their topic without explicitly knowing what those topics are in advance. Using Doc2Vec, you can:

1.  Train a Doc2Vec model on a corpus of news articles.
2.  Generate document embeddings for each article.
3.  Use a clustering algorithm (e.g., k-means) on the document embeddings to automatically group articles into clusters representing different news topics.

Another scenario: You are building a question answering system based on a knowledge base of documents. You can encode the question using Doc2Vec and compare the resulting vector to the vector of each document in your database.  The documents with the highest similarity scores are the best candidates to use for answering the question.

## 3) Python method (if possible)

We can use the `gensim` library in Python to implement Doc2Vec.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Sample documents
documents = [
    "This is the first document about natural language processing.",
    "The second document discusses machine learning algorithms.",
    "A third document explores deep learning techniques.",
    "Natural language processing and machine learning are related fields."
]

# Tokenize the documents and create TaggedDocument objects
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)]

# Initialize and train the Doc2Vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs=100)

# Get the vector for a document
vector_doc_0 = model.dv['0']  # Access by tag

# Infer the vector for a new document
new_doc = "This is a new document about machine learning."
vector_new_doc = model.infer_vector(word_tokenize(new_doc.lower()))

# Print the vectors (optional - just to see some output)
print("Vector for document 0:", vector_doc_0)
print("Vector for the new document:", vector_new_doc)

# Save and load the model
model.save("doc2vec.model")
loaded_model = Doc2Vec.load("doc2vec.model")

#Perform Similarity
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

similarity = cosine_similarity(vector_doc_0, vector_new_doc)
print(f"Cosine similarity between document 0 and new document: {similarity}")
```

Key points:

*   `TaggedDocument` is used to associate a tag (usually the document ID) with the tokenized words of the document.
*   `vector_size` defines the dimensionality of the document vectors.
*   `min_count` ignores all words with total frequency lower than this.
*   `epochs` specifies the number of training iterations over the corpus.
*   `model.dv['0']` accesses the vector of the document tagged with '0'.
*   `model.infer_vector()` is used to generate a vector for a new, unseen document.  This method does not update the model; it leverages the existing word vectors to estimate a document vector.

## 4) Follow-up question

How does the performance of Doc2Vec compare to simpler methods of document representation like TF-IDF followed by dimensionality reduction (e.g., PCA or SVD) for document similarity tasks?  Under what conditions would one method be preferred over the other?