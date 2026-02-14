---
title: "Visualizing Embeddings (t-SNE, PCA)"
date: "2026-02-14"
week: 7
lesson: 5
slug: "visualizing-embeddings-t-sne-pca"
---

# Topic: Visualizing Embeddings (t-SNE, PCA)

## 1) Formal definition (what is it, and how can we use it?)

**Embedding Visualization** refers to the process of projecting high-dimensional embeddings (like word embeddings or document embeddings) into lower-dimensional space (typically 2D or 3D) so they can be visually represented on a graph or plot.  This allows us to qualitatively assess the relationships captured by the embeddings.

**How we can use it:**

*   **Understanding Relationships:** Visualize whether semantically similar words/concepts are located near each other in the embedding space. This can help validate the quality of the embedding model and identify potential biases.
*   **Identifying Clusters:**  See if words/documents with similar characteristics cluster together. This can reveal underlying topics or categories within the dataset.
*   **Model Debugging:** If embeddings don't align with expectations (e.g., antonyms are close together), it can indicate issues with the training data, model architecture, or hyperparameters.
*   **Communication:** Provides a visually intuitive way to explain complex embedding spaces to stakeholders.

**t-SNE (t-distributed Stochastic Neighbor Embedding):** A non-linear dimensionality reduction technique particularly well-suited for visualizing high-dimensional data. It attempts to preserve the local structure of the data, meaning that points that are close to each other in the high-dimensional space will also tend to be close to each other in the low-dimensional space.  T-SNE focuses on capturing the neighborhood relationships rather than preserving global distances. It works by:
    1.  Constructing a probability distribution over pairs of high-dimensional data points such that the probability is proportional to the similarity of the points.
    2.  Defining a similar probability distribution over points in the low-dimensional space.
    3.  Minimizing the Kullback-Leibler (KL) divergence between the two probability distributions with respect to the locations of the points in the low-dimensional space.

**PCA (Principal Component Analysis):** A linear dimensionality reduction technique that identifies the principal components (directions of maximum variance) in the data. It projects the data onto these principal components, effectively reducing the dimensionality while retaining as much variance as possible. PCA is useful for identifying the most important features in the data and for visualizing the overall structure.  PCA transforms the data into a new coordinate system where the axes are orthogonal and ordered by the amount of variance they explain.  The steps are:
    1.  Standardize the data.
    2.  Compute the covariance matrix.
    3.  Compute the eigenvectors and eigenvalues of the covariance matrix.
    4.  Select the principal components based on the eigenvalues.
    5.  Project the data onto the selected principal components.

The key difference is that t-SNE is non-linear and better at preserving local structure, while PCA is linear and aims to preserve global variance. PCA is generally faster and more deterministic than t-SNE.

## 2) Application scenario

Let's say we've trained a word embedding model (e.g., Word2Vec, GloVe, or FastText) on a large corpus of text. We want to understand if the embedding space learned meaningful relationships between words.

*   **Scenario 1 (t-SNE):** We want to visualize the relationships between words related to "animals". We take the embeddings of words like "dog," "cat," "lion," "tiger," "elephant," "mouse," "bird," "snake," etc.  We apply t-SNE to reduce the dimensionality of these embeddings to 2D.  If the visualization shows clusters of "domestic animals" (dog, cat, mouse), "big cats" (lion, tiger), etc., it indicates the embedding model has captured some semantic similarities.

*   **Scenario 2 (PCA):**  We want to get a general sense of the global structure of the word embeddings across a large vocabulary (e.g., several thousand words).  Using PCA to reduce the dimensionality to 2D allows us to see the overall variance explained by the first two principal components. This might reveal broad categories of words that are well-separated, providing a high-level overview.

*   **Scenario 3 (Document Embeddings):** We have document embeddings generated using Doc2Vec or a transformer model.  We can use t-SNE or PCA to visualize these embeddings and identify clusters of documents that share similar topics or themes. This could be used for topic modeling or document categorization.

## 3) Python method (if possible)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Assume 'word_embeddings' is a dictionary where keys are words and values are their embeddings
# For example:
# word_embeddings = {'dog': np.array([0.1, 0.2, 0.3, ...]), 'cat': np.array([0.4, 0.5, 0.6, ...]), ...}

def visualize_embeddings(embeddings, method='t-SNE', perplexity=30, n_components=2, random_state=42):
    """
    Visualizes word embeddings using t-SNE or PCA.

    Args:
        embeddings (dict): A dictionary of word embeddings.
        method (str): 't-SNE' or 'PCA'.
        perplexity (int): Perplexity parameter for t-SNE.
        n_components (int): Number of dimensions to reduce to (usually 2 or 3).
        random_state (int): Random state for reproducibility.
    """
    words = list(embeddings.keys())
    embedding_matrix = np.array(list(embeddings.values()))

    if method == 't-SNE':
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init='pca')
        reduced_embeddings = tsne.fit_transform(embedding_matrix)
    elif method == 'PCA':
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embedding_matrix)
    else:
        raise ValueError("Method must be 't-SNE' or 'PCA'")

    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])  # Assuming 2D visualization
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.title(f"{method} Visualization of Word Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# Example Usage (replace with your actual word embeddings)
example_embeddings = {
    'dog': np.array([0.1, 0.2]),
    'cat': np.array([0.15, 0.25]),
    'lion': np.array([0.8, 0.7]),
    'tiger': np.array([0.75, 0.65]),
    'apple': np.array([0.2, 0.8]),
    'banana': np.array([0.25, 0.75]),
}


visualize_embeddings(example_embeddings, method='t-SNE') #Using t-SNE
visualize_embeddings(example_embeddings, method='PCA') #Using PCA

```

## 4) Follow-up question

What are the limitations of using t-SNE for visualizing embeddings, and how can we mitigate them? For example, what happens if we change the perplexity parameter significantly? How should we interpret the distances between points in a t-SNE plot?