---
title: "Text Similarity Measures (Cosine, Jaccard)"
date: "2026-02-13"
week: 7
lesson: 4
slug: "text-similarity-measures-cosine-jaccard"
---

# Topic: Text Similarity Measures (Cosine, Jaccard)

## 1) Formal definition (what is it, and how can we use it?)

Text similarity measures are quantitative metrics that assess how alike two pieces of text are. They provide a numerical score indicating the degree of similarity between textual data, allowing us to compare documents, sentences, or even individual words. These measures are crucial for a wide range of NLP tasks where identifying semantic relatedness is important. Two common measures are Cosine similarity and Jaccard similarity.

*   **Cosine Similarity:** Cosine similarity measures the angle between two non-zero vectors in a multi-dimensional space. In the context of text, each document is represented as a vector, where each dimension corresponds to a word (or n-gram) and the value represents the frequency or TF-IDF weight of that word. The cosine similarity score is calculated as the cosine of the angle between these vectors. A cosine similarity of 1 indicates perfect similarity (same direction), 0 indicates orthogonality (no similarity), and -1 indicates complete opposition. The formula is:

    `cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)`

    where:
    *   `A . B` is the dot product of vectors A and B.
    *   `||A||` and `||B||` are the magnitudes (Euclidean norms) of vectors A and B.

    We can use cosine similarity to:
    *   Find similar documents in a corpus.
    *   Build recommendation systems based on text content.
    *   Identify plagiarism or duplicate content.
    *   Cluster documents based on their content.

*   **Jaccard Similarity:** Jaccard similarity (also known as the Jaccard index) measures the size of the intersection of two sets divided by the size of the union of the sets.  In text analysis, these sets often represent the unique words (or n-grams) present in two documents. The Jaccard similarity ranges from 0 (no common elements) to 1 (identical sets). The formula is:

    `Jaccard(A, B) = |A ∩ B| / |A ∪ B|`

    where:
    *   `A ∩ B` is the intersection of sets A and B (elements present in both sets).
    *   `A ∪ B` is the union of sets A and B (all elements present in either set).

    We can use Jaccard similarity to:
    *   Compare the similarity of sets of words (or other features) extracted from texts.
    *   Identify near-duplicate documents based on their vocabulary.
    *   Evaluate the overlap between topics discussed in different documents.
    *   In information retrieval, Jaccard Similarity is commonly used for document comparison where the document's text is preprocessed into a bag-of-words.

## 2) Application scenario

**Scenario:** Imagine you're building a news aggregator that groups similar news articles together.

*   **Cosine Similarity:** You could use cosine similarity to compare the TF-IDF vectors of each news article. Articles with a high cosine similarity score would be grouped together, indicating they likely cover the same topic.  Articles discussing, for example, "Inflation rates" will form a cluster and news articles that do not will have a lower similarity score.

*   **Jaccard Similarity:**  You could use Jaccard similarity to compare the set of keywords extracted from each news article. Articles with a high Jaccard similarity would be grouped together, indicating they share a significant number of common keywords and thus are likely about the same topic. If two articles share the same keywords "interest", "rates", and "economic growth", their Jaccard similarity would be high.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import jaccard_distance
import nltk
nltk.download('punkt') # Necessary for tokenizing

def calculate_cosine_similarity(text1, text2):
    """Calculates cosine similarity between two texts using TF-IDF."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def calculate_jaccard_similarity(text1, text2):
    """Calculates Jaccard similarity between two texts using set of words."""
    set1 = set(nltk.word_tokenize(text1))
    set2 = set(nltk.word_tokenize(text2))
    return 1 - jaccard_distance(set1, set2)  # jaccard_distance returns distance, so we subtract from 1 to get similarity


# Example usage:
text1 = "This is the first document."
text2 = "This document is the second document."
text3 = "This is a completely different sentence."

cosine_sim = calculate_cosine_similarity(text1, text2)
jaccard_sim = calculate_jaccard_similarity(text1, text2)

cosine_sim_diff = calculate_cosine_similarity(text1, text3)
jaccard_sim_diff = calculate_jaccard_similarity(text1, text3)


print(f"Cosine Similarity (text1, text2): {cosine_sim}")
print(f"Jaccard Similarity (text1, text2): {jaccard_sim}")

print(f"Cosine Similarity (text1, text3): {cosine_sim_diff}")
print(f"Jaccard Similarity (text1, text3): {jaccard_sim_diff}")

```

## 4) Follow-up question

How does the choice of pre-processing steps (e.g., stemming, lemmatization, stop word removal) affect the performance and interpretation of cosine and Jaccard similarity scores?  How would these choices differ based on the specific application?