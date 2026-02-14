---
title: "Semantic Analogies with Word Vectors"
date: "2026-02-14"
week: 7
lesson: 6
slug: "semantic-analogies-with-word-vectors"
---

# Topic: Semantic Analogies with Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

Semantic analogies with word vectors leverage the distributional hypothesis: words that occur in similar contexts tend to have similar meanings. Word embeddings, like Word2Vec, GloVe, or fastText, encode these contextual similarities into vector spaces.  The core idea is that if "A is to B as C is to D" holds true semantically (e.g., "king is to man as queen is to woman"), then the vector relationship `vector(B) - vector(A) ≈ vector(D) - vector(C)` should approximately hold in the embedding space.

More formally:

*   Given words A, B, and C, we want to find a word D such that the analogy "A is to B as C is to D" holds.
*   We represent each word with its corresponding vector in the embedding space: `v_A`, `v_B`, `v_C`.
*   We compute a target vector `v_target = v_B - v_A + v_C`.
*   We then search for the word vector `v_D` in the embedding space that is closest to `v_target`, typically using cosine similarity. The word corresponding to this nearest vector is our predicted D.

**How we can use it:**

*   **Testing embedding quality:**  Analogies serve as a diagnostic tool for evaluating how well word embeddings capture semantic relationships. High accuracy on analogy tasks suggests a better quality embedding.
*   **Relationship discovery:** Exploring the embedding space to uncover hidden or unexpected relationships between words.  For example, discovering analogies related to specific concepts or domains.
*   **Word prediction/completion:** Filling in missing words in sentences or completing analogies based on context.
*   **Knowledge graph completion:**  Inferring relationships between entities in knowledge graphs by representing entities as vectors and leveraging analogy reasoning.

## 2) Application scenario

Consider the analogy "Paris is to France as Berlin is to ______".

*   A high-quality word embedding should capture the "capital of" relationship.
*   Therefore, the vector `vector("France") - vector("Paris")` should be similar to the vector `vector("Germany") - vector("Berlin")`.
*   By computing `vector("France") - vector("Paris") + vector("Berlin")`, and searching for the word vector closest to the result, we expect to find "Germany" as the answer.

Another application could be in medical diagnosis. Imagine having word embeddings representing medical concepts (diseases, symptoms, treatments).  One might ask: "Diabetes is to Insulin as Hyperthyroidism is to ______".  The system could potentially suggest treatment options based on the learned relationships.  This is a more speculative scenario and requires carefully constructed embeddings and validation.

## 3) Python method (if possible)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analogy(word_vectors, word_a, word_b, word_c):
  """
  Performs the analogy A:B :: C:D using word vectors.

  Args:
      word_vectors: A dictionary mapping words to their vector representations.
      word_a: The word A in the analogy.
      word_b: The word B in the analogy.
      word_c: The word C in the analogy.

  Returns:
      The predicted word D in the analogy, or None if any word is not found.
  """

  if word_a not in word_vectors or word_b not in word_vectors or word_c not in word_vectors:
    print("One or more words not found in vocabulary.")
    return None

  v_a = word_vectors[word_a]
  v_b = word_vectors[word_b]
  v_c = word_vectors[word_c]

  v_target = v_b - v_a + v_c

  best_word = None
  best_similarity = -1

  for word, vector in word_vectors.items():
    if word in [word_a, word_b, word_c]:  # Avoid returning A, B, or C
      continue

    similarity = cosine_similarity(v_target.reshape(1, -1), vector.reshape(1, -1))[0][0]

    if similarity > best_similarity:
      best_similarity = similarity
      best_word = word

  return best_word


# Example usage (requires a word_vectors dictionary to be loaded)
# This example uses randomly initialized vectors for demonstration.
# In a real scenario, you would load pretrained word vectors (e.g., from Gensim).

# Create a dummy word_vectors dictionary for demonstration:
word_vectors = {
    "king": np.array([1.0, 2.0, 3.0]),
    "man": np.array([4.0, 5.0, 6.0]),
    "queen": np.array([7.0, 8.0, 9.0]),
    "woman": np.array([10.0, 11.0, 12.0]),
    "prince": np.array([13.0, 14.0, 15.0]),
    "princess": np.array([16.0, 17.0, 18.0])
}

predicted_word = analogy(word_vectors, "king", "man", "queen")

if predicted_word:
  print(f"king is to man as queen is to {predicted_word}")
```

## 4) Follow-up question

What are some limitations of using semantic analogies with word vectors, and how can these limitations be addressed? For example, are there biases that the system can pick up on, or certain types of analogies it struggles with?