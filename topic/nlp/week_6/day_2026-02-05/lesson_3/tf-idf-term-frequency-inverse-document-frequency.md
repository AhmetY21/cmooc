Topic: TF-IDF (Term Frequency-Inverse Document Frequency)

1- Provide formal definition, what is it and how can we use it?

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus. It's a weighting factor used in information retrieval and text mining. The TF-IDF value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus, which helps to control for the fact that some words are generally more common than others.

Formally, TF-IDF is calculated as the product of two metrics: Term Frequency (TF) and Inverse Document Frequency (IDF).

*   **Term Frequency (TF):** This measures how frequently a term occurs in a document.  Several variations exist for calculating TF:

    *   *Raw Count:* Simply the number of times the term appears in the document.
    *   *Frequency:* Raw count divided by the total number of terms in the document.
    *   *Log normalization:* The logarithm of the raw count (or the raw count + 1 to avoid log(0)).
    *   *Double normalization K:*  TF = K + (1-K) * (raw count / max frequency of any term in the document). K is typically set to 0.5.

*   **Inverse Document Frequency (IDF):** This measures how important a term is across the entire corpus. It is calculated as the logarithm of the number of documents in the corpus divided by the number of documents that contain the term.  A common formula is:

    IDF = log(N / df)

    where:
    *   N is the total number of documents in the corpus.
    *   df is the document frequency of the term (i.e., the number of documents containing the term).

    Variations exist, for example, adding 1 to both the numerator and denominator (log((1 + N)/(1 + df))), or just the numerator (log((N+1) / df)) to avoid division by zero errors.

**TF-IDF = TF * IDF**

We use TF-IDF to:

*   **Rank documents:**  In information retrieval, documents are ranked based on their TF-IDF scores for a given query. Documents with higher scores are considered more relevant.
*   **Keyword extraction:**  Terms with high TF-IDF scores in a document are often considered important keywords that represent the document's content.
*   **Document similarity:** TF-IDF vectors can be used to calculate the similarity between documents using techniques like cosine similarity.
*   **Feature engineering for machine learning:** TF-IDF scores can be used as features in machine learning models for tasks such as text classification and sentiment analysis.

2- Provide an application scenario

Imagine we have a search engine and a user searches for "quantum physics".  The search engine has indexed a large collection of documents.  To find relevant documents, the search engine can use TF-IDF.

1.  **Calculate TF-IDF:** For each document in the collection, the search engine calculates the TF-IDF score for the terms "quantum" and "physics".

2.  **Ranking:** Documents where "quantum" and "physics" appear frequently, and where these words are relatively uncommon in the rest of the document collection, will have higher TF-IDF scores.  The search engine then ranks the documents based on the sum of the TF-IDF scores for "quantum" and "physics" in each document.

3.  **Results:** Documents with the highest TF-IDF scores are presented to the user as the most relevant results for the search query "quantum physics".  A document that prominently discusses quantum physics will likely rank higher than a document that mentions it only in passing.

3- Provide a method to apply in python

python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array for easier viewing (optional)
tfidf_array = tfidf_matrix.toarray()

# Print the TF-IDF matrix
print("Feature Names:", feature_names)
print("\nTF-IDF Matrix:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")
    for j, word in enumerate(feature_names):
        print(f"  {word}: {tfidf_array[i][j]:.4f}") # Print the TF-IDF value for each word in each document
    print("-" * 20)

#Example Usage : Get the TF-IDF value for the word 'document' in the first document
doc_index = 0 #first document
word_index = list(feature_names).index('document')
tfidf_value = tfidf_array[doc_index][word_index]
print(f"\nTF-IDF value for 'document' in Document 1: {tfidf_value:.4f}")


Explanation:

1.  **Import `TfidfVectorizer`:** We import the `TfidfVectorizer` class from the `sklearn.feature_extraction.text` module.
2.  **Sample Documents:**  We create a list of sample documents.
3.  **Create Vectorizer:**  We create an instance of `TfidfVectorizer`.  We can customize this by specifying parameters like stop words, minimum document frequency (`min_df`), maximum document frequency (`max_df`), and n-gram ranges (`ngram_range`).
4.  **Fit and Transform:**  We call `fit_transform` on the documents.  This does two things:
    *   `fit`: Learns the vocabulary and IDF from the documents.
    *   `transform`: Transforms the documents into a TF-IDF matrix.
5.  **Get Feature Names:**  We use `get_feature_names_out()` to retrieve the vocabulary (the words in the corpus).
6.  **Convert to Array:**  The `tfidf_matrix` is a sparse matrix. We convert it to a dense array using `toarray()` for easier printing and examination.  This is optional, and working with sparse matrices is generally more efficient for large datasets.
7. **Print the TF-IDF Matrix** We iterate through the matrix and print out the TF-IDF score for each word in each document.
8. **Example Usage** We show how to retrieve a specific TF-IDF value from the matrix.

4- Provide a follow up question about that topic

How does TF-IDF handle synonyms and different forms of the same word (e.g., "run", "running", "ran"), and what techniques can be used to improve TF-IDF's performance in such scenarios?