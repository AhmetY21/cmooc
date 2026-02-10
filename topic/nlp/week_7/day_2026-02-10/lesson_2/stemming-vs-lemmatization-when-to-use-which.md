---
title: "Stemming vs. Lemmatization: When to use which?"
date: "2026-02-10"
week: 7
lesson: 2
slug: "stemming-vs-lemmatization-when-to-use-which"
---

# Topic: Stemming vs. Lemmatization: When to use which?

## 1) Formal definition (what is it, and how can we use it?)

**Stemming:**

Stemming is a text normalization technique that reduces words to their root form by chopping off prefixes and suffixes. It's a heuristic process that aims to group together words with similar meanings, even if the resulting stem isn't a valid dictionary word. The primary goal is to reduce vocabulary size and improve the effectiveness of information retrieval and text analysis. Stemming algorithms operate based on rules, often iteratively removing known prefixes and suffixes.

**How we use it:** Stemming is used to simplify words, allowing us to treat variations of a word (e.g., "running," "runs," "run") as the same entity. This improves recall in search engines (finding more relevant documents) and reduces dimensionality in machine learning models by reducing the number of unique words.

**Lemmatization:**

Lemmatization, unlike stemming, aims to find the *lemma* or dictionary form of a word. The lemma represents the base or canonical form of a word as it would appear in a dictionary. This process uses a vocabulary and morphological analysis to achieve the correct form, considering the word's context and part of speech (POS). It transforms words to their meaningful root form.

**How we use it:** Lemmatization is used when it's important to preserve the meaning and readability of the text. It's beneficial when you need the "real" root form of the word and want to avoid creating meaningless stems. Like stemming, it reduces vocabulary size, but with more attention to correctness and meaningfulness.

**Key Differences Summary:**

| Feature       | Stemming                               | Lemmatization                            |
|---------------|----------------------------------------|------------------------------------------|
| Method        | Rule-based, heuristic chopping         | Vocabulary & morphological analysis      |
| Output        | Might not be a valid word              | Always a valid dictionary word (lemma)   |
| Complexity    | Simpler, faster                        | More complex, slower                     |
| Accuracy      | Lower, potential for over-stemming     | Higher, more accurate root form identification |
| Meaning       | Meaning not always preserved           | Meaning preserved                        |

## 2) Application scenario

**When to use Stemming:**

*   **Information Retrieval (Search Engines):** When recall is more important than precision. You want to find as many relevant documents as possible, even if some are slightly off-topic. The speed of stemming is also advantageous here.
*   **Text Categorization/Classification (Especially with high dimensionality):** When you need to reduce the feature space significantly. The slightly lower accuracy is acceptable for the gains in speed and dimensionality reduction.
*   **Sentiment Analysis (Sometimes):** Can be useful as a preprocessing step, especially when combined with other techniques. However, lemmatization might be preferred if precise word meanings are crucial.

**When to use Lemmatization:**

*   **Question Answering:** When understanding the exact meaning of the question and answer is critical.
*   **Text Summarization:** When maintaining readability and coherence is vital for the summary.
*   **Machine Translation:** When translating between languages, it is crucial to understand the grammatical structure and meaning of words.
*   **Chatbots/Conversational AI:** When ensuring the chatbot understands the intent of the user correctly.
*   **Sentiment Analysis (If High Accuracy Required):** When subtle differences in word meanings can significantly impact sentiment (e.g., "good" vs. "well").

**Example demonstrating the difference:**

Consider the words "better" and "good".

*   **Stemming:** Might reduce "better" to something like "bett", which is not a valid word and doesn't necessarily link it to "good".
*   **Lemmatization:** Would correctly reduce "better" to "good".

## 3) Python method (if possible)

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') # Required for POS tagging in lemmatization

# Stemming
stemmer = PorterStemmer()
words = ["running", "runs", "run", "easily", "caring", "better"]
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed words:", stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()

# Function to convert nltk tag to wordnet tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default to noun if unknown

# Lemmatization with POS tagging
lemmatized_words = []
for word in words:
    pos_tag = nltk.pos_tag([word])[0][1]  # Get POS tag
    wordnet_tag = get_wordnet_pos(pos_tag)  # Convert to WordNet tag
    lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_tag)) # Lemmatize with POS
print("Lemmatized words:", lemmatized_words)

# Lemmatization without POS (defaults to noun)
lemmatized_words_no_pos = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized words (no POS):", lemmatized_words_no_pos)

#Example with a sentence, demonstrating the usage in context
sentence = "The cats were running across the field."
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
lemmatized_sentence = " ".join([lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags])
print("Lemmatized sentence:", lemmatized_sentence)
```

**Explanation:**

1.  **Stemming (PorterStemmer):** The `PorterStemmer` from `nltk.stem` is used.  It applies a series of rules to chop off suffixes. Note the results (e.g., 'run' becomes 'runn', 'easily' becomes 'easili').
2.  **Lemmatization (WordNetLemmatizer):** The `WordNetLemmatizer` from `nltk.stem` is used.
3.  **POS Tagging:** Crucially, lemmatization benefits significantly from Part-of-Speech (POS) tagging. The example shows how to tag words using `nltk.pos_tag` and how to map these tags to WordNet's POS tags. Lemmatizing with the correct POS tag gives much better results.  `get_wordnet_pos` converts NLTK's POS tags to WordNet POS tags.
4.  **Lemmatization without POS:**  If no POS tag is provided to `lemmatize()`, it defaults to treating the word as a noun, which may not always be correct (e.g., "running" becomes "running" when treated as a noun, instead of "run" if treated as a verb).
5. **Sentence Example:** Demonstrates the correct usage of lemmatization in a full sentence by combining the tokenization, pos tagging, and lemmatization.

## 4) Follow-up question

How do more advanced techniques like Transformers (e.g., BERT, RoBERTa) handle word variations (morphological differences like tense, number, etc.) compared to traditional stemming and lemmatization? Do these models eliminate the need for explicit stemming/lemmatization as a preprocessing step, or can it still be beneficial? What are the trade-offs?