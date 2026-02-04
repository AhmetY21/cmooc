## Topic: Stemming vs. Lemmatization: When to use which?

1- **Provide formal definition, what is it and how can we use it?**

*   **Stemming:** Stemming is a text normalization technique in NLP that reduces words to their root form by chopping off affixes (prefixes and suffixes). The goal is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. Stemming algorithms typically operate based on heuristic rules without understanding the context or meaning of the word. Because of this, the resulting stem may not be a valid word.

    *   **Use:** Stemming is useful for reducing the vocabulary size in a corpus, improving the efficiency of search engines, and grouping similar words together for analysis.

*   **Lemmatization:** Lemmatization, on the other hand, is a more sophisticated text normalization technique that aims to find the *lemma* of a word. The lemma is the dictionary form of a word (the base or canonical form). Lemmatization involves considering the context of the word and using morphological analysis to identify the lemma. This process requires a vocabulary and morphological analysis to correctly derive the lemma. Therefore, the resulting lemma is always a valid word.

    *   **Use:** Lemmatization is used when it's important to have valid words as output and when the context of the word is important for the analysis. It's more accurate than stemming and provides meaningful base forms for words.

| Feature         | Stemming                                     | Lemmatization                                      |
|-----------------|----------------------------------------------|----------------------------------------------------|
| Process         | Rule-based chopping of affixes              | Morphological analysis and vocabulary lookup      |
| Output          | May not be a valid word                       | Always a valid word                                |
| Accuracy        | Lower                                        | Higher                                             |
| Complexity      | Simpler, faster                               | More complex, slower                                |
| Context aware  | No                                          | Yes                                                  |

2- **Provide an application scenario**

*   **Stemming Scenario:** Consider a search engine application. A user searches for "running". If we use stemming, both "running" and "ran" would be reduced to the stem "run". This means the search engine will return documents containing both "running" and "ran", increasing the recall (the proportion of relevant documents retrieved). Because search engines process vast amounts of data, the speed advantage of stemming is beneficial.

*   **Lemmatization Scenario:** In sentiment analysis, understanding the precise meaning of words is crucial. If we are analyzing the sentence "The cat is running happily", lemmatizing "running" to "run" retains the verbal form, contributing to a more accurate sentiment score. If we used stemming and ended up with "runn" it would be much harder to contextualize the sentiment. Furthermore, lemmatization ensures that the resulting word is a valid word, which can be beneficial for human readability in post-processing or analysis.

3- **Provide a method to apply in python (if possible)**

python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt') # needed for word_tokenize
nltk.download('wordnet') # needed for WordNetLemmatizer
nltk.download('omw-1.4') # needed for WordNetLemmatizer

# Sample sentence
sentence = "The cats are running happily, and ran earlier."

# Tokenization
tokens = word_tokenize(sentence)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]
print("Stemmed words:", stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
print("Lemmatized words:", lemmatized_words)


**Explanation:**

*   We use `nltk` (Natural Language Toolkit), a popular Python library for NLP.
*   `PorterStemmer` is used for stemming.  It's a widely used stemming algorithm.
*   `WordNetLemmatizer` is used for lemmatization. It uses WordNet's lexical database to look up the correct lemma.
*   `word_tokenize` splits the sentence into individual words.

**Output:**


Stemmed words: ['the', 'cat', 'are', 'run', 'happili', ',', 'and', 'ran', 'earlier', '.']
Lemmatized words: ['The', 'cat', 'are', 'running', 'happily', ',', 'and', 'ran', 'earlier', '.']


Notice that stemming converts "running" to "run" and "happily" to "happili". Lemmatization leaves "running" as "running" (because it correctly identifies it as a verb in its present participle form) and "cats" to cat, but does not change "ran."

4- **Provide a follow up question about that topic**

How does Part-of-Speech (POS) tagging enhance lemmatization, and why is it crucial for achieving accurate results in complex sentences?  What are the practical limitations of using POS-tagged lemmatization in real-world NLP pipelines?

5- **Schedule a chatgpt chat to send notification (Simulated)**

**Simulation:**

*Notification:*
Subject: NLP Deep Dive - Stemming vs. Lemmatization Follow-up
Body:  Hi there!  Just a reminder about our follow-up chat on Stemming vs. Lemmatization. We'll be discussing how POS tagging impacts lemmatization accuracy and its real-world limitations. Ready to continue the NLP learning journey?
- Sincerely, ChatGPT