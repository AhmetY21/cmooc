---
title: "Ambiguity in Natural Language: Lexical, Syntactic, Semantic"
date: "2026-02-09"
week: 7
lesson: 2
slug: "ambiguity-in-natural-language-lexical-syntactic-semantic"
---

# Topic: Ambiguity in Natural Language: Lexical, Syntactic, Semantic

## 1) Formal definition (what is it, and how can we use it?)

Ambiguity in natural language refers to the property where a word, phrase, or sentence can have more than one meaning or interpretation. This inherent characteristic of human language poses a significant challenge for NLP systems, as these systems must be able to correctly discern the intended meaning in a given context. There are three primary types of ambiguity: lexical, syntactic, and semantic.

*   **Lexical Ambiguity:** Occurs when a single word has multiple meanings. This is also known as "word sense ambiguity." For example, the word "bank" can refer to a financial institution or the edge of a river.

*   **Syntactic Ambiguity:** Arises when a sentence has multiple possible grammatical structures, leading to different interpretations.  This is also called "structural ambiguity." For example, "I saw the man with the telescope."  Did I use the telescope to see him, or did he possess the telescope?

*   **Semantic Ambiguity:** This involves ambiguity at the level of meaning, often involving the relationships between words and phrases in a sentence, even when the syntactic structure is clear.  This can arise from vagueness or unresolved references.  For example, "The pen is mightier than the sword." Is this literally about pens and swords, or is it a metaphorical statement about the power of communication over violence?

We can use the understanding of ambiguity to:

*   **Improve parsing accuracy:** By incorporating disambiguation techniques into parsers, we can choose the correct syntactic structure for ambiguous sentences.
*   **Enhance machine translation:** Correctly resolving ambiguity is crucial for accurately translating text from one language to another.
*   **Build better information retrieval systems:**  By understanding the multiple meanings of search queries, we can provide more relevant search results.
*   **Develop more robust chatbot applications:**  Chatbots need to understand the intent of the user, even when their input is ambiguous.
*   **Evaluate NLP model performance:** The ability of an NLP model to resolve ambiguity is a key measure of its overall performance and understanding of language.

## 2) Application scenario

Consider a customer service chatbot. A user types the query: "I want to book a flight to see my relatives."

*   **Lexical Ambiguity:** The word "book" can mean either to reserve something (a flight) or a physical book.
*   **Syntactic Ambiguity:** While less prominent here, one could technically parse "to see my relatives" as either the *reason* for the flight or a description of the destination. For example "I want to book a flight to a place like [see my relatives] Florida."
*   **Semantic Ambiguity:** The phrase "relatives" is ambiguous. Who are these relatives? Where do they live? The chatbot needs to understand the *intent* to book a flight and then prompt the user to specify the destination and dates to resolve the semantic and underlying pragmatic ambiguity.

Without addressing the ambiguity, the chatbot might misinterpret the user's request (e.g., try to sell them a book instead of a flight) or fail to gather enough information to book a suitable flight. A good chatbot would need to employ disambiguation techniques to understand the user's intent accurately and provide relevant assistance.

## 3) Python method (if possible)

While there isn't a single Python function that magically *solves* ambiguity, several NLP libraries offer tools that can help *mitigate* its effects. One common approach involves using Word Sense Disambiguation (WSD) techniques.  Here's an example using NLTK and WordNet, demonstrating a simplified Lesk algorithm for Lexical Ambiguity:

```python
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

def simplified_lesk(word, sentence):
    """
    Simplified Lesk algorithm for word sense disambiguation.
    """
    best_sense = None
    max_overlap = 0
    sentence_context = set(word_tokenize(sentence))

    for synset in wn.synsets(word):
        synset_context = set(word_tokenize(synset.definition()))
        overlap = len(sentence_context.intersection(synset_context))

        for example in synset.examples():
            synset_context.update(word_tokenize(example))

        overlap = len(sentence_context.intersection(synset_context))

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = synset

    return best_sense

# Example usage
sentence = "I went to the bank to deposit money."
ambiguous_word = "bank"

sense = simplified_lesk(ambiguous_word, sentence)

if sense:
    print(f"The sense of '{ambiguous_word}' in the sentence is: {sense.definition()}")
else:
    print(f"Could not disambiguate '{ambiguous_word}'.")

# another Example
sentence = "I went to the bank of the river to relax."
ambiguous_word = "bank"

sense = simplified_lesk(ambiguous_word, sentence)

if sense:
    print(f"The sense of '{ambiguous_word}' in the sentence is: {sense.definition()}")
else:
    print(f"Could not disambiguate '{ambiguous_word}'.")


nltk.download('punkt') # first time only.
nltk.download('wordnet') # first time only.

```

**Explanation:**

1.  **`simplified_lesk(word, sentence)`:** This function takes the ambiguous word and the sentence as input.
2.  **`wn.synsets(word)`:**  It retrieves all possible synsets (sets of synonyms representing distinct concepts) for the given word from WordNet.
3.  **Context Overlap:** It calculates the overlap between the context of each synset (definition and examples) and the context of the sentence. This is a simplified version, as more sophisticated methods would consider part-of-speech tagging, stemming, and other factors.
4.  **Best Sense Selection:** The synset with the highest overlap is chosen as the most likely sense of the word in the given sentence.
5.  **Prints results:** If a suitable synset can be determined it prints out the corresponding definition.

**Important Notes:**

*   This is a *simplified* Lesk algorithm. Real-world WSD systems are significantly more complex and often rely on machine learning models trained on large corpora.
*   NLTK requires downloading resources the first time you use it. The included code downloads `punkt` for tokenization, and `wordnet` for word sense information.
*   More sophisticated approaches to disambiguation might use pre-trained language models like BERT or transformer networks to capture contextual information more effectively.  These models can be fine-tuned for specific WSD tasks.

## 4) Follow-up question

How can deep learning models, such as transformer networks, be used to address syntactic and semantic ambiguity more effectively than traditional rule-based or statistical methods? What are the limitations of using deep learning for ambiguity resolution?