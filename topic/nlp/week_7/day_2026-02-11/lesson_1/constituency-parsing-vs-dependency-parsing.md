---
title: "Constituency Parsing vs Dependency Parsing"
date: "2026-02-11"
week: 7
lesson: 1
slug: "constituency-parsing-vs-dependency-parsing"
---

# Topic: Constituency Parsing vs Dependency Parsing

## 1) Formal definition (what is it, and how can we use it?)

Both constituency parsing and dependency parsing are syntactic parsing techniques used in Natural Language Processing (NLP) to analyze the grammatical structure of a sentence. They aim to represent how words in a sentence relate to each other, but they do so in different ways.

**Constituency Parsing (Phrase Structure Parsing):**

*   **Definition:** Constituency parsing decomposes a sentence into its constituent parts (phrases or chunks), building a tree structure based on formal grammar rules. These rules define how words group together to form phrases, which in turn group together to form larger phrases, ultimately forming the entire sentence. The resulting tree is a **constituency tree** or **phrase structure tree**. Nodes in the tree represent constituents (e.g., noun phrase (NP), verb phrase (VP), prepositional phrase (PP), etc.), and the leaves represent the individual words.  Constituency parsing emphasizes hierarchical structure and grammatical relations based on these predefined phrase types.

*   **Usage:** Constituency parsing is used to understand the syntactic structure of a sentence, identify phrases and their types, and build representations for semantic analysis, machine translation, and grammatical error detection.  It is particularly useful when the grammatical correctness and well-formedness of sentences are important. For example, it can help determine if a verb is in the correct tense or if a noun phrase has the correct determiner.

**Dependency Parsing:**

*   **Definition:** Dependency parsing focuses on the relationships between individual words in a sentence. It represents the syntactic structure as a graph (usually a tree) where words are nodes, and the edges represent dependencies between words. Each edge is labeled with a grammatical relation, indicating the type of dependency (e.g., subject, object, modifier). The root of the tree is typically the main verb of the sentence. Dependency parsing represents the relationships between words directly, rather than through intermediate phrases.

*   **Usage:** Dependency parsing is useful for tasks that require understanding the semantic relationships between words, such as information extraction, question answering, and text summarization. It is particularly effective at identifying arguments of verbs and modifiers of nouns. It can also be used in machine translation and relation extraction. For example, in information extraction, we might want to know which noun is the subject of a verb, or which adjective modifies a given noun. Dependency parsing makes these relationships explicit.

In essence, constituency parsing tells us "what *kinds* of phrases exist in a sentence and how they are nested," while dependency parsing tells us "which words *depend* on other words and how."

## 2) Application scenario

**Constituency Parsing Application: Grammar Checking and Text Generation**

Imagine building a grammar checker. Constituency parsing can identify malformed phrases or sentences that violate grammatical rules. For example, if a sentence lacks a verb phrase where one is expected, the constituency parser will fail to produce a valid tree according to the grammar rules. Similarly, in text generation, constituency parsing can be used to ensure that generated sentences adhere to a predefined grammar, leading to more natural and grammatically correct output.

**Dependency Parsing Application: Information Extraction**

Consider the task of extracting relationships between entities from text. For example, given the sentence "Elon Musk founded SpaceX," dependency parsing can identify "Elon Musk" as the nominal subject (nsubj) of the verb "founded," and "SpaceX" as the direct object (dobj). This information can then be used to extract the relationship "founder-of" between Elon Musk and SpaceX.  This is crucial in building knowledge graphs and other information retrieval systems.

## 3) Python method (if possible)

```python
import spacy
from nltk.tree import Tree #optional. For visualization in constituency parsing

# Load a spaCy model (for dependency parsing)
nlp = spacy.load("en_core_web_sm")

# Example sentence
text = "The quick brown fox jumps over the lazy dog."

# Dependency Parsing using spaCy
doc = nlp(text)

print("Dependency Parsing:")
for token in doc:
    print(f"{token.text} --{token.dep_}-> {token.head.text}")

# Constituency Parsing using nltk and benepar (requires additional setup)

try:
    import benepar
    benepar.download('benepar_en3')  # Download the model (first time only)
    parser = benepar.Parser("benepar_en3")

    #Note that you would typically need to perform tokenization separately here
    #Using the sentence from before, but must be tokenized.
    tokenized_text = text.split() #A simple tokenization
    tree = parser.parse(tokenized_text)

    print("\nConstituency Parsing:")
    tree.pretty_print() #Optional for visualization

    #To get the tree as a string:
    #print(tree)

except ImportError:
    print("\nConstituency Parsing: Please install benepar.  `pip install benepar`")
except LookupError:
    print("\nConstituency Parsing: Please run benepar.download('benepar_en3')")
except Exception as e:
    print(f"Constituency Parsing Error: {e}")
```

**Explanation:**

*   **Dependency Parsing (spaCy):** The `spacy` library is used for dependency parsing.  The code iterates through each token (word) in the sentence and prints the word, its dependency relation (`token.dep_`), and the word it depends on (`token.head.text`).
*   **Constituency Parsing (benepar and nltk):**  The `benepar` library (built on top of PyTorch) is used in conjunction with `nltk` for constituency parsing. `benepar` provides a pre-trained model for English.  The code loads the model, parses the sentence (assuming a simple tokenization), and prints the resulting tree structure. `nltk`'s `Tree.pretty_print()` provides a visual representation of the tree, making it easier to understand the hierarchical structure.  If `benepar` is not installed, it gives instructions to do so.

**Note:** The `benepar` approach requires you to first tokenize the sentence before parsing. Also, be aware that setting up `benepar` can be slightly more involved than `spaCy`.

## 4) Follow-up question

Given a specific NLP task, how would you choose between using constituency parsing and dependency parsing, and are there situations where combining both approaches would be beneficial? Explain your reasoning with examples. For example, what kind of task would require using both?