---
title: "Named Entity Recognition (NER) Fundamentals"
date: "2026-02-10"
week: 7
lesson: 6
slug: "named-entity-recognition-ner-fundamentals"
---

# Topic: Named Entity Recognition (NER) Fundamentals

## 1) Formal definition (what is it, and how can we use it?)

Named Entity Recognition (NER), also known as entity identification, entity chunking, and entity extraction, is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, times, quantities, monetary values, percentages, etc.

Formally, given a text sequence *T* = (w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>), NER aims to identify contiguous spans of words (chunks) that represent named entities and assign a specific category *c* from a predefined set of entity types *C* to each identified entity.  The output is a set of (start_index, end_index, entity_type) tuples, where start_index and end_index define the span of the entity within the text, and entity_type is the assigned category.

We can use NER for several purposes:

*   **Information Retrieval:** Improve search relevance by identifying specific entities of interest within documents.
*   **Question Answering:** Extract key entities from a question to better understand the user's intent and formulate a more precise query against a knowledge base.
*   **Knowledge Base Construction:** Automatically populate knowledge bases with entities and their relationships extracted from text.
*   **Customer Service:** Identify customer needs and issues by extracting entities related to products, services, and complaints from customer interactions.
*   **Content Recommendation:** Suggest relevant content based on the entities discussed in a document or user's reading history.
*   **Fraud Detection:** Identifying suspicious patterns and relationships between entities involved in fraudulent activities.

## 2) Application scenario

Imagine a news article:

"Apple Inc. announced a new iPhone 15 at an event in Cupertino, California. CEO Tim Cook highlighted the phone's advanced features. The price will start at $799."

Applying NER to this text would identify the following entities:

*   "Apple Inc." - ORGANIZATION
*   "iPhone 15" - PRODUCT
*   "Cupertino, California" - GPE (Geopolitical Entity - a location)
*   "Tim Cook" - PERSON
*   "$799" - MONEY

This information can then be used to understand the article's subject (Apple's new iPhone), where the event took place, who announced it, and its price. This allows for easier indexing, searching, and summarization of the news article.  A news aggregator could categorize this article as "Technology" or "Business" based on the identified entities.

## 3) Python method (if possible)

We can use the `spaCy` library for NER in Python. Here's a basic example:

```python
import spacy

# Load the English language model (you might need to download it first: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. announced a new iPhone 15 at an event in Cupertino, California. CEO Tim Cook highlighted the phone's advanced features. The price will start at $799."

# Process the text with spaCy
doc = nlp(text)

# Iterate over the entities and print them
for ent in doc.ents:
    print(ent.text, ent.label_)

#Alternatively, to get the start and end character indices:
for ent in doc.ents:
    print(ent.text, ent.label_, ent.start_char, ent.end_char)
```

This code will output:

```
Apple Inc. ORG
iPhone 15 PRODUCT
Cupertino GPE
California GPE
Tim Cook PERSON
$799 MONEY
```

The `spacy.load("en_core_web_sm")` line loads a small English language model.  Larger models (e.g., "en_core_web_lg", "en_core_web_trf") generally provide more accurate NER performance, but require more computational resources.  You can download different models using `python -m spacy download en_core_web_lg` (or the desired model name).  The `doc.ents` attribute contains a list of `Span` objects, each representing a recognized entity. Each `Span` object has attributes like `text` (the entity text) and `label_` (the entity type).  The `start_char` and `end_char` attributes give the character indices of the entity within the original text.

## 4) Follow-up question

What are some common challenges and errors encountered in NER, and what techniques can be used to mitigate them? For example, how can we address issues like ambiguity in entity types (e.g., "Amazon" as a company vs. a river), or handling nested entities? How does the performance of NER systems vary across different languages and domains?