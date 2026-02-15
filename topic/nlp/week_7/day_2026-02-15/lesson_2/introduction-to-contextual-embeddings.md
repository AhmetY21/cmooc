---
title: "Introduction to Contextual Embeddings"
date: "2026-02-15"
week: 7
lesson: 2
slug: "introduction-to-contextual-embeddings"
---

# Topic: Introduction to Contextual Embeddings

## 1) Formal definition (what is it, and how can we use it?)

Contextual word embeddings are representations of words where the meaning of a word is determined by its surrounding words (the context). Unlike static word embeddings (like Word2Vec or GloVe) which assign a single vector to each word regardless of its usage, contextual embeddings generate different vectors for the same word based on the sentence or document it appears in.

Formally, let `w` be a word in a sentence `S = [w_1, w_2, ..., w_n]`. A contextual embedding model `f` takes the sentence `S` as input and produces a vector `v_i = f(S, i)` for the word `w_i` at position `i`. This vector `v_i` represents the contextual meaning of `w_i` given the sentence `S`.

We can use contextual embeddings for various downstream tasks, including:

*   **Text Classification:** Representing documents or sentences as the average or concatenation of contextual embeddings.
*   **Named Entity Recognition (NER):**  Using the contextual embedding of a word as a feature for classifying its entity type.
*   **Question Answering:**  Embedding questions and passages, then using attention mechanisms to find relevant information based on the context.
*   **Sentiment Analysis:**  Determining the overall sentiment of a sentence by analyzing the contextual embeddings of its words.
*   **Machine Translation:** Improving translation accuracy by considering the context of words in both the source and target languages.
*   **Text Generation:**  Predicting the next word in a sequence based on the contextual embeddings of the preceding words.
## 2) Application scenario

Consider the word "bank." In static word embeddings, "bank" would have a single vector representation. However, "bank" can have different meanings:

*   "I went to the **bank** to deposit money." (Financial institution)
*   "The river **bank** was overgrown with weeds." (Land alongside a river)

With contextual embeddings, the embedding for "bank" in the first sentence would be different from the embedding for "bank" in the second sentence. A model like BERT would capture the semantic difference based on the surrounding words.

Imagine a sentiment analysis task where we want to determine if a sentence expresses a positive or negative sentiment.  The sentence "That was a **sick** performance!"  could be misinterpreted by a model using static word embeddings because "sick" often has a negative connotation. A contextual embedding model, however, would recognize that in this context, "sick" likely indicates something positive or impressive, leading to a more accurate sentiment prediction.

## 3) Python method (if possible)

We can use the `transformers` library in Python to generate contextual embeddings using models like BERT, RoBERTa, or DistilBERT.

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Model and tokenizer names
model_name = "bert-base-uncased"  # Or any other pre-trained model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example sentence
sentence = "The river bank was overgrown with weeds."

# Tokenize the sentence
tokens = tokenizer(sentence, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    outputs = model(**tokens)

# Extract the contextual embeddings (last hidden state)
embeddings = outputs.last_hidden_state

# 'embeddings' is a tensor of shape (batch_size, sequence_length, hidden_size)
# In this case, batch_size is 1 and sequence_length is the number of tokens.
# hidden_size is the dimensionality of the embeddings (e.g., 768 for bert-base-uncased).

# To get the embedding of a specific word, you need to identify its token index.
# For example, to get the embedding of "bank":
bank_index = tokens["input_ids"][0].tolist().index(tokenizer.encode("bank", add_special_tokens=False)[0])

bank_embedding = embeddings[0, bank_index, :]

# Print the shape of the 'bank' embedding
print(bank_embedding.shape) # Output: torch.Size([768])

# You can now use this 'bank_embedding' for downstream tasks.

#Example with another sentence:
sentence2 = "I went to the bank to deposit money."
tokens2 = tokenizer(sentence2, return_tensors="pt")

with torch.no_grad():
  outputs2 = model(**tokens2)

embeddings2 = outputs2.last_hidden_state
bank_index2 = tokens2["input_ids"][0].tolist().index(tokenizer.encode("bank", add_special_tokens=False)[0])
bank_embedding2 = embeddings2[0, bank_index2, :]

#Check if the embeddings are different:
print(torch.equal(bank_embedding, bank_embedding2)) #output: False
```

## 4) Follow-up question

How are contextual embeddings typically pre-trained, and what are the advantages and disadvantages of different pre-training objectives (e.g., Masked Language Modeling, Next Sentence Prediction)?