---
id: crag
title: Corrective Retrieval Augmented Generation (CRAG)
source: Yan et al.
year: 2024
---
Corrective Retrieval Augmented Generation (CRAG) was proposed by Shi-Qi Yan and colleagues to make RAG robust when retrieval goes wrong. CRAG adds a lightweight retrieval evaluator, fine-tuned from T5-large, that estimates the quality of the retrieved documents for a query.

Based on the evaluator's confidence, CRAG triggers one of three actions: Correct, Incorrect or Ambiguous. When retrieval is judged Correct, the documents are refined with a decompose-then-recompose algorithm that keeps the key knowledge strips and filters out irrelevant text. When retrieval is judged Incorrect, the documents are discarded and large-scale web search is used instead. The Ambiguous action combines both sources of knowledge. CRAG is plug-and-play and can be added to standard RAG as well as to Self-RAG.
