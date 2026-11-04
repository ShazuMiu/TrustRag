---
id: selfrag
title: Self-RAG
source: Asai et al., ICLR
year: 2024
---
Self-RAG (Self-Reflective Retrieval-Augmented Generation) was proposed by Akari Asai and colleagues from the University of Washington, the Allen Institute for AI and IBM Research. Instead of always retrieving a fixed number of passages, Self-RAG trains a single language model to decide when retrieval is needed and to critique its own output.

The model generates special reflection tokens. The Retrieve token signals whether retrieval is required. The ISREL token judges whether a retrieved passage is relevant, the ISSUP token judges whether the generated segment is supported by the passage, and the ISUSE token rates the overall usefulness of the response. The authors trained Self-RAG on Llama 2 models with 7 billion and 13 billion parameters. Self-RAG improved factuality and citation accuracy for long-form generation compared with standard retrieval-augmented models, but it requires fine-tuning the generator.
