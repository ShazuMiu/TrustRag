---
id: rag
title: Retrieval-Augmented Generation (RAG)
source: Lewis et al., NeurIPS
year: 2020
---
Retrieval-Augmented Generation (RAG) was introduced by Patrick Lewis and colleagues at Facebook AI Research in 2020. RAG combines parametric memory, stored in the weights of a pre-trained sequence-to-sequence model, with non-parametric memory, a dense vector index of Wikipedia. The retriever is a Dense Passage Retriever (DPR) built on BERT encoders, and the generator is BART. The Wikipedia index contains about 21 million passages of 100 words each. Relevant passages are found with Maximum Inner Product Search using the FAISS library.

The paper proposes two variants. RAG-Sequence uses the same retrieved passage to generate the complete output sequence, while RAG-Token can use a different passage for each generated token. The retriever query encoder and the generator are fine-tuned jointly, while the document encoder is kept fixed. RAG set new state-of-the-art results on open-domain question answering benchmarks such as Natural Questions, TriviaQA and WebQuestions.
