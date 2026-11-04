---
id: bert
title: BERT language model
source: Devlin et al., NAACL
year: 2019
---
BERT, which stands for Bidirectional Encoder Representations from Transformers, was introduced by Jacob Devlin and colleagues at Google AI Language. Unlike left-to-right language models, BERT is pre-trained to use context from both the left and the right of every token.

BERT is pre-trained with two objectives. In masked language modeling, 15 percent of the input tokens are selected for prediction and the model must recover them. In next sentence prediction, the model learns whether a second sentence actually follows the first one. BERT-Base has 12 layers and about 110 million parameters, while BERT-Large has 24 layers and about 340 million parameters. The pre-training data was the BooksCorpus with 800 million words and English Wikipedia with 2,500 million words. BERT encoders are widely used inside dense retrievers and cross-encoder rerankers.
