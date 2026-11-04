---
id: dpr
title: Dense Passage Retrieval (DPR)
source: Karpukhin et al., EMNLP
year: 2020
---
Dense Passage Retrieval (DPR) was introduced by Vladimir Karpukhin and colleagues at Facebook AI in 2020. DPR uses two independent BERT encoders, one for questions and one for passages, and scores a passage by the dot product of the two embeddings. The encoders are trained with in-batch negatives, where the passages paired with other questions in the same batch serve as negative examples.

On the Natural Questions benchmark, DPR reached a top-20 passage retrieval accuracy of 78.4 percent, compared with 59.1 percent for BM25. Dense retrievers capture paraphrases well but can miss rare names and exact numbers that lexical retrievers find easily.
