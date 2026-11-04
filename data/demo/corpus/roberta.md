---
id: roberta
title: RoBERTa
source: Liu et al.
year: 2019
---
RoBERTa (Robustly Optimized BERT Pretraining Approach) was released by Yinhan Liu and colleagues at Facebook AI in 2019. It keeps the BERT architecture but changes the pre-training recipe. RoBERTa removes the next sentence prediction objective, uses dynamic masking that changes the masked tokens in every epoch, trains with much larger batches and uses about 160 GB of text, roughly ten times more data than the original BERT. These changes improved results on the GLUE, SQuAD and RACE benchmarks.
