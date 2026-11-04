---
id: transformer
title: The Transformer architecture
source: Vaswani et al., NeurIPS
year: 2017
---
The Transformer was introduced in the 2017 paper "Attention Is All You Need" by Ashish Vaswani and colleagues at Google Brain and Google Research. It replaced recurrence and convolutions entirely with attention mechanisms. The original model follows an encoder-decoder structure in which both the encoder and the decoder are stacks of 6 identical layers.

The base model uses a model dimension of 512 and multi-head attention with 8 heads. Because the Transformer contains no recurrence, positional encodings based on sine and cosine functions are added to the input embeddings to give the model information about token order. On the WMT 2014 English-to-German translation task the big Transformer model reached a BLEU score of 28.4. The big model was trained for 3.5 days on 8 NVIDIA P100 GPUs.
