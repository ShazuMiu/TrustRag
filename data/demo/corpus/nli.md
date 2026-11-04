---
id: nli
title: Natural Language Inference datasets and models
source: NLP survey notes
year: 2021
---
Natural Language Inference (NLI) is the task of deciding whether a hypothesis sentence is entailed by, contradicts, or is neutral with respect to a premise sentence. The three labels are therefore entailment, contradiction and neutral.

The Stanford Natural Language Inference corpus (SNLI), released by Samuel Bowman and colleagues in 2015, contains about 570,000 human-written English sentence pairs. The Multi-Genre NLI corpus (MultiNLI), released in 2018, contains about 433,000 sentence pairs drawn from ten genres of spoken and written text. Adversarial NLI (ANLI) was collected with a human-and-model-in-the-loop procedure to make examples harder for models. Models such as DeBERTa, developed by Microsoft, achieve strong NLI accuracy and are frequently used as automatic fact-checkers that test whether a generated claim is supported by a source passage.
