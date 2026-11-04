---
id: hallucination
title: Hallucination in large language models
source: Ji et al., ACM Computing Surveys
year: 2023
---
In natural language generation, hallucination refers to generated content that is nonsensical or unfaithful to the provided source content. A widely cited survey by Ziwei Ji and colleagues, published in ACM Computing Surveys in 2023, distinguishes two types. Intrinsic hallucinations are outputs that contradict the source content. Extrinsic hallucinations are outputs that cannot be verified from the source, neither supported nor contradicted by it.

Large language models can also produce imitative falsehoods learned from their training data. The TruthfulQA benchmark, introduced by Stephanie Lin and colleagues, contains 817 questions spanning 38 categories, including health, law, finance and politics, designed so that some humans would answer falsely because of misconceptions. Retrieval-augmented generation reduces hallucination by grounding answers in retrieved evidence, but it does not remove the problem when the retrieved evidence is irrelevant or when the model ignores it.
