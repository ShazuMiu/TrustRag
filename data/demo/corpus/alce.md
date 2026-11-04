---
id: alce
title: ALCE benchmark for citation evaluation
source: Gao et al., EMNLP
year: 2023
---
ALCE (Automatic LLMs' Citation Evaluation) is a benchmark introduced by Tianyu Gao and colleagues at Princeton University to evaluate whether language models generate text with correct citations. It is built from three datasets: ASQA, QAMPARI and ELI5.

ALCE evaluates three dimensions: fluency, correctness and citation quality. Fluency is measured with MAUVE. Citation quality is measured with citation recall, which checks whether each output statement is fully supported by its cited passages, and citation precision, which identifies irrelevant citations. Both are computed automatically with TRUE, a T5-11B model fine-tuned for natural language inference. The authors found that even the best models lacked complete citation support about 50 percent of the time on the ELI5 dataset.
