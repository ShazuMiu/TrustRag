---
id: citefix
title: CiteFix post-processing citation correction
source: Maheshwari et al., ACL Industry Track
year: 2025
---
CiteFix is a post-processing framework for correcting citations in retrieval-augmented generation, proposed by Harsh Maheshwari and colleagues. Instead of changing the retriever or the generator, CiteFix splits a generated response into factual points and checks which retrieved document actually supports each point.

The method compares each factual point with the candidate documents using hybrid lexical-semantic matching, including keyword matching, semantic similarity and BERTScore, and reassigns the citation to the best matching document. The authors reported a relative improvement of 15.46 percent in citation accuracy, measured with their MQLA metric. CiteFix adds little latency because it only runs lightweight matching after generation, but similarity matching can fail for complex multi-hop attributions.
