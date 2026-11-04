---
id: faiss
title: FAISS similarity search library
source: Johnson, Douze and Jegou
year: 2017
---
FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search and clustering of dense vectors, released by Facebook AI Research in 2017. It is written in C++ with complete wrappers for Python, and many of its algorithms have GPU implementations.

FAISS supports exact search as well as approximate nearest neighbor indexes. Inverted file indexes (IVF) partition the vectors into clusters so that only a few clusters are scanned for each query. Product quantization (PQ) compresses vectors into short codes to reduce memory use, and graph-based HNSW indexes are also available. The accompanying paper, "Billion-scale similarity search with GPUs", showed that the library can search collections of a billion vectors. FAISS is commonly used as the vector index inside retrieval-augmented generation systems.
