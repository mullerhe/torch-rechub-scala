package torchrec.serving

/**
 * Serving module for vector search and ANN indexing.
 *
 * Supports multiple backend implementations via [[BuilderFactory]]:
 *  - "vector": pure brute-force indexer (default)
 *  - "annoy": ANNOY-style (random projection tree forest)
 *  - "faiss": FAISS-style (FLAT/HNSW/IVF indexes)
 *  - "milvus": Milvus-style (FLAT/HNSW/IVF indexes)
 *
 * @example
 * {{{
 * val builder = BuilderFactory("vector", Map("metric" -> IndexMetric.L2))
 * val indexer = builder.from_embeddings(embeddings)
 * val (ids, distances) = indexer.query(queryEmbeddings, topK = 10)
 * }}}
 */
package object serving
