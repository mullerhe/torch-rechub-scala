package tutorials

import org.bytedeco.pytorch.global.torch

import torchrec.serving._
import torchrec.TorchRec.{toFloatArray, toLongArray}

/**
 * Tutorial: Vector Indexing and Serving
 */
object ServingVectorIndex {

  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("Vector Indexing Tutorial")
    println("=" * 70)

    // Generate sample embeddings
    val n = 1000
    val dim = 64
    println(s"\n1. Generating $n random embeddings (dim=$dim)...")
    val opts = new org.bytedeco.pytorch.TensorOptions()
    val embeddings = torch.randn(Array(n.toLong, dim.toLong), opts)

    // Test VectorIndexer (brute-force)
    println("\n2. Testing VectorIndexer (brute-force)...")
    val vectorBuilder = new VectorIndexerBuilder(IndexMetric.L2)
    val vectorIndexer = vectorBuilder.from_embeddings(embeddings)
    val queryVec = embeddings.select(0, 0).unsqueeze(0)
    val vectorResult = vectorIndexer.query(queryVec, topK = 5)
    val idsVec = toLongArray(vectorResult._1)
    val distsVec = toFloatArray(vectorResult._2)
    println(s"  Top-5 nearest to item 0: indices=${idsVec.take(5).mkString(", ")}, dists=${distsVec.take(5).map("%.4f".format(_)).mkString(", ")}")

    // Test FaissBuilder (Flat)
    println("\n3. Testing FaissBuilder (FLAT)...")
    val faissBuilder = new FaissBuilder(IndexType.FLAT, IndexMetric.L2)
    val faissIndexer = faissBuilder.from_embeddings(embeddings)
    val idsFaiss = toLongArray(faissIndexer.query(queryVec, topK = 5)._1)
    println(s"  Top-5: ${idsFaiss.take(5).mkString(", ")}")

    // Test FaissBuilder HNSW
    println("\n4. Testing FaissBuilder (HNSW)...")
    val hnswBuilder = new FaissBuilder(IndexType.HNSW, IndexMetric.L2, m = 16)
    val hnswIndexer = hnswBuilder.from_embeddings(embeddings)
    val idsHnsw = toLongArray(hnswIndexer.query(queryVec, topK = 5)._1)
    println(s"  Top-5: ${idsHnsw.take(5).mkString(", ")}")

    // Test BuilderFactory
    println("\n5. Testing BuilderFactory...")
    val b1 = torchrec.serving.BuilderFactory("vector", Map("metric" -> IndexMetric.L2))
    val b2 = torchrec.serving.BuilderFactory("annoy", Map("d" -> dim, "nTrees" -> 10))
    val b3 = torchrec.serving.BuilderFactory("faiss", Map("indexType" -> IndexType.HNSW))
    val b4 = torchrec.serving.BuilderFactory("milvus", Map("d" -> dim, "metric" -> IndexMetric.COSINE))
    println(s"  Vector builder: ${b1.getClass.getSimpleName}")
    println(s"  Annoy builder: ${b2.getClass.getSimpleName}")
    println(s"  Faiss builder: ${b3.getClass.getSimpleName}")
    println(s"  Milvus builder: ${b4.getClass.getSimpleName}")

    // Test AnnoyBuilder
    println("\n6. Testing AnnoyBuilder...")
    val annoyBuilder = new AnnoyBuilder(dim, IndexMetric.COSINE, nTrees = 5)
    val annoyIndexer = annoyBuilder.from_embeddings(embeddings)
    val idsAnnoy = toLongArray(annoyIndexer.query(queryVec, topK = 5)._1)
    println(s"  Top-5: ${idsAnnoy.take(5).mkString(", ")}")

    // Test MilvusBuilder
    println("\n7. Testing MilvusBuilder (FLAT)...")
    val milvusBuilder = new MilvusBuilder(dim, IndexType.FLAT, IndexMetric.COSINE)
    val milvusIndexer = milvusBuilder.from_embeddings(embeddings)
    val idsMilvus = toLongArray(milvusIndexer.query(queryVec, topK = 5)._1)
    println(s"  Top-5: ${idsMilvus.take(5).mkString(", ")}")

    println("\n" + "=" * 70)
    println("All serving tests passed!")
    println("=" * 70)
  }
}
