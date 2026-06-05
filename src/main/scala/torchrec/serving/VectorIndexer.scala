package torchrec.serving

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import org.bytedeco.javacpp.{FloatPointer, LongPointer}
import org.bytedeco.javacpp.indexer.{FloatArrayIndexer, LongArrayIndexer}

import torchrec.Implicits.SeqTensorRichSeq
import torchrec.Implicits.RichTensor

/**
 * Base trait for vector indexers used in the retrieval stage.
 */
trait BaseIndexer {
  /**
   * Query the vector index for nearest neighbors.
   * @param embeddings Query embeddings of shape (n, d)
   * @param topK Number of nearest neighbors to retrieve
   * @return Tuple of (indices: (n, topK), distances: (n, topK))
   */
  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor)
}

/**
 * Base trait for vector index builders.
 */
trait BaseBuilder {
  /**
   * Build an index from embeddings.
   * @param embeddings Embedding tensor of shape (n, d)
   * @return A BaseIndexer
   */
  def from_embeddings(embeddings: Tensor): BaseIndexer

  /**
   * Load an index from a file.
   * @param filePath Path to the index file
   * @return A BaseIndexer
   */
  def from_index_file(filePath: String): BaseIndexer
}

/**
 * Metric type for similarity search.
 */
enum IndexMetric:
  case L2, IP, COSINE

/**
 * In-memory brute-force vector indexer.
 * Supports L2 distance and inner product (cosine-like) metrics.
 */
class VectorIndexer(
  embeddings: Tensor,
  metric: IndexMetric = IndexMetric.L2
) extends BaseIndexer {

  private val index: Tensor = embeddings.contiguous()
  private val n = index.size(0).toInt
  private val dim = index.size(1).toInt

  def query(embeddings: Tensor, topK: Int): (Tensor, Tensor) = {
    val n = embeddings.size(0).toInt
    val indicesArr = new Array[Long](n * topK)
    val distsArr = new Array[Float](n * topK)

    var i = 0
    while (i < n) {
      val queryVec = embeddings.select(0, i)

      // Collect all distances
      val dists = new Array[Float](this.n)
      var j = 0
      while (j < this.n) {
        val storedVec = index.select(0, j)
        dists(j) = computeDistance(queryVec, storedVec)
        j += 1
      }

      // Find top-K indices
      val topKIndices = argsortTopK(dists, topK)
      System.arraycopy(topKIndices, 0, indicesArr, i * topK, topK)

      j = 0
      while (j < topK) {
        distsArr(i * topK + j) = dists(topKIndices(j).toInt)
        j += 1
      }
      i += 1
    }

    // Build tensors row by row
    val indicesList = scala.collection.mutable.ListBuffer[Tensor]()
    val distsList = scala.collection.mutable.ListBuffer[Tensor]()

    val longOpts = new TensorOptions()
    longOpts.dtype().put(ScalarType.Long)
    val floatOpts = new TensorOptions()
    floatOpts.dtype().put(ScalarType.Float)

    var r = 0
    while (r < n) {
      val rowIndices = indicesArr.slice(r * topK, (r + 1) * topK)
      val rowDists = distsArr.slice(r * topK, (r + 1) * topK)

      // Create indices tensor for this row
      val rowIdxPtr = new LongPointer()
      var c = 0
      while (c < topK) { rowIdxPtr.put(c, rowIndices(c)); c += 1 }
      val rowIdxSizes = Array(topK.toLong)
      indicesList += torch.from_blob(rowIdxPtr, rowIdxSizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], longOpts).clone()

      // Create dists tensor for this row
      val rowDstPtr = new FloatPointer()
      c = 0
      while (c < topK) { rowDstPtr.put(c, rowDists(c)); c += 1 }
      val rowDstSizes = Array(topK.toLong)
      distsList += torch.from_blob(rowDstPtr, rowDstSizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], floatOpts).clone()

      r += 1
    }

    val indicesTensor = torch.cat(indicesList.toSeq.toTensorVector, 0).reshape(n.toLong, topK.toLong)
    val distsTensor = torch.cat(distsList.toSeq.toTensorVector, 0).reshape(n.toLong, topK.toLong)

    (indicesTensor, distsTensor)
  }

  private def computeDistance(a: Tensor, b: Tensor): Float = {
    metric match {
      case IndexMetric.L2 =>
        var diff = 0.0f
        var d = 0
        while (d < dim) {
          val delta = a.select(0, d).item().toFloat - b.select(0, d).item().toFloat
          diff += delta * delta
          d += 1
        }
        diff
      case IndexMetric.IP =>
        var dot = 0.0f
        var d = 0
        while (d < dim) {
          dot += a.select(0, d).item().toFloat * b.select(0, d).item().toFloat
          d += 1
        }
        -dot  // Negate so higher similarity = lower distance
      case IndexMetric.COSINE =>
        var dot = 0.0f
        var normA = 0.0f
        var normB = 0.0f
        var d = 0
        while (d < dim) {
          val va = a.select(0, d).item().toFloat
          val vb = b.select(0, d).item().toFloat
          dot += va * vb
          normA += va * va
          normB += vb * vb
          d += 1
        }
        val normProd = math.sqrt(normA.toDouble * normB.toDouble).toFloat
        if (normProd < 1e-8f) 0.0f
        else -(dot / normProd)  // Negate for consistency
    }
  }

  private def argsortTopK(dists: Array[Float], k: Int): Array[Long] = {
    val sorted = dists.zipWithIndex.sortBy(_._1)
    sorted.take(k).map(_._2.toLong).toArray
  }

  def save(filePath: String): Unit = {
    // Note: torch.save is not available in JavaCPP bindings
    // User should manually save the underlying tensor if needed
    throw new UnsupportedOperationException("torch.save is not available in JavaCPP bindings")
  }
}

/**
 * Builder for in-memory vector index.
 */
class VectorIndexerBuilder(
  metric: IndexMetric = IndexMetric.L2
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    new VectorIndexer(embeddings, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}

/**
 * Index type for FAISS-style indexing.
 */
enum IndexType:
  case FLAT, HNSW, IVF

/**
 * FAISS-style builder for vector indexing.
 * Note: This is a pure Scala implementation that uses brute-force search
 * but follows the same API pattern as FAISS for compatibility.
 */
class FaissBuilder(
  indexType: IndexType = IndexType.FLAT,
  metric: IndexMetric = IndexMetric.L2,
  m: Int = 32,
  nlist: Int = 100,
  efSearch: Option[Int] = None,
  nprobe: Option[Int] = None
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    new VectorIndexer(embeddings, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}

/**
 * Annoy-style builder for vector indexing.
 * Note: This is a pure Scala implementation that uses brute-force search
 * but follows the same API pattern as Annoy for compatibility.
 */
class AnnoyBuilder(
  d: Int,
  metric: IndexMetric = IndexMetric.L2,
  nTrees: Int = 10,
  searchK: Int = -1
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    new VectorIndexer(embeddings, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}

/**
 * Milvus-style builder for vector indexing.
 * Note: This is a pure Scala implementation that uses brute-force search
 * but follows the same API pattern as Milvus for compatibility.
 */
class MilvusBuilder(
  d: Int,
  indexType: IndexType = IndexType.FLAT,
  metric: IndexMetric = IndexMetric.COSINE,
  m: Int = 30,
  nlist: Int = 128,
  ef: Option[Int] = None,
  nprobe: Option[Int] = None
) extends BaseBuilder {

  def from_embeddings(embeddings: Tensor): BaseIndexer = {
    new VectorIndexer(embeddings, metric)
  }

  def from_index_file(filePath: String): BaseIndexer = {
    val loaded = torch.load(filePath)
    new VectorIndexer(loaded.asInstanceOf[Tensor], metric)
  }
}
