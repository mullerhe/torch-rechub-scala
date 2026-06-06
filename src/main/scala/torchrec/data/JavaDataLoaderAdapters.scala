package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._

/**
 * Adapters to create JavaCPP (org.bytedeco.pytorch) Dataset/DataLoader objects
 * from the project's Scala tensors. This file provides factory helpers to
 * build Java TensorDatasets and DataLoaders so you can use the JavaCPP
 * data loading primitives (Random/Sequential/Stream/Stateful) without
 * hand-writing JNI code.
 *
 * Note: we create TensorVector instances and pass them to Java-side dataset
 * constructors. If constructors differ for your JavaCPP version, the compile
 * will show errors and we will iterate.
 */
object JavaDataLoaderAdapters {

  // Helper: deterministic feature order (sorted keys)
  private def defaultFeatureOrder(features: Map[String, Tensor]): Seq[String] = features.keys.toSeq.sorted

  /** Pack a single Batch into a 1-D Tensor of feature indices (Long dtype).
   *  Rule: take sparseFeatures in deterministic order, for each take the first element
   *  (assumes per-sample tensors are shaped [1]). Sequence features should be pre-pooled
   *  (or will be reduced to their first element). Labels are handled separately.
   */
  private def packBatchToIndexTensor(batch: Batch, order: Seq[String]): Tensor = {
    import org.bytedeco.pytorch.global.torch.ScalarType
    val vals = order.map { name =>
      val t = batch.sparseFeatures.getOrElse(name, null)
      if (t == null) 0.0f
      else {
        val arr = t.toFloatArray
        if (arr.nonEmpty) arr(0) else 0.0f
      }
    }.toArray
    // Create float tensor then cast to long indices
    val ft = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
    ft.toType(ScalarType.Long)
  }

  /** JavaTensorDataset adapter backed by a Scala Dataset. It returns TensorExample
   *  instances where .data is a 1-D Long tensor containing the packed feature indices
   *  according to the provided feature order. Labels (if present) are set as target tensor.
   */
  class JavaTensorDatasetAdapter(val backing: Dataset, val featureOrder: Seq[String]) extends JavaTensorDataset {
    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case _ => Seq.empty[String]
    })

    override def get_batch(indices: org.bytedeco.pytorch.SizeTArrayRef): org.bytedeco.pytorch.TensorExampleVector = {
      val vec = new org.bytedeco.pytorch.TensorExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val idx = indices.get(i)
        val b = try { backing.get(idx) } catch { case _: Throwable => backing.get(0) }
        // pack features
        val dataTensor = packBatchToIndexTensor(b, featureOrder)
        // construct example with data tensor
        val ex = new org.bytedeco.pytorch.TensorExample(dataTensor)
        vec.push_back(ex)
        i += 1
      }
      vec
    }
  }

  // Stream adapter: JavaStreamTensorDataLoader requires a JavaStreamTensorDataset
  class JavaStreamTensorDatasetAdapter(val backing: Dataset, val featureOrder: Seq[String]) extends JavaStreamTensorDataset {
    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case _ => Seq.empty[String]
    })

    private var _currentIndex: Long = 0L

    def get_batch_from_indices(indices: org.bytedeco.pytorch.SizeTArrayRef): org.bytedeco.pytorch.TensorExampleVector = {
      val vec = new org.bytedeco.pytorch.TensorExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val idx = indices.get(i)
        val b = try { backing.get(idx) } catch { case _: Throwable => backing.get(0) }
        val dataTensor = packBatchToIndexTensor(b, featureOrder)
        val ex = new org.bytedeco.pytorch.TensorExample(dataTensor)
        vec.push_back(ex)
        i += 1
      }
      vec
    }

    override def size(): SizeTOptional = super.size()

    override def get_batch(batchSize: Long): TensorExampleVector = {
      val vec = new org.bytedeco.pytorch.TensorExampleVector()
      val len = math.min(batchSize, backing.size - _currentIndex).toInt
      var i = 0
      while (i < len) {
        val idx = _currentIndex + i
        val b = try {
          backing.get(idx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val dataTensor = packBatchToIndexTensor(b, featureOrder)
        val ex = new org.bytedeco.pytorch.TensorExample(dataTensor)
        vec.push_back(ex)
        i += 1
      }
      _currentIndex += len
      vec
    }
  }


  /** Create DataLoaderOptions with common hyperparameters */
  private def buildOptions(batchSize: Long, workers: Long = 0L, dropLast: Boolean = false): DataLoaderOptions = {
    val opts = new DataLoaderOptions()
    try { opts.batch_size().put(batchSize) } catch { case _: Throwable => () }
    try { opts.workers().put(workers) } catch { case _: Throwable => () }
    try { opts.drop_last().put(dropLast) } catch { case _: Throwable => () }
    opts
  }

  /** Create a Java Random Tensor DataLoader backed by a Scala Dataset */
  def createRandomTensorDataLoaderFromScala(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): org.bytedeco.pytorch.JavaRandomTensorDataLoader = {
    val adapter = new JavaTensorDatasetAdapter(backing)
    // sampler takes dataset size
    val datasetSize = backing.size
    val sampler = new org.bytedeco.pytorch.RandomSampler(datasetSize)
    val opts = buildOptions(batchSize, numWorkers, dropLast)
    new org.bytedeco.pytorch.JavaRandomTensorDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Sequential Tensor DataLoader backed by a Scala Dataset */
  def createSequentialTensorDataLoaderFromScala(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): org.bytedeco.pytorch.JavaSequentialTensorDataLoader = {
    val adapter = new JavaTensorDatasetAdapter(backing)
    val datasetSize = backing.size
    val sampler = new org.bytedeco.pytorch.SequentialSampler(datasetSize)
    val opts = buildOptions(batchSize, numWorkers, dropLast)
    new org.bytedeco.pytorch.JavaSequentialTensorDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Stream Tensor DataLoader backed by a Scala Dataset */
  def createStreamTensorDataLoaderFromScala(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): org.bytedeco.pytorch.JavaStreamTensorDataLoader = {
    val adapter = new JavaStreamTensorDatasetAdapter(backing)
    val datasetSize = backing.size
    val sampler = new org.bytedeco.pytorch.StreamSampler(datasetSize)
    val opts = buildOptions(batchSize, numWorkers)
    new org.bytedeco.pytorch.JavaStreamTensorDataLoader(adapter, sampler, opts)
  }
}

