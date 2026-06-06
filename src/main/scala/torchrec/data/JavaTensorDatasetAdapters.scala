package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.Implicits._
import scala.collection.mutable

/**
 * JavaCPP Tensor Dataset adapters for TensorExample-based datasets.
 */
object JavaTensorDatasetAdapters {

  /** Default feature order: sorted keys for deterministic ordering */
  private def defaultFeatureOrder(features: Map[String, Tensor]): Seq[String] = features.keys.toSeq.sorted

  /** Pack a Batch into a TensorExample (1-D Long tensor of feature indices) */
  private def packBatchToTensorExample(batch: Batch, order: Seq[String]): TensorExample = {
    import org.bytedeco.pytorch.global.torch.ScalarType
    val vals = order.map { name =>
      val t = batch.sparseFeatures.getOrElse(name, null)
      if (t == null) 0.0f
      else {
        val arr = t.toFloatArray
        if (arr.nonEmpty) arr(0) else 0.0f
      }
    }.toArray
    val ft = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
    val dataTensor = ft.toType(ScalarType.Long)
    new TensorExample(dataTensor)
  }

  /**
   * JavaTensorDataset adapter - wraps a Scala Dataset to provide TensorExample objects.
   */
  class JavaTensorDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String]
  ) extends JavaTensorDataset {

    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    })

    override def size(): SizeTOptional = new SizeTOptional(backing.size)

    override def get_batch(indices: SizeTArrayRef): TensorExampleVector = {
      val vec = new TensorExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val idx = indices.get(i)
        val b = try {
          backing.get(idx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val ex = packBatchToTensorExample(b, featureOrder)
        vec.push_back(ex)
        i += 1
      }
      vec
    }

    override def get(index: Long): TensorExample = {
      val b = backing.get(index)
      packBatchToTensorExample(b, featureOrder)
    }
  }

  /**
   * JavaStatefulTensorDataset adapter - maintains state across batches.
   * Note: StatefulTensorDataset.get_batch takes batch size and returns TensorExampleVectorOptional.
   */
  class JavaStatefulTensorDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String],
    private val _state: mutable.Map[String, Any] = mutable.Map.empty
  ) extends JavaStatefulTensorDataset {

    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    }, mutable.Map.empty)

    private var _currentIndex: Long = 0L

    override def size(): SizeTOptional = new SizeTOptional(backing.size)

    def state: mutable.Map[String, Any] = _state

    // StatefulTensorDataset.get_batch returns TensorExampleVectorOptional and takes batch size
    override def get_batch(batchSize: Long): TensorExampleVectorOptional = {
      val vec = new TensorExampleVector()
      val maxIndex = math.min(_currentIndex + batchSize, backing.size)
      var i = _currentIndex
      while (i < maxIndex) {
        val b = try {
          backing.get(i)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val ex = packBatchToTensorExample(b, featureOrder)
        vec.push_back(ex)
        i += 1
      }
      _currentIndex = i
      new TensorExampleVectorOptional(vec)
    }
  }

  /**
   * JavaStreamTensorDataset adapter - for streaming data sources.
   */
  class JavaStreamTensorDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String]
  ) extends JavaStreamTensorDataset {

    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    })

    private var _currentIndex: Long = 0L

    override def size(): SizeTOptional = new SizeTOptional(backing.size)

    // StreamTensorDataset.get_batch takes batch size
    override def get_batch(batchSize: Long): TensorExampleVector = {
      val vec = new TensorExampleVector()
      val len = math.min(batchSize, backing.size - _currentIndex).toInt
      var i = 0
      while (i < len) {
        val idx = _currentIndex + i
        val b = try {
          backing.get(idx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val ex = packBatchToTensorExample(b, featureOrder)
        vec.push_back(ex)
        i += 1
      }
      _currentIndex += len
      vec
    }

    def reset(): Unit = {
      _currentIndex = 0L
    }
  }

  // ============= Factory Methods =============

  /** Create a Java Tensor Dataset adapter from a Scala Dataset */
  def createTensorDatasetAdapter(backing: Dataset): JavaTensorDatasetAdapter = {
    new JavaTensorDatasetAdapter(backing)
  }

  /** Create a Java Stateful Tensor Dataset adapter from a Scala Dataset */
  def createStatefulTensorDatasetAdapter(backing: Dataset): JavaStatefulTensorDatasetAdapter = {
    new JavaStatefulTensorDatasetAdapter(backing)
  }

  /** Create a Java Stream Tensor Dataset adapter from a Scala Dataset */
  def createStreamTensorDatasetAdapter(backing: Dataset): JavaStreamTensorDatasetAdapter = {
    new JavaStreamTensorDatasetAdapter(backing)
  }

  /** Create a Java Random Tensor DataLoader from a Scala Dataset */
  def createRandomTensorDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaRandomTensorDataLoader = {
    val adapter = new JavaTensorDatasetAdapter(backing)
    val sampler = new RandomSampler(backing.size)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaRandomTensorDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Sequential Tensor DataLoader from a Scala Dataset */
  def createSequentialTensorDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaSequentialTensorDataLoader = {
    val adapter = new JavaTensorDatasetAdapter(backing)
    val sampler = new SequentialSampler(backing.size)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaSequentialTensorDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Stateful Tensor DataLoader from a Scala Dataset */
  def createStatefulTensorDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStatefulTensorDataLoader = {
    val adapter = new JavaStatefulTensorDatasetAdapter(backing)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers)
    new JavaStatefulTensorDataLoader(adapter, opts)
  }

  /** Create a Java Stream Tensor DataLoader from a Scala Dataset */
  def createStreamTensorDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStreamTensorDataLoader = {
    val adapter = new JavaStreamTensorDatasetAdapter(backing)
    val sampler = new StreamSampler(backing.size)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers)
    new JavaStreamTensorDataLoader(adapter, sampler, opts)
  }
}