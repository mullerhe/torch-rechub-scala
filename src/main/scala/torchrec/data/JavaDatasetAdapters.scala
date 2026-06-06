package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.Implicits._
import scala.collection.mutable

/**
 * JavaCPP Dataset adapters for non-Tensor Example-based datasets.
 * Provides implementations for:
 * - JavaDataset (generic Example)
 * - JavaStatefulDataset (with state)
 * - JavaStreamDataset (streaming)
 */
object JavaDatasetAdapters {

  /** Default feature order: sorted keys for deterministic ordering */
  private def defaultFeatureOrder(features: Map[String, Tensor]): Seq[String] = features.keys.toSeq.sorted

  /**
   * JavaDataset adapter - wraps a Scala Dataset to provide Example objects.
   */
  class JavaDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String]
  ) extends JavaDataset {

    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    })

    override def size(): SizeTOptional = new SizeTOptional(backing.size)

    override def get_batch(indices: SizeTArrayRef): ExampleVector = {
      val vec = new ExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val idx = indices.get(i)
        val b = try {
          backing.get(idx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val dataTensor = packBatchToTensor(b, featureOrder)
        vec.push_back(new Example(dataTensor))
        i += 1
      }
      vec
    }

    override def get(index: Long): Example = {
      val b = backing.get(index)
      val dataTensor = packBatchToTensor(b, featureOrder)
      new Example(dataTensor)
    }

    private def packBatchToTensor(batch: Batch, order: Seq[String]): Tensor = {
      import org.bytedeco.pytorch.global.torch.ScalarType
      val vals = order.flatMap { name =>
        batch.sparseFeatures.get(name).map { t =>
          val arr = t.toFloatArray
          if (arr.nonEmpty) arr else Array(0.0f)
        }.getOrElse(Array(0.0f))
      }.toArray
      if (vals.isEmpty) {
        torch.zeros(Array(1L)*)
      } else {
        val ft = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
        ft.toType(ScalarType.Long)
      }
    }
  }

  /**
   * JavaStatefulDataset adapter - maintains state across batches.
   * Note: StatefulDataset.get_batch takes batch size and returns ExampleVectorOptional.
   */
  class JavaStatefulDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String],
    private val _state: mutable.Map[String, Any] = mutable.Map.empty
  ) extends JavaStatefulDataset {

    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    }, mutable.Map.empty)

    private var _currentIndex: Long = 0L

    override def size(): SizeTOptional = new SizeTOptional(backing.size)

    def state: mutable.Map[String, Any] = _state

    // StatefulDataset.get_batch returns ExampleVectorOptional and takes batch size
    override def get_batch(batchSize: Long): ExampleVectorOptional = {
      val vec = new ExampleVector()
      val maxIndex = math.min(_currentIndex + batchSize, backing.size)
      var i = _currentIndex
      while (i < maxIndex) {
        val b = try {
          backing.get(i)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val dataTensor = packBatchToTensor(b, featureOrder)
        vec.push_back(new Example(dataTensor))
        i += 1
      }
      _currentIndex = i
      new ExampleVectorOptional(vec)
    }

    override def reset(): Unit = {
      _currentIndex = 0L
    }

    private def packBatchToTensor(batch: Batch, order: Seq[String]): Tensor = {
      import org.bytedeco.pytorch.global.torch.ScalarType
      val vals = order.flatMap { name =>
        batch.sparseFeatures.get(name).map { t =>
          val arr = t.toFloatArray
          if (arr.nonEmpty) arr else Array(0.0f)
        }.getOrElse(Array(0.0f))
      }.toArray
      if (vals.isEmpty) {
        torch.zeros(Array(1L)*)
      } else {
        val ft = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
        ft.toType(ScalarType.Long)
      }
    }
  }

  /**
   * JavaStreamDataset adapter - for streaming data sources.
   * Note: StreamDataset.get_batch takes batch size.
   */
  class JavaStreamDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String]
  ) extends JavaStreamDataset {

    def this(backing: Dataset) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    })

    private var _currentIndex: Long = 0L

    override def size(): SizeTOptional = new SizeTOptional(backing.size)

    // StreamDataset.get_batch takes batch size
    override def get_batch(batchSize: Long): ExampleVector = {
      val vec = new ExampleVector()
      val len = math.min(batchSize, backing.size - _currentIndex).toInt
      var i = 0
      while (i < len) {
        val idx = _currentIndex + i
        val b = try {
          backing.get(idx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val dataTensor = packBatchToTensor(b, featureOrder)
        vec.push_back(new Example(dataTensor))
        i += 1
      }
      _currentIndex += len
      vec
    }

    private def packBatchToTensor(batch: Batch, order: Seq[String]): Tensor = {
      import org.bytedeco.pytorch.global.torch.ScalarType
      val vals = order.flatMap { name =>
        batch.sparseFeatures.get(name).map { t =>
          val arr = t.toFloatArray
          if (arr.nonEmpty) arr else Array(0.0f)
        }.getOrElse(Array(0.0f))
      }.toArray
      if (vals.isEmpty) {
        torch.zeros(Array(1L)*)
      } else {
        val ft = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
        ft.toType(ScalarType.Long)
      }
    }
  }

  // ============= Factory Methods =============

  /** Create a Java Dataset adapter from a Scala Dataset */
  def createDatasetAdapter(backing: Dataset): JavaDatasetAdapter = {
    new JavaDatasetAdapter(backing)
  }

  /** Create a Java Stateful Dataset adapter from a Scala Dataset */
  def createStatefulDatasetAdapter(backing: Dataset): JavaStatefulDatasetAdapter = {
    new JavaStatefulDatasetAdapter(backing)
  }

  /** Create a Java Stream Dataset adapter from a Scala Dataset */
  def createStreamDatasetAdapter(backing: Dataset): JavaStreamDatasetAdapter = {
    new JavaStreamDatasetAdapter(backing)
  }

  /** Create a Java Random DataLoader from a Scala Dataset */
  def createRandomDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaRandomDataLoader = {
    val adapter = new JavaDatasetAdapter(backing)
    val sampler = new RandomSampler(backing.size)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaRandomDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Sequential DataLoader from a Scala Dataset */
  def createSequentialDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaSequentialDataLoader = {
    val adapter = new JavaDatasetAdapter(backing)
    val sampler = new SequentialSampler(backing.size)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaSequentialDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Stateful DataLoader from a Scala Dataset */
  def createStatefulDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStatefulDataLoader = {
    val adapter = new JavaStatefulDatasetAdapter(backing)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers)
    new JavaStatefulDataLoader(adapter, opts)
  }

  /** Create a Java Stream DataLoader from a Scala Dataset */
  def createStreamDataLoader(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStreamDataLoader = {
    val adapter = new JavaStreamDatasetAdapter(backing)
    val sampler = new StreamSampler(backing.size)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers)
    new JavaStreamDataLoader(adapter, sampler, opts)
  }
}

/**
 * DataLoaderOptions builder with supported hyperparameters.
 */
object DataLoaderOptionsBuilder {

  def build(
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false,
    enforce_ordering: Boolean = true,
    max_jobs: Option[Long] = None
           
  ): DataLoaderOptions = {
    val opts = new DataLoaderOptions()

    try { opts.batch_size().put(batchSize) } catch { case _: Throwable => () }
    try { opts.workers().put(numWorkers) } catch { case _: Throwable => () }
    try { opts.drop_last().put(dropLast) } catch { case _: Throwable => () }
    opts.enforce_ordering().put(enforce_ordering)
    if (max_jobs.isDefined) {
      opts.max_jobs().put(max_jobs.get) }
    opts
  }

  def buildFull(
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false,
    enforce_ordering: Boolean = true,
    max_jobs: Option[Long] = None
  ): FullDataLoaderOptions = {
    val opt = new DataLoaderOptions()
    try {
      opt.batch_size().put(batchSize)
    } catch {
      case _: Throwable => ()
    }
    try {
      opt.workers().put(numWorkers)
    } catch {
      case _: Throwable => ()
    }
    try {
      opt.drop_last().put(dropLast)
    } catch {
      case _: Throwable => ()
    }
    opt.enforce_ordering().put(enforce_ordering)
    if (max_jobs.isDefined) {
        opt.max_jobs().put(max_jobs.get) }
    val opts = new FullDataLoaderOptions(opt)
    opts
  }
}