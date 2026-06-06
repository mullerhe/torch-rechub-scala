package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.Implicits._

/**
 * JavaCPP Distributed DataLoader adapters for distributed training.
 */
object JavaDistributedAdapters {

  /** Default feature order: sorted keys for deterministic ordering */
  private def defaultFeatureOrder(features: Map[String, Tensor]): Seq[String] = features.keys.toSeq.sorted

  /** Pack a Batch into an Example */
  private def packBatchToExample(batch: Batch, order: Seq[String]): Example = {
    import org.bytedeco.pytorch.global.torch.ScalarType
    val vals = order.map { name =>
      val t = batch.sparseFeatures.getOrElse(name, null)
      if (t == null) 0.0f
      else {
        val arr = t.toFloatArray
        if (arr.nonEmpty) arr(0) else 0.0f
      }
    }.toArray
    val dataTensor = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
    val dataTensorLong = dataTensor.toType(ScalarType.Long)
    new Example(dataTensorLong)
  }

  /** Pack a Batch into a TensorExample */
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
    val dataTensor = torchrec.Implicits.tensor(vals, Array(vals.length.toLong))
    val dataTensorLong = dataTensor.toType(ScalarType.Long)
    new TensorExample(dataTensorLong)
  }

  /**
   * JavaDistributedRandomDatasetAdapter - distributed random sampling Dataset.
   * Note: This adapter extends JavaDataset so it can be used with JavaDistributedRandomDataLoader.
   */
  class JavaDistributedRandomDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String],
    val rank: Int,
    val worldSize: Int
  ) extends JavaDataset {

    def this(backing: Dataset, rank: Int, worldSize: Int) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    }, rank, worldSize)

    override def size(): SizeTOptional = {
      val perRank = backing.size / worldSize.toLong
      new SizeTOptional(perRank)
    }

    override def get_batch(indices: SizeTArrayRef): ExampleVector = {
      val vec = new ExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val idx = indices.get(i) * worldSize.toLong + rank.toLong
        val b = try {
          backing.get(idx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val ex = packBatchToExample(b, featureOrder)
        vec.push_back(ex)
        i += 1
      }
      vec
    }

    override def get(index: Long): Example = {
      val adjustedIndex = index * worldSize.toLong + rank.toLong
      val b = backing.get(adjustedIndex)
      packBatchToExample(b, featureOrder)
    }
  }

  /**
   * JavaDistributedSequentialDatasetAdapter - distributed sequential sampling Dataset.
   */
  class JavaDistributedSequentialDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String],
    val rank: Int,
    val worldSize: Int
  ) extends JavaDataset {

    def this(backing: Dataset, rank: Int, worldSize: Int) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    }, rank, worldSize)

    override def size(): SizeTOptional = {
      val baseSize = backing.size / worldSize.toLong
      val remainder = backing.size % worldSize.toLong
      val perRank = if (rank < remainder) baseSize + 1 else baseSize
      new SizeTOptional(perRank)
    }

    override def get_batch(indices: SizeTArrayRef): ExampleVector = {
      val vec = new ExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val startIdx = i * worldSize.toLong + rank.toLong
        val b = try {
          backing.get(startIdx)
        } catch {
          case _: Throwable => backing.get(0)
        }
        val ex = packBatchToExample(b, featureOrder)
        vec.push_back(ex)
        i += 1
      }
      vec
    }

    override def get(index: Long): Example = {
      val startIdx = index * worldSize.toLong + rank.toLong
      val b = backing.get(startIdx)
      packBatchToExample(b, featureOrder)
    }
  }

  /**
   * JavaDistributedRandomTensorDatasetAdapter - distributed random sampling Tensor Dataset.
   */
  class JavaDistributedRandomTensorDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String],
    val rank: Int,
    val worldSize: Int
  ) extends JavaTensorDataset {

    def this(backing: Dataset, rank: Int, worldSize: Int) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    }, rank, worldSize)

    override def size(): SizeTOptional = {
      val perRank = backing.size / worldSize.toLong
      new SizeTOptional(perRank)
    }

    override def get_batch(indices: SizeTArrayRef): TensorExampleVector = {
      val vec = new TensorExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val idx = indices.get(i) * worldSize.toLong + rank.toLong
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
      val adjustedIndex = index * worldSize.toLong + rank.toLong
      val b = backing.get(adjustedIndex)
      packBatchToTensorExample(b, featureOrder)
    }
  }

  /**
   * JavaDistributedSequentialTensorDatasetAdapter - distributed sequential sampling Tensor Dataset.
   */
  class JavaDistributedSequentialTensorDatasetAdapter(
    val backing: Dataset,
    val featureOrder: Seq[String],
    val rank: Int,
    val worldSize: Int
  ) extends JavaTensorDataset {

    def this(backing: Dataset, rank: Int, worldSize: Int) = this(backing, backing match {
      case td: torchrec.data.TensorDataset => defaultFeatureOrder(td.sparseFeatures)
      case md: torchrec.data.MatchingDataset => defaultFeatureOrder(md.userFeatures)
      case sd: torchrec.data.SequenceDataset => defaultFeatureOrder(sd.features)
      case _ => Seq.empty[String]
    }, rank, worldSize)

    override def size(): SizeTOptional = {
      val baseSize = backing.size / worldSize.toLong
      val remainder = backing.size % worldSize.toLong
      val perRank = if (rank < remainder) baseSize + 1 else baseSize
      new SizeTOptional(perRank)
    }

    override def get_batch(indices: SizeTArrayRef): TensorExampleVector = {
      val vec = new TensorExampleVector()
      val len = indices.size().toInt
      var i = 0
      while (i < len) {
        val startIdx = i * worldSize.toLong + rank.toLong
        val b = try {
          backing.get(startIdx)
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
      val startIdx = index * worldSize.toLong + rank.toLong
      val b = backing.get(startIdx)
      packBatchToTensorExample(b, featureOrder)
    }
  }

  // ============= Factory Methods =============

  /** Create a Java Distributed Random DataLoader */
  def createDistributedRandomDataLoader(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedRandomDataLoader = {
    val adapter = new JavaDistributedRandomDatasetAdapter(backing, rank, worldSize)
    val sampler = new DistributedRandomSampler(backing.size, worldSize.toLong, rank.toLong, false)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaDistributedRandomDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Distributed Sequential DataLoader */
  def createDistributedSequentialDataLoader(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedSequentialDataLoader = {
    val adapter = new JavaDistributedSequentialDatasetAdapter(backing, rank, worldSize)
    val sampler = new DistributedSequentialSampler(backing.size, worldSize.toLong, rank.toLong, false)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaDistributedSequentialDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Distributed Random Tensor DataLoader */
  def createDistributedRandomTensorDataLoader(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedRandomTensorDataLoader = {
    val adapter = new JavaDistributedRandomTensorDatasetAdapter(backing, rank, worldSize)
    val sampler = new DistributedRandomSampler(backing.size, worldSize.toLong, rank.toLong, false)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaDistributedRandomTensorDataLoader(adapter, sampler, opts)
  }

  /** Create a Java Distributed Sequential Tensor DataLoader */
  def createDistributedSequentialTensorDataLoader(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedSequentialTensorDataLoader = {
    val adapter = new JavaDistributedSequentialTensorDatasetAdapter(backing, rank, worldSize)
    val sampler = new DistributedSequentialSampler(backing.size, worldSize.toLong, rank.toLong, false)
    val opts = DataLoaderOptionsBuilder.build(batchSize, numWorkers, dropLast)
    new JavaDistributedSequentialTensorDataLoader(adapter, sampler, opts)
  }
}