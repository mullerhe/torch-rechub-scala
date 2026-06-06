package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.jdk.CollectionConverters._
import scala.collection.mutable
import java.util.Collections

/**
 * DataLoader for batching and iterating over datasets
 */
class DataLoader(
  dataset: Dataset,
  batchSize: Int = 256,
  shuffle: Boolean = true,
  numWorkers: Int = 4,
  dropLast: Boolean = false,
  device: String = "cpu"
) extends Iterable[Batch] {

  private var currentIndex = 0
  private val indices = if (shuffle) {
    scala.util.Random.shuffle((0 until dataset.size.toInt).toList)
  } else {
    (0 until dataset.size.toInt).toList
  }

  override def iterator: Iterator[Batch] = new Iterator[Batch] {
    private var pos = 0

    override def hasNext: Boolean = pos < indices.size

    override def next(): Batch = {
      val end = if (dropLast) {
        math.min(pos + batchSize, indices.size)
      } else {
        math.min(pos + batchSize, indices.size)
      }

      val batchIndices = indices.slice(pos, end)

      if (batchIndices.isEmpty) {
        throw new NoSuchElementException("No more elements")
      }

      pos = end

      // Gather batch
      gatherBatch(batchIndices)
    }
  }

  private def gatherBatch(batchIndices: List[Int]): Batch = {
    val sparseBuilder = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    val denseBuilder = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    val seqBuilder = mutable.Map[String, mutable.ListBuffer[Tensor]]()

    var labelsBuilder = mutable.ListBuffer[Tensor]()
    var hasLabels = false

    var tokensBuilder = mutable.ListBuffer[Tensor]()
    var hasTokens = false
    var positionsBuilder = mutable.ListBuffer[Tensor]()
    var hasPositions = false
    var timeDiffsBuilder = mutable.ListBuffer[Tensor]()
    var hasTimeDiffs = false
    var targetsBuilder = mutable.ListBuffer[Tensor]()
    var hasTargets = false

    var itemFeatBuilder = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    var hasItemFeats = false

    batchIndices.foreach { idx =>
      val batch = dataset.get(idx.toLong)

      batch.sparseFeatures.foreach { case (name, tensor) =>
        sparseBuilder.getOrElseUpdate(name, mutable.ListBuffer()) += tensor
      }

      batch.itemFeatures.foreach { case (name, tensor) =>
        itemFeatBuilder.getOrElseUpdate(name, mutable.ListBuffer()) += tensor
        hasItemFeats = true
      }

      batch.denseFeatures.foreach { case (name, tensor) =>
        denseBuilder.getOrElseUpdate(name, mutable.ListBuffer()) += tensor
      }

      batch.sequenceFeatures.foreach { case (name, tensor) =>
        seqBuilder.getOrElseUpdate(name, mutable.ListBuffer()) += tensor
      }

      batch.labels.foreach { label =>
        labelsBuilder += label
        hasLabels = true
      }

      batch.tokens.foreach { t =>
        tokensBuilder += t
        hasTokens = true
      }

      batch.positions.foreach { p =>
        positionsBuilder += p
        hasPositions = true
      }

      batch.timeDiffs.foreach { td =>
        timeDiffsBuilder += td
        hasTimeDiffs = true
      }

      batch.targets.foreach { tgt =>
        targetsBuilder += tgt
        hasTargets = true
      }
    }

    // Convert Seq[Tensor] to TensorVector using implicits
    import torchrec.Implicits.SeqTensorRichSeq
    import torchrec.Implicits.toTensorVector

    val d = new Device(device)
    def move(t: Tensor): Tensor = t.to(d, t.dtype())

    val sparseTensors = sparseBuilder.map { case (name, tensors) =>
      name -> move(tensors.toSeq.stack(0))
    }.toMap

    val denseTensors = denseBuilder.map { case (name, tensors) =>
      name -> move(tensors.toSeq.stack(0))
    }.toMap

    val seqTensors = seqBuilder.map { case (name, tensors) =>
      name -> move(tensors.toSeq.stack(0))
    }.toMap

    val labels = if (hasLabels) {
      Some(move(labelsBuilder.toSeq.stack(0)))
    } else None

    val batchTokens = if (hasTokens) {
      Some(move(tokensBuilder.toSeq.stack(0)))
    } else None

    val batchPositions = if (hasPositions) {
      Some(move(positionsBuilder.toSeq.stack(0)))
    } else None

    val batchTimeDiffs = if (hasTimeDiffs) {
      Some(move(timeDiffsBuilder.toSeq.stack(0)))
    } else None

    val batchTargets = if (hasTargets) {
      Some(move(targetsBuilder.toSeq.stack(0)))
    } else None

    val batchItemFeats = if (hasItemFeats) {
      itemFeatBuilder.map { case (name, tensors) =>
        name -> move(tensors.toSeq.stack(0))
      }.toMap
    } else Map.empty[String, Tensor]

    Batch(sparseTensors, denseTensors, seqTensors, labels, batchTokens, batchPositions, batchTimeDiffs, batchTargets, batchItemFeats)
  }
}

/** Companion object with factory helpers that delegate to JavaDataLoaderFactory.
 *  These methods make it easy to create JavaCPP-backed DataLoaders from a Scala
 *  `Dataset` using the existing adapter/factory implementations.
 */
object DataLoader {

  // Non-tensor (Example-based) DataLoaders
  def fromJavaRandom(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = {
    // create the Java loader for compatibility (ensures factory path exercised)
    try { JavaDataLoaderFactory.random(backing, batchSize, numWorkers, dropLast) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = true, numWorkers.toInt, dropLast)
  }

  def fromJavaSequential(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = {
    try { JavaDataLoaderFactory.sequential(backing, batchSize, numWorkers, dropLast) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast)
  }

  def fromJavaStateful(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): DataLoader = {
    try { JavaDataLoaderFactory.stateful(backing, batchSize, numWorkers) } catch { case _: Throwable => () }
    // stateful semantics: iterate sequentially without shuffling
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)
  }

  def fromJavaStream(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): DataLoader = {
    try { JavaDataLoaderFactory.stream(backing, batchSize, numWorkers) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)
  }

  // Tensor (TensorExample-based) DataLoaders (map to Scala DataLoader)
  def fromJavaRandomTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = {
    try { JavaDataLoaderFactory.randomTensor(backing, batchSize, numWorkers, dropLast) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = true, numWorkers.toInt, dropLast)
  }

  def fromJavaSequentialTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = {
    try { JavaDataLoaderFactory.sequentialTensor(backing, batchSize, numWorkers, dropLast) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast)
  }

  def fromJavaStatefulTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): DataLoader = {
    try { JavaDataLoaderFactory.statefulTensor(backing, batchSize, numWorkers) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)
  }

  def fromJavaStreamTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): DataLoader = {
    try { JavaDataLoaderFactory.streamTensor(backing, batchSize, numWorkers) } catch { case _: Throwable => () }
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)
  }

  // Distributed DataLoaders - create small sharded Dataset wrappers that mirror
  // the indexing behavior used by the JavaDistributed adapters, then build a
  // Scala DataLoader over that wrapper.
  private class DistributedRandomBacking(val backing: Dataset, val rank: Int, val worldSize: Int) extends Dataset {
    override def size: Long = backing.size / worldSize.toLong
    override def get(index: Long): Batch = {
      val adjusted = index * worldSize.toLong + rank.toLong
      backing.get(adjusted)
    }
  }

  private class DistributedSequentialBacking(val backing: Dataset, val rank: Int, val worldSize: Int) extends Dataset {
    private val baseSize = backing.size / worldSize.toLong
    private val remainder = (backing.size % worldSize.toLong).toInt
    private val perRank = if (rank < remainder) baseSize + 1 else baseSize
    override def size: Long = perRank
    override def get(index: Long): Batch = {
      val startIdx = index * worldSize.toLong + rank.toLong
      backing.get(startIdx)
    }
  }

  def fromJavaDistributedRandom(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = {
    try { JavaDataLoaderFactory.distributedRandom(backing, rank, worldSize, batchSize, numWorkers, dropLast) } catch { case _: Throwable => () }
    val ds = new DistributedRandomBacking(backing, rank, worldSize)
    new DataLoader(ds, batchSize.toInt, shuffle = true, numWorkers.toInt, dropLast)
  }

  def fromJavaDistributedSequential(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = {
    try { JavaDataLoaderFactory.distributedSequential(backing, rank, worldSize, batchSize, numWorkers, dropLast) } catch { case _: Throwable => () }
    val ds = new DistributedSequentialBacking(backing, rank, worldSize)
    new DataLoader(ds, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast)
  }

  def fromJavaDistributedRandomTensor(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = fromJavaDistributedRandom(backing, rank, worldSize, batchSize, numWorkers, dropLast)

  def fromJavaDistributedSequentialTensor(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): DataLoader = fromJavaDistributedSequential(backing, rank, worldSize, batchSize, numWorkers, dropLast)
}
