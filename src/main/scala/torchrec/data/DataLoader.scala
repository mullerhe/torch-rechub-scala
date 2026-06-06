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

/**
 * Random sampler
 */
class RandomSampler(datasetSize: Long) extends Sampler {
  override def sample(): Iterator[Long] = {
    scala.util.Random.shuffle((0 until datasetSize.toInt).toList).iterator.map(_.toLong)
  }
}

/**
 * Sequential sampler
 */
class SequentialSampler(datasetSize: Long) extends Sampler {
  override def sample(): Iterator[Long] = {
    (0 until datasetSize.toInt).iterator.map(_.toLong)
  }
}

/**
 * Base trait for samplers
 */
trait Sampler {
  def sample(): Iterator[Long]
}

/**
 * BatchSampler
 */
class BatchSampler(
  sampler: Sampler,
  batchSize: Int,
  dropLast: Boolean = false
) extends org.bytedeco.pytorch.Module {
  def sampleBatches(): Iterator[List[Long]] = {
    val indices = sampler.sample().toList
    indices.grouped(batchSize).filter(_.size == batchSize || !dropLast).map(_.toList)
  }
}