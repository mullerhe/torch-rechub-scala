package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.jdk.CollectionConverters._
import scala.collection.mutable

/**
 * Base trait for datasets
 */
trait Dataset {
  def size: Long
  def get(index: Long): Batch
}

/**
 * Batch containing features and labels
 */
case class Batch(
  sparseFeatures: Map[String, Tensor],
  denseFeatures: Map[String, Tensor] = Map.empty,
  sequenceFeatures: Map[String, Tensor] = Map.empty,
  labels: Option[Tensor] = None,
  // Sequence model fields
  tokens: Option[Tensor] = None,
  positions: Option[Tensor] = None,
  timeDiffs: Option[Tensor] = None,
  targets: Option[Tensor] = None,
  // Two-tower matching fields
  itemFeatures: Map[String, Tensor] = Map.empty
) {
  def to(device: String): Batch = {
    val d = new Device(device)
    def move(t: Tensor): Tensor = t.to(d, t.dtype())
    Batch(
      sparseFeatures.map { case (k, v) => k -> move(v) },
      denseFeatures.map { case (k, v) => k -> move(v) },
      sequenceFeatures.map { case (k, v) => k -> move(v) },
      labels.map(move),
      tokens.map(move),
      positions.map(move),
      timeDiffs.map(move),
      targets.map(move),
      itemFeatures.map { case (k, v) => k -> move(v) }
    )
  }

  def numSamples: Long = {
    sparseFeatures.headOption.map(_._2.size(0))
      .orElse(denseFeatures.headOption.map(_._2.size(0)))
      .orElse(sequenceFeatures.headOption.map(_._2.size(0)))
      .orElse(tokens.map(_.size(0)))
      .getOrElse(0)
  }
}

/**
 * TensorDataset - wraps in-memory tensors
 */
class TensorDataset(
  val sparseFeatures: Map[String, Tensor],
  val denseFeatures: Map[String, Tensor] = Map.empty,
  val labels: Option[Tensor] = None
) extends Dataset {

  override def size: Long = {
    sparseFeatures.headOption.map(_._2.size(0)).getOrElse(0)
  }

  override def get(index: Long): Batch = {
    // Ensure index is within bounds
    val safeIndex = index.min(size - 1).max(0)
    // Use slice to get 1D sub-tensor, then unsqueeze to ensure consistent shape
    def getFeature(v: Tensor): Tensor = {
      val sliced = v.narrow(0, safeIndex, 1)
      if (sliced.dim() == 0) sliced.unsqueeze(0) else sliced
    }
    Batch(
      sparseFeatures.map { case (k, v) => k -> getFeature(v) },
      denseFeatures.map { case (k, v) => k -> getFeature(v) },
      Map.empty,
      labels.map(l => getFeature(l))
    )
  }
}

/**
 * SequenceDataset for sequence models
 */
class SequenceDataset(
  val features: Map[String, Tensor] = Map.empty,
  val sequenceFeatures: Map[String, Tensor] = Map.empty,
  val labels: Option[Tensor] = None,
  val positions: Option[Tensor] = None,
  val timeDiffs: Option[Tensor] = None,
  val tokens: Option[Tensor] = None,
  val targets: Option[Tensor] = None
) extends Dataset {

  override def size: Long = {
    tokens.map(_.size(0))
      .orElse(features.headOption.map(_._2.size(0)))
      .orElse(sequenceFeatures.headOption.map(_._2.size(0)))
      .getOrElse(0)
  }

  override def get(index: Long): Batch = {
    Batch(
      features.map { case (k, v) => k -> v.select(0, index) },
      Map.empty,
      sequenceFeatures.map { case (k, v) => k -> v.select(0, index) },
      labels.map(_.select(0, index)),
      tokens.map(_.select(0, index)),
      positions.map(_.select(0, index)),
      timeDiffs.map(_.select(0, index)),
      targets.map(_.select(0, index))
    )
  }
}

/**
 * MatchingDataset for two-tower models
 */
class MatchingDataset(
  val userFeatures: Map[String, Tensor],
  val itemFeatures: Map[String, Tensor],
  val labels: Option[Tensor] = None,
  val negItemFeatures: Option[Map[String, Tensor]] = None
) extends Dataset {

  override def size: Long = {
    // Return the max of user and item feature counts so the dataset can be
    // iterated either by users or by items (used for embedding extraction).
    val u = userFeatures.headOption.map(_._2.size(0)).getOrElse(0L)
    val it = itemFeatures.headOption.map(_._2.size(0)).getOrElse(0L)
    math.max(u, it)
  }

  override def get(index: Long): Batch = {
    // Use safe selection to avoid out-of-range when item and user counts differ.
    def safeSelect(v: Tensor, idx: Long): Tensor = {
      val safeIdx = math.min(idx, v.size(0) - 1)
      v.select(0, safeIdx)
    }

    Batch(
      userFeatures.map { case (k, v) => k -> safeSelect(v, index) },
      itemFeatures = itemFeatures.map { case (k, v) => k -> safeSelect(v, index) },
      labels = labels.map(l => safeSelect(l, index))
    )
  }
}

/**
 * MultiTaskDataset for multi-task learning
 */
class MultiTaskDataset(
  val features: Map[String, Tensor],
  val taskLabels: Map[String, Tensor]
) extends Dataset {

  override def size: Long = {
    features.headOption.map(_._2.size(0)).getOrElse(0)
  }

  override def get(index: Long): Batch = {
    Batch(
      features.map { case (k, v) => k -> v.select(0, index) },
      Map.empty,
      Map.empty,
      None  // Labels handled separately for multi-task
    )
  }
}