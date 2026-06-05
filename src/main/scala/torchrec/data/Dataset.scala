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
  targets: Option[Tensor] = None
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
      targets.map(move)
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
    Batch(
      sparseFeatures.map { case (k, v) => k -> v.select(0, index) },
      denseFeatures.map { case (k, v) => k -> v.select(0, index) },
      Map.empty,
      labels.map(_.select(0, index))
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
    userFeatures.headOption.map(_._2.size(0)).getOrElse(0)
  }

  override def get(index: Long): Batch = {
    Batch(
      userFeatures.map { case (k, v) => k -> v.select(0, index) },
      itemFeatures.map { case (k, v) => k -> v.select(0, index) },
      Map.empty,
      labels.map(_.select(0, index))
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