package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._
import torchrec.TensorImplicits._

/**
 * YouTubeDNN Matching Model
 * Reference: YouTube
 */
class YoutubeDNN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  towerDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  // Feature embeddings
  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  // Sequence embeddings
  private val sequenceEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // MLP tower
  private val featSparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim
  private val seqSparseDim = sequenceFeatures.size * embedDim
  private val totalInputDim = featSparseDim + seqSparseDim

  private val tower = new MLP(totalInputDim, towerDims, embedDim, "relu", dropout, device = device)
  register_module("tower", tower)

  def forward(
    features: Map[String, Tensor],
    sequenceFeatures: Map[String, Tensor]
  ): Tensor = {
    val featEmb = featureEmbedding.forward(features)
    val seqEmbs = sequenceFeatures.map { case (name, indices) =>
      sequenceEmbedding.getEmbedding(name, indices)
    }.toSeq
    val seqPooled = if (seqEmbs.nonEmpty) {
      seqEmbs.cat(1).mean(1)
    } else {
      zeros(Array(totalInputDim))
    }

    val combined = torch.cat(new TensorVector(featEmb, seqPooled), 1L)
    tower.forward(combined)
  }

  def userTowerForward(
    features: Map[String, Tensor],
    sequenceFeatures: Map[String, Tensor]
  ): Tensor = {
    forward(features, sequenceFeatures)
  }
}
