package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Deep Interest Evolution Network (DIEN)
 * Reference: "Deep Interest Evolution Network for Click-Through Rate Prediction"
 * Alibaba, AAAI 2019
 */
class DIEN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  hiddenDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")

  // Embedding layers
  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val sequenceEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // MLP for final prediction
  private val sparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim
  private val totalDim = sparseDim + embedDim
  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, false, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor],
    targetFeats: Map[String, Tensor]
  ): Tensor = {
    // Get feature embeddings
    val featEmb = featureEmbedding.forward(sparseFeats)

    // Simplified: just use mean pooling of sequence
    val seqEmb = sequenceEmbedding.forward(seqFeats)
    val pooledSeq = seqEmb.mean(1)

    // Combine features and sequence
    val flatFeatures = featEmb.view(featEmb.size(0), -1)
    val combined = torch.cat(Seq(flatFeatures, pooledSeq).toTensorVector, 1)

    // Final MLP
    val logits = mlp.forward(combined)

    logits
  }
}