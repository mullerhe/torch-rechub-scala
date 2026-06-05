package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

/**
 * Multi-Interest Network with Dynamic Routing
 * Reference: Alibaba, CIKM 2019
 */
class MIND(
  features: List[Feature],
  sequenceFeature: SequenceFeature,
  embedDim: Int = 8,
  numInterests: Int = 4,
  capsuleDim: Int = 4,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  private val sequenceEmbedding = new EmbeddingLayer(List(sequenceFeature), embedDim, device)
  register_module("sequenceEmbedding", sequenceEmbedding)

  // Capsule routing
  private val capsuleNet = new CapsuleNetwork(embedDim, numInterests, capsuleDim)
  register_module("capsuleNet", capsuleNet)

  // MLP
  private val featSparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim
  private val totalInputDim = featSparseDim + numInterests * capsuleDim

  private val tower = new MLP(totalInputDim, mlpDims, embedDim, "relu", dropout, device = device)
  register_module("tower", tower)

  def forward(
    features: Map[String, Tensor],
    sequenceIndices: Tensor  // (batch, seq_len)
  ): Tensor = {
    val featEmb = featureEmbedding.forward(features)
    val seqEmb = sequenceEmbedding.getEmbedding(sequenceFeature.name, sequenceIndices)
    // seqEmb: (batch, seq_len, embed_dim)

    // Capsule routing to get multiple interests
    val interests = capsuleNet.forward(seqEmb)
    // interests: (batch, num_interests, capsule_dim)

    // Concatenate features and interests
    val interestsFlat = interests.view(interests.size(0), -1)
    val combined = torch.cat(new TensorVector(featEmb, interestsFlat), 1L)

    tower.forward(combined)
  }
}

/**
 * Capsule Network for Multi-Interest Learning
 */
class CapsuleNetwork(
  embedDim: Int,
  numInterests: Int,
  capsuleDim: Int,
  numRoutings: Int = 3
) extends Module {

  private val inputDim = embedDim
  private val W = new LinearImpl(inputDim, numInterests * capsuleDim)
  register_module("W", W)

  def forward(x: Tensor): Tensor = {
    // x: (batch, seq_len, embed_dim)
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    // Project to capsule space
    val u = W.forward(x) // (batch, seq_len, num_interests * capsule_dim)
    val reshaped = u.view(batchSize, seqLen, numInterests, capsuleDim)
    // (batch, seq_len, num_interests, capsule_dim)

    // Simple attention-based pooling
    val attnWeights = reshaped.mean(2).softmax(2).unsqueeze(2) // (batch, seq_len, 1, num_interests)
    val interests = reshaped.mul(attnWeights).sum(1) // (batch, num_interests, capsule_dim)

    interests
  }
}