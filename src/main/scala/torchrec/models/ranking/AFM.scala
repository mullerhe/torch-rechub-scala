package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.TensorImplicits.RichTensor

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Attentional Factorization Machine (AFM)
 * Reference: "Attentional Factorization Machine: Learning the Weight of Feature Interactions via Attention Networks"
 * Zhejiang University, IJCAI 2017
 *
 * @param features List of sparse features
 * @param embedDim Embedding dimension
 * @param attentionDim Attention network hidden dimension
 * @param dropout Dropout rate
 * @param device Device to run on
 */
class AFM(
  features: List[Feature],
  embedDim: Int = 8,
  attentionDim: Int = 8,
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "AFM requires at least 2 sparse features for interaction")

  // Embedding layer for sparse features
  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Attention network for pairwise interactions
  private val attentionNet = new MLP(
    embedDim * 3,  // [embed_i, embed_j, element-wise product]
    List(attentionDim.toLong),
    1,
    "relu",
    dropout,
    false,
    device = device
  )
  register_module("attentionNet", attentionNet)

  // Second-order FM layer
  private val fmInteraction = new FMInteraction(embedDim)
  register_module("fmInteraction", fmInteraction)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim)
    val embeddings = embedding.forward(sparseFeats)

    // Get attention weights for each pair of field interactions
    val attentionWeights = computeAttentionWeights(embeddings)

    // Apply attention-weighted FM interactions
    val fmOut = fmInteraction.forward(embeddings)
    val attendedInteractions = fmOut * attentionWeights

    attendedInteractions.sigmoid()
  }

  private def computeAttentionWeights(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)
    val embedDim = embeddings.size(2)

    // Collect all pairwise interactions: (batch, num_pairs, embed_dim)
    val pairwiseList = scala.collection.mutable.ListBuffer[Tensor]()
    var i = 0
    while (i < numFields) {
      var j = i + 1
      while (j < numFields) {
        // Element-wise product of embeddings from field i and j
        val embI = embeddings.narrow(1, i, 1).squeeze(1)
        val embJ = embeddings.narrow(1, j, 1).squeeze(1)
        val product = embI.mul(embJ)
        pairwiseList += product
        j += 1
      }
      i += 1
    }

    if (pairwiseList.isEmpty) {
      return torch.ones(batchSize, 1)
    }

    val pairwise = torch.cat(pairwiseList.toList.toTensorVector,  1)
    // pairwise: (batch, num_pairs, embed_dim)

    // Attention network
    val attentionScores = attentionNet.forward(pairwise)
    // attentionScores: (batch, num_pairs, 1)

    // Softmax over attention scores
    val attnWeights = attentionScores.softmax(1)
    attnWeights
  }
}