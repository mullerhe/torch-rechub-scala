package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.pytorch._
import org.bytedeco.pytorch._

/**
 * Attentional Factorization Machine (AFM)
 * Reference: "Attentional Factorization Machine: Learning the Weight of Feature Interactions via Attention Networks"
 * Zhejiang University, IJCAI 2017
 */
class AFM(
  features: List[Feature],
  embedDim: Int = 8,
  attentionDim: Int = 8,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "AFM requires at least 2 sparse features for interaction")

  // Embedding layer for sparse features
  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Attention network: input=embed_dim, hidden=attentionDim, output=1
  private val attentionNet = new MLP(
    embedDim,
    List(attentionDim.toLong),
    1,
    "relu",
    dropout,
    false,
    device = device
  )
  register_module("attentionNet", attentionNet)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields * embed_dim)
    val embeddingsFlat = embedding.forward(sparseFeats)
    val batchSize = embeddingsFlat.size(0).toInt
    val dev = embeddingsFlat.device()

    // Reshape to (batch, num_fields, embed_dim)
    val embeddings = embeddingsFlat.view(batchSize, numFields, embedDim)

    // Compute attention-weighted pairwise interactions
    val weightedList = scala.collection.mutable.ListBuffer[Tensor]()
    var i = 0
    while (i < numFields) {
      var j = i + 1
      while (j < numFields) {
        val vi = embeddings.narrow(1, i, 1).squeeze(1)  // (batch, embed_dim)
        val vj = embeddings.narrow(1, j, 1).squeeze(1)  // (batch, embed_dim)
        val ip = vi.mul(vj)  // (batch, embed_dim)
        val attnScore = attentionNet.forward(ip)  // (batch, 1)
        val weighted = ip.mul(attnScore)  // (batch, embed_dim)
        weightedList += weighted
        j += 1
      }
      i += 1
    }

    if (weightedList.isEmpty) {
      return torch.zeros(batchSize.toLong, 1L).to(dev, ScalarType.Float)
    }

    // Sum all weighted interactions
    val vec = new TensorVector()
    weightedList.foreach(vec.push_back)
    val combined = torch.cat(vec, 1L)  // (batch, num_pairs * embed_dim)
    combined.sum(1L).unsqueeze(1)  // (batch, 1)
  }
}
