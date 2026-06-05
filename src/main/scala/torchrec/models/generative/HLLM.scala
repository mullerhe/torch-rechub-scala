package torchrec.models.generative

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.Implicits._

/**
 * Hierarchical Large Language Model for Recommendations
 * Reference: ByteDance, 2024
 */
class HLLM(
  itemEmbeddings: Tensor,  // Pre-computed frozen item embeddings (vocab_size, embed_dim)
  features: List[Feature],
  embedDim: Int = 512,
  numHeads: Int = 8,
  numLayers: Int = 6,
  dropout: Float = 0.1f,
  device: String = "cpu"
) extends Module {

  // Frozen item embeddings (no gradient)
  private val itemEmb = itemEmbeddings.clone().detach()
  itemEmb.requires_grad_(false)

  // User feature embedding
  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
  register_module("featureEmbedding", featureEmbedding)

  // Simple feedforward layers instead of transformer encoder
  private val encoderLayers = (0 until numLayers).map { i =>
    val mlp = new MLP(embedDim, List(embedDim * 2L), embedDim, "silu", dropout, device = device)
    register_module(s"encoder_$i", mlp)
    mlp
  }

  private val layernorm = new LayerNormImpl(new LongVector(embedDim.toLong))
  register_module("layernorm", layernorm)

  def forward(
    seqTokens: Tensor,  // (batch, seq_len) - item IDs
    featMap: Map[String, Tensor]
  ): Tensor = {
    val batchSize = seqTokens.size(0)
    val seqLen = seqTokens.size(1)

    // Get item embeddings from frozen table
    val itemEmbLocal = itemEmb.index_select(0, seqTokens.view(-1).toType(ScalarType.Long))
    val itemEmbReshaped = itemEmbLocal.view(batchSize, seqLen, embedDim)

    // User feature embeddings
    val featEmb = featureEmbedding.forward(featMap)
    val featExpanded = featEmb.unsqueeze(1).repeat(1, seqLen, 1)

    // Combine
    val combined = itemEmbReshaped.add(featExpanded)

    // Encode - apply encoder layers
    var x = combined.mean(1) // Pool sequence dimension
    encoderLayers.foreach { layer =>
      x = layer.forward(x)
    }

    layernorm.forward(x)
  }

  def getItemEmbedding(itemId: Tensor): Tensor = {
    itemEmb.index_select(0, itemId.toType(ScalarType.Long))
  }
}
