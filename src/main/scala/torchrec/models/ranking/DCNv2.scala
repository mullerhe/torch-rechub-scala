package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Deep & Cross Network V2
 * Reference: Stanford/Huawei
 */
class DCNv2(
  features: List[Feature],
  embedDim: Int = 8,
  numCrossLayers: Int = 3,
  useCrossNetMix: Boolean = true,
  lowRank: Int = 4,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val sparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim

  // Cross network
  private val crossNet = if (useCrossNetMix) {
    new CrossNetMix(sparseDim, numCrossLayers, lowRank, device)
  } else {
    new CrossNetV2(sparseDim, numCrossLayers, device)
  }
  register_module("crossNet", crossNet)

  // Deep network
  private val mlp = new MLP(sparseDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  // Final combination layer
  private val combo = new LinearImpl(sparseDim + 1, 1)
  register_module("combo", combo)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    combo.to(dev, false)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)

    val crossOut = crossNet match {
      case c: CrossNetMix => c.forward(embeddings)
      case c: CrossNetV2 => c.forward(embeddings)
    }

    val deepOut = mlp.forward(embeddings)

    val combined = torch.cat(new TensorVector(crossOut, deepOut), 1)
    val logits = combo.forward(combined)
    logits
  }
}
