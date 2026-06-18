package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Deep & Cross Network
 * Reference: Stanford/Huawei
 */
class DCN(
  features: List[Feature],
  embedDim: Int = 8,
  numCrossLayers: Int = 3,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val sparseDim = Features.calcSparseDim(features)

  // Cross network
  private val crossNet = new CrossNetwork(sparseDim, numCrossLayers, device)
  register_module("crossNet", crossNet)

  // Deep network
  private val mlp = new MLP(sparseDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  // Final combination layer: cross output (sparseDim) + MLP last hidden dim
  private val combo = new LinearImpl(sparseDim + mlpDims.last, 1)
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
    val dev = embeddings.device()

    // Cross network
    val crossOut = crossNet.forward(embeddings)

    // Deep network
    val deepOut = mlp.forward(embeddings)

    // Combine via concat + final linear (standard DCN architecture)
    val combined = torch.cat(new TensorVector(crossOut, deepOut), 1)
    val logits = combo.forward(combined)
    logits
  }
}
