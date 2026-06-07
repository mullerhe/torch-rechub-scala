package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.Implicits._

/**
 * Enhanced DCN with Bridge and Regulation
 * Reference: 2021
 */
class EDCN(
  features: List[Feature],
  embedDim: Int = 8,
  numCrossLayers: Int = 3,
  mlpDims: List[Long] = List(256L, 128L),
  bridgeType: String = "add",  // "add", "concat", "hadamard"
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  private val sparseDim = numFields * embedDim

  // Cross network
  private val crossNet = new CrossNetwork(sparseDim, numCrossLayers, device)
  register_module("crossNet", crossNet)

  // Deep network
  private val mlp = new MLP(sparseDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  // Bridge module
  private val bridge = new Bridge(sparseDim, mlpDims.last.toInt, bridgeType, device)
  register_module("bridge", bridge)

  // Final combo layer (output dim depends on bridge type)
  private val bridgeOutDim = bridgeType match {
    case "concat" => mlpDims.last + 1
    case _ => mlpDims.last
  }
  private val combo = new LinearImpl(sparseDim + bridgeOutDim, 1)
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
    val batchSize = embeddings.size(0)
    val x0 = embeddings.view(batchSize, -1)

    // Cross network
    val crossOut = crossNet.forward(x0)

    // Deep network
    val mlpOut = mlp.forward(embeddings)

    // Bridge
    val bridged = bridge.forward(crossOut, mlpOut)

    // Final combination via concat + linear
    val combined = torch.cat(new TensorVector(crossOut, bridged), 1)
    val logits = combo.forward(combined)
    logits
  }
}

/**
 * Bridge module for connecting cross and deep networks
 */
class Bridge(
  inputDim: Long,
  hiddenDim: Long,
  bridgeType: String,
  device: String = "cpu"
) extends Module {

  private val projection = new LinearImpl(inputDim, hiddenDim)
  register_module("projection", projection)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    projection.to(dev, false)
  }

  def forward(crossOut: Tensor, deepOut: Tensor): Tensor = {
    bridgeType match {
      case "add" =>
        val projected = projection.forward(crossOut)
        projected.add(deepOut)

      case "concat" =>
        val projected = projection.forward(crossOut)
        torch.cat(new TensorVector(projected, deepOut), 1)

      case "hadamard" =>
        val projected = projection.forward(crossOut)
        projected.mul(deepOut)

      case _ => deepOut
    }
  }
}
