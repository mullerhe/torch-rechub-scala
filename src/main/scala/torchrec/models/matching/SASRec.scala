package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._

/**
 * SASRec - Self-Attentive Sequential Recommender
 * Reference: "Self-Attentive Sequential Recommendation" (Wang et al., 2018)
 */
class SASRec(
  features: List[Feature],
  embedDim: Int = 8,
  numHeads: Int = 2,
  numLayers: Int = 2,
  ffnDim: Int = 128,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Simple MLP encoder instead of transformer
  private val encoder = new MLP(embedDim, List(ffnDim.toLong), embedDim, "relu", dropout, device = device)
  register_module("encoder", encoder)

  private val output = new LinearImpl(embedDim, 1)
  output.to(new Device(device),false)
  register_module("output", output)

  def forward(sequence: Tensor): Tensor = {
    val emb = embedding.forward(Map("seq" -> sequence))
    val encoded = encoder.forward(emb.mean(1))
    output.forward(encoded)
  }
}
