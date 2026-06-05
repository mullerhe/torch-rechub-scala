package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._

/**
 * GRU4Rec - GRU-based Recurrent Recommender
 * Reference: "GRU4Rec: Gated Recurrent Unit for Recommendations" (Hidasi et al., 2015)
 */
class GRU4Rec(
  features: List[Feature],
  embedDim: Int = 8,
  hiddenDim: Int = 8,
  numLayers: Int = 1,
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  private val encoder = new MLP(embedDim, List(hiddenDim.toLong), hiddenDim, "relu", dropout, device = device)
  register_module("encoder", encoder)

  private val output = new LinearImpl(hiddenDim, embedDim)
  register_module("output", output)

  def forward(sequence: Tensor): Tensor = {
    val emb = embedding.forward(Map("seq" -> sequence))
    val encoded = encoder.forward(emb.mean(1))
    output.forward(encoded)
  }
}
