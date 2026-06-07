package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._

/**
 * NARM - Neural Attentive Sequential Recommender
 * Reference: "Neural Attentive Sequential Recommender" (Li et al., 2017)
 */
class NARM(
  features: List[Feature],
  embedDim: Int = 8,
  hiddenDim: Int = 8,
  attentionDim: Int = 8,
  device: String = DeviceSupport.backend
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  private val encoder = new MLP(embedDim, List(hiddenDim.toLong), hiddenDim, "relu", 0f, device = device)
  register_module("encoder", encoder)

  private val attention = new MLP(hiddenDim * 2, List(attentionDim), hiddenDim, "relu", 0f, device = device)
  register_module("attention", attention)

  private val output = new LinearImpl(hiddenDim * 2, embedDim)
  register_module("output", output)

  def forward(sequence: Tensor): Tensor = {
    val emb = embedding.forward(Map("seq" -> sequence))
    val encoded = encoder.forward(emb.mean(1)).unsqueeze(1).repeat(1, emb.size(1), 1)
    val last = encoded.select(1, encoded.size(1) - 1)
    val attn = attention.forward(emb.mean(1))
    val combined = torch.cat(new TensorVector(last, attn), 1L)
    output.forward(combined)
  }
}
