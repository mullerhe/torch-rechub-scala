package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.Implicits._

/**
 * STAMP - Short-Term Memory Attentive Sequential Model
 * Reference: "STAMP: Short-Term Attention/Memory Priority Model" (Liu et al., 2018)
 */
class STAMP(
  features: List[Feature],
  embedDim: Int = 8,
  attentionDim: Int = 8,
  device: String = "cpu"
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  private val mlp1 = new MLP(embedDim * 2, List(attentionDim), embedDim, "relu", 0f, device = device)
  private val mlp2 = new MLP(embedDim * 2, List(attentionDim), embedDim, "relu", 0f, device = device)
  register_module("mlp1", mlp1)
  register_module("mlp2", mlp2)

  private val output = new LinearImpl(embedDim, 1)
  register_module("output", output)

  def forward(sequence: Tensor): Tensor = {
    val emb = embedding.forward(Map("seq" -> sequence))
    val seqLen = emb.size(1)
    val last = emb.select(1, seqLen - 1)
    val first = emb.select(1, 0)

    val attn1 = mlp1.forward(torch.cat(List(last, emb.mean(1)).toSeq.toTensorVector, 1))
    val attn2 = mlp2.forward(torch.cat(List(first, emb.mean(1)).toSeq.toTensorVector, 1))

    val weighted = attn1.mul(last).add(attn2.mul(first))
    output.forward(weighted)
  }
}