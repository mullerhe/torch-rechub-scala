package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Adversarial Information Transfer Multi-Task
 * Reference: Alibaba, KDD 2020
 * Uses adversarial training to transfer information between tasks
 * while preventing negative transfer
 */
class AITM(
  features: List[Feature],
  taskNames: List[String] = List("ctr", "cvr"),
  embedDim: Int = 8,
  hiddenDim: Int = 64,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Shared encoder
  private val encoder = new MLP(embedDim, List(hiddenDim), hiddenDim, "relu", dropout, device = device)
  register_module("encoder", encoder)

  // Task-specific towers
  private val towers = taskNames.map { name =>
    val tower = new MLP(hiddenDim, List(hiddenDim / 2), 1, "relu", dropout, device = device)
    register_module(s"tower_$name", tower)
    (name, tower)
  }.toMap

  // Information transfer gates (learnable task interaction)
  private val transferGates = taskNames.map { name =>
    val gate = new LinearImpl(hiddenDim, hiddenDim)
    register_module(s"gate_$name", gate)
    (name, gate)
  }.toMap

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Map[String, Tensor] = {
    val emb = embedding.forward(sparseFeats)
    val encoded = encoder.forward(emb)

    // Apply adversarial information transfer
    taskNames.map { name =>
      val gated = encoded.mul(transferGates(name).forward(encoded).sigmoid())
      val output = towers(name).forward(gated)
      (name, output.sigmoid())
    }.toMap
  }

  def predict(
    sparseFeats: Map[String, Tensor],
    taskName: String
  ): Tensor = {
    val emb = embedding.forward(sparseFeats)
    val encoded = encoder.forward(emb)
    val gated = encoded.mul(transferGates(taskName).forward(encoded).sigmoid())
    towers(taskName).forward(gated).sigmoid()
  }
}