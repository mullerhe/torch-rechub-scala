package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * One-gate Mixture-of-Experts (OMoE)
 *
 * A simplified MoE variant with a single shared gate across all tasks.
 * Unlike MMOE which has per-task gates, OMoE uses one gate that routes
 * all tasks through the same expert selection.
 *
 * Architecture:
 *   Input → Shared Gate → Expert Routing → Per-Task Towers → Outputs
 *
 * Reference: "Modeling Task Relationships in Multi-Task Learning with Multi-gate Mixture-of-Experts" (Ma et al., KDD 2018)
 *            The OMoE is the single-gate variant from the same paper.
 *
 * @param features     List of sparse features
 * @param taskNames    Names of tasks
 * @param embedDim     Embedding dimension
 * @param numExperts   Number of expert networks
 * @param expertDims   Hidden layer dimensions for expert networks
 * @param towerDims    Hidden layer dimensions for task towers
 * @param dropout      Dropout rate
 * @param device       Device to run on
 */
class OMoE(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  numExperts: Int = 4,
  expertDims: List[Long] = List(128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskNames.nonEmpty, "taskNames cannot be empty")
  require(numExperts >= 1, s"numExperts must be >= 1, got $numExperts")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Expert networks
  private val experts = (0 until numExperts).map { i =>
    val expert = new MLP(embedDim, expertDims, embedDim, "relu", dropout, device = device)
    register_module(s"expert_$i", expert)
    expert
  }

  // Single shared gate (different from MMOE which has per-task gates)
  private val gate = new LinearImpl(embedDim, numExperts)
  gate.to(new Device(device),false)
  register_module("gate", gate)

  // Task-specific towers
  private val taskTowers = taskNames.map { name =>
    val tower = new MLP(embedDim, towerDims, 1, "relu", dropout, device = device)
    register_module(s"tower_$name", tower)
    (name, tower)
  }.toMap

  if (device != "cpu") {
    gate.to(new org.bytedeco.pytorch.Device(device), false)
    ()
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Map[String, Tensor] = {
    // EmbeddingLayer.forward() returns (batch, numFields * embedDim)
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0)

    // Reshape to (batch, numFields, embedDim) then sum-pool over fields
    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val reshaped = embeddings.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
    val pooled = reshaped.sum(1)  // (batch, embedDim)

    // Single shared gate (softmax over experts)
    val gateWeights = gate.forward(pooled).softmax(1)  // (batch, num_experts)

    // Compute expert outputs
    val expertOutputs = experts.map(_.forward(pooled))  // List[(batch, embedDim)]

    // Route through shared gate
    val weightedExperts = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
      gateWeights.select(1, i).unsqueeze(1).mul(expertOut)
    }
    val combined = weightedExperts.reduce(_.add(_))  // (batch, embedDim)

    // Pass through task-specific towers
    taskNames.map { name =>
      val taskOut = taskTowers(name).forward(combined)
      (name, taskOut)
    }.toMap
  }
}
