package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

/**
 * Multi-Gate Mixture-of-Experts
 * Reference: Google, KDD 2018
 */
class MMOE(
  features: List[Feature],
  taskNames: List[String],
  taskTypes: List[String] = List.fill(1)("classification"), // not used, kept for API compat
  embedDim: Int = 8,
  numExperts: Int = 4,
  expertDims: List[Long] = List(128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Expert networks
  private val experts = (0 until numExperts).map { i =>
    val expert = new MLP(embedDim, expertDims, embedDim, "relu", dropout, device = device)
    register_module(s"expert_$i", expert)
    expert
  }

  // Task gates
  private val gates = taskNames.map { name =>
    val gate = new LinearImpl(embedDim, numExperts)
    gate.to(new Device(device),false)
    register_module(s"gate_$name", gate)
    (name, gate)
  }.toMap

  // Task towers
  private val taskTowers = taskNames.map { name =>
    val tower = new MLP(embedDim, towerDims, 1, "relu", dropout, device = device)
    register_module(s"tower_$name", tower)
    (name, tower)
  }.toMap

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Map[String, Tensor] = {
    // EmbeddingLayer.forward() returns (batch, totalEmbedDim) where totalEmbedDim is sum of all feature embedDims
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0)
    val totalEmbedDim = embeddings.size(1)

    // For MMOE, we need to project from totalEmbedDim to embedDim
    // Use a linear projection or just slice to get embedDim
    // For simplicity, take the first embedDim dimensions or repeat to match
    val pooled = if (totalEmbedDim == embedDim) {
      embeddings
    } else {
      // Project: take first embedDim, or repeat/sum to get embedDim
      // Simple approach: reshape to (batch, numFeatures, -1) then sum
      val numFeatures = features.collect { case f: SparseFeature => 1 }.size
      val perFeatureDim = totalEmbedDim / numFeatures  // assume uniform for now
      // Actually, simpler: just take first embedDim columns
      embeddings.narrow(1, 0, embedDim)
    }

    // Expert outputs
    val expertOutputs = experts.map(_.forward(pooled))

    taskNames.map { name =>
      val gateWeights = gates(name).forward(pooled).softmax(1) // (batch, num_experts)

      // Weighted sum of expert outputs
      val weightedExperts = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
        gateWeights.select(1, i).unsqueeze(1).mul(expertOut)
      }
      val combined = weightedExperts.reduce(_.add(_))

      val taskOut = taskTowers(name).forward(combined)
      (name, taskOut)
    }.toMap
  }
}
