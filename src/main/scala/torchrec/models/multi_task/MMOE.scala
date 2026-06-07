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
    val embeddings = embeddingLayer.forward(sparseFeats)

    // Expert outputs
    val expertOutputs = experts.map(_.forward(embeddings))

    taskNames.map { name =>
      val gateWeights = gates(name).forward(embeddings).softmax(1) // (batch, num_experts)

      // Weighted sum of expert outputs
      val weightedExperts = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
        gateWeights.select(1, i).unsqueeze(1).mul(expertOut)
      }
      val combined = weightedExperts.reduce(_.add(_))

      val taskOut = taskTowers(name).forward(combined)
      (name, taskOut.sigmoid())
    }.toMap
  }
}
