package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.TensorImplicits.RichTensor

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

/**
 * Progressive Layered Extraction
 * Reference: Tencent, RecSys 2020
 */
class PLE(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  numSharedExperts: Int = 2,
  numTaskExperts: Int = 2,
  numLayers: Int = 3,
  expertDims: List[Long] = List(128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Expert layers
  private val expertLayers = (0 until numLayers).map { layerIdx =>
    val numExperts = if (layerIdx == numLayers - 1) numTaskExperts * taskNames.size else numSharedExperts + numTaskExperts * taskNames.size

    val experts = (0 until numExperts).map { i =>
      val expert = new MLP(embedDim, expertDims, embedDim, "relu", dropout, device = device)
      register_module(s"expert_layer${layerIdx}_$i", expert)
      expert
    }
    experts
  }

  // Task-specific gates
  private val taskGates = taskNames.map { name =>
    val gate = new LinearImpl(embedDim, numSharedExperts + numTaskExperts)
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
    var currentInput = embeddings
    var result: Map[String, Tensor] = Map.empty

    // Progressive extraction layers
    var layerIdx = 0
    while (layerIdx < numLayers) {
      val experts = expertLayers(layerIdx)
      val isLastLayer = layerIdx == numLayers - 1

      val numExperts = experts.size
      val expertOutputs = experts.map(_.forward(currentInput))

      if (isLastLayer) {
        // Split experts by task
        val taskExpertOutputs = taskNames.map { name =>
          val startIdx = taskNames.indexOf(name) * numTaskExperts
          expertOutputs.slice(startIdx, startIdx + numTaskExperts)
        }

        result = taskNames.map { name =>
          val taskExperts = taskExpertOutputs(taskNames.indexOf(name))

          // Gate
          val gateWeights = taskGates(name).forward(currentInput).softmax(1)
          val weightedExperts = taskExperts.zipWithIndex.map { case (expertOut, i) =>
            gateWeights.select(1, i).unsqueeze(1).mul(expertOut)
          }
          val combined = weightedExperts.reduce(_.add(_))
          val taskOut = taskTowers(name).forward(combined)
          (name, taskOut.sigmoid())
        }.toMap
        layerIdx = numLayers // Exit loop

      } else {
        // Shared extraction
        val allExpertOuts = expertOutputs

        // Update input for next layer
        val nextInputs = taskNames.map { name =>
          val gateWeights = taskGates(name).forward(currentInput).softmax(1)
          val weightedExperts = allExpertOuts.zipWithIndex.map { case (expertOut, i) =>
            gateWeights.select(1, i).unsqueeze(1).mul(expertOut)
          }
          weightedExperts.reduce(_.add(_))
        }

        val tensorVec = new TensorVector(nextInputs.size.toLong)
        nextInputs.foreach(tensorVec.push_back)
        val scale = new Scalar(taskNames.size.toFloat)
        currentInput = torch.cat(tensorVec, 1L).div(scale)
        layerIdx += 1
      }
    }
    result
  }
}
