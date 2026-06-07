package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Shared Bottom Model (Hard Parameter Sharing)
 */
class SharedBottom(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  sharedDims: List[Long] = List(256L, 128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val sparseDim = Features.calcSparseDim(features)

  // Shared bottom
  private val sharedBottom = new MLP(sparseDim, sharedDims, sharedDims.last, "relu", dropout, device = device)
  register_module("sharedBottom", sharedBottom)

  // Task-specific towers
  private val taskTowers = taskNames.map { name =>
    val tower = new MLP(sharedDims.last, towerDims, 1, "relu", dropout, device = device)
    register_module(s"tower_$name", tower)
    (name, tower)
  }.toMap

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Map[String, Tensor] = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val sharedOut = sharedBottom.forward(embeddings)

    taskNames.map { name =>
      val taskOut = taskTowers(name).forward(sharedOut)
      (name, taskOut)
    }.toMap
  }
}
