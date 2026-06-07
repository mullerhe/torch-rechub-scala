package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Entire Space Multi-Task Model
 * Reference: Alibaba, KDD 2018
 */
class ESMM(
  features: List[Feature],
  taskNames: List[String] = List("cvr", "ctr"),
  embedDim: Int = 8,
  towerDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val sparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim

  // Shared bottom
  private val sharedBottom = new MLP(sparseDim, towerDims.init, towerDims.last, "relu", dropout, device = device)
  register_module("sharedBottom", sharedBottom)

  // Task-specific towers
  private val taskTowers = taskNames.map { name =>
    val tower = new MLP(towerDims.last, List(towerDims.last), 1, "relu", dropout, device = device)
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
      (name, taskOut.sigmoid())
    }.toMap
  }

  def predict(
    sparseFeats: Map[String, Tensor],
    taskName: String
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val sharedOut = sharedBottom.forward(embeddings)
    taskTowers(taskName).forward(sharedOut).sigmoid()
  }
}
