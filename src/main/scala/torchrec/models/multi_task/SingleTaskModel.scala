package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Single-Task Model (Independent Multi-Task)
 *
 * Each task has its own independent embedding layer and tower.
 * No parameter sharing between tasks — the opposite of SharedBottom.
 * This provides an upper bound for multi-task model performance.
 *
 * Architecture:
 *   Input → [Embedding_i + Tower_i for each task] → Task Outputs
 *
 * @param features     List of sparse features
 * @param taskNames    Names of tasks
 * @param embedDim     Embedding dimension
 * @param bottomDims   Hidden layer dimensions for task-specific bottom networks
 * @param towerDims    Hidden layer dimensions for task-specific towers
 * @param dropout      Dropout rate
 * @param device       Device to run on
 */
class SingleTaskModel(
  features: List[Feature],
  taskNames: List[String],
  embedDim: Int = 8,
  bottomDims: List[Long] = List(128L),
  towerDims: List[Long] = List(64L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskNames.nonEmpty, "taskNames cannot be empty")

  // Embedding dimension per field
  private val sparseDim = Features.calcSparseDim(features)

  // Per-task embedding layers (independent)
  private val taskEmbeddings = taskNames.map { name =>
    val emb = new EmbeddingLayer(features, embedDim, device)
    register_module(s"embedding_$name", emb)
    (name, emb)
  }.toMap

  // Per-task bottom networks (independent)
  private val taskBottoms = taskNames.map { name =>
    val bottom = new MLP(sparseDim, bottomDims, bottomDims.last, "relu", dropout, device = device)
    register_module(s"bottom_$name", bottom)
    (name, bottom)
  }.toMap

  // Per-task towers (independent)
  private val taskTowers = taskNames.map { name =>
    val tower = new MLP(bottomDims.last, towerDims, 1, "relu", dropout, device = device)
    register_module(s"tower_$name", tower)
    (name, tower)
  }.toMap

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Map[String, Tensor] = {
    taskNames.map { name =>
      val emb = taskEmbeddings(name).forward(sparseFeats)
      val bottomOut = taskBottoms(name).forward(emb)
      val taskOut = taskTowers(name).forward(bottomOut)
      (name, taskOut)
    }.toMap
  }
}
