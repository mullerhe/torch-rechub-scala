package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * Deep Structured Semantic Model (DSSM) - Two Tower Architecture
 * Reference: Microsoft
 */
class DSSM(
  userFeatures: List[Feature],
  itemFeatures: List[Feature],
  embedDim: Int = 8,
  towerDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  // User tower
  private val userEmbedding = new EmbeddingLayer(userFeatures, embedDim, device)
  register_module("userEmbedding", userEmbedding)

  private val userSparseDim = userFeatures.collect { case f: SparseFeature => 1 }.size * embedDim
  private val userTower = new MLP(userSparseDim, towerDims, embedDim, "relu", dropout, device = device)
  register_module("userTower", userTower)

  // Item tower
  private val itemEmbedding = new EmbeddingLayer(itemFeatures, embedDim, device)
  register_module("itemEmbedding", itemEmbedding)

  private val itemSparseDim = itemFeatures.collect { case f: SparseFeature => 1 }.size * embedDim
  private val itemTower = new MLP(itemSparseDim, towerDims, embedDim, "relu", dropout, device = device)
  register_module("itemTower", itemTower)

  def forward(
    userFeats: Map[String, Tensor],
    itemFeats: Map[String, Tensor]
  ): Tensor = {
    val userEmb = userEmbedding.forward(userFeats)
    val itemEmb = itemEmbedding.forward(itemFeats)

    val userOut = userTower.forward(userEmb)
    val itemOut = itemTower.forward(itemEmb)

    // Cosine similarity
    // Compute L2 norm manually: sqrt(sum(x^2))
    val userNorm = userOut.pow(new Scalar(2)).sum(1).sqrt()
    val itemNorm = itemOut.pow(new Scalar(2)).sum(1).sqrt()
    val prodNorms = userNorm.mul(itemNorm)
    val cosSim = userOut.mul(itemOut).sum(1).div(prodNorms.add(new Scalar(1e-8f)))
    cosSim.unsqueeze(1)
  }

  def userTowerForward(userFeats: Map[String, Tensor]): Tensor = {
    val sparseOnly = userFeats.filter(_._2.dim() == 1L)
    val userEmb = userEmbedding.forward(sparseOnly)
    userTower.forward(userEmb)
  }

  def itemTowerForward(itemFeats: Map[String, Tensor]): Tensor = {
    val sparseOnly = itemFeats.filter(_._2.dim() == 1L)
    val itemEmb = itemEmbedding.forward(sparseOnly)
    itemTower.forward(itemEmb)
  }
}
