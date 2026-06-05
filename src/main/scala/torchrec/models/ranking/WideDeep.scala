package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Wide & Deep Learning
 * Reference: Google
 */
class WideDeep(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val totalVocab = features.collect { case f: SparseFeature => f.vocabSize }.sum.toInt

  // Wide (linear) part
  private val wide = new LinearImpl(totalVocab, 1)
  register_module("wide", wide)

  // Deep part
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val sparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim
  private val mlp = new MLP(sparseDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)

    // Wide
    val indices = torch.cat(sparseFeats.values.toSeq.map(_.toType(ScalarType.Long)).toTensorVector, 1)
    val wideOut = wide.forward(indices.toType(ScalarType.Float))

    // Deep
    val deepOut = mlp.forward(embeddings)

    // Combine
    val logits = wideOut.add(deepOut)
    logits.sigmoid()
    logits
  }
}