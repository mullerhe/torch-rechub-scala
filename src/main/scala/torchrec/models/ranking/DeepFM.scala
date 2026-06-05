package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * DeepFM: Combine FM (2nd-order) with DNN
 * Reference: IJCAI 2017
 */
class DeepFM(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  dropout: Float = 0.2f,
  device: String = "cpu"
) extends Module {

  private val totalVocab = features.collect { case f: SparseFeature => f.vocabSize }.sum.toInt

  // First-order: linear part
  private val linear = new LinearImpl(totalVocab, 1)
  register_module("linear", linear)

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // FM layer for 2nd-order interactions
  private val fm = new FM(embedDim, device)
  register_module("fm", fm)

  // Deep part
  private val sparseDim = features.collect { case f: SparseFeature => 1 }.size * embedDim
  private val mlp = new MLP(sparseDim, mlpDims.map(_.toLong), 1, "relu", dropout, false, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings
    val embeddings = embeddingLayer.forward(sparseFeats)

    // First-order: linear
    val sparseIndices = sparseFeats.values.toSeq
    if (sparseIndices.isEmpty) throw new IllegalArgumentException("No sparse features")
    val tensorVec = new TensorVector(sparseIndices.size.toLong)
    sparseIndices.foreach(t => tensorVec.push_back(t.toType(ScalarType.Long)))
    val indices = torch.cat(tensorVec, 1L)
    val linearOut = linear.forward(indices.toType(ScalarType.Float))

    // Second-order: FM interactions
    val fmOut = fm.forward(embeddings)

    // Deep part
    val mlpOut = mlp.forward(embeddings)

    // Combine and sigmoid
    val logits = linearOut.add(fmOut).add(mlpOut)
    logits.sigmoid()
    logits
  }
}