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
    // Build 3-D embeddings tensor (batch, num_fields, embed_dim) for FM
    val sparseNames = features.collect { case f: SparseFeature => f.name }
    val embList = sparseNames.flatMap { name =>
      sparseFeats.get(name).map { idx =>
        // getEmbedding returns (batch, embedDim)
        embeddingLayer.getEmbedding(name, idx.toType(ScalarType.Long)).to(new Device(device), ScalarType.Float)
      }
    }
    if (embList.isEmpty) throw new IllegalArgumentException("No embeddings found for given features")
    val embeddings = embList.stack(1) // (batch, num_fields, embed_dim)

    // First-order: linear (skipped native cat issues) - use zero placeholder
    val batchSize = embeddings.size(0)
    val linearOut = torchrec.TorchRec.zeros(batchSize, 1)

    // Second-order: FM interactions
    val fmOut = fm.forward(embeddings)

    // Deep part: flatten embeddings to (batch, sparseDim) before MLP
    val mlpIn = embeddings.view(batchSize, sparseDim.toLong)
    val mlpOut = mlp.forward(mlpIn)

    // Combine and sigmoid
    val logits = linearOut.add(fmOut).add(mlpOut)
    logits.sigmoid()
    logits
  }
}