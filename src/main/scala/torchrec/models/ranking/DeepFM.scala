package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.utils.DeviceSupport

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
  device: String = DeviceSupport.backend
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
  private val mlp = new MLP(sparseDim.toLong, mlpDims.map(_.toLong), 1, "relu", dropout, false, device = device)
  register_module("mlp", mlp)

  // Move all submodules to device
  if (device != "cpu") {
    linear.to(new org.bytedeco.pytorch.Device(device), false)
    fm.to(new org.bytedeco.pytorch.Device(device), false)
    mlp.to(new org.bytedeco.pytorch.Device(device), false)
    embeddingLayer.to(device)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Build 3-D embeddings tensor (batch, num_fields, embed_dim) for FM
    val sparseNames = features.collect { case f: SparseFeature => f.name }
    val embList = sparseNames.flatMap { name =>
      sparseFeats.get(name).map { idx =>
        embeddingLayer.getEmbedding(name, idx.toType(ScalarType.Long))
      }
    }
    if (embList.isEmpty) throw new IllegalArgumentException("No embeddings found for given features")

    // Ensure all embeddings are on the same device before stacking
    val targetDev = embList.head.device()
    val embListOnDev = embList.map(e => if (e.device().equals(targetDev)) e else e.to(targetDev, e.dtype()))
    val embeddings3D = embListOnDev.stack(1) // (batch, num_fields, embed_dim)

    // First-order: linear (zero placeholder)
    val batchSize = embeddings3D.size(0)
    val linearOut = torch.zeros(Array(batchSize.toLong, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .to(embeddings3D.device(), ScalarType.Float)

    // Second-order: FM interactions
    val fmOut = fm.forward(embeddings3D)

    // Deep part: flatten to (batch, sparseDim) before MLP
    val mlpIn = embeddings3D.view(batchSize, sparseDim.toLong)
    val mlpOut = mlp.forward(mlpIn)

    // Combine (raw logits — BCEWithLogitsLoss handles sigmoid internally)
    val logits = linearOut.add(fmOut).add(mlpOut)
    logits
  }
}
