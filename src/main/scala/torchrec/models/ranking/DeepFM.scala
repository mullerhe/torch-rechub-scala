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

  // Deep part — sparseDim is computed dynamically in forward() from the actual
  // embedding tensor shape to guarantee consistency with the incoming sparseFeats.
  // MLP in-features computed using numFields * maxEmbedDim for uniform embedding dim.
  private val numSparseFeatures = features.collect { case f: SparseFeature => f }.size
  private val maxEmbedDimFeature = features.collect { case f: SparseFeature => f.embedDim }.max
  private val sparseDimConstructor = numSparseFeatures * maxEmbedDimFeature
  private val mlp = new MLP(sparseDimConstructor.toLong, mlpDims.map(_.toLong), 1, "relu", dropout, false, device = device)
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

    // Determine the maximum embed dim across all features for padding
    val maxEmbedDim = embList.map(_.size(1)).max.toInt
    val batchSize = embList.head.size(0)

    // Pad embeddings to maxEmbedDim using zero-padding (manual approach for compatibility)
    val paddedEmbeddings = embList.map { emb =>
      val embDim = emb.size(1).toInt
      if (embDim < maxEmbedDim) {
        // Create a new tensor with zeros and copy original data
        val paddedData = Array.fill[Float](batchSize.toInt * maxEmbedDim)(0f)
        val origData = emb.reshape(batchSize.toLong * embDim.toLong).toFloatArray
        System.arraycopy(origData, 0, paddedData, 0, origData.length)
        torch.tensor(paddedData*).view( batchSize.toLong, maxEmbedDim.toLong).to(emb.device(), ScalarType.Float)
      } else {
        emb
      }
    }

    // Ensure all embeddings are on the same device before stacking
    val targetDev = paddedEmbeddings.head.device()
    val embListOnDev = paddedEmbeddings.map(e => if (e.device().equals(targetDev)) e else e.to(targetDev, e.dtype()))
    val embeddings3D = embListOnDev.stack(1) // (batch, num_fields, embed_dim)

    // Derive sparseDim dynamically from the actual embedding tensor so it always
    // matches the number of fields that are present in sparseFeats.
    val numFields = embeddings3D.size(1)
    val embedDimActual = embeddings3D.size(2)
    val sparseDimDyn = (numFields * embedDimActual).toLong

    // First-order: linear (zero placeholder)
    val linearOut = torch.zeros(Array(batchSize.toLong, 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .to(embeddings3D.device(), ScalarType.Float)

    // Second-order: FM interactions
    val fmOut = fm.forward(embeddings3D)

    // Deep part: flatten to (batch, sparseDimDyn) before MLP
    val mlpIn = embeddings3D.view(batchSize, sparseDimDyn)
    val mlpOut = mlp.forward(mlpIn)

    // Combine (raw logits — BCEWithLogitsLoss handles sigmoid internally)
    val logits = linearOut.add(fmOut).add(mlpOut)
    logits
  }
}
