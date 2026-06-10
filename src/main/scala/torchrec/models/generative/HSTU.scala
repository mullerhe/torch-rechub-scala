package torchrec.models.generative

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

import scala.collection.mutable

/**
 * Hierarchical Sequential Transduction Units
 * Reference: Meta, 2024
 */
class HSTU(
  vocabSize: Long,
  embedDim: Int = 512,
  numHeads: Int = 8,
  numLayers: Int = 12,
  maxSeqLen: Int = 512,
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  // Token embeddings
  private val tokenEmbedding = new EmbeddingImpl(vocabSize, embedDim)
  tokenEmbedding.to(new Device(device),false)
  register_module("tokenEmbedding", tokenEmbedding)

  // Position embeddings
  private val positionEmbedding = new EmbeddingImpl(maxSeqLen, embedDim)
  positionEmbedding.to(new Device(device),false)
  register_module("positionEmbedding", positionEmbedding)

  // Time difference embeddings
  private val timeEmbedding = new EmbeddingImpl(512, embedDim)  // max_time_bins
  timeEmbedding.to(new Device(device),false)
  register_module("timeEmbedding", timeEmbedding)

  // HSTU layers
  private val layers = (0 until numLayers).map { i =>
    val layer = new HSTULayer(embedDim, numHeads, dropout, device)
    register_module(s"hstu_$i", layer)
    layer
  }

  // Output projection
  private val outputProj = new LinearImpl(embedDim, vocabSize)
  outputProj.to(new Device(device),false)
  register_module("outputProj", outputProj)

  private val layernorm = new LayerNormImpl(new LongVector(embedDim.toLong))
  layernorm.to(new Device(device),false)
  register_module("layernorm", layernorm)

  private val dropoutLayer = new DropoutImpl(dropout)

  def forward(
    tokens: Tensor,  // (batch, seq_len)
    positions: Tensor,  // (batch, seq_len)
    timeDiffs: Tensor  // (batch, seq_len)
  ): Tensor = {
    val batchSize = tokens.size(0)
    val seqLen = tokens.size(1)

    // Embeddings
    var x = tokenEmbedding.forward(tokens.toType(ScalarType.Long))
    val posEmb = positionEmbedding.forward(positions.toType(ScalarType.Long))
    val timeEmb = timeEmbedding.forward(timeDiffs.toType(ScalarType.Long))

    x = x.add(posEmb).add(timeEmb)
    x = dropoutLayer.forward(x)

    // HSTU layers
    layers.foreach { layer =>
      x = layer.forward(x).add(x)  // Residual
    }

    x = layernorm.forward(x)

    // Output projection
    outputProj.forward(x)
  }
}

/**
 * HSTU Layer
 */
class HSTULayer(
  embedDim: Int,
  numHeads: Int,
  dropout: Float,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)
  private val headDim = embedDim / numHeads

  // Unified feature projector (UVQK)
  private val uvqk = new LinearImpl(embedDim, 4 * embedDim)
  uvqk.to(new Device(device),false)
  register_module("uvqk", uvqk)

  // Output projector
  private val o = new LinearImpl(embedDim, embedDim)
  o.to(new Device(device),false)
  register_module("o", o)

  // SiLU activation
  private val layernorm1 = new LayerNormImpl(new LongVector(embedDim.toLong))
  private val layernorm2 = new LayerNormImpl(new LongVector(embedDim.toLong))
  layernorm1.to(new Device(device),false)
  layernorm2.to(new Device(device),false)
  register_module("layernorm1", layernorm1)
  register_module("layernorm2", layernorm2)

  private val dropoutLayer = new DropoutImpl(dropout)

  def forward(x: Tensor): Tensor = {
    val batchSize = x.size(0)
    val seqLen = x.size(1)

    // Normalize and project
    val normed = layernorm1.forward(x)
    val uvqkOut = uvqk.forward(normed)

    // Split into U, V, Q, K
    val u = uvqkOut.narrow(2, 0, embedDim)
    val v = uvqkOut.narrow(2, embedDim, embedDim)
    val q = uvqkOut.narrow(2, 2 * embedDim, embedDim)
    val k = uvqkOut.narrow(2, 3 * embedDim, embedDim)

    // SiLU activation on U
    val uAct = u.mul(u.sigmoid())  // SiLU

    // Reshape for multi-head attention
    val uBatched = uAct.view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val vBatched = v.view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val qBatched = q.view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)
    val kBatched = k.view(batchSize, seqLen, numHeads, headDim).transpose(1, 2)

    // Attention
    val invScale = new Scalar(1.0f / scala.math.sqrt(headDim).toFloat)
    val attnScores = torch.matmul(qBatched, kBatched.transpose(2, 3)).mul(invScale)
    val attnWeights = attnScores.softmax(-1)

    val attnOut = torch.matmul(attnWeights, vBatched)
    val attnFlat = attnOut.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)

    val output = o.forward(dropoutLayer.forward(attnOut))

    layernorm2.forward(x.add(output))
  }
}
