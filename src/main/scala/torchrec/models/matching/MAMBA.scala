package torchrec.models.matching

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * MAMBA: State Space Model for Sequential Recommendation
 *
 * A simplified implementation of the Mamba/SSM architecture for recommendation.
 * Uses a selective state space model with input-dependent dynamics.
 * This is a Scala 3 adaptation of the Mamba SSM architecture.
 *
 * Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
 *            (Gu & Dao, 2023) -- adapted for recommendation.
 *
 * Architecture:
 *   Token Input → Input Projection (x, z) → Conv1D (local context)
 *   → SSM Dynamics (A, B, C, dt) → Output Projection → [CLS] pooling → MLP
 *
 * @param vocabSize         Item vocabulary size
 * @param embedDim          Embedding dimension
 * @param dState            SSM state dimension (default 16)
 * @param numLayers         Number of MAMBA layers
 * @param maxSeqLen         Maximum sequence length
 * @param mlpDims           MLP hidden dimensions
 * @param dropout           Dropout rate
 * @param device            Device
 */
class MAMBA(
  vocabSize: Long,
  embedDim: Int = 64,
  dState: Int = 16,
  numLayers: Int = 2,
  maxSeqLen: Int = 50,
  mlpDims: List[Long] = List(128L, 64L),
  dropout: Float = 0.1f,
  device: String = DeviceSupport.backend
) extends Module {

  private val dInner = embedDim * 2  // Inner dimension

  // Token embedding
  private val tokenEmbedding = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim))
  tokenEmbedding.to(new Device(device),false)
  register_module("tokenEmbedding", tokenEmbedding)

  // Learned [CLS] token
  private val clsTensor = {
    val t = torch.zeros(
      Array(1L, 1L, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    t.fill_(new Scalar(0.02f))
    if (device != "cpu") {
      val dev = new org.bytedeco.pytorch.Device(device)
      t.to(dev, ScalarType.Float)
    } else t
  }
  register_parameter("clsToken", clsTensor)

  // MAMBA layers
  private val layers = (0 until numLayers).map { i =>
    val layer = new MAMBABlock(embedDim, dState, dInner, dropout, device)
    register_module(s"mamba_$i", layer)
    layer
  }

  // Layer norm over last dim: use varargs LongVector
  private val layerNorm = {
    val ln = new LayerNormImpl(new LongVector(embedDim.toLong))
    register_module("layerNorm", ln)
    ln
  }

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // Final MLP
  private val mlp = new MLP(embedDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    tokenEmbedding.to(dev, false)
    layers.foreach(_.to(dev, false))
    layerNorm.to(dev, false)
    dropoutLayer.to(dev, false)
    mlp.to(dev, false)
  }

  def forward(
    seqTokens: Tensor,   // (batch, seq_len) -- item IDs
    positions: Tensor     // (batch, seq_len) -- position indices
  ): Tensor = {
    val batchSize = seqTokens.size(0).toInt

    // Token embeddings: (batch, seq_len, embed_dim)
    val tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long))

    // Prepend [CLS]
    val clsBatched = clsTensor.expand(batchSize.toLong, 1L, embedDim.toLong)
    val tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1)

    // MAMBA layers
    var x = tokenEmbWithCls
    x = dropoutLayer.forward(x)
    layers.foreach { layer =>
      x = layer.forward(x).add(x)  // Residual
    }

    // Layer norm
    val normed = layerNorm.forward(x)  // (batch, seq_len+1, embed_dim)

    // Extract [CLS] representation
    val clsRep = normed.select(1, 0)  // (batch, embed_dim)

    // MLP
    val logits = mlp.forward(clsRep)  // (batch, 1)
    logits
  }
}

/**
 * A single MAMBA Block with simplified SSM dynamics.
 *
 * The key insight of Mamba: use input-dependent (selective) SSM parameters.
 * Instead of using fixed A, B, C matrices, they are computed from the input.
 *
 * Structure:
 *   LayerNorm → Input Projection (x, z) → SSM-inspired Recurrence → Gated Residual → Linear
 */
class MAMBABlock(
  embedDim: Int,
  dState: Int,
  dInner: Int,
  dropout: Float,
  device: String
) extends Module {

  // Input projection: embedDim -> dInner * 2 (for x channel and gate)
  private val xProj = new LinearImpl(embedDim, dInner * 2)
  xProj.to(new Device(device),false)
  register_module("xProj", xProj)

  // SSM parameters projection: dInner -> dt_rank + 2*dState
  private val dtRank = scala.math.ceil(embedDim / 16.0).toInt
  private val ssmProj = new LinearImpl(dInner, dtRank + 2 * dState)
  ssmProj.to(new Device(device),false)
  register_module("ssmProj", ssmProj)

  // dt projection: dt_rank -> dInner
  private val dtProj = new LinearImpl(dtRank, dInner)
  dtProj.to(new Device(device),false)
  register_module("dtProj", dtProj)

  // SSM D parameter (skip connection)
  private val dParam = {
    val t = torch.ones(Array(dInner.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    register_parameter("dParam", t)
    t
  }

  // Output projection: dInner -> embedDim
  private val outProj = new LinearImpl(dInner, embedDim)
  outProj.to(new Device(device),false)
  register_module("outProj", outProj)

  // Layer norm over embed_dim
  private val norm = {
    val ln = new LayerNormImpl(new LongVector(embedDim.toLong))
    register_module("norm", ln)
    ln
  }

  // Dropout
  private val dropoutLayer = new DropoutImpl(dropout)
  register_module("dropout", dropoutLayer)

  // A matrix (SSM state transition) - initialized to negative for stability
  // Shape: [dInner, dState]. Use torch.arange to create a clean [dInner, dState] tensor.
  private val aLog = {
    val vals = (0 until dInner * dState).map { i =>
      val d = i % dState + 1
      -math.log(d.toFloat + 1.0f)
    }.toArray
    val t = torch.tensor(vals, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
      .view(dInner.toLong, dState.toLong)
    register_parameter("aLog", t)
    t
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    xProj.to(dev, false)
    ssmProj.to(dev, false)
    dtProj.to(dev, false)
    outProj.to(dev, false)
    norm.to(dev, false)
    dropoutLayer.to(dev, false)
  }

  def forward(x: Tensor): Tensor = {
    // x: (batch, seq_len, embed_dim)
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt

    // Layer norm
    val normed = norm.forward(x)  // (batch, seq_len, embed_dim)

    // Input projection: (batch, seq_len, embed_dim) -> (batch, seq_len, dInner*2)
    val xz = xProj.forward(normed)
    val xHalf = xz.narrow(2, 0, dInner)
    val zHalf = xz.narrow(2, dInner, dInner)
    val gate = zHalf.sigmoid()

    // Local SSM-inspired dynamics: simplified selective SSM
    // Project to SSM parameters
    val ssmParams = ssmProj.forward(xHalf)  // (batch, seq_len, dt_rank + 2*dState)

    val dtPart = ssmParams.narrow(2, 0, dtRank)
    val bPart = ssmParams.narrow(2, dtRank, dState)
    val cPart = ssmParams.narrow(2, dtRank + dState, dState)

    // dt: discretize using SiLU (softplus)
    val dt = dtProj.forward(dtPart)  // (batch, seq_len, dInner)
    val dtAct = torch.silu(dt)  // SiLU discretization

    // A matrix (from log A)
    val a = torch.exp(aLog)  // (dInner, dState) - always positive

    // Simplified SSM: element-wise gating with A-decay
    // h_t = A @ h_{t-1} + B @ x_t (simplified element-wise)
    // Use recurrent approximation: h_t = a_avg * h_{t-1} + x_t
    val aAvg = a.mean(1).unsqueeze(1)  // (dInner, 1) — average A across state dimension
    val aRep = aAvg.expand(batchSize.toLong, seqLen.toLong, dInner.toLong)

    // Recurrence: apply A decay along sequence
    val xHalfT = xHalf.transpose(0, 1)  // (seq_len, batch, dInner)
    var h = xHalfT.select(0, 0)  // (batch, dInner) — init with first position
    val hiddenSeq = mutable.ArrayBuffer[Tensor]()
    hiddenSeq += h

    for (pos <- 1 until seqLen) {
      val xPos = xHalfT.select(0, pos)
      val aDecay = aRep.select(1, pos)  // (batch, dInner)
      h = h.mul(aDecay).add(xPos)
      hiddenSeq += h
    }

    // Stack hidden states: (seq_len, batch, dInner)
    val hiddenStacked = hiddenSeq(0).unsqueeze(0)
    for (pos <- 1 until seqLen) {
      val expanded = hiddenSeq(pos).unsqueeze(0)
      val combined = torch.cat(new TensorVector(hiddenStacked, expanded), 0)
      hiddenSeq(pos - 1).close()
      if (pos > 1) hiddenStacked.close()
    }

    val ssmOut = hiddenStacked.transpose(0, 1)  // (batch, seq_len, dInner)

    // Add skip connection (D parameter)
    val skipOut = xHalf.mul(dParam)

    // Output projection with gate
    val out = outProj.forward(ssmOut.add(skipOut)).mul(gate.unsqueeze(1))

    // Residual
    val result = dropoutLayer.forward(out)
    hiddenStacked.close()
    result
  }
}
