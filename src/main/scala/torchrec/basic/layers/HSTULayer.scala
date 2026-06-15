package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * Single HSTU "sequential transduction unit" layer (paper-faithful).
 *
 * Implements HSTU Eq. 2-4 from *Actions Speak Louder than Words*:
 * - Eq. 2 — U, V, Q, K = Split(SiLU(f_1(LayerNorm(x))))
 * - Eq. 3 — A(x) V(x) = (SiLU(Q K^T * alpha + rab^{p,t}) / N) V(x)
 * - Eq. 4 — Y(x) = f_2(LayerNorm(A V) * U)
 *
 * The external residual x + Layer(x) is added by HSTUBlock.
 *
 * Parameters
 * ----------
 * dModel : int
 *   Hidden dimension. Must be divisible by n_heads.
 * nHeads : int
 *   Number of attention heads.
 * dqk : int
 *   Query/key dim per head.
 * dv : int
 *   Value/u dim per head.
 * dropout : float
 *   Dropout applied to the gated attention output before the final projection.
 * maxSeqLen : int
 *   Used for the silu(scores) / max_seq_len scaling.
 * numTimeBuckets : int
 *   Number of time-difference buckets in rab.
 * timeBucketFn : String
 *   'sqrt' or 'log' bucketization.
 * timeBucketDivisor : float
 *   Bucket-range divisor.
 * timeBucketUnit : String
 *   'minutes' or 'seconds'.
 *
 * Shape
 * -----
 * Input: x: (batch_size, seq_len, d_model)
 *        padding_mask: (batch_size, seq_len) bool, True for valid tokens
 *        time_diffs: (batch_size, seq_len) seconds delta
 * Output: (batch_size, seq_len, d_model)
 */
class HSTULayer(
  dModel: Int = 512,
  nHeads: Int = 8,
  dqk: Int = 64,
  dv: Int = 64,
  dropout: Float = 0.1f,
  maxSeqLen: Int = 200,
  numTimeBuckets: Int = 128,
  timeBucketFn: String = "sqrt",
  timeBucketDivisor: Float = 1.0f,
  timeBucketUnit: String = "minutes",
  device: String = DeviceSupport.backend
) extends Module {

  require(dModel % nHeads == 0, s"d_model ($dModel) must be divisible by n_heads ($nHeads)")

  private val attnAlpha = 1.0f / math.sqrt(dqk).toFloat

  private val normIn = new LayerNormImpl(new LongVector(dModel.toLong))
  register_module("norm_in", normIn)
  normIn.to(new org.bytedeco.pytorch.Device(device), false)

  private val proj1 = new LinearImpl(dModel, 2 * nHeads * dqk + 2 * nHeads * dv)
  register_module("proj1", proj1)
  proj1.to(new org.bytedeco.pytorch.Device(device), false)

  private val rab = new RelativeBucketedTimeAndPositionBias(
    nHeads = nHeads,
    maxSeqLen = maxSeqLen,
    numTimeBuckets = numTimeBuckets,
    timeBucketFn = timeBucketFn,
    timeBucketDivisor = timeBucketDivisor,
    timeBucketUnit = timeBucketUnit
  )
  register_module("rab", rab)

  private val normAttn = new LayerNormImpl(new LongVector(nHeads * dv.toLong))
  register_module("norm_attn", normAttn)
  normAttn.to(new org.bytedeco.pytorch.Device(device), false)

  private val proj2 = new LinearImpl(nHeads * dv, dModel)
  register_module("proj2", proj2)
  proj2.to(new org.bytedeco.pytorch.Device(device), false)

  private val dropoutLayer = new DropoutImpl(dropout)

  def forward(
    x: Tensor,
    paddingMask: Option[Tensor] = None,
    timeDiffs: Option[Tensor] = None
  ): Tensor = {
    val batchSize = x.size(0).toInt
    val seqLen = x.size(1).toInt
    val H = nHeads

    val xNormed = normIn.forward(x)

    // Eq. 2: SiLU on the whole UVQK projection BEFORE split
    val projOut = torch.silu(proj1.forward(xNormed))

    val q = projOut.narrow(2, 0, H * dqk).reshape(batchSize, seqLen, H, dqk).transpose(1, 2)
    val k = projOut.narrow(2, H * dqk, H * dqk).reshape(batchSize, seqLen, H, dqk).transpose(1, 2)
    val u = projOut.narrow(2, 2 * H * dqk, H * dv).reshape(batchSize, seqLen, H, dv)
    val v = projOut.narrow(2, 2 * H * dqk + H * dv, H * dv).reshape(batchSize, seqLen, H, dv).transpose(1, 2)

    // scores: (B, H, L, L)
    val scores = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(attnAlpha))

    // Eq. 3: rab^{p,t} is added to scores BEFORE the SiLU/N activation
    val rabBias = rab.forward(timeDiffs, seqLen)
    val scoresWithBias = scores.add(rabBias)

    // Causal mask
    val causalMask = torch.tril(torch.ones(Array(seqLen.toLong, seqLen.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(x.device()))))
    val validMask = causalMask.unsqueeze(0).unsqueeze(0)

    val finalMask = if (paddingMask.isDefined) {
      val keyMask = paddingMask.get.unsqueeze(1).unsqueeze(1)
      torch.logical_and(validMask, keyMask)
    } else {
      validMask
    }

    val maskedScores = scoresWithBias.masked_fill(torch.eq(finalMask, new Scalar(0)), new Scalar(-1e4f))

    // HSTU: silu then / max_seq_len
    val attnWeights = torch.silu(maskedScores).div(new Scalar(maxSeqLen.toFloat))

    val attnOutput = torch.matmul(attnWeights, v)
    val attnOutputReshaped = attnOutput.transpose(1, 2).reshape(batchSize, seqLen, H * dv)
    val uFlat = u.reshape(batchSize, seqLen, H * dv)

    // Eq. 4: U is already SiLU'd via the joint pre-split SiLU above
    val gated = normAttn.forward(attnOutputReshaped).mul(uFlat)
    val dropped = dropoutLayer.forward(gated)
    proj2.forward(dropped)
  }
}

/**
 * HSTULayer companion object with factory methods.
 */
object HSTULayer {
  def apply(
    dModel: Int = 512,
    nHeads: Int = 8,
    dqk: Int = 64,
    dv: Int = 64,
    dropout: Float = 0.1f,
    maxSeqLen: Int = 200,
    numTimeBuckets: Int = 128,
    timeBucketFn: String = "sqrt",
    timeBucketDivisor: Float = 1.0f,
    timeBucketUnit: String = "minutes",
    device: String = DeviceSupport.backend
  ): HSTULayer = {
    new HSTULayer(dModel, nHeads, dqk, dv, dropout, maxSeqLen, numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device)
  }
}