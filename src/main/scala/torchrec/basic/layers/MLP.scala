package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

import scala.jdk.CollectionConverters.*
import scala.collection.mutable

/**
 * Multi-Layer Perceptron
 */
class MLP(
  inputDim: Long,
  hiddenDims: List[Long],
  outputDim: Long = 1,
  activation: String = "relu",
  dropout: Float = 0.0f,
  useBatchNorm: Boolean = false,
  useLayerNorm: Boolean = false,
  device: String = DeviceSupport.backend
) extends Module {

  private val layers = mutable.ListBuffer[Module]()
  private var prevDim = inputDim

  // Build MLP layers
  hiddenDims.foreach { dim =>
    val linear = new LinearImpl(prevDim, dim)
    register_module(s"linear_${layers.size}", linear)
    layers += linear

    if (useLayerNorm) {
      val ln = new LayerNormImpl(new LongVector(dim))
      register_module(s"ln_${layers.size}", ln)
      layers += ln
    } else if (useBatchNorm && !activation.equals("relu")) {
      val bn = new BatchNorm1dImpl(new BatchNormOptions(dim))
      register_module(s"bn_${layers.size}", bn)
      layers += bn
    }

    layers += createActivation(activation)

    if (dropout > 0) {
      val drop = new DropoutImpl(dropout)
      register_module(s"dropout_${layers.size}", drop)
      layers += drop
    }

    prevDim = dim
  }

  // Output layer
  val outputLinear = new LinearImpl(prevDim, outputDim)
  register_module("output", outputLinear)
  layers += outputLinear

  // Move all layers to target device
  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    layers.foreach { m =>
      m.to(dev, false)
    }
    this.to(dev, false)
  }

  def forward(x: Tensor): Tensor = {
    var result: Tensor = x
    layers.foreach { layer =>
      result = layer match {
        case mlp: LinearImpl => mlp.forward(result)
        case bn: BatchNorm1dImpl => bn.forward(result)
        case ln: LayerNormImpl => ln.forward(result)
        case act: ReLUImpl => act.forward(result)
        case act: SigmoidImpl => act.forward(result)
        case act: TanhImpl => act.forward(result)
        case act: SiLUImpl => act.forward(result)
        case act: GELUImpl => act.forward(result)
        case drop: DropoutImpl => drop.forward(result)
        case _ => result
      }
    }
    result
  }

  private def createActivation(act: String): Module = {
    act.toLowerCase match {
      case "relu" => new ReLUImpl()
      case "sigmoid" => new SigmoidImpl()
      case "tanh" => new TanhImpl()
      case "silu" | "swish" => new SiLUImpl()
      case "gelu" => new GELUImpl()
      case "prelu" => new PReLUImpl()
      case "leaky_relu" | "leakyrelu" => new LeakyReLUImpl()
      case "none" | "identity" => new IdentityImpl()
      case _ => new ReLUImpl()
    }
  }
}

/**
 * DNN (alias for MLP)
 */
object DNN {
  def apply(
    inputDim: Long,
    hiddenDims: List[Long],
    outputDim: Long = 1,
    activation: String = "relu",
    dropout: Float = 0.0f,
    useBatchNorm: Boolean = false,
    device: String = DeviceSupport.backend
  ): MLP = new MLP(inputDim, hiddenDims, outputDim, activation, dropout, useBatchNorm, device = device)
}

/**
 * Identity layer (no-op)
 */
class IdentityImpl extends Module {
  def forward(x: Tensor): Tensor = x
}

/**
 * PReLU activation
 */
class PReLUImpl extends Module {
  private var weight: Tensor = _

  def forward(x: Tensor): Tensor = {
    if (weight == null || weight.device() != x.device()) {
      if (weight != null) weight.close()
      weight = torch.zeros(Array[Long](1L), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
        .to(x.device(), ScalarType.Float)
    }
    torch.prelu(x, weight)
  }
}

/**
 * Leaky ReLU
 */
class LeakyReLUImpl extends Module {
  def forward(x: Tensor): Tensor = torch.leaky_relu(x, new Scalar(0.01f))
}

/**
 * Dice activation function
 * Reference: Alibaba DIN paper, KDD 2018
 *
 * Formula: output = p * x + (1 - p) * alpha * x
 * where p = sigmoid(beta * bn(x)), and alpha, beta are learnable
 *
 * Input: (batch, embed) or (batch, seq, embed)
 * Output: same shape as input
 */
class DiceActivation(
  val embedSize: Int,
  val eps: Float = 1e-8f
) extends Module {

  private val bn = new BatchNorm1dImpl(new BatchNormOptions(embedSize))
  register_module("bn", bn)

  // Learnable alpha and beta per dimension
  private val alpha = torch.zeros(Array[Long](embedSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  private val beta = torch.zeros(Array[Long](embedSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  register_parameter("alpha", alpha)
  register_parameter("beta", beta)

  def forward(x: Tensor): Tensor = {
    val dim = x.dim()
    val batchSize = x.size(0)

    if (dim == 2L) {
      // 2D: (batch, embed)
      val bnOut = bn.forward(x)
      val p = bnOut.mul(beta).sigmoid()
      val alphaB = alpha.reshape(1L, embedSize.toLong).expand(batchSize, embedSize.toLong)
      // Dice: alpha * (1-p) * x + p * x
      val term1 = alphaB.mul(p.neg().add(new Scalar(1.0)))
      val term2 = p.mul(x)
      term1.add(term2)
    } else {
      // 3D: (batch, seq, embed)
      val seqLen = x.size(1)
      val xFlat = x.transpose(1, 2).reshape(batchSize * seqLen, embedSize.toLong)
      val bnOut = bn.forward(xFlat)
      val p = bnOut.mul(beta).sigmoid()
      val alphaB = alpha.reshape(1L, embedSize.toLong).expand(batchSize * seqLen, embedSize.toLong)
      val term1 = alphaB.mul(p.neg().add(new Scalar(1.0)))
      val term2 = p.mul(xFlat)
      term1.add(term2).reshape(batchSize, embedSize.toLong, seqLen).transpose(1, 2)
    }
  }
}