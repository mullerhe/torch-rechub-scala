package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.jdk.CollectionConverters._
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
  device: String = "cpu"
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
    device: String = "cpu"
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
  private val weightTensor = torch.zeros(Array[Long](1L), null.asInstanceOf[org.bytedeco.pytorch.TensorOptions])
  private val weight = weightTensor

  def forward(x: Tensor): Tensor = {
    torch.prelu(x, weight)
  }
}

/**
 * Leaky ReLU
 */
class LeakyReLUImpl extends Module {
  def forward(x: Tensor): Tensor = torch.leaky_relu(x, new Scalar(0.01f))
}