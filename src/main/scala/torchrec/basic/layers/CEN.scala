package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport

/**
 * The Compose-Excitation Network module, mentioned in the FAT-DeepFFM paper.
 * A modified version of Squeeze-and-Excitation Network (SENet).
 * It is used to highlight the importance of second-order feature crosses.
 *
 * Parameters
 * ----------
 * embedDim : int
 *   The dimensionality of categorical value embedding.
 * numFieldCrosses : int
 *   The number of second order crosses between feature fields.
 * reductionRatio : int
 *   The ratio between the dimensions of input layer and hidden layer of the MLP module.
 *
 * Shape
 * -----
 * Input: ``(batch_size, num_field_crosses, num_field_crosses, embed_dim)``
 * Output: ``(batch_size, num_field_crosses * embed_dim)``
 */
class CEN(
  embedDim: Int,
  numFieldCrosses: Int,
  reductionRatio: Int,
  device: String = DeviceSupport.backend
) extends Module {

  // Convolution weight (Eq.7 FAT-DeepFFM)
  // u shape: (num_field_crosses, embed_dim)
  private val u = torch.rand(Array(numFieldCrosses.toLong, embedDim.toLong),
    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  register_parameter("u", u)

  // Two FC layers that computes the field attention
  private val mlpAttention = new MLP(
    inputDim = numFieldCrosses,
    hiddenDims = List((numFieldCrosses / reductionRatio).toLong),
    outputDim = numFieldCrosses,
    activation = "relu",
    useBatchNorm = false,
    useLayerNorm = false,
    outputLayer = false,
    device = device
  )
  register_module("mlp_att", mlpAttention)

  def forward(em: Tensor): Tensor = {
    // em: (batch, num_field_crosses, num_field_crosses, embed_dim)
    // u: (num_field_crosses, embed_dim) -> squeeze(0) -> (embed_dim)
    // uSqueezed * em: (batch, num_field_crosses, num_field_crosses, embed_dim)
    // sum(-1): (batch, num_field_crosses, num_field_crosses)
    val uSqueezed = u.squeeze(0)
    val d = torch.relu(uSqueezed.mul(em).sum(-1))

    // Compute field attention (Eq.9), output shape [batch_size, num_field_crosses]
    val s = mlpAttention.forward(d)

    // Rescale original embedding with field attention (Eq.10)
    // s: (batch, num_field_crosses, num_field_crosses)
    // s.unsqueeze(-1): (batch, num_field_crosses, num_field_crosses, 1)
    // aem: (batch, num_field_crosses, num_field_crosses, embed_dim)
    val aem = s.unsqueeze(-1).mul(em)
    // flatten from dim 1: (batch, num_field_crosses * num_field_crosses * embed_dim)
    aem.flatten(1L,1l)
  }
}

/**
 * CEN companion object with factory methods.
 */
object CEN {
  def apply(embedDim: Int, numFieldCrosses: Int, reductionRatio: Int, device: String = DeviceSupport.backend): CEN = {
    new CEN(embedDim, numFieldCrosses, reductionRatio, device)
  }
}