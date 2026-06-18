package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.utils.DeviceSupport
import torchrec.Implicits._

/**
 * Fixed Compressed Interaction Network from xDeepFM.
 * Creates all layers in the constructor so they are properly registered
 * with the module and visible to the optimizer.
 *
 * splitHalf: if true, halves the output size of intermediate layers (standard xDeepFM behavior).
 * Reference: "xDeepFM: Combining Explicit and Implicit Feature Interactions in Recommender Systems"
 * (KDD 2018)
 */
class CINFixed(
  numFields: Int,
  embedDim: Int,
  crossLayerSizes: List[Int],
  splitHalf: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  require(crossLayerSizes.nonEmpty, "crossLayerSizes cannot be empty")
  require(embedDim > 0, "embedDim must be positive")
  require(numFields > 0, "numFields must be positive")

  // Build layers eagerly in constructor so they are registered properly
  private val convLayers = collection.mutable.ListBuffer[org.bytedeco.pytorch.LinearImpl]()
  private val layerOutputDims = collection.mutable.ListBuffer[Int]()
  private var prevOutDim = numFields

  for (i <- crossLayerSizes.indices) {
    val outDim = crossLayerSizes(i)
    val inChannels = numFields * prevOutDim
    val conv = new org.bytedeco.pytorch.LinearImpl(inChannels, outDim)
    register_module(s"conv_$i", conv)
    conv.to(new org.bytedeco.pytorch.Device(device), false)
    convLayers += conv
    layerOutputDims += outDim

    prevOutDim = if (splitHalf && i != crossLayerSizes.length - 1) {
      math.floor(outDim / 2.0).toInt max 1
    } else {
      outDim
    }
  }

  // Final linear projection to scalar
  private val totalDim = crossLayerSizes.sum
  private val fc = new org.bytedeco.pytorch.LinearImpl(totalDim, 1)
  register_module("fc", fc)
  fc.to(new org.bytedeco.pytorch.Device(device), false)

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)

    // x0: original embeddings (stays constant throughout all layers)
    // hk: output of previous layer (transposed)
    val x0 = embeddings
    var hk = x0.transpose(1, 2)  // (batch, embed_dim, num_fields) for first iteration

    val outputs = collection.mutable.ListBuffer[Tensor]()
    var prevOutDim = numFields

    for (i <- crossLayerSizes.indices) {
      // Outer product of x0 (original embeddings) and hk (previous layer output)
      // x0: (batch, F, D) where F=num_fields, D=embed_dim (constant)
      // hk: (batch, D_prev, F) where D_prev is output dim of previous layer
      // Result: (batch, F, F)
      val xh = torch.bmm(x0, hk.transpose(1, 2))
      // Flatten: (batch, F * prev_out_dim)
      val flat = xh.view(batchSize, numFields * prevOutDim)

      // Convolution (1x1): compress to cross_layer_size
      var convOut = convLayers(i).forward(flat)
      convOut = convOut.relu()

      // Split-half: halve for next layer input
      val actualNextDim = if (splitHalf && i != crossLayerSizes.length - 1) {
        math.floor(convOut.size(1) / 2.0).toInt max 1
      } else {
        convOut.size(1).toInt
      }

      // Pool this layer's output and store
      val pooled = convOut.sum(1).unsqueeze(1)  // (batch, 1)
      outputs += pooled

      // Prepare hk for next layer using convOut output
      // x0 stays as original embeddings, only hk changes
      if (i < crossLayerSizes.length - 1 || splitHalf) {
        val half = if (splitHalf && convOut.size(1) > actualNextDim) {
          convOut.narrow(1, 0, actualNextDim)
        } else {
          convOut
        }
        // Reshape: (batch, actual_next_dim) -> (batch, num_fields, actual_next_dim)
        // Then transpose: (batch, actual_next_dim, num_fields)
        hk = half.view(batchSize, numFields, actualNextDim).transpose(1, 2)
      }

      prevOutDim = actualNextDim
    }

    // Concatenate all pooled layer outputs
    val tensorVec = new TensorVector(outputs.size.toLong)
    outputs.foreach(tensorVec.push_back)
    val cinOut = torch.cat(tensorVec, 1L)  // (batch, total_cross_dim)

    fc.forward(cinOut).squeeze(1)  // (batch,)
  }
}
