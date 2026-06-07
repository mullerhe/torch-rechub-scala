package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import torchrec.utils.DeviceSupport

/**
 * Squeeze-and-Excitation Network Layer
 * Used in FiBiNet for feature gating
 */
class SENETLayer(
  reduction: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private var inputDim: Long = 0

  private val fc1 = new LinearImpl(0, 0) // Placeholder
  private val fc2 = new LinearImpl(0, 0) // Placeholder

  def setInputDim(dim: Long): this.type = {
    if (inputDim != dim) {
      inputDim = dim
      val reducedDim = math.max(1, dim / reduction).toLong

      val fc1New = new LinearImpl(dim, reducedDim)
      val fc2New = new LinearImpl(reducedDim, dim)

      register_module("fc1", fc1New)
      register_module("fc2", fc2New)

      fc1New.to(new org.bytedeco.pytorch.Device(device), false)
      fc2New.to(new org.bytedeco.pytorch.Device(device), false)
    }
    this
  }

  def forward(x: Tensor): Tensor = {
    // x: (batch_size, num_fields, embed_dim)
    if (inputDim == 0) setInputDim(x.size(2))

    val batchSize = x.size(0)
    val numFields = x.size(1)
    val embedDim = x.size(2)

    // Squeeze: global average pooling
    val squeezed = x.mean(1) // (batch, embed_dim)

    // Excitation: FC -> ReLU -> FC -> Sigmoid
    val excited = fc1.forward(squeezed).relu()
    val out = fc2.forward(excited).sigmoid()

    // Scale: multiply back to original shape
    val scaled = x.mul(out.view(batchSize, 1, embedDim))

    scaled
  }
}

/**
 * Context-Adjusted SE (CARD) - SE with context vector
 */
class ContextSENET(
  reduction: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private var inputDim: Long = 0

  private val context_proj = new LinearImpl(0, 0) // Placeholder
  private val fc1 = new LinearImpl(0, 0)
  private val fc2 = new LinearImpl(0, 0)

  def setInputDim(dim: Long, contextDim: Long): this.type = {
    if (inputDim != dim) {
      inputDim = dim
      val reducedDim = math.max(1, dim / reduction).toLong

      val cp = new LinearImpl(contextDim, dim)
      val fc1New = new LinearImpl(dim, reducedDim)
      val fc2New = new LinearImpl(reducedDim, dim)

      register_module("context_proj", cp)
      register_module("fc1", fc1New)
      register_module("fc2", fc2New)

      cp.to(new org.bytedeco.pytorch.Device(device), false)
      fc1New.to(new org.bytedeco.pytorch.Device(device), false)
      fc2New.to(new org.bytedeco.pytorch.Device(device), false)
    }
    this
  }

  def forward(x: Tensor, context: Tensor): Tensor = {
    // x: (batch, num_fields, embed)
    // context: (batch, context_dim)
    if (inputDim == 0) setInputDim(x.size(2), context.size(1))

    val batchSize = x.size(0)
    val numFields = x.size(1)
    val embedDim = x.size(2)

    // Project context to field dimension
    val projected = context_proj.forward(context) // (batch, embed)
    val weighted = x.mul(projected.view(batchSize, 1, embedDim))

    // Squeeze
    val squeezed = weighted.mean(1)

    // Excitation
    val excited = fc1.forward(squeezed).relu()
    val out = fc2.forward(excited).sigmoid()

    weighted.mul(out.view(batchSize, 1, embedDim))
  }
}
