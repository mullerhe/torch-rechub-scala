package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

/**
 * Anova Kernel for computing high-order polynomial interactions.
 *
 * Implements the ANOVA kernel expansion for polynomial interactions of configurable order.
 * Each order-k term represents the interaction between k features.
 *
 * Reference: "Factorization Machines" (Rendle, 2010) - High-order extension
 *
 * The kernel uses a dynamic programming approach with cumsum for efficiency:
 *   a_t = sum_{S subset of {1..m}, |S|=t} prod_{i in S} x_i
 *
 * @param order      The polynomial order (2 = FM, 3 = 3rd order, etc.)
 * @param embedDim   Embedding dimension
 * @param reduceSum  If true, sum over the final dimension; otherwise return full tensor
 * @param device     Device to run on
 */
class AnovaKernel(
  order: Int,
  embedDim: Int,
  reduceSum: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  require(order >= 2, s"order must be >= 2, got $order")
  require(embedDim > 0, s"embedDim must be positive, got $embedDim")

  // No learnable parameters - pure tensor operations

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)
    val eDim = embeddings.size(2)

    val dev = embeddings.device()

    // Initialize: a_prev starts as all-ones tensor (batch, num_fields+1, embed_dim)
    val onesOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
    val zerosOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))

    var aPrev = torch.ones(Array(batchSize, numFields + 1, eDim), onesOpts).to(dev, ScalarType.Float)
    var a = torch.zeros(Array(batchSize, numFields + 1, eDim), zerosOpts).to(dev, ScalarType.Float)

    // Iterative kernel computation
    // For each order t:
    //   a[:, t+1, :] += embeddings[:, t, :] * a_prev[:, t, :]
    //   a = cumsum(a, dim=1)
    //   a_prev = a
    for (t <- 0 until order) {
      val aPrevSlice = aPrev.narrow(1, t, 1).squeeze(1)
      val embSlice = embeddings.narrow(1, t, 1).squeeze(1)
      val product = aPrevSlice.mul(embSlice)

      // Add to a at index t+1
      val aTarget = a.narrow(1, t + 1, 1).squeeze(1)
      val updatedASlice = aTarget.add(product)
      a.narrow(1, t + 1, 1).squeeze(1).copy_(updatedASlice)

      // Cumulative sum along field dimension
      a = torch.cumsum(a, 1L)

      // Update aPrev for next iteration
      aPrev = a
    }

    // Output: a[:, order, :] (0-indexed at index order)
    val result = a.narrow(1, order, 1).squeeze(1)

    if (reduceSum) {
      val summed = result.sum(1L)
      summed.squeeze(1)
    } else {
      result
    }
  }
}
