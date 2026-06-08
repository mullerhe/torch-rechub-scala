package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.RichTensor
import torchrec.utils.DeviceSupport

/**
 * Cross Network from DCN
 * x_{l+1} = x_0 * W_l^T * h(x_l) + b_l + x_l
 */
class CrossNetwork(
  inputDim: Long,
  numLayers: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private val weights = List.tabulate(numLayers) { i =>
    val w = new LinearImpl(inputDim, 1)
    register_module(s"weight_$i", w)
    w.to(new org.bytedeco.pytorch.Device(device), false)
    w
  }

  private var biases: List[Tensor] = _

  def forward(x0: Tensor): Tensor = {
    if (biases == null) {
      biases = List.tabulate(numLayers) { _ =>
        torch.zeros(1L).to(x0.device(), ScalarType.Float)
      }
    }
    var xl = x0
    for (i <- 0 until numLayers) {
      val wtxl = weights(i).forward(xl) // (batch, 1)
      val dot = x0.mul(wtxl).squeeze(1) // (batch, dim)
      xl = dot.add(biases(i)).add(xl)
    }
    xl
  }
}

/**
 * Cross Network V2 from DCN v2
 */
class CrossNetV2(
  inputDim: Long,
  numLayers: Int = 3,
  device: String = DeviceSupport.backend
) extends Module {

  private val weights = List.tabulate(numLayers) { i =>
    val w = new LinearImpl(inputDim, inputDim)
    register_module(s"weight_$i", w)
    w.to(new org.bytedeco.pytorch.Device(device), false)
    w
  }

  def forward(x0: Tensor): Tensor = {
    var xl = x0
    for (i <- 0 until numLayers) {
      // x_{l+1} = x_0 * sigmoid(W_l * x_l) + x_l
      val wxl = weights(i).forward(xl)
      val sig = wxl.sigmoid()
      xl = x0.mul(sig).add(xl)
    }
    xl
  }
}

/**
 * Cross Network Mix from DCN v2 (with low-rank approximation)
 */
class CrossNetMix(
  inputDim: Long,
  numLayers: Int = 3,
  lowRank: Int = 4,
  device: String = DeviceSupport.backend
) extends Module {

  private val U_kernels = List.tabulate(numLayers) { i =>
    val u = new LinearImpl(inputDim, lowRank)
    register_module(s"U_$i", u)
    u.to(new org.bytedeco.pytorch.Device(device), false)
    u
  }

  private val V_kernels = List.tabulate(numLayers) { i =>
    val v = new LinearImpl(inputDim, lowRank)
    register_module(s"V_$i", v)
    v.to(new org.bytedeco.pytorch.Device(device), false)
    v
  }

  private val C_kernels = List.tabulate(numLayers) { i =>
    val c = new LinearImpl(lowRank, inputDim)
    register_module(s"C_$i", c)
    c.to(new org.bytedeco.pytorch.Device(device), false)
    c
  }

  def forward(x0: Tensor): Tensor = {
    var xl = x0
    for (i <- 0 until numLayers) {
      val uxl = U_kernels(i).forward(xl)           // (batch, rank)
      val vxl = V_kernels(i).forward(xl)            // (batch, rank)
      val uvxl = uxl.mul(vxl)                      // (batch, rank) element-wise
      // unsqueeze(1): (batch, rank) -> (batch, 1, rank)
      // C(rank, dim) @ (batch, 1, rank)^T: (batch, 1, rank) @ (rank, dim) -> (batch, 1, dim)
      // squeeze(1) -> (batch, dim); then x0 * cx + xl
      val uvxl1d = uvxl.unsqueeze(1)  // (batch, 1, rank)
      val cx = C_kernels(i).forward(uvxl1d).squeeze(1)  // (batch, dim)
      xl = x0.mul(cx).add(xl)
    }
    xl
  }
}
