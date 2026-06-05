package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

import torchrec.Implicits._

/**
 * Compressed Interaction Network from xDeepFM
 */
class CIN(
  embedDim: Int = 8,
  numLayers: Int = 3,
  activation: String = "relu",
  device: String = "cpu"
) extends Module {

  private val fcLayers = mutable.ListBuffer[LinearImpl]()

  // CIN layer sizes (number of filters at each layer)
  private val layerSizes: List[Long] = (0 until numLayers).map { i =>
    math.max(1, embedDim / (i + 1)).toLong
  }.toList

  def forward(embeddings: Tensor): Tensor = {
    // embeddings: (batch_size, num_fields, embed_dim)
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)

    var xk = embeddings // (batch, num_fields, embed_dim)
    val splitResult = embeddings.split(1, 1)
    var hk = splitResult.get(0) // (batch, 1, embed_dim)

    val outputs = mutable.ListBuffer[Tensor]()
    outputs += hk.squeeze(1) // First layer output

    for (layerIdx <- 0 until numLayers) {
      val xh = torch.matmul(xk.transpose(1, 2), hk) // (batch, embed, fields_k * fields_0)
      val xhSqueezed = xh.squeeze(1) // (batch, embed * fields_prev)

      // Reshape for convolution
      val outSize = layerSizes(layerIdx)
      val fc = new LinearImpl(xhSqueezed.size(1), numFields * outSize)
      register_module(s"fc_$layerIdx", fc)

      val out = fc.forward(xhSqueezed)
      val reshaped = out.view(batchSize, numFields, outSize) // (batch, fields, out_size)

      // Activation
      val activated = activation match {
        case "relu" => out.relu()
        case "sigmoid" => out.sigmoid()
        case "tanh" => out.tanh()
        case _ => out
      }

      // Sum pooling over field dimension
      val pooled = activated.sum(1) // (batch, out_size)

      // Prepare for next layer
      val newHk = pooled.unsqueeze(1).repeat(1, numFields, 1) // (batch, fields, out_size)
      xk = xk // Keep x_0
      hk = newHk.transpose(1, 2) // (batch, out_size, fields) for next iteration

      outputs += pooled
    }

    // Concat all layer outputs and reduce
    val tensorVec = new TensorVector(outputs.size.toLong)
    outputs.foreach(tensorVec.push_back)
    torch.cat(tensorVec, 1L)
  }
}
