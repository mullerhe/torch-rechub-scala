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

  private val layerSizes: List[Long] = (0 until numLayers).map { i =>
    math.max(1, embedDim / (i + 1)).toLong
  }.toList

  // Lazily created FC layers (input dim only known at first forward)
  private var fcLayers: List[LinearImpl] = _
  private var lastNumFields: Int = -1

  def forward(embeddings: Tensor): Tensor = {
    val batchSize = embeddings.size(0)
    val numFields = embeddings.size(1)

    // Lazily build layers on first call (input dim depends on numFields)
    if (fcLayers == null || lastNumFields != numFields) {
      fcLayers = List.tabulate(numLayers) { layerIdx =>
        val inDim = numFields * embeddings.size(2)
        val outSize = layerSizes(layerIdx)
        val fc = new LinearImpl(inDim, numFields * outSize)
        register_module(s"fc_$layerIdx", fc)
        fc.to(new org.bytedeco.pytorch.Device(device), false)
        fc
      }
      lastNumFields = numFields.toInt
      
    }

    var xk = embeddings
    val splitResult = embeddings.split(1, 1)
    var hk = splitResult.get(0)

    val outputs = mutable.ListBuffer[Tensor]()
    outputs += hk.squeeze(1)

    for (layerIdx <- 0 until numLayers) {
      val xh = torch.matmul(xk.transpose(1, 2), hk)
      val xhSqueezed = xh.squeeze(1)

      val out = fcLayers(layerIdx).forward(xhSqueezed)

      val activated = activation match {
        case "relu" => out.relu()
        case "sigmoid" => out.sigmoid()
        case "tanh" => out.tanh()
        case _ => out
      }

      val pooled = activated.sum(1)

      val newHk = pooled.unsqueeze(1).repeat(1, numFields, 1)
      xk = xk
      hk = newHk.transpose(1, 2)

      outputs += pooled
    }

    val tensorVec = new TensorVector(outputs.size.toLong)
    outputs.foreach(tensorVec.push_back)
    torch.cat(tensorVec, 1L)
  }
}
