package torchrec.models.generative

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport

import scala.collection.mutable

class RQVAE(
  embedDim: Int = 256,
  numCodebooks: Int = 8,
  codebookSize: Int = 256,
  latentDim: Int = 64,
  device: String = DeviceSupport.backend
) extends Module {

  private val codebooks = (0 until numCodebooks).map { i =>
    val codebook = new EmbeddingImpl(codebookSize, embedDim)
    register_module(s"codebook_$i", codebook)
    codebook
  }

  def forward(x: Tensor): (Tensor, Tensor, List[Tensor]) = {
    val (zQ, indices) = quantize(x)
    (x, zQ, indices)
  }

  def encode(x: Tensor): Tensor = x.mean(-1)
  def decode(z: Tensor): Tensor = z
  def quantize(x: Tensor): (Tensor, List[Tensor]) = {
    val batchSize = x.size(0).toInt
    val seqLen = if (x.dim() == 3) x.size(1).toInt else 1
    val flat = if (x.dim() == 3) x.view(-1, embedDim) else x
    var residual = flat
    val indices = mutable.ListBuffer[Tensor]()

    for (i <- 0 until numCodebooks) {
      val codebook = codebooks(i)
      val similarities = torch.matmul(residual, codebook.weight.t())
      // Get indices by finding max along dimension 1 using reshape and max
      val maxVals = similarities.max(1)
      val minIdx = maxVals.get1
      val quantized = codebook.weight.index_select(0, minIdx.toType(ScalarType.Long))
      residual = residual.sub(quantized)
      indices += minIdx
    }

    val output = residual.view(batchSize, seqLen, embedDim)

    (output, indices.toList)
  }
}
