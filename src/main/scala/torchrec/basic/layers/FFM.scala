package torchrec.basic.layers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import torchrec.utils.DeviceSupport
import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq
import torchrec.TensorImplicits.RichTensor

import scala.collection.mutable

/**
 * The Field-aware Factorization Machine module.
 *
 * Parameters
 * ----------
 * numFields : int
 *   Number of feature fields.
 * reduceSum : bool
 *   Whether to sum in embed_dim (default = True).
 *
 * Shape
 * -----
 * Input: ``(batch_size, num_fields, num_fields, embed_dim)``
 * Output: ``(batch_size, num_fields*(num_fields-1)/2, 1)`` or ``(batch_size, num_fields*(num_fields-1)/2, embed_dim)``
 */
class FFM(
  numFields: Int,
  reduceSum: Boolean = true,
  device: String = DeviceSupport.backend
) extends Module {

  def forward(x: Tensor): Tensor = {
    // compute (non-redundant) second order field-aware feature crossings
    val crossedEmbeddings = mutable.ListBuffer[Tensor]()

    for (i <- 0 until numFields - 1; j <- i + 1 until numFields) {
      // x: (batch, num_fields, num_fields, embed_dim)
      // vi, vj: (batch, num_fields, embed_dim) after narrow and squeeze
      val vi = x.select(1, i)  // (batch, num_fields, embed_dim)
      val vj = x.select(1, j)
      // Hadamard product: (batch, num_fields, num_fields, embed_dim)
      crossedEmbeddings += vi * vj
    }

    // Stack: (num_interactions, batch, num_fields, num_fields, embed_dim)
    val stacked = torch.stack(new TensorVector(crossedEmbeddings.toSeq*), 0L)

    // if reduce_sum is true, the crossing operation is effectively inner
    // product, otherwise Hadamard-product
    if (reduceSum) {
      // Sum over embed_dim (last dim): (num_interactions, batch, num_fields, num_fields, 1)
      torch.sum(stacked,Array(-1l) , true,new ScalarTypeOptional())
    } else {
      stacked
    }
  }
}

/**
 * FFM companion object with factory methods.
 */
object FFM {
  def apply(numFields: Int, reduceSum: Boolean = true, device: String = DeviceSupport.backend): FFM = {
    new FFM(numFields, reduceSum, device)
  }
}