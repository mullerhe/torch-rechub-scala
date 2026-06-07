package torchrec

import org.bytedeco.pytorch.{Device, Scalar, Tensor, TensorVector}
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.language.implicitConversions

object TensorImplicits {
  implicit class RichTensor(val tensor: Tensor) extends AnyVal {
    def shape: Array[Long] = tensor.shape()
    def ndim: Int = tensor.dim().toInt
    def numel: Long = tensor.numel()

    def reshape(sizes: Array[Long]): Tensor = tensor.reshape(sizes)
    def view(sizes: Array[Long]): Tensor = tensor.view(sizes)

    def squeeze(dim: Int = -1): Tensor =
      if (dim < 0) tensor.squeeze() else tensor.squeeze(dim)

    def unsqueeze(dim: Int): Tensor = tensor.unsqueeze(dim)

    def flatten(startDim: Int = 0, endDim: Int = -1): Tensor =
      if (endDim < 0) tensor.flatten(startDim)
      else tensor.flatten(startDim, endDim)

    def transpose(dim0: Int, dim1: Int): Tensor = tensor.t()

    def clone(): Tensor = tensor.clone()
    def detach(): Tensor = tensor.detach()

    def to(device: String): Tensor = {
      val d = new Device(device)
      tensor.to(d, tensor.dtype())
    }
    def cuda(device: Int = 0): Tensor = {
      val d = new Device(s"cuda:$device")
      tensor.to(d, tensor.dtype())
    }
    def cpu(): Tensor = tensor.cpu()
    def mps(): Tensor = tensor.mps()

    def npu(): Tensor = tensor.npu()
    def xpu(): Tensor = tensor.xpu()
    def toType(dtype: ScalarType): Tensor = tensor.toType(dtype)

    def relu(): Tensor = tensor.relu()
    def sigmoid(): Tensor = tensor.sigmoid()
    def tanh(): Tensor = tensor.tanh()
    def softmax(dim: Int): Tensor = tensor.softmax(dim)
    def log_softmax(dim: Int): Tensor = tensor.log_softmax(dim)

    def exp(): Tensor = tensor.exp()
    def log(): Tensor = tensor.log()
    def sqrt(): Tensor = tensor.sqrt()
    def abs(): Tensor = tensor.abs()
    def neg(): Tensor = tensor.neg()
    def pow(exp: Float): Tensor = tensor.pow(new Scalar(exp))

    def +(other: Tensor): Tensor = tensor.add(other)
    def -(other: Tensor): Tensor = tensor.sub(other)
    def *(other: Tensor): Tensor = tensor.mul(other)
    def /(other: Tensor): Tensor = tensor.div(other)

    def +(scalar: Float): Tensor = tensor.add(new Scalar(scalar))
    def -(scalar: Float): Tensor = tensor.sub(new Scalar(scalar))
    def *(scalar: Float): Tensor = tensor.mul(new Scalar(scalar))
    def /(scalar: Float): Tensor = tensor.div(new Scalar(scalar))

    def mean(): Tensor = tensor.mean()
    def mean(dim: Int): Tensor = tensor.mean(dim)

    def sum(): Tensor = tensor.sum()
    def sum(dim: Int): Tensor = tensor.sum(dim)

    def argmax(): Tensor = tensor.argmax()
    def argmax(dim: Int): Tensor = tensor.argmax(dim)
    def argmin(): Tensor = tensor.argmin()
    def argmin(dim: Int): Tensor = tensor.argmin(dim)

    def masked_fill(mask: Tensor, value: Float): Tensor =
      tensor.masked_fill(mask, new Scalar(value))

    def masked_select(mask: Tensor): Tensor = tensor.masked_select(mask)

    def select(dim: Int, index: Long): Tensor = tensor.select(dim, index)

    def toFloat: Float = try { tensor.item().toFloat } catch { case _: Throwable => 0.0f }
    def toInt: Int = try { tensor.item().toInt } catch { case _: Throwable => 0 }
    def toLong: Long = try { tensor.item().toLong } catch { case _: Throwable => 0L }

    def toFloatArray: Array[Float] = {
      if (tensor.is_contiguous()) {
        val ptr = tensor.data_ptr().asByteBuffer()
        val size = tensor.numel().toInt
        val result = new Array[Float](size)
        var i = 0
        while (i < size) {
          result(i) = ptr.getFloat(i * 4)
          i += 1
        }
        result
      } else {
        clone().toFloatArray
      }
    }

    def toLongArray: Array[Long] = {
      val longTensor = if (tensor.dtype().toScalarType() != ScalarType.Long) {
        tensor.clone().toType(ScalarType.Long)
      } else tensor
      val ptr = longTensor.data_ptr().asByteBuffer()
      val size = tensor.numel().toInt
      val result = new Array[Long](size)
      var i = 0
      while (i < size) {
        result(i) = ptr.getLong(i * 8)
        i += 1
      }
      result
    }

    def print(shape: Boolean = true): Unit = {
      if (shape) println(s"Shape: ${tensor.shape().mkString(", ")}")
    }
  }

  implicit class SeqTensorRichSeq(val tensors: Seq[Tensor]) extends AnyVal {
    def toTensorVector: TensorVector = {
      val vec = new TensorVector(tensors.size.toLong)
      tensors.foreach(vec.push_back)
      vec
    }
  }
}
