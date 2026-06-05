package torchrec

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.javacpp.{FloatPointer, LongPointer}

object TorchRec {
  def tensor(data: Array[Float], sizes: Long*): Tensor = {
    val ptr = new FloatPointer(data.length)
    var i = 0
    while (i < data.length) {
      ptr.put(i, data(i))
      i += 1
    }
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Float)
    torch.from_blob(ptr, sizes.toArray, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
  }

  def tensor(data: Array[Int], sizes: Long*): Tensor = {
    val floatData = data.map(_.toFloat)
    tensor(floatData, sizes*)
  }

  def zeros(sizes: Long*): Tensor = torch.zeros(sizes.toArray, null.asInstanceOf[TensorOptions])
  def ones(sizes: Long*): Tensor = torch.ones(sizes.toArray, null.asInstanceOf[TensorOptions])
  def randn(sizes: Long*): Tensor = torch.randn(sizes.toArray, null.asInstanceOf[TensorOptions])
  def rand(sizes: Long*): Tensor = torch.rand(sizes.toArray, null.asInstanceOf[TensorOptions])

  def longTensor(data: Array[Long]): Tensor = {
    val ptr = new LongPointer(data.length)
    var i = 0
    while (i < data.length) {
      ptr.put(i, data(i))
      i += 1
    }
    val sizesArr = Array(data.length.toLong)
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Long)
    torch.from_blob(ptr, sizesArr, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
  }

  def arange(start: Int, end: Int): Tensor =
    torch.arange(new Scalar(start), new Scalar(end), new Scalar(1))

  def cpu(): String = "cpu"
  def cuda(device: Int = 0): String = s"cuda:$device"

  def relu(x: Tensor): Tensor = x.relu()
  def sigmoid(x: Tensor): Tensor = x.sigmoid()
  def tanh(x: Tensor): Tensor = x.tanh()
  def softmax(x: Tensor, dim: Int): Tensor = x.softmax(dim)
  def log_softmax(x: Tensor, dim: Int): Tensor = x.log_softmax(dim)

  def mean(x: Tensor): Tensor = x.mean()
  def mean(x: Tensor, dim: Int): Tensor = x.mean(dim)
  def sum(x: Tensor): Tensor = x.sum()
  def sum(x: Tensor, dim: Int): Tensor = x.sum(dim)

  def add(x: Tensor, y: Tensor): Tensor = x.add(y)
  def sub(x: Tensor, y: Tensor): Tensor = x.sub(y)
  def mul(x: Tensor, y: Tensor): Tensor = x.mul(y)
  def div(x: Tensor, y: Tensor): Tensor = x.div(y)

  def addScalar(x: Tensor, s: Float): Tensor = x.add(new Scalar(s))
  def subScalar(x: Tensor, s: Float): Tensor = x.sub(new Scalar(s))
  def mulScalar(x: Tensor, s: Float): Tensor = x.mul(new Scalar(s))
  def divScalar(x: Tensor, s: Float): Tensor = x.div(new Scalar(s))

  def pow(x: Tensor, exp: Float): Tensor = x.pow(new Scalar(exp))
  def sqrt(x: Tensor): Tensor = x.sqrt()
  def abs(x: Tensor): Tensor = x.abs()
  def neg(x: Tensor): Tensor = x.neg()
  def exp(x: Tensor): Tensor = x.exp()
  def log(x: Tensor): Tensor = x.log()

  def reshape(x: Tensor, sizes: Long*): Tensor = x.reshape(sizes*)
  def view(x: Tensor, sizes: Long*): Tensor = x.view(sizes*)

  def squeeze(x: Tensor, dim: Int = -1): Tensor =
    if (dim < 0) x.squeeze() else x.squeeze(dim)

  def unsqueeze(x: Tensor, dim: Int): Tensor = x.unsqueeze(dim)
  def flatten(x: Tensor, startDim: Int = 0): Tensor = x.flatten(startDim, x.dim() - 1)
  def transpose(x: Tensor): Tensor = x.t()

  def cat(tensors: Seq[Tensor], dim: Int = 0): Tensor = {
    val vec = new TensorVector(tensors.size.toLong)
    tensors.foreach(vec.push_back)
    torch.cat(vec, dim)
  }

  def stack(tensors: Seq[Tensor], dim: Int = 0): Tensor = {
    val vec = new TensorVector(tensors.size.toLong)
    tensors.foreach(vec.push_back)
    torch.stack(vec, dim)
  }

  def toFloat(x: Tensor): Float = x.item().toFloat
  def toInt(x: Tensor): Int = x.item().toInt
  def toLong(x: Tensor): Long = x.item().toLong

  def toFloatArray(x: Tensor): Array[Float] = {
    if (x.is_contiguous()) {
      val ptr = x.data_ptr().asByteBuffer()
      val size = x.numel().toInt
      val result = new Array[Float](size)
      var i = 0
      while (i < size) {
        result(i) = ptr.getFloat(i * 4)
        i += 1
      }
      result
    } else {
      val cloned = x.clone()
      toFloatArray(cloned)
    }
  }

  type Tensor = org.bytedeco.pytorch.Tensor
  type Module = org.bytedeco.pytorch.Module
  type ScalarType = org.bytedeco.pytorch.global.torch.ScalarType
}
