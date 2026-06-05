package torchrec

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.javacpp.{FloatPointer, DoublePointer, LongPointer}

package object torchrec {
  type Tensor = org.bytedeco.pytorch.Tensor
  type Module = org.bytedeco.pytorch.Module
  type ScalarType = org.bytedeco.pytorch.global.torch.ScalarType

  def cpu(): String = "cpu"
  def cuda(device: Int = 0): String = s"cuda:$device"

  def randn(sizes: Long*): Tensor = torch.randn(sizes.toArray, null.asInstanceOf[TensorOptions])
  def rand(sizes: Long*): Tensor = torch.rand(sizes.toArray, null.asInstanceOf[TensorOptions])
  def zeros(sizes: Long*): Tensor = torch.zeros(sizes.toArray, null.asInstanceOf[TensorOptions])
  def ones(sizes: Long*): Tensor = torch.ones(sizes.toArray, null.asInstanceOf[TensorOptions])

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

  def tensor(data: Array[Long], sizes: Long*): Tensor = {
    val ptr = new LongPointer(data.length)
    var i = 0
    while (i < data.length) {
      ptr.put(i, data(i))
      i += 1
    }
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Long)
    torch.from_blob(ptr, sizes.toArray, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
  }

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

  def arange(start: Int, end: Int, step: Int = 1): Tensor =
    torch.arange(new Scalar(start), new Scalar(end), new Scalar(step))

  def toTensorVector(tensors: Seq[Tensor]): TensorVector = {
    val vec = new TensorVector(tensors.size.toLong)
    tensors.foreach(vec.push_back)
    vec
  }

  def toParameterVector(params: Seq[Tensor]): TensorVector = {
    val pv = new TensorVector(params.size.toLong)
    params.foreach(pv.push_back)
    pv
  }

  implicit class DeviceContext(val device: String) extends AnyVal {
    def asImplicit: String = device
  }

  object Implicits {
    implicit val cpuDevice: String = "cpu"
  }
}
