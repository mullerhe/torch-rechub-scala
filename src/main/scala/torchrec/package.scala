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

  def randn(sizes: Long*): Tensor =
    torch.randn(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def rand(sizes: Long*): Tensor =
    torch.rand(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def zeros(sizes: Long*): Tensor =
    torch.zeros(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
  def ones(sizes: Long*): Tensor =
    torch.ones(sizes.toArray, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))

  def tensor(data: Array[Float], sizes: Long*): Tensor = {
    val flat = torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    if (sizes.length == 1 && sizes(0) == data.length) flat
    else flat.reshape(sizes.toArray: _*)
  }

  def tensor(data: Array[Int], sizes: Long*): Tensor = {
    val floatData = data.map(_.toFloat)
    tensor(floatData, sizes*)
  }

  def tensor(data: Array[Long], sizes: Long*): Tensor = {
    val n = data.length
    val f = data.map(_.toFloat)
    val flat = torch.tensor(f, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    val t = flat.toType(ScalarType.Long)
    flat.close()
    if (sizes.length == 1 && sizes(0) == n) t
    else { val r = t.reshape(sizes.toArray: _*); t.close(); r }
  }

  def longTensor(data: Array[Long]): Tensor = {
    val n = data.length
    val f = data.map(_.toFloat)
    val flat = torch.tensor(f, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
    val t = flat.toType(ScalarType.Long)
    flat.close()
    t
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
