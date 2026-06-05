package torchrec

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import org.bytedeco.javacpp.{FloatPointer, LongPointer}

import scala.language.implicitConversions

object Implicits {
  def tensor(data: Array[Float], sizes: Array[Long]): Tensor = {
    val ptr = new FloatPointer(data.length)
    var i = 0
    while (i < data.length) {
      ptr.put(i, data(i))
      i += 1
    }
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Float)
    torch.from_blob(ptr, sizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
  }

  def zeros(sizes: Array[Long]): Tensor = torch.zeros(sizes, null.asInstanceOf[TensorOptions])
  def ones(sizes: Array[Long]): Tensor = torch.ones(sizes, null.asInstanceOf[TensorOptions])
  def randn(sizes: Array[Long]): Tensor = torch.randn(sizes, null.asInstanceOf[TensorOptions])
  def rand(sizes: Array[Long]): Tensor = torch.rand(sizes, null.asInstanceOf[TensorOptions])

  def longTensor(data: Array[Long]): Tensor = {
    val ptr = new LongPointer(data.length)
    var i = 0
    while (i < data.length) {
      ptr.put(i, data(i))
      i += 1
    }
    val sizesArr = new Array[Long](1)
    sizesArr(0) = data.length.toLong
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Long)
    torch.from_blob(ptr, sizesArr, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
  }

  def floatTensor(data: Array[Float]): Tensor = {
    val ptr = new FloatPointer(data.length)
    var i = 0
    while (i < data.length) {
      ptr.put(i, data(i))
      i += 1
    }
    val sizesArr = new Array[Long](1)
    sizesArr(0) = data.length.toLong
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Float)
    torch.from_blob(ptr, sizesArr, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
  }

  def toTensorVector(tensors: Seq[Tensor]): TensorVector = {
    val vec = new TensorVector(tensors.size.toLong)
    tensors.foreach(vec.push_back)
    vec
  }

  def toParameterVector(params: Seq[Tensor]): TensorVector = {
    val vec = new TensorVector(params.size.toLong)
    params.foreach(vec.push_back)
    vec
  }

  def cpu(): String = "cpu"
  def cuda(device: Int = 0): String = s"cuda:$device"

  implicit class RichTensor(val tensor: Tensor) extends AnyVal {
    def shape: Array[Long] = tensor.shape()
    def ndim: Int = tensor.dim().toInt
    def numel: Long = tensor.numel()

    def clone(): Tensor = tensor.clone()
    def detach(): Tensor = tensor.detach()

    def to(device: String): Tensor = tensor.to(new Device(device), tensor.dtype())

    def toType(dtype: ScalarType): Tensor = tensor.toType(dtype)

    def mean(): Tensor = tensor.mean()
    def sum(): Tensor = tensor.sum()
    def toFloat: Float = tensor.item().toFloat

    def toFloatArray: Array[Float] = {
      val contig = if (tensor.is_contiguous()) tensor else tensor.contiguous()
      val size = contig.numel().toInt
      if (size == 0) return Array.empty[Float]
      try {
        val ptr = contig.data_ptr()
        val buf = ptr.asByteBuffer()
        buf.limit(size * 4)
        val floatBuf = buf.asFloatBuffer()
        val result = new Array[Float](size)
        var i = 0
        while (i < size) {
          result(i) = floatBuf.get(i)
          i += 1
        }
        result
      } catch {
        case _: Exception =>
          val result = new Array[Float](size)
          var i = 0
          while (i < size) {
            result(i) = contig.select(0, i).item().toFloat
            i += 1
          }
          result
      }
    }
  }

  implicit class SeqTensorRichSeq(val tensors: Seq[Tensor]) extends AnyVal {
    def toTensorVector: TensorVector = {
      val vec = new TensorVector(tensors.size.toLong)
      tensors.foreach(vec.push_back)
      vec
    }

    def cat(dim: Int = 0): Tensor = torch.cat(toTensorVector, dim)
    def stack(dim: Int = 0): Tensor = torch.stack(toTensorVector, dim)
  }
}

type TRTensor = Tensor
type TRModule = Module
type TRScalarType = ScalarType
