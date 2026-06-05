package torchrec.basic.losses

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.immutable.Seq

import torchrec.Implicits.SeqTensorRichSeq
import torchrec.Implicits.RichTensor

class BCELoss(
  reduction: String = "mean",
  posWeight: Option[Tensor] = None
) {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    // Compute BCE without using clamp - clip manually
    val epsTensor = torch.full(Array(1L), new Scalar(1e-7f))
    val oneMinusEpsTensor = torch.full(Array(1L), new Scalar(1.0f - 1e-7f))
    val oneTensor = torch.full(Array(1L), new Scalar(1.0f))
    val clipped = predictions.maximum(epsTensor).minimum(oneMinusEpsTensor)
    val logClipped = clipped.log()
    val oneMinusClipped = clipped.neg().add(oneTensor)
    val logOneMinusClipped = oneMinusClipped.log()
    val loss = targets.mul(logClipped.neg()).add(targets.neg().add(oneTensor).mul(logOneMinusClipped.neg()))

    reduction match {
      case "sum" => loss.sum()
      case "none" => loss
      case _ => loss.mean()
    }
  }

  def apply(predictions: Array[Float], targets: Array[Float]): Float = {
    val predPtr = new org.bytedeco.javacpp.FloatPointer()
    var i = 0
    while (i < predictions.length) { predPtr.put(i, predictions(i)); i += 1 }
    val targetPtr = new org.bytedeco.javacpp.FloatPointer()
    i = 0
    while (i < targets.length) { targetPtr.put(i, targets(i)); i += 1 }
    val sizes = Array(predictions.length.toLong)
    val opts = new TensorOptions()
    opts.dtype().put(ScalarType.Float)
    val predTensor = torch.from_blob(predPtr, sizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
    val targetTensor = torch.from_blob(targetPtr, sizes, null.asInstanceOf[org.bytedeco.pytorch.PointerConsumer], opts).clone()
    apply(predTensor, targetTensor).item().toFloat
  }
}

class BCEWithLogitsLoss(reduction: String = "mean") {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    torch.binary_cross_entropy_with_logits(predictions, targets) match {
      case loss if reduction == "sum" => loss.sum()
      case loss if reduction == "none" => loss
      case loss => loss.mean()
    }
  }
}

class CrossEntropyLoss(
  reduction: String = "mean",
  labelSmoothing: Float = 0.0f
) {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    val loss = torch.cross_entropy(predictions, targets)
    reduction match {
      case "sum" => loss.sum()
      case "none" => loss
      case _ => loss.mean()
    }
  }
}

class BPRLoss(margin: Float = 1.0f) {
  def apply(posScores: Tensor, negScores: Tensor): Tensor = {
    val diff = posScores.sub(negScores)
    val losses = diff.sigmoid().log().neg()
    losses.mean()
  }

  def apply(posScore: Float, negScore: Float): Float = {
    val diff = posScore - negScore
    val sigmoid = 1.0f / (1.0f + scala.math.exp(-diff).toFloat)
    -scala.math.log(sigmoid + 1e-8).toFloat
  }
}

class HingeLoss(margin: Float = 1.0f) {
  def apply(posScores: Tensor, negScores: Tensor): Tensor = {
    val diff = posScores.sub(negScores)
    val mTensor = torch.full(Array(1L), new Scalar(margin))
    val zeroTensor = torch.full(Array(1L), new Scalar(0.0f))
    // ReLU-like: max(0, margin - diff)
    val losses = diff.neg().add(mTensor).maximum(zeroTensor)
    losses.mean()
  }
}

class TripletMarginLoss(margin: Float = 1.0f) {
  def apply(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor = {
    val distPos = anchor.sub(positive).pow(new Scalar(2.0f)).sum(1)
    val distNeg = anchor.sub(negative).pow(new Scalar(2.0f)).sum(1)
    val mTensor = torch.full(Array(1L), new Scalar(margin))
    val zeroTensor = torch.full(Array(1L), new Scalar(0.0f))
    val losses = distPos.sub(distNeg).add(mTensor).maximum(zeroTensor)
    losses.mean()
  }
}

class InBatchNCELoss(temperature: Float = 0.07f) {
  def apply(userEmbeds: Tensor, posItemEmbeds: Tensor, negItemEmbeds: Tensor): Tensor = {
    val posScores = userEmbeds.mul(posItemEmbeds).sum(1)
    val negScores = torch.bmm(
      negItemEmbeds,
      userEmbeds.unsqueeze(2)
    ).squeeze(2)
    val allScores = torch.cat(Seq(posScores.unsqueeze(1), negScores).toTensorVector, 1)
    val scaledScores = allScores.div(new Scalar(temperature))
    val labels = torch.zeros(userEmbeds.size(0).toLong).toType(ScalarType.Long)
    torch.cross_entropy(scaledScores, labels)
  }
}

class MaskedCrossEntropyLoss {
  def apply(logits: Tensor, targets: Tensor, mask: Tensor): Tensor = {
    val loss = torch.cross_entropy(logits, targets.toType(ScalarType.Long))
    val maskedLoss = loss.mul(mask)
    maskedLoss.sum().div(mask.sum())
  }
}

class FocalLoss(
  alpha: Float = 0.25f,
  gamma: Float = 2.0f
) {
  def apply(predictions: Tensor, targets: Tensor): Tensor = {
    val p = predictions.sigmoid()
    val ceLoss = torch.binary_cross_entropy(predictions, targets)
    val one = new Scalar(1.0f)
    val pTerm = p.mul(targets).add(p.neg().add(one).mul(targets.neg().add(one)))
    val focalWeight = pTerm.neg().add(one).pow(new Scalar(gamma.toDouble))
    val alphaWeight = targets.mul(new Scalar(alpha.toDouble)).add(targets.neg().add(one).mul(new Scalar((1.0 - alpha).toDouble)))
    alphaWeight.mul(focalWeight).mul(ceLoss).mean()
  }
}
