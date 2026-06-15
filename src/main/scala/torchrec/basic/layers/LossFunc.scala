package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable
import scala.jdk.CollectionConverters._

/**
 * Unified L1/L2 regularization for embedding and dense parameters.
 *
 * @param embeddingL1 L1 coefficient for embedding parameters (default: 0.0)
 * @param embeddingL2 L2 coefficient for embedding parameters (default: 0.0)
 * @param denseL1 L1 coefficient for dense (non-embedding) parameters (default: 0.0)
 * @param denseL2 L2 coefficient for dense (non-embedding) parameters (default: 0.0)
 */
class RegularizationLoss(
  embeddingL1: Float = 0.0f,
  embeddingL2: Float = 0.0f,
  denseL1: Float = 0.0f,
  denseL2: Float = 0.0f
) extends Module {

  def apply(model: Module): Tensor = {
    var regLoss = 0.0f

    // Collect normalization layer parameter ids using iterator pattern
    val normParamIds = mutable.Set[Long]()
    val normModules = List(
      classOf[BatchNorm1dImpl], classOf[BatchNorm2dImpl], classOf[BatchNorm3dImpl],
      classOf[LayerNormImpl], classOf[GroupNormImpl],
      classOf[InstanceNorm1dImpl], classOf[InstanceNorm2dImpl], classOf[InstanceNorm3dImpl]
    )

    var modBegin = model.modules().begin()
    var modEnd = model.modules().end()
    while (!modBegin.equals(modEnd)) {
      val module = modBegin.get()
      if (module != null && !module.isNull) {
        val moduleClass = module.getClass
        if (normModules.exists(_.isAssignableFrom(moduleClass))) {
          var paramBegin = module.parameters().begin()
          var paramEnd = module.parameters().end()
          while (!paramBegin.equals(paramEnd)) {
            val p = paramBegin.get()
            if (p != null && !p.isNull) {
              normParamIds.add(System.identityHashCode(p))
            }
            paramBegin.increment()
          }
        }
      }
      modBegin.increment()
    }

    // Collect embedding layer parameter ids using iterator pattern
    val embeddingParamIds = mutable.Set[Long]()
    val embedModules = List(classOf[EmbeddingImpl], classOf[EmbeddingBagImpl])

    modBegin = model.modules().begin()
    modEnd = model.modules().end()
    while (!modBegin.equals(modEnd)) {
      val module = modBegin.get()
      if (module != null && !module.isNull) {
        val moduleClass = module.getClass
        if (embedModules.exists(_.isAssignableFrom(moduleClass))) {
          var paramBegin = module.parameters().begin()
          var paramEnd = module.parameters().end()
          while (!paramBegin.equals(paramEnd)) {
            val p = paramBegin.get()
            if (p != null && !p.isNull) {
              embeddingParamIds.add(System.identityHashCode(p))
            }
            paramBegin.increment()
          }
        }
      }
      modBegin.increment()
    }

    // Compute regularization loss using iterator pattern
    var paramBegin = model.parameters().begin()
    var paramEnd = model.parameters().end()
    while (!paramBegin.equals(paramEnd)) {
      val param = paramBegin.get()
      if (param != null && !param.isNull && param.requires_grad()) {
        val paramId = System.identityHashCode(param)
        // Skip normalization layer parameters
        if (!normParamIds.contains(paramId)) {
          if (embeddingParamIds.contains(paramId)) {
            if (embeddingL1 > 0.0f) {
              val absSum = torch.sum(torch.abs(param)).item().toFloat()
              regLoss += embeddingL1 * absSum
            }
            if (embeddingL2 > 0.0f) {
              val sqSum = torch.sum(param.pow(new Scalar(2.0f))).item().toFloat()
              regLoss += embeddingL2 * sqSum
            }
          } else {
            if (denseL1 > 0.0f) {
              val absSum = torch.sum(torch.abs(param)).item().toFloat()
              regLoss += denseL1 * absSum
            }
            if (denseL2 > 0.0f) {
              val sqSum = torch.sum(param.pow(new Scalar(2.0f))).item().toFloat()
              regLoss += denseL2 * sqSum
            }
          }
        }
      }
      paramBegin.increment()
    }

    torch.tensor(Array(regLoss)*)
  }
}

/**
 * Hinge loss for pairwise learning.
 *
 * @param margin Margin value for hinge loss (default: 2.0)
 * @param numItems Total number of items for rank-based weighting (optional)
 */
class HingeLoss(margin: Float = 2.0f, numItems: Option[Long] = None) extends Module {

  def forward(posScore: Tensor, negScore: Tensor, inBatchNeg: Boolean = false): Tensor = {
    val posFlat = posScore.view(-1)

    val maxNeg = torch.max(negScore, -1).get0
    val marginTensor = torch.tensor(Array(margin)*)
    val loss = torch.maximum(maxNeg.sub(posFlat).add(marginTensor), torch.zeros_like(maxNeg))

    numItems match {
      case Some(n) =>
        val impostors = (negScore.sub(posFlat.view(-1, 1)).add(new Scalar(margin))).gt(new Scalar(0.0f))
        val rank = torch.mean(impostors.toType(ScalarType.Float), -1).mul(new Scalar(n.toFloat)) // * n.toFloat
        torch.mean(loss.mul(torch.log(rank.sub(new Scalar(1.0f)))))
      case None =>
        torch.mean(loss)
    }
  }
}

/**
 * Bayesian Personalized Ranking (BPR) Loss.
 */
class BPRLoss extends Module {

  def forward(posScore: Tensor, negScore: Tensor, inBatchNeg: Boolean = false): Tensor = {
    val posFlat = posScore.view(-1)
    val diff = if (negScore.dim() == 1L) {
      posFlat.sub(negScore)
    } else {
      posFlat.view(-1, 1).sub(negScore)
    }
    torch.sigmoid(diff.negative()).log().mean()
//    (- diff.sigmoid()).log().mean()
  }
}

/**
 * Noise Contrastive Estimation (NCE) loss for recommender systems.
 *
 * @param temperature Temperature for scaling logits (default: 1.0)
 * @param ignoreIndex Target index to ignore (default: 0)
 * @param reduction Reduction applied to output: 'mean', 'sum', or 'none' (default: 'mean')
 */
class NCELoss(
  temperature: Float = 1.0f,
  ignoreIndex: Long = 0L,
  reduction: String = "mean"
) extends Module {

  def forward(logits: Tensor, targets: Tensor): Tensor = {
    // Scale logits by temperature
    val scaledLogits =  logits.div(new Scalar(temperature) )

    // Compute log softmax
    val logProbs = torch.log_softmax(scaledLogits, -1)

    // Get log probability of target class
    val batchSize = targets.size(0).toInt
    val batchIndices = torch.arange(new Scalar(batchSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))

    // Use select-based indexing instead of fancy indexing
    var targetLogProbsArr = new Array[Float](batchSize)
    for (i <- 0 until batchSize) {
      val idx = batchIndices.select(0, i.toLong)
      val targetIdx = targets.select(0, i.toLong)
      targetLogProbsArr(i) = logProbs.select(0, i.toLong).select(0, targetIdx.item().toLong()).item().toFloat()
    }
    var lossTensor = torch.tensor(targetLogProbsArr*).neg()

    // Create mask for ignore_index
    var lossSum = 0.0f
    var lossCount = 0
    for (i <- 0 until batchSize) {
      val targetVal = targets.select(0, i.toLong).item().toLong()
      if (targetVal != ignoreIndex) {
        lossSum += targetLogProbsArr(i)
        lossCount += 1
      }
    }

    // Apply reduction
    reduction match {
      case "mean" => torch.tensor(Array(lossSum / lossCount)*)
      case "sum" => torch.tensor(Array(lossSum)*)
      case _ => lossTensor
    }
  }
}

/**
 * In-batch NCE loss with explicit negatives.
 *
 * @param temperature Temperature for scaling logits (default: 0.1)
 * @param ignoreIndex Target index to ignore (default: 0)
 * @param reduction Reduction applied to output: 'mean', 'sum', or 'none' (default: 'mean')
 */
class InBatchNCELoss(
  temperature: Float = 0.1f,
  ignoreIndex: Long = 0L,
  reduction: String = "mean"
) extends Module {

  def forward(embeddings: Tensor, itemEmbeddings: Tensor, targets: Tensor): Tensor = {
    // Compute logits: (batch_size, vocab_size)
    val logits = torch.matmul(embeddings, itemEmbeddings.t()).div(new Scalar(temperature)) 

    // Compute log softmax
    val logProbs = torch.log_softmax(logits, -1)

    // Get log probability of target class
    val batchSize = targets.size(0).toInt

    var targetLogProbsArr = new Array[Float](batchSize)
    for (i <- 0 until batchSize) {
      val targetIdx = targets.select(0, i.toLong).item().toLong()
      targetLogProbsArr(i) = logProbs.select(0, i.toLong).select(0, targetIdx).item().toFloat()
    }
    var lossTensor = torch.tensor(targetLogProbsArr*).neg()

    // Apply mask for ignore_index and compute loss
    var lossSum = 0.0f
    var lossCount = 0
    for (i <- 0 until batchSize) {
      val targetVal = targets.select(0, i.toLong).item().toLong()
      if (targetVal != ignoreIndex) {
        lossSum += targetLogProbsArr(i)
        lossCount += 1
      }
    }

    // Apply reduction
    reduction match {
      case "mean" => torch.tensor(Array(lossSum / lossCount)*)
      case "sum" => torch.tensor(Array(lossSum)*)
      case _ => lossTensor
    }
  }
}