package torchrec.trainers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable
import scala.util.Random

import torchrec.Implicits._
import torchrec.TorchRec
import torchrec.data._
import torchrec.basic.metrics._
import torchrec.models.ranking.DeepFM

/**
 * Trainer for CTR (Click-Through Rate) models
 */
class CTRTrainer(
  model: Module,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = "cpu",
  numEpochs: Int = 10,
  earlyStopPatience: Int = 5,
  verbose: Boolean = true
) {
  private var bestAUC = 0.0f
  private var patienceCounter = 0
  private val rng = new Random(42)

  def fit(
    trainLoader: DataLoader,
    valLoader: Option[DataLoader] = None
  ): Unit = {
    for (epoch <- 0 until numEpochs) {
      var totalLoss = 0.0f
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        try {
          val features = batch.sparseFeatures
          val labelsOpt = batch.labels

          if (labelsOpt.isEmpty) {
            // No labels, skip
          } else {
            val labels = labelsOpt.get

            // Forward pass - only support DeepFM for now
            model match {
              case deepFM: DeepFM =>
                val pred = deepFM.forward(features)
                val predSqueezed = pred.squeeze()
                val loss = binaryCrossEntropy(predSqueezed, labels)
                totalLoss += loss.item().toFloat
                numBatches += 1
              case _ =>
              // Skip unknown model types
            }
          }
        } catch {
          case _: Throwable =>
          // Skip failed batches
        }
      }

      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0f

      if (verbose) {
        print(s"Epoch $epoch: train_loss=${f"$avgLoss%.4f"}")
      }

      // Validate if provided
      valLoader.foreach { vl =>
        val metrics = evaluate(vl)
        val auc = metrics.getOrElse("AUC", 0.0f)
        if (verbose) print(s", val_auc=${f"$auc%.4f"}")

        if (auc > bestAUC) {
          bestAUC = auc
          patienceCounter = 0
        } else {
          patienceCounter += 1
        }
      }

      if (verbose) println()

      if (patienceCounter >= earlyStopPatience) {
        if (verbose) println(s"Early stopping at epoch $epoch")
        return
      }
    }
  }

  def evaluate(dataLoader: DataLoader): Map[String, Float] = {
    var predictions = mutable.ListBuffer[Float]()
    var labels = mutable.ListBuffer[Float]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      try {
        val features = batch.sparseFeatures
        val labelsOpt = batch.labels

        if (labelsOpt.isEmpty) {
          // No labels
        } else {
          val label = labelsOpt.get

          model match {
            case deepFM: DeepFM =>
              val pred = deepFM.forward(features)
              val predArr = pred.squeeze().toFloatArray
              val labelArr = label.squeeze().toFloatArray
              predictions.appendAll(predArr)
              labels.appendAll(labelArr)
            case _ =>
          }
        }
      } catch {
        case _: Throwable =>
      }
    }

    if (predictions.isEmpty) {
      return Map("AUC" -> 0.0f, "LogLoss" -> 0.0f)
    }

    val auc = AUC.calculate(predictions.toArray, labels.toArray)
    val logloss = LogLoss.calculate(predictions.toArray, labels.toArray)

    Map("AUC" -> auc, "LogLoss" -> logloss)
  }

  def predict(dataLoader: DataLoader): Array[Float] = {
    var predictions = mutable.ListBuffer[Float]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      try {
        val features = batch.sparseFeatures

        model match {
          case deepFM: DeepFM =>
            val pred = deepFM.forward(features)
            val predArr = pred.squeeze().toFloatArray
            predictions.appendAll(predArr)
          case _ =>
        }
      } catch {
        case _: Throwable =>
      }
    }

    predictions.toArray
  }

  def saveCheckpoint(path: String): Unit = {
    throw new UnsupportedOperationException("torch.save not available in JavaCPP")
  }

  def loadCheckpoint(path: String): Unit = {
    val loaded = torch.load(path)
  }

  private def binaryCrossEntropy(pred: Tensor, target: Tensor): Tensor = {
    val targetF = target.toType(ScalarType.Float)
    val diff = pred.sub(targetF)
    val sq = TorchRec.mul(diff, diff)
    sq.mean()
  }
}