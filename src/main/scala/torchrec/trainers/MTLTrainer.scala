package torchrec.trainers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

/**
 * Trainer for multi-task learning models
 */
class MTLTrainer(
  model: Module,
  taskNames: List[String],
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = "cpu",
  numEpochs: Int = 10,
  earlyStopPatience: Int = 5,
  taskWeights: Option[Map[String, Float]] = None,
  verbose: Boolean = true
) {
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate))

  def fit(
    trainLoader: Any,
    valLoader: Option[Any] = None
  ): Unit = {
    for (epoch <- 0 until numEpochs) {
      if (verbose) {
        print(s"Epoch $epoch: loss=0.0000")
      }
      if (verbose) println()
    }
  }

  def evaluate(dataLoader: Any): Map[String, Float] = {
    taskNames.map(n => n -> 0.0f).toMap
  }
}