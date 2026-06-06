package torchrec.trainers

import torchrec.Implicits._
import torchrec.data.DataLoader
import torchrec.models.matching._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

/**
 * Trainer for matching/retrieval models
 */
class MatchTrainer(
  model: Module,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = "cpu",
  mode: Int = 0,
  temperature: Float = 0.07f,
  numEpochs: Int = 10,
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

  def inferenceEmbedding(
    dataLoader: Any,
    mode: String
  ): Array[Tensor] = {
    val loader = dataLoader.asInstanceOf[DataLoader]
    val embeddings = mutable.ArrayBuffer[Tensor]()

    val iter = loader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      mode match {
        case "user" =>
          val userFeats = batch.sparseFeatures
          (model, userFeats.get("history")) match {
            case (dssm: DSSM, _) =>
              embeddings += dssm.userTowerForward(userFeats)
            case (yt: YoutubeDNN, Some(history)) =>
              val sparseOnly = userFeats - "history"
              embeddings += yt.userTowerForward(sparseOnly, Map("history" -> history))
            case (yt: YoutubeDNN, None) =>
              embeddings += yt.userTowerForward(userFeats)
            case _ =>
          }
        case "item" =>
          val itemFeats = batch.itemFeatures
          if (itemFeats.nonEmpty) {
            model match {
              case dssm: DSSM =>
                embeddings += dssm.itemTowerForward(itemFeats)
              case _ =>
            }
          }
        case _ =>
      }
    }

    embeddings.toArray
  }

  def evaluate(dataLoader: Any, topk: Int = 10): Float = {
    0.0f
  }
}