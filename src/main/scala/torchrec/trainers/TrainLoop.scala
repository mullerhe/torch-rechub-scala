package torchrec.trainers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.collection.mutable

/**
 * Generic training loop with callbacks support
 */
class TrainLoop(
  model: Module,
  optimizer: Optimizer,
  device: String = "cpu",
  gradientClip: Option[Float] = None
) {
  private var trainStep = 0

  def step(batch: Any, lossFn: Any => Tensor): (Tensor, Map[String, Float]) = {
    optimizer.zero_grad()
    val loss = lossFn(batch)
    loss.backward()
    optimizer.step()
    trainStep += 1
    val metrics = Map("loss" -> loss.item().toFloat)
    (loss, metrics)
  }

  def trainEpoch(
    dataLoader: Any,
    lossFn: Any => Tensor
  ): Map[String, Float] = {
    Map("loss" -> 0.0f)
  }

  def evaluate(
    dataLoader: Any,
    evalFn: Any => Map[String, Float]
  ): Map[String, Float] = {
    Map()
  }

  def getTrainStep: Int = trainStep
  def reset(): Unit = { trainStep = 0 }
}