package torchrec.trainers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

import torchrec.Implicits._
import torchrec.data.DataLoader
import torchrec.models.multi_task._
import torchrec.basic.metrics._
import torchrec.basic.losses.{BCELoss, BCEWithLogitsLoss}
import torchrec.utils.DeviceSupport

/**
 * Trainer for multi-task learning models (MMOE, SharedBottom, PLE, AITM, ESMM)
 */
class MTLTrainer(
  model: Module,
  taskNames: List[String],
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = DeviceSupport.backend,
  numEpochs: Int = 10,
  earlyStopPatience: Int = 5,
  taskWeights: Option[Map[String, Float]] = None,
  verbose: Boolean = true
) {
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
  private val bceLoss = new BCEWithLogitsLoss()
  private val defaultWeights = taskWeights.getOrElse(taskNames.map(_ -> 1.0f).toMap)

  def fit(
    trainLoader: DataLoader,
    valLoader: Option[DataLoader] = None
  ): Unit = {
    model.train(true)
    var bestMetric = 0.0f
    var patienceCounter = 0

    for (epoch <- 0 until numEpochs) {
      var totalLoss = 0.0f
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        val sparseFeats = batch.sparseFeatures
        val taskLabelsMap = batch.taskLabels
        if (sparseFeats.isEmpty || taskLabelsMap.isEmpty) { /* skip */ }
        else {
          optimizer.zero_grad()

          model match {
            case mmoe: MMOE =>
              val outputs = mmoe.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case sb: SharedBottom =>
              val outputs = sb.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case ple: torchrec.models.multi_task.PLE =>
              val outputs = ple.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case esmm: ESMM =>
              val outputs = esmm.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case aitm: torchrec.models.multi_task.AITM =>
              val outputs = aitm.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case omoe: torchrec.models.multi_task.OMoE =>
              val outputs = omoe.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case stm: torchrec.models.multi_task.SingleTaskModel =>
              val outputs = stm.forward(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case metaheac: torchrec.models.multi_task.MetaHeac =>
              val outputs = metaheac.forwardByName(sparseFeats)
              var totalWeightedLoss: Tensor = null.asInstanceOf[Tensor]
              var totalWeight = 0.0f

              for (taskName <- taskNames) {
                outputs.get(taskName).foreach { pred =>
                  val label = taskLabelsMap.get(taskName)
                  if (label != null) {
                    val actualBatchSize = label.size(0).toInt
                    val target2D = if (pred.dim() == 1) pred.reshape(actualBatchSize, 1) else pred
                    val label2D = label.view(actualBatchSize, 1).toType(ScalarType.Float)
                    val taskLoss = bceLoss.apply(target2D, label2D)
                    val weight = defaultWeights.getOrElse(taskName, 1.0f).toFloat
                    if (totalWeightedLoss == null) {
                      totalWeightedLoss = taskLoss.mul(new Scalar(weight.toDouble))
                    } else {
                      totalWeightedLoss = totalWeightedLoss.add(taskLoss.mul(new Scalar(weight.toDouble)))
                    }
                    totalWeight += weight
                  }
                }
              }

              if (totalWeightedLoss != null && totalWeight > 0) {
                val avgLoss = totalWeightedLoss.div(new Scalar(totalWeight.toDouble))
                avgLoss.backward()
                optimizer.step()
                totalLoss += avgLoss.item().toFloat
              }
              numBatches += 1

            case _ =>
            // Unknown model type — skip
          }
        }
      }

      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0f

      if (verbose) print(s"Epoch $epoch: loss=${f"$avgLoss%.4f"}")

      valLoader.foreach { vl =>
        model.eval()
        val metrics = evaluate(vl)
        val auc = metrics.values.headOption.getOrElse(0.0f)
        if (verbose) print(s", ${taskNames.head}_AUC=${f"$auc%.4f"}")

        if (auc > bestMetric) {
          bestMetric = auc
          patienceCounter = 0
        } else {
          patienceCounter += 1
        }
        model.train(true)
      }

      if (verbose) println()

      if (patienceCounter >= earlyStopPatience) {
        if (verbose) println(s"Early stopping at epoch $epoch")
        return
      }
    }
  }

  def evaluate(dataLoader: DataLoader): Map[String, Float] = {
    model.eval()

    val taskMetrics = mutable.Map[String, (AUC, LogLoss, Accuracy)]()
    for (name <- taskNames) {
      taskMetrics(name) = (new AUC(), new LogLoss(), new Accuracy())
    }

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val sparseFeats = batch.sparseFeatures
      val taskLabelsMap = batch.taskLabels

      if (sparseFeats.isEmpty || taskLabelsMap.isEmpty) { /* skip */ }
      else {
        model match {
          case mmoe: MMOE =>
            val outputs = mmoe.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case sb: SharedBottom =>
            val outputs = sb.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case ple: torchrec.models.multi_task.PLE =>
            val outputs = ple.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case esmm: ESMM =>
            val outputs = esmm.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case aitm: torchrec.models.multi_task.AITM =>
            val outputs = aitm.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case omoe: torchrec.models.multi_task.OMoE =>
            val outputs = omoe.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case stm: torchrec.models.multi_task.SingleTaskModel =>
            val outputs = stm.forward(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case metaheac: torchrec.models.multi_task.MetaHeac =>
            val outputs = metaheac.forwardByName(sparseFeats)
            for (taskName <- taskNames) {
              outputs.get(taskName).foreach { pred =>
                val label = taskLabelsMap.get(taskName)
                if (label != null) {
                  val labelArr = label.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val predArr = pred.squeeze().to(ScalarType.Float).contiguous().cpu().toFloatArray
                  val (auc, logloss, acc) = taskMetrics(taskName)
                  auc.update(predArr, labelArr)
                  logloss.update(predArr, labelArr)
                  acc.update(predArr, labelArr)
                }
              }
            }

          case _ =>
        }
      }
    }

    model.train(true)

    taskMetrics.flatMap { case (name, (auc, logloss, acc)) =>
      Map(
        s"${name}_AUC" -> auc.compute(),
        s"${name}_LogLoss" -> logloss.compute(),
        s"${name}_Accuracy" -> acc.compute()
      )
    }.toMap
  }
}
