package torchrec.trainers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable
import scala.util.Random
import torchrec.Implicits.*
import torchrec.TorchRec
import torchrec.data.*
import torchrec.basic.metrics.*
import torchrec.models.ranking.DeepFM
import torchrec.models.ranking.DCN
import torchrec.models.ranking.DCNv2
import torchrec.models.ranking.AFM
import torchrec.models.ranking.WideDeep
import torchrec.models.ranking.DIN
import torchrec.models.ranking.BST
import torchrec.basic.losses.BCELoss
import torchrec.distributed.ModuleForward

/**
 * Trainer for CTR (Click-Through Rate) models
 */
class CTRTrainer(
  model: Module,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = "cuda",
  numEpochs: Int = 10,
  earlyStopPatience: Int = 5,
  verbose: Boolean = true
) {
  private var bestAUC = 0.0f
  private var patienceCounter = 0
  private val rng = new Random(42)
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
  private val bceLoss = new BCELoss()
  private val forward: ModuleForward = ModuleForward.of(model)

  def fit(
    trainLoader: DataLoader,
    valLoader: Option[DataLoader] = None
  ): Unit = {
    model.train(true)

    for (epoch <- 0 until numEpochs) {
      var totalLoss = 0.0f
      var numBatches = 0

      val iter = trainLoader.iterator
      while (iter.hasNext) {
        val batch = iter.next()
        val features = batch.sparseFeatures
        val labelsOpt = batch.labels

        if (labelsOpt.isEmpty) {
          // No labels, skip
        } else {
          val labels = labelsOpt.get

          // Zero gradients before backward
          optimizer.zero_grad()

          model match {
            case deepFM: DeepFM =>
              val pred = deepFM.forward(features)
              val batchSize = pred.size(0).toInt

              // Ensure shapes match: pred=(batch,1), labels=(batch,) — view both to (batch,1)
              val pred2D = pred.view(batchSize, 1)
              val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)

              // DeepFM.forward already applies sigmoid, use BCELoss
              val loss = bceLoss.apply(pred2D, target2D)

              // Backward pass and optimizer step
              loss.backward()
              optimizer.step()

              totalLoss += loss.item().toFloat
              numBatches += 1
            case _ =>
            // Skip unknown model types
          }
        }
      }

      val avgLoss = if (numBatches > 0) totalLoss / numBatches else 0.0f

      if (verbose) {
        print(s"Epoch $epoch: train_loss=${f"$avgLoss%.4f"}")
      }

      // Validate if provided
      valLoader.foreach { vl =>
        model.eval()
        val metrics = evaluate(vl)
        val auc = metrics.getOrElse("AUC", 0.0f)
        if (verbose) print(s", val_auc=${f"$auc%.4f"}")

        if (auc > bestAUC) {
          bestAUC = auc
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
    val predList = mutable.ListBuffer[Float]()
    val labelList = mutable.ListBuffer[Float]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val features = batch.sparseFeatures
      val seqFeats = batch.sequenceFeatures
      val labelsOpt = batch.labels

      if (labelsOpt.nonEmpty) {
        val label = labelsOpt.get
        model match {
          case deepFM: DeepFM =>
            val pred = deepFM.forward(features)
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case dcn: DCN =>
            val pred = dcn.forward(features)
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case dcnv2: DCNv2 =>
            val pred = dcnv2.forward(features)
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case afm: AFM =>
            val pred = afm.forward(features)
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case wd: WideDeep =>
            val pred = wd.forward(features)
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case din: DIN =>
            if (seqFeats.nonEmpty) {
              val targetIdx = label.view(label.size(0), 1).toType(ScalarType.Long)
              val pred = din.forward(features, seqFeats, targetIdx)
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
              targetIdx.close()
            }

          case bst: BST =>
            if (seqFeats.nonEmpty) {
              val pred = bst.forward(features, seqFeats)
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }

          case module: Module =>
            // Generic fallback for unknown CTR models
            try {
              val pred = forward.apply(module,features).asInstanceOf[Tensor]
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            } catch { case _: Throwable => }
        }
      }
    }

    if (predList.isEmpty || labelList.isEmpty) {
      return Map("AUC" -> 0.0f, "LogLoss" -> 0.0f, "Accuracy" -> 0.0f)
    }

    val minLen = math.min(predList.length, labelList.length)
    val predArray = predList.take(minLen).toArray
    val labelArray = labelList.take(minLen).toArray

    val aucMetric = new AUC()
    val loglossMetric = new LogLoss()
    val accMetric = new Accuracy()
    val hitRateMetric = new HitRate[Float](10)
    val ndcgMetric = new NDCG[Float](10)
    val mrrMetric = new MRR()

    aucMetric.update(predArray, labelArray)
    loglossMetric.update(predArray, labelArray)
    accMetric.update(predArray, labelArray)
    hitRateMetric.update(predArray, labelArray)
    ndcgMetric.update(predArray, labelArray)
    mrrMetric.update(predArray, labelArray)

    Map(
      "AUC" -> aucMetric.compute(),
      "LogLoss" -> loglossMetric.compute(),
      "Accuracy" -> accMetric.compute(),
      "Hit@10" -> hitRateMetric.compute(),
      "NDCG@10" -> ndcgMetric.compute(),
      "MRR" -> mrrMetric.compute()
    )
  }

  def predict(dataLoader: DataLoader): Array[Float] = {
    val predictions = mutable.ListBuffer[Float]()

    val iter = dataLoader.iterator
    while (iter.hasNext) {
      val batch = iter.next()
      val features = batch.sparseFeatures

      model match {
        case deepFM: DeepFM =>
          val pred = deepFM.forward(features)
          val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
          predictions.appendAll(predHost.toFloatArray)
          predHost.close()
        case _ =>
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
}