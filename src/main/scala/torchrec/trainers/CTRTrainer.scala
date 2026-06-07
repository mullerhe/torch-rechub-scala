package torchrec.trainers

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable
import scala.util.Random
import torchrec.Implicits.*
import torchrec.TorchRec
import torchrec.utils.DeviceSupport
import torchrec.data.*
import torchrec.basic.metrics.*
import torchrec.models.ranking.DeepFM
import torchrec.models.ranking.DCN
import torchrec.models.ranking.DCNv2
import torchrec.models.ranking.AFM
import torchrec.models.ranking.WideDeep
import torchrec.models.ranking.DIN
import torchrec.models.ranking.BST
import torchrec.basic.losses.{BCELoss, BCEWithLogitsLoss}

/**
 * Trainer for CTR (Click-Through Rate) models
 */
class CTRTrainer(
  model: Module,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f,
  device: String = DeviceSupport.backend,
  numEpochs: Int = 10,
  earlyStopPatience: Int = 5,
  verbose: Boolean = true
) {
  private var bestAUC = 0.0f
  private var patienceCounter = 0
  private val rng = new Random(42)
  private val optimizer = new Adam(model.parameters(), new AdamOptions(learningRate.toDouble))
  private val bceLoss = new BCEWithLogitsLoss()

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
              val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
              val loss = bceLoss.apply(pred, target2D)
              loss.backward(); optimizer.step()
              totalLoss += loss.item().toFloat; numBatches += 1

            case dcn: DCN =>
              val pred = dcn.forward(features)
              val batchSize = pred.size(0).toInt
              val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
              val loss = bceLoss.apply(pred, target2D)
              loss.backward(); optimizer.step()
              totalLoss += loss.item().toFloat; numBatches += 1

            case dcnv2: DCNv2 =>
              val pred = dcnv2.forward(features)
              val batchSize = pred.size(0).toInt
              val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
              val loss = bceLoss.apply(pred, target2D)
              loss.backward(); optimizer.step()
              totalLoss += loss.item().toFloat; numBatches += 1

            case afm: AFM =>
              val pred = afm.forward(features)
              val batchSize = pred.size(0).toInt
              val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
              val loss = bceLoss.apply(pred, target2D)
              loss.backward(); optimizer.step()
              totalLoss += loss.item().toFloat; numBatches += 1

            case wd: WideDeep =>
              val pred = wd.forward(features)
              val batchSize = pred.size(0).toInt
              val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
              val loss = bceLoss.apply(pred, target2D)
              loss.backward(); optimizer.step()
              totalLoss += loss.item().toFloat; numBatches += 1

            case din: DIN =>
              val seqFeats = batch.sequenceFeatures
              if (seqFeats.nonEmpty) {
                val targetIdx = labels.view(labels.size(0), 1).toType(ScalarType.Long)
                val pred = din.forward(features, seqFeats, targetIdx)
                val batchSize = pred.size(0).toInt
                val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                val loss = bceLoss.apply(pred, target2D)
                loss.backward(); optimizer.step()
                totalLoss += loss.item().toFloat; numBatches += 1
                targetIdx.close()
              }

            case bst: BST =>
              val seqFeats = batch.sequenceFeatures
              if (seqFeats.nonEmpty) {
                val pred = bst.forward(features, seqFeats)
                val batchSize = pred.size(0).toInt
                val target2D = labels.view(batchSize, 1).toType(ScalarType.Float)
                val loss = bceLoss.apply(pred, target2D)
                loss.backward(); optimizer.step()
                totalLoss += loss.item().toFloat; numBatches += 1
              }

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
            val pred = deepFM.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case dcn: DCN =>
            val pred = dcn.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case dcnv2: DCNv2 =>
            val pred = dcnv2.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case afm: AFM =>
            val pred = afm.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case wd: WideDeep =>
            val pred = wd.forward(features).sigmoid()
            val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
            val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
            predList.appendAll(predHost.toFloatArray)
            labelList.appendAll(labelHost.toFloatArray)
            predHost.close(); labelHost.close()

          case din: DIN =>
            if (seqFeats.nonEmpty) {
              val targetIdx = label.view(label.size(0), 1).toType(ScalarType.Long)
              val pred = din.forward(features, seqFeats, targetIdx).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
              targetIdx.close()
            }

          case bst: BST =>
            if (seqFeats.nonEmpty) {
              val pred = bst.forward(features, seqFeats).sigmoid()
              val predHost = pred.squeeze().to(ScalarType.Float).contiguous().cpu()
              val labelHost = label.squeeze().to(ScalarType.Float).contiguous().cpu()
              predList.appendAll(predHost.toFloatArray)
              labelList.appendAll(labelHost.toFloatArray)
              predHost.close(); labelHost.close()
            }

          case _ =>
            // Unknown model type — skip evaluation
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
          val pred = deepFM.forward(features).sigmoid()
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