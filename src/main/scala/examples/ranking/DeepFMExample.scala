package examples.ranking

import torchrec.Implicits._
import torchrec.TorchRec
import torchrec.models.ranking._
import torchrec.basic.features._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * DeepFM Example for CTR Prediction
 */
object DeepFMExample {

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("DeepFM CTR Prediction Example")
    println("=" * 60)

    // Configuration
    val batchSize = 256
    val embedDim = 8
    val numEpochs = 3

    println(s"\nConfiguration:")
    println(s"  Batch Size: $batchSize")
    println(s"  Embed Dim: $embedDim")
    println(s"  Epochs: $numEpochs")

    // Create features
    val numSparseFeatures = 10
    val vocabSize = 100
    println("\n[1] Defining features...")
    val features = (0 until numSparseFeatures).map { i =>
      SparseFeature(name = s"feat_$i", vocabSize = vocabSize, embedDim = embedDim)
    }.toList
    features.foreach { f =>
      println(s"  - ${f.name}: vocab=${f.vocabSize}, embed=${f.embedDim}")
    }

    // Create model
    println("\n[2] Creating DeepFM model...")
    val model = new DeepFM(
      features = features,
      embedDim = embedDim,
      mlpDims = List(128L, 64L),
      dropout = 0.2f,
      device = "cuda"
    )
    println(s"  Model created: DeepFM(embedDim=$embedDim)")

    // Generate simple training data
    println("\n[3] Generating synthetic data...")
    val random = new scala.util.Random(42)
    val batchFeatures = mutable.Map[String, Tensor]()
    for (feat <- features) {
      val data = Array.tabulate(batchSize) { _ =>
        random.nextInt(vocabSize).toFloat
      }
      // Ensure tensors have shape (batch,1) and are on cpu
      batchFeatures(feat.name) = TorchRec.tensor(data, batchSize.toLong, 1L).to("cpu").toType(ScalarType.Long)
    }
    val labels = Array.tabulate(batchSize) { _ =>
      if (random.nextFloat() > 0.5f) 1.0f else 0.0f
    }
    // Create labels tensor with shape (batch,1) on cpu
    val labelsTensor = TorchRec.tensor(labels, batchSize.toLong, 1L).to("cpu").toType(ScalarType.Float)
    println(s"  Generated $batchSize samples")

    // Training loop
    println("\n[4] Training model...")
    val startTime = System.currentTimeMillis()

    for (epoch <- 0 until numEpochs) {
      try {
        val pred = model.forward(batchFeatures.toMap)
        val predSqueezed = pred.squeeze()
        val labelsOnDevice = labelsTensor.to(pred.device(), ScalarType.Float)
        val diff = predSqueezed.sub(labelsOnDevice)
        val loss = TorchRec.mul(diff, diff).mean()
        println(s"  Epoch $epoch: loss=${f"${loss.item().toFloat}%.4f"}")
      }
//      catch {
//        case e: Throwable =>
//          println(s"  Epoch $epoch: error=${e.getMessage}")
//      }
    }

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // Evaluate
    println("\n[5] Evaluating...")
    try {
      val pred = model.forward(batchFeatures.toMap)
      println("\n[5] Evaluating...1")
      val predArr = pred.squeeze().toFloatArray
      println("\n[5] Evaluating...2")
      val labelArr = labelsTensor.toFloatArray
      println("\n[5] Evaluating..3.")
      val auc = torchrec.basic.metrics.AUC.calculate(predArr, labelArr)
      val logloss = torchrec.basic.metrics.LogLoss.calculate(predArr, labelArr)
      println(f"  AUC: $auc%.4f")
      println(f"  LogLoss: $logloss%.4f")
    } catch {
      case e: Throwable =>
        println(s"  Evaluation error: ${e.getMessage}")
    }

    println("\n" + "=" * 60)
    println("DeepFM Example Completed Successfully!")
    println("=" * 60)
  }
}