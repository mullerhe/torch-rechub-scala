package examples.generative

import torchrec.Implicits._
import torchrec.data._
import torchrec.data.DataGenerator
import torchrec.models.generative._
import torchrec.trainers._
import torchrec.basic.features._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * MovieLens Sequential/Generative Example
 * Demonstrates training HSTU and HLLM sequential models on MovieLens-like data.
 */
object MovieLensSeqExample {

  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("MovieLens Sequential/Generative Model Example")
    println("=" * 70)

    // Configuration
    val numSamples = 5000
    val vocabSize = 500
    val embedDim = 64
    val numHeads = 2
    val numLayers = 2
    val maxSeqLen = 50
    val batchSize = 256
    val learningRate = 1e-3f
    val numEpochs = 3
    val dropout = 0.2f
    val device = "cpu"
    val seed = 2022

    val modelName = if (args.length > 0) args(0).toLowerCase else "hstu"

    println(s"\nConfiguration:")
    println(s"  Model: $modelName")
    println(s"  Samples: $numSamples")
    println(s"  Vocab Size: $vocabSize")
    println(s"  Embed Dim: $embedDim")
    println(s"  Epochs: $numEpochs")

    // 1. Generate synthetic ranking data (reusing for sequential model demo)
    println("\n[1] Generating synthetic data...")
    val (trainData, valData, testData) = DataGenerator.generateRankingData(
      numSamples = numSamples,
      numSparseFeatures = 10,
      numDenseFeatures = 5,
      vocabSize = vocabSize,
      trainRatio = 0.7f,
      valRatio = 0.1f,
      seed = seed
    )
    println(s"  Train: ${trainData.size}")
    println(s"  Val: ${valData.size}")
    println(s"  Test: ${testData.size}")

    // 2. Create data loaders
    println("\n[2] Creating data loaders...")
    val trainLoader = new DataLoader(trainData, batchSize, shuffle = true)
    val valLoader = new DataLoader(valData, batchSize, shuffle = false)
    val testLoader = new DataLoader(testData, batchSize, shuffle = false)

    // 3. Create model
    println("\n[3] Creating model...")
    val effectiveVocabSize = vocabSize + 2

    val model = modelName match {
      case "hstu" | "hstu_model" =>
        new HSTU(
          vocabSize = effectiveVocabSize,
          embedDim = embedDim,
          numHeads = numHeads,
          numLayers = numLayers,
          maxSeqLen = maxSeqLen,
          dropout = dropout,
          device = device
        )

      case "hllm" =>
        val frozenEmbeddings = torch.randn(effectiveVocabSize, embedDim)
        val userFeatures = List(
          SparseFeature(name = "feat_0", vocabSize = 50, embedDim = embedDim)
        )
        new HLLM(
          itemEmbeddings = frozenEmbeddings,
          features = userFeatures,
          embedDim = embedDim,
          numHeads = numHeads,
          numLayers = numLayers,
          dropout = dropout,
          device = device
        )

      case _ =>
        println(s"  Warning: Unknown model '$modelName', defaulting to HSTU")
        new HSTU(
          vocabSize = effectiveVocabSize,
          embedDim = embedDim,
          numHeads = numHeads,
          numLayers = numLayers,
          maxSeqLen = maxSeqLen,
          dropout = dropout,
          device = device
        )
    }

    println(s"  Model: $modelName")
    println(s"  Vocab Size: $effectiveVocabSize")

    // 4. Train model
    println("\n[4] Training model...")
    val startTime = System.currentTimeMillis()

    val trainer = new CTRTrainer(
      model = model,
      learningRate = learningRate,
      device = device,
      numEpochs = numEpochs,
      earlyStopPatience = 3,
      verbose = true
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    println(f"\n  Training completed in $trainingTime%.2f seconds")

    // 5. Evaluate
    println("\n[5] Evaluating...")
    val testMetrics = trainer.evaluate(testLoader)
    println("  Test metrics:")
    testMetrics.foreach { case (name, value) =>
      println(f"    $name: $value%.4f")
    }

    // 6. Make predictions
    println("\n[6] Making predictions...")
    val predictions = trainer.predict(testLoader)
    println(s"  Predicted ${predictions.length} samples")
    if (predictions.length > 0) {
      println(s"  Sample predictions: ${predictions.take(5).map(v => f"$v%.3f").mkString(", ")}")
    }

    println("\n" + "=" * 70)
    println(s"MovieLens Sequential Example ($modelName) Completed!")
    println("=" * 70)
  }
}