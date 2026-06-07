package tutorials

import torchrec.Implicits._
import torchrec.Implicits._
import torchrec.data.DataGenerator
import torchrec.basic.features._
import torchrec.data._
import torchrec.models.multi_task._
import torchrec.trainers._

/**
 * Tutorial: Multi-Task Learning with MMOE
 *
 * Multi-Gate Mixture-of-Experts (MMOE) enables learning multiple
 * related tasks with shared experts while maintaining task-specific
 * gating mechanisms.
 */
object MultiTaskMMOE {

  def main(args: Array[String]): Unit = {
    println("""
      |=============================================================
      | Tutorial: Multi-Task Learning with MMOE
      |=============================================================
      |
      | MMOE uses multiple experts with task-specific gates to
      | handle task relationships and conflicts in multi-task learning.
      |
      |""".stripMargin)

    // Define task configuration
    val taskNames = List("cvr", "ctr", "like")
    val taskTypes = List("classification", "classification", "classification")

    // Define shared features
    println("Defining shared features...")
    val features = List(
      SparseFeature("user_id", 10000, 16),
      SparseFeature("item_id", 50000, 16),
      SparseFeature("category", 1000, 8),
      SparseFeature("brand", 5000, 8),
      SparseFeature("price_level", 10, 4)
    )
    features.foreach(f => println(f"  - ${f.name}: vocab=${f.vocabSize}%,d"))

    // Create MMOE model
    println("\nCreating MMOE model...")
    val model = new MMOE(
      features = features,
      taskNames = taskNames,
      taskTypes = taskTypes,
      embedDim = 16,
      numExperts = 4,
      expertDims = List(128),
      towerDims = List(64L)
    )
    println(s"  Model: MMOE(tasks=${taskNames.mkString(", ")})")
    println("  Experts: 4")
    println("  Expert dims: [128]")

    // Generate multi-task data
    println("\nGenerating multi-task data...")
    val (trainData, valData, testData) = DataGenerator.generateMultiTaskData(
      numSamples = 10000,
      numFeatures = features.size,
      taskNames = taskNames,
      vocabSize = 100
    )
    println(f"  Train: ${trainData.size}%,d")
    println(f"  Val: ${valData.size}%,d")
    println(f"  Test: ${testData.size}%,d")

    // Create data loaders
    val trainLoader = new DataLoader(trainData, batchSize = 256, shuffle = true)
    val valLoader = new DataLoader(valData, batchSize = 256, shuffle = false)

    // Define task weights
    val taskWeights = Map(
      "cvr" -> 1.0f,  // Conversion - primary task
      "ctr" -> 1.0f,  // Click-through - secondary
      "like" -> 0.8f  // Like - auxiliary task
    )

    // Create trainer
    println("\nCreating MTLTrainer...")
    val trainer = new MTLTrainer(
      model = model,
      taskNames = taskNames,
      learningRate = 1e-3f,
      numEpochs = 5,
      earlyStopPatience = 3,
      taskWeights = Some(taskWeights),
      verbose = true
    )

    // Train
    println("\nTraining multi-task model...")
    trainer.fit(trainLoader, Some(valLoader))

    // Evaluate
    println("\nEvaluating...")
    val metrics = trainer.evaluate(valLoader)
    println("  Task Metrics:")
    taskNames.foreach { task =>
      val weight = taskWeights(task)
      val auc = metrics(s"${task}_AUC")
      val logloss = metrics(s"${task}_LogLoss")
      val acc = metrics(s"${task}_Accuracy")
      println(f"    $task (weight=$weight): AUC=$auc%.4f, LogLoss=$logloss%.4f, Accuracy=$acc%.4f")
    }

    println("""
      |
      |=============================================================
      | Tutorial Complete!
      |=============================================================
      |
      | Try other multi-task models:
      | - PLE: Progressive Layered Extraction
      | - ESMM: Entire Space Multi-Task Model
      | - SharedBottom: Hard parameter sharing
      |""".stripMargin)
  }
}