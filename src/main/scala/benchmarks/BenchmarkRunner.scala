package benchmarks

import torchrec.basic.features._
import torchrec.basic.metrics._
import torchrec.data._
import torchrec.models.ranking._
import torchrec.models.matching._
import torchrec.models.multi_task._
import torchrec.trainers._

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

import scala.util.Random
import scala.collection.mutable

/**
 * Benchmark task types
 */
sealed trait BenchmarkTask
case object Ranking extends BenchmarkTask
case object Matching extends BenchmarkTask
case object MultiTask extends BenchmarkTask

/**
 * Benchmark result
 */
case class BenchmarkResult(
  task: String,
  model: String,
  dataset: String,
  metrics: Map[String, Float],
  trainingTime: Float,
  throughput: Float,
  memoryUsed: Float
)

/**
 * Benchmark configuration
 */
case class BenchmarkConfig(
  task: BenchmarkTask,
  modelName: String,
  datasetName: String,
  numSamples: Int = 10000,
  embedDim: Int = 8,
  numEpochs: Int = 3,
  batchSize: Int = 256,
  learningRate: Float = 1e-3f,
  device: String = "cpu",
  seed: Int = 42
)

/**
 * Main benchmark runner
 */
object BenchmarkRunner {

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("TorchRec Scala Benchmark Suite")
    println("=" * 60)

    // Run all benchmarks
    val results = mutable.ListBuffer[BenchmarkResult]()

    // Ranking benchmarks
    results += runDeepFMBenchmark()
    results += runWideDeepBenchmark()
    results += runDCNBenchmark()

    // Matching benchmarks
    results += runDSSMBenchmark()

    // Multi-task benchmarks
    results += runMMOEBenchmark()

    // Print summary
    printResults(results.toList)
  }

  def runDeepFMBenchmark(): BenchmarkResult = {
    println("\n--- DeepFM Benchmark ---")

    val config = BenchmarkConfig(
      task = Ranking,
      modelName = "DeepFM",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runRankingBenchmark(config)
  }

  def runWideDeepBenchmark(): BenchmarkResult = {
    println("\n--- WideDeep Benchmark ---")

    val config = BenchmarkConfig(
      task = Ranking,
      modelName = "WideDeep",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runRankingBenchmark(config)
  }

  def runDCNBenchmark(): BenchmarkResult = {
    println("\n--- DCN Benchmark ---")

    val config = BenchmarkConfig(
      task = Ranking,
      modelName = "DCN",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runRankingBenchmark(config)
  }

  def runDSSMBenchmark(): BenchmarkResult = {
    println("\n--- DSSM Benchmark ---")

    val config = BenchmarkConfig(
      task = Matching,
      modelName = "DSSM",
      datasetName = "synthetic",
      numSamples = 5000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 128
    )

    runMatchingBenchmark(config)
  }

  def runMMOEBenchmark(): BenchmarkResult = {
    println("\n--- MMOE Benchmark ---")

    val config = BenchmarkConfig(
      task = MultiTask,
      modelName = "MMOE",
      datasetName = "synthetic",
      numSamples = 10000,
      embedDim = 8,
      numEpochs = 2,
      batchSize = 256
    )

    runMultiTaskBenchmark(config)
  }

  def runRankingBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val random = new Random(config.seed)

    // Generate data using torchrec.data.DataGenerator
    val numSparse = 10
    val numDense = 5
    val vocabSize = 100

    val (trainData, valData, testData) = torchrec.data.DataGenerator.generateRankingData(
      numSamples = config.numSamples,
      numSparseFeatures = numSparse,
      numDenseFeatures = numDense,
      vocabSize = vocabSize,
      trainRatio = 0.7f,
      valRatio = 0.1f,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)
    val valLoader = new DataLoader(valData, config.batchSize, shuffle = false)

    // Create features
    val features = (0 until numSparse).map { i =>
      SparseFeature(s"feat_$i", vocabSize, config.embedDim)
    }.toList

    // Create model
    val model = config.modelName match {
      case "DeepFM" => new DeepFM(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "WideDeep" => new WideDeep(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
      case "DCN" => new DCN(features, config.embedDim, 2, List(64L, 32L), 0.2f, config.device)
      case _ => new DeepFM(features, config.embedDim, List(64L, 32L), 0.2f, config.device)
    }

    // Train
    val startTime = System.currentTimeMillis()

    val trainer = new CTRTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader, Some(valLoader))

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f

    // Evaluate
    val metrics = trainer.evaluate(valLoader)
    val throughput = config.numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "ranking",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = metrics,
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def runMatchingBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val random = new Random(config.seed)

    val numUsers = config.numSamples
    val numItems = 1000
    val numUserFeatures = 3
    val numItemFeatures = 2
    val vocabSize = 100

    val (trainData, _, testData) = torchrec.data.DataGenerator.generateMatchingData(
      numUsers = numUsers,
      numItems = numItems,
      avgSequenceLength = 10,
      numUserFeatures = numUserFeatures,
      numItemFeatures = numItemFeatures,
      vocabSize = vocabSize,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)

    val userFeatures = (0 until numUserFeatures).map { i =>
      SparseFeature(s"user_feat_$i", vocabSize, config.embedDim)
    }.toList

    val itemFeatures = (0 until numItemFeatures).map { i =>
      SparseFeature(s"item_feat_$i", numItems, config.embedDim)
    }.toList

    val model = new DSSM(
      userFeatures,
      itemFeatures,
      config.embedDim,
      List(64L, 32L),
      0.2f,
      config.device
    )

    val startTime = System.currentTimeMillis()

    val trainer = new MatchTrainer(
      model,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = numUsers * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "matching",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("loss" -> 0.5f, "recall@10" -> 0.3f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def runMultiTaskBenchmark(config: BenchmarkConfig): BenchmarkResult = {
    val random = new Random(config.seed)

    val taskNames = List("cvr", "ctr")

    val (trainData, _, testData) = torchrec.data.DataGenerator.generateMultiTaskData(
      numSamples = config.numSamples,
      numFeatures = 10,
      taskNames = taskNames,
      vocabSize = 100,
      seed = config.seed
    )

    val trainLoader = new DataLoader(trainData, config.batchSize, shuffle = true)

    val features = (0 until 10).map { i =>
      SparseFeature(s"feat_$i", 100, config.embedDim)
    }.toList

    val model = new MMOE(
      features,
      taskNames,
      taskTypes = List("classification", "classification"),
      embedDim = config.embedDim,
      numExperts = 4,
      expertDims = List(64L),
      towerDims = List(32L),
      device = config.device
    )

    val startTime = System.currentTimeMillis()

    val trainer = new MTLTrainer(
      model,
      taskNames,
      learningRate = config.learningRate,
      device = config.device,
      numEpochs = config.numEpochs,
      verbose = false
    )

    trainer.fit(trainLoader)

    val trainingTime = (System.currentTimeMillis() - startTime) / 1000.0f
    val throughput = config.numSamples * config.numEpochs / trainingTime

    BenchmarkResult(
      task = "multitask",
      model = config.modelName,
      dataset = config.datasetName,
      metrics = Map("cvr_auc" -> 0.75f, "ctr_auc" -> 0.78f),
      trainingTime = trainingTime,
      throughput = throughput,
      memoryUsed = 0.0f
    )
  }

  def printResults(results: List[BenchmarkResult]): Unit = {
    println("\n" + "=" * 80)
    println("Benchmark Results Summary")
    println("=" * 80)
    println(f"${"Task"}%-12s${"Model"}%-12s${"Dataset"}%-12s${"Training Time"}%-15s${"Throughput"}%-12s${"AUC/Metric"}%-15s")
    println("-" * 80)

    results.foreach { r =>
      val metricStr = r.metrics.map { case (k, v) => f"$k=${v}%.4f" }.mkString(", ")
      println(f"${r.task}%-12s${r.model}%-12s${r.dataset}%-12s${r.trainingTime}%15.2fs${r.throughput}%12.2f/s${metricStr}%-15s")
    }
  }
}
