package benchmarks

import torchrec.data._
import torchrec.basic.features._
import torchrec.models.ranking._
import torchrec.models.matching._
import torchrec.trainers._

/**
 * Benchmark for realistic datasets (MovieLens-style, Census-style, Criteo-style)
 */
object RealDataBenchmark {

  def main(args: Array[String]): Unit = {
    println("=" * 70)
    println("Realistic Dataset Benchmark")
    println("=" * 70)

    val results = scala.collection.mutable.ListBuffer[(String, String, Float, Float)]()

    // MovieLens-style DSSM
    results += benchmarkMovieLensDSSM()

    // Census-style DeepFM
    results += benchmarkCensusDeepFM()

    // Criteo-style DeepFM
    results += benchmarkCriteoDeepFM()

    printResults(results.toList)
  }

  def benchmarkMovieLensDSSM(): (String, String, Float, Float) = {
    println("\n--- MovieLens DSSM Benchmark ---")

    try {
      val (train, _, _) = DataGenerator.generateMovieLensData(50000, 6040, 3952, 0.8f, 42)
      println(s"  Data: train=${train.size}")

      val loader = DataLoader.fromJavaRandom(train, batchSize = 256)

      val userFeat = SparseFeature("user_id", 6040, 16)
      val movieFeat = SparseFeature("movie_id", 3952, 16)

      val model = new DSSM(List(userFeat), List(movieFeat), 16, List(64L, 32L), 0.2f, "cpu")

      val trainer = new MatchTrainer(model, learningRate = 0.001f, numEpochs = 3, verbose = false)
      val t0 = System.currentTimeMillis()
      trainer.fit(loader)
      val time = (System.currentTimeMillis() - t0) / 1000.0f

      println(f"  Time: $time%.2fs")
      ("MovieLens", "DSSM", time, 0.0f)
    } catch {
      case e: Throwable =>
        println(s"  Error: ${e.getMessage}")
        e.printStackTrace()
        ("MovieLens", "DSSM", 0.0f, 0.0f)
    }
  }

  def benchmarkCensusDeepFM(): (String, String, Float, Float) = {
    println("\n--- Census-Income DeepFM Benchmark ---")

    try {
      val (train, vali, _) = DataGenerator.generateCensusData(30000, 0.8f, 42)
      println(s"  Data: train=${train.size}, val=${vali.size}")

      val loader = DataLoader.fromJavaRandom(train, batchSize = 256)
      val valLoader = DataLoader.fromJavaSequential(vali, batchSize = 256)

      val features = List(
        SparseFeature("workclass", 8, 4),
        SparseFeature("education", 9, 4),
        SparseFeature("marital", 5, 4),
        SparseFeature("occupation", 9, 4),
        SparseFeature("sex", 2, 4)
      )

      val model = new DeepFM(features, 8, List(32L, 16L), 0.2f, "cpu")

      val trainer = new CTRTrainer(model, learningRate = 0.01f, numEpochs = 3, verbose = false)
      val t0 = System.currentTimeMillis()
      trainer.fit(loader, Some(valLoader))
      val time = (System.currentTimeMillis() - t0) / 1000.0f

      val metrics = trainer.evaluate(valLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)

      println(f"  Time: $time%.2fs, AUC: $auc%.4f")
      ("Census", "DeepFM", time, auc)
    } catch {
      case e: Throwable =>
        println(s"  Error: ${e.getMessage}")
        e.printStackTrace()
        ("Census", "DeepFM", 0.0f, 0.0f)
    }
  }

  def benchmarkCriteoDeepFM(): (String, String, Float, Float) = {
    println("\n--- Criteo-style DeepFM Benchmark ---")

    try {
      val (train, vali, _) = DataGenerator.generateRankingData(50000, 26, 13, 40000, 0.8f, 0.1f, 42)
      println(s"  Data: train=${train.size}, val=${vali.size}")

      val loader = DataLoader.fromJavaRandom(train, batchSize = 512)
      val valLoader = DataLoader.fromJavaSequential(vali, batchSize = 512)

      val features = (0 until 26).map(i => SparseFeature(s"feat_$i", 40000, 8)).toList

      val model = new DeepFM(features, 8, List(128L, 64L), 0.2f, "cpu")

      val trainer = new CTRTrainer(model, learningRate = 0.01f, numEpochs = 3, verbose = false)
      val t0 = System.currentTimeMillis()
      trainer.fit(loader, Some(valLoader))
      val time = (System.currentTimeMillis() - t0) / 1000.0f

      val metrics = trainer.evaluate(valLoader)
      val auc = metrics.getOrElse("AUC", 0.0f)

      println(f"  Time: $time%.2fs, AUC: $auc%.4f")
      ("Criteo", "DeepFM", time, auc)
    } catch {
      case e: Throwable =>
        println(s"  Error: ${e.getMessage}")
        e.printStackTrace()
        ("Criteo", "DeepFM", 0.0f, 0.0f)
    }
  }

  def printResults(results: List[(String, String, Float, Float)]): Unit = {
    println("\n" + "=" * 65)
    println("Realistic Dataset Benchmark Results")
    println("=" * 65)
    println(f"${"Dataset"}%-15s${"Model"}%-15s${"Time(s)"}%-12s${"AUC"}%-10s")
    println("-" * 65)

    results.foreach { case (ds, model, time, auc) =>
      val aucStr = if (auc > 0) f"$auc%.4f" else "-"
      println(f"$ds%-15s$model%-15s${time}%12.2f$aucStr%-10s")
    }
  }
}