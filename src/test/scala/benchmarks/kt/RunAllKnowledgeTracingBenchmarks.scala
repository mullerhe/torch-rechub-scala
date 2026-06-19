package benchmarks.kt
object RunAllKnowledgeTracingBenchmarks {
  private val models = Vector(
    "DKT", "DKTForget", "DKVMN", "DeepIRT", "GKT", "AKT", "ATDKT", "ATKT", "CSKT", "DIMKT",
    "IEKT", "LPKT", "MTKT", "PromptKT", "QDKT", "RKT", "RobustKT", "SAINT", "SAINTPlusPlus",
    "SAKT", "SKVMN", "StableKT", "UKT"
  )
  private val datasets = Vector("assist0910", "assist15", "KDD", "statics11")
  def main(args: Array[String]): Unit = {
    val p = KnowledgeTracingHarness.Args.parse(args)
    val datasetRoot = p.getOrElse("dataset_root", "/home/muller/IdeaProjects/Knowledge-Tracing-Datasets")
    val epochs = p.getOrElse("epochs", "1").toInt
    val batchSize = p.getOrElse("batch_size", "32").toInt
    val maxSeqLen = p.getOrElse("max_seq_len", "200").toInt
    val stride = p.getOrElse("window_stride", p.getOrElse("max_seq_len", "200")).toInt
    val lr = p.getOrElse("lr", "0.001").toFloat
    val embedDim = p.getOrElse("embed_dim", "64").toInt
    val device = p.getOrElse("device", "cpu")
    val seed = p.getOrElse("seed", "2026").toInt
    val limitTrain = p.getOrElse("limit_train_sequences", "0").toInt
    val limitTest = p.getOrElse("limit_test_sequences", "0").toInt
    var failures = Vector.empty[String]
    for (dataset <- datasets; model <- models) {
      try {
        val result = KnowledgeTracingHarness.run(
          KnowledgeTracingHarness.KTRunConfig(
            datasetRoot = datasetRoot,
            datasetName = dataset,
            modelName = model,
            epochs = epochs,
            batchSize = batchSize,
            maxSeqLen = maxSeqLen,
            windowStride = stride,
            learningRate = lr,
            embedDim = embedDim,
            device = device,
            seed = seed,
            limitTrainSequences = limitTrain,
            limitTestSequences = limitTest
          )
        )
        val metricStr = result.metrics.toSeq.sortBy(_._1).map { case (k, v) => f"$k=$v%.4f" }.mkString(", ")
        println(s"[PASS] $dataset / $model -> $metricStr")
      } catch {
        case e: Throwable =>
          val msg = Option(e.getMessage).getOrElse(e.getClass.getSimpleName)
          failures :+= s"$dataset / $model -> $msg"
          println(s"[FAIL] $dataset / $model -> $msg")
      }
    }
    if (failures.nonEmpty) {
      println("\nFailures:")
      failures.foreach(f => println(s"  - $f"))
      throw new RuntimeException(s"Knowledge tracing suite failed: ${failures.size} case(s)")
    }
  }
}
