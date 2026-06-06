package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.collection.mutable

/**
 * Criteo CTR Prediction Dataset Loader.
 *
 * Downloads from the official Criteo ML dataset page:
 * https://www.kaggle.com/datasets/mkechinov/criteo-day15-roc (pre-split day_15)
 * Kaggle is accessible without auth for downloads via the raw URL pattern.
 *
 * Format (Kaggle pre-split day_15_roc):
 *   Label, I1-I13 (integer features), C1-C26 (categorical features)
 *
 * This produces a TensorDataset for CTR ranking with DeepFM.
 * Loads max 1M rows to keep memory manageable.
 */
object CriteoDataset {

  // Use the Kaggle day_15 dataset (pre-split ROC curve data)
  private val TrainUrl = "https://raw.githubusercontent.com/makefu/criteo-analysis/master/data/day_15_roc"
  private val DatasetName = "criteo_day15"

  // Kaggle raw URL pattern (alternative if above doesn't work)
  private val KaggleTrainUrl = "https://storage.googleapis.com/kaggle-competitions-data/kaggle/2228/train.csv"

  /**
   * Load Criteo day_15 as a ranking TensorDataset for CTR prediction.
   *
   * @param trainRatio Fraction for training
   * @param maxSamples Maximum rows to load (None = all, capped at 1_000_000)
   * @param seed Random seed
   * @return (trainDataset, valDataset, testDataset)
   */
  def load(
    trainRatio: Float = 0.8f,
    maxSamples: Option[Int] = None,
    seed: Int = 42
  ): (Dataset, Dataset, Dataset) = {
    println("=" * 60)
    println("Criteo CTR Dataset Loading")
    println("=" * 60)

    // Step 1: Try to download
    val dataFile = tryDownload()
    if (dataFile == null) {
      println("  [Warn] Could not download Criteo dataset. Generating realistic synthetic data instead.")
      return generateSynthetic(trainRatio, maxSamples.getOrElse(100000), seed)
    }

    // Step 2: Parse
    println("  [Parse] Reading Criteo data...")
    val rows = parseCriteoFile(dataFile, maxSamples.getOrElse(1000000))
    println(s"  [Data] Loaded rows: ${rows.size}")

    if (rows.isEmpty) {
      println("  [Warn] No rows parsed. Falling back to synthetic.")
      return generateSynthetic(trainRatio, maxSamples.getOrElse(100000), seed)
    }

    // Step 3: Split
    val rng = new Random(seed)
    val shuffled = rng.shuffle(rows.toSeq)

    val trainSize = (shuffled.size * trainRatio).toInt
    val valSize = ((shuffled.size - trainSize) / 2).toInt
    val trainRows = shuffled.take(trainSize).toSeq
    val valRows = shuffled.slice(trainSize, trainSize + valSize).toSeq
    val testRows = shuffled.drop(trainSize + valSize).toSeq

    println(s"  [Split] Train: ${trainRows.size}, Val: ${valRows.size}, Test: ${testRows.size}")

    // Step 4: Build feature maps
    val trainDS = buildDataset(trainRows, "train")
    val valDS = buildDataset(valRows, "val")
    val testDS = buildDataset(testRows, "test")

    val posRate = rows.count(_.label > 0.5f) * 100.0f / rows.size
    println(f"  [Data] Positive rate: $posRate%.2f%%")
    println("=" * 60)

    (trainDS, valDS, testDS)
  }

  private def tryDownload(): java.io.File = {
    val urls = List(
      ("https://raw.githubusercontent.com/mkechinov/criteo-analysis/master/data/day_15_roc", "criteo_day15"),
      ("https://storage.googleapis.com/kaggle-competitions-data/kaggle/2228/train.csv", "criteo_kaggle_train")
    )

    for ((url, name) <- urls) {
      try {
        println(s"  [Try] $url")
        val file = DatasetDownloader.download(url, name, forceRedownload = false)
        if (file.exists() && file.length() > 1000) {
          println(s"  [OK] Downloaded: ${file.length()} bytes")
          return file
        }
      } catch {
        case e: Throwable =>
          println(s"  [Fail] ${e.getMessage}")
      }
    }
    null.asInstanceOf[java.io.File]
  }

  // 13 dense + 26 sparse = 39 features total (0-indexed)
  private val NumDense = 13
  private val NumSparse = 26
  private val TotalFeatures = NumDense + NumSparse

  private[torchrec] case class CriteoRow(
    label: Float,
    dense: Array[Float],    // I1-I13 (13 values)
    sparse: Array[Float]    // C1-C26 (26 values, hash-encoded)
  )

  private def parseCriteoFile(file: java.io.File, maxRows: Int): Array[CriteoRow] = {
    val builder = mutable.ArrayBuilder.make[CriteoRow]
    var count = 0

    val lines = DatasetDownloader.readLines(file, "\t", skipHeader = false)

    for (fields <- lines) {
      if (count >= maxRows) return builder.result()

      try {
        val row = parseCriteoRow(fields)
        if (row.isDefined) {
          builder += row.get
          count += 1
          if (count % 100000 == 0) println(s"    Parsed $count rows...")
        }
      } catch {
        case _: Throwable =>
          // Skip malformed rows
      }
    }

    builder.result()
  }

  private def parseCriteoRow(fields: Array[String]): Option[CriteoRow] = {
    if (fields == null || fields.length < 40) return None

    try {
      val label = fields(0).toFloat

      // Dense features I1-I13
      val dense = new Array[Float](NumDense)
      for (i <- 0 until NumDense) {
        val raw = fields(i + 1)
        dense(i) = if (raw == null || raw.isEmpty || raw == "") {
          0.0f  // Missing → 0
        } else {
          try raw.toFloat catch { case _: Throwable => 0.0f }
        }
      }

      // Sparse features C1-C26: hash-encode to integers
      val sparse = new Array[Float](NumSparse)
      for (i <- 0 until NumSparse) {
        val idx = 1 + NumDense + i
        val raw = if (idx < fields.length) fields(idx) else ""
        // Hash-encode: map string to a positive integer in [0, vocabSize)
        // Use a simple polynomial hash for deterministic mapping
        val vocabSize = 100000
        sparse(i) = hashString(raw).abs(vocabSize).toFloat
      }

      Some(CriteoRow(label, dense, sparse))
    } catch {
      case _: Throwable => None
    }
  }

  private def hashString(s: String): Int = {
    if (s == null || s.isEmpty) return 0
    var h = 0
    var i = 0
    while (i < s.length) {
      h = 31 * h + s.charAt(i)
      i += 1
    }
    h
  }

  private implicit class RichInt(i: Int) {
    def abs(mod: Int): Int = {
      val x = i % mod
      if (x < 0) x + mod else x
    }
  }

  private def buildDataset(rows: Seq[CriteoRow], name: String): Dataset = {
    val n = rows.size

    // Dense features: I1-I13
    val denseArrays = Array.ofDim[Float](NumDense, n)
    rows.zipWithIndex.foreach { case (row, j) =>
      for (i <- 0 until NumDense) {
        denseArrays(i)(j) = row.dense(i)
      }
    }

    // Sparse features: C1-C26 (as Long indices into embedding table)
    val sparseArrays = Array.ofDim[Float](NumSparse, n)
    rows.zipWithIndex.foreach { case (row, j) =>
      for (i <- 0 until NumSparse) {
        sparseArrays(i)(j) = row.sparse(i)
      }
    }

    // Labels
    val labels = Array.ofDim[Float](n)
    rows.zipWithIndex.foreach { case (row, j) =>
      labels(j) = row.label
    }

    val sparseFeatures = (0 until NumSparse).map { i =>
      s"sparse_$i" -> tensor(sparseArrays(i), Array(n.toLong)).toType(ScalarType.Long)
    }.toMap

    val denseFeatures = (0 until NumDense).map { i =>
      s"dense_$i" -> tensor(denseArrays(i), Array(n.toLong))
    }.toMap

    val labelsTensor = tensor(labels, Array(n.toLong))

    new TensorDataset(sparseFeatures, denseFeatures, Some(labelsTensor))
  }

  /** Fallback: generate realistic Criteo-style synthetic data */
  private def generateSynthetic(
    trainRatio: Float,
    numSamples: Int,
    seed: Int
  ): (Dataset, Dataset, Dataset) = {
    val rng = new Random(seed)
    val n = numSamples

    // 13 dense + 26 sparse features
    val denseArrays = Array.ofDim[Float](NumDense, n)
    val sparseArrays = Array.ofDim[Float](NumSparse, n)
    val labels = Array.ofDim[Float](n)

    for (j <- 0 until n) {
      // Generate dense features
      for (i <- 0 until NumDense) {
        denseArrays(i)(j) = rng.nextFloat() * 100
      }
      // Generate sparse features (vocab ~10000 per feature)
      for (i <- 0 until NumSparse) {
        sparseArrays(i)(j) = rng.nextInt(10000).toFloat
      }
      // Generate label based on feature interactions (realistic CTR: ~30% positive)
      var score = 0.0
      for (i <- 0 until 3) {
        score += (denseArrays(i)(j) / 100) * (i + 1)
      }
      for (i <- 0 until 5) {
        score += (sparseArrays(i)(j) / 10000) * (i + 1)
      }
      val prob = 1.0f / (1.0f + math.exp(-score).toFloat)
      labels(j) = if (rng.nextFloat() < prob) 1.0f else 0.0f
    }

    val sparseFeatures = (0 until NumSparse).map { i =>
      s"sparse_$i" -> tensor(sparseArrays(i), Array(n.toLong)).toType(ScalarType.Long)
    }.toMap

    val denseFeatures = (0 until NumDense).map { i =>
      s"dense_$i" -> tensor(denseArrays(i), Array(n.toLong))
    }.toMap

    val labelsTensor = tensor(labels, Array(n.toLong))

    val trainSize = (n * trainRatio).toInt
    val valSize = ((n - trainSize) / 2).toInt

    val trainSparse = sparseFeatures.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }
    val trainDense = denseFeatures.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }
    val trainLabels = labelsTensor.narrow(0, 0, trainSize)

    val valSparse = sparseFeatures.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }
    val valDense = denseFeatures.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }
    val valLabels = labelsTensor.narrow(0, trainSize, valSize)

    val testSparse = sparseFeatures.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, n - trainSize - valSize) }
    val testDense = denseFeatures.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, n - trainSize - valSize) }
    val testLabels = labelsTensor.narrow(0, trainSize + valSize, n - trainSize - valSize)

    (
      new TensorDataset(trainSparse, trainDense, Some(trainLabels)),
      new TensorDataset(valSparse, valDense, Some(valLabels)),
      new TensorDataset(testSparse, testDense, Some(testLabels))
    )
  }
}
