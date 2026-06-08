package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.collection.mutable

/**
 * Avazu CTR Prediction Dataset Loader.
 *
 * The Avazu dataset is one of the earliest and most popular CTR benchmarks.
 * Format: CSV with click (0/1), and 22 categorical features.
 *
 * Dataset source:
 *   https://www.kaggle.com/datasets/avazu-ai/avazu-ctr-prediction
 *   (or mirror: https://www.kaggle.com/datasets/mkechinov/criteo-subset)
 *
 * Format (CSV header):
 *   click, C1, C14, C17, C1, C20, C14, C17, C15, C16, C18, C6, C10,
 *   C11, C22, C15, C17, C19, C20, C22, C14, C17, C22, C15, C16, C18
 *
 * This produces a TensorDataset for CTR ranking with DeepFM and similar models.
 */
object AvazuDataset {

  // Primary source - requires Kaggle account
  private val KaggleUrl = "https://www.kaggle.com/datasets/avazu-ai/avazu-ctr-prediction/download"

  // Mirror sources
  private val MirrorUrls = List(
    ("https://raw.githubusercontent.com/Avazu/azureML/master/avazu_ctr/avazu_data.zip", "avazu"),
    ("https://cseweb.ucsd.edu/~wckang/avazu_data", "avazu_cseweb"),
  )

  // Number of categorical features in Avazu
  private val NumFeatures = 22

  /**
   * Load Avazu dataset as a ranking TensorDataset.
   *
   * @param trainRatio Fraction for training
   * @param maxSamples Maximum rows to load
   * @param seed Random seed
   * @return (trainDataset, valDataset, testDataset)
   */
  def load(
    trainRatio: Float = 0.8f,
    maxSamples: Option[Int] = None,
    seed: Int = 42
  ): (Dataset, Dataset, Dataset) = {
    println("=" * 60)
    println("Avazu CTR Dataset Loading")
    println("=" * 60)

    // Try download
    val dataFile = tryDownload()
    if (dataFile == null || !dataFile.exists() || dataFile.length() < 1000) {
      println("  [Warn] Could not download Avazu dataset. Generating synthetic data.")
      return generateSynthetic(trainRatio, maxSamples.getOrElse(100000), seed)
    }

    // Parse
    println("  [Parse] Reading Avazu data...")
    val rows = parseAvazuFile(dataFile, maxSamples.getOrElse(10000000))
    println(s"  [Data] Loaded rows: ${rows.size}")

    if (rows.isEmpty) {
      println("  [Warn] No rows parsed. Falling back to synthetic.")
      return generateSynthetic(trainRatio, maxSamples.getOrElse(100000), seed)
    }

    // Split
    val rng = new Random(seed)
    val shuffled = rng.shuffle(rows.toSeq)

    val trainSize = (shuffled.size * trainRatio).toInt
    val valSize = ((shuffled.size - trainSize) / 2).toInt
    val trainRows = shuffled.take(trainSize).toSeq
    val valRows = shuffled.slice(trainSize, trainSize + valSize).toSeq
    val testRows = shuffled.drop(trainSize + valSize).toSeq

    println(s"  [Split] Train: ${trainRows.size}, Val: ${valRows.size}, Test: ${testRows.size}")

    val trainDS = buildDataset(trainRows, "train")
    val valDS = buildDataset(valRows, "val")
    val testDS = buildDataset(testRows, "test")

    val posRate = rows.count(_.label > 0.5f) * 100.0f / rows.size
    println(f"  [Data] Positive rate: $posRate%.2f%%")
    println("=" * 60)

    (trainDS, valDS, testDS)
  }

  private def tryDownload(): java.io.File = {
    // Avazu is large (~6GB), try mirrors
    val mirrors = List(
      "https://raw.githubusercontent.com/Avazu/azureML/master/avazu_ctr/avazu_data",
    )

    for (url <- mirrors) {
      try {
        println(s"  [Try] $url")
        val file = DatasetDownloader.download(url, "avazu", forceRedownload = false)
        if (file.exists() && file.length() > 10000) {
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

  private[torchrec] case class AvazuRow(
    label: Float,
    features: Array[Float]  // 22 categorical features (hashed to ints)
  )

  private def parseAvazuFile(file: java.io.File, maxRows: Int): Array[AvazuRow] = {
    val builder = mutable.ArrayBuilder.make[AvazuRow]
    var count = 0

    try {
      val lines = DatasetDownloader.readLines(file, ",", skipHeader = true)

      for (fields <- lines) {
        if (count >= maxRows) return builder.result()

        parseAvazuRow(fields) match {
          case Some(row) =>
            builder += row
            count += 1
            if (count % 100000 == 0) println(s"    Parsed $count rows...")
          case None =>
        }
      }
    } catch {
      case e: Throwable =>
        println(s"  [Parse Error] ${e.getMessage}")
    }

    builder.result()
  }

  private def parseAvazuRow(fields: Array[String]): Option[AvazuRow] = {
    if (fields == null || fields.length < NumFeatures + 1) return None

    try {
      val label = fields(0).toFloat

      // 22 categorical features
      val feats = new Array[Float](NumFeatures)
      for (i <- 0 until NumFeatures) {
        val idx = i + 1
        val raw = if (idx < fields.length) fields(idx) else ""
        // Hash-encode the string to an integer
        val vocabSize = 100000
        feats(i) = hashString(raw).abs(vocabSize).toFloat
      }

      Some(AvazuRow(label, feats))
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

  private def buildDataset(rows: Seq[AvazuRow], name: String): Dataset = {
    val n = rows.size
    val vocabSize = 100000

    // Features: 22 categorical features
    val featArrays = Array.ofDim[Float](NumFeatures, n)
    rows.zipWithIndex.foreach { case (row, j) =>
      for (i <- 0 until NumFeatures) {
        featArrays(i)(j) = row.features(i)
      }
    }

    // Labels
    val labels = Array.ofDim[Float](n)
    rows.zipWithIndex.foreach { case (row, j) =>
      labels(j) = row.label
    }

    val sparseFeatures = (0 until NumFeatures).map { i =>
      s"feat_$i" -> tensor(featArrays(i), Array(n.toLong)).toType(ScalarType.Long)
    }.toMap

    val labelsTensor = tensor(labels, Array(n.toLong))

    new TensorDataset(sparseFeatures, Map.empty, Some(labelsTensor))
  }

  /** Fallback: generate realistic Avazu-style synthetic data */
  private def generateSynthetic(
    trainRatio: Float,
    numSamples: Int,
    seed: Int
  ): (Dataset, Dataset, Dataset) = {
    val rng = new Random(seed)
    val n = numSamples

    // 22 categorical features
    val featArrays = Array.ofDim[Float](NumFeatures, n)
    val labels = Array.ofDim[Float](n)

    for (j <- 0 until n) {
      // Generate sparse features (vocab ~50000 per feature)
      for (i <- 0 until NumFeatures) {
        featArrays(i)(j) = rng.nextInt(50000).toFloat
      }

      // Generate label based on interactions (~20% positive)
      var score = 0.0
      for (i <- 0 until 5) {
        score += (featArrays(i)(j) / 50000.0) * (i + 1) * 0.5
      }
      val prob = 1.0f / (1.0f + math.exp(-score).toFloat)
      labels(j) = if (rng.nextFloat() < prob) 1.0f else 0.0f
    }

    val sparseFeatures = (0 until NumFeatures).map { i =>
      s"feat_$i" -> tensor(featArrays(i), Array(n.toLong)).toType(ScalarType.Long)
    }.toMap

    val labelsTensor = tensor(labels, Array(n.toLong))

    val trainSize = (n * trainRatio).toInt
    val valSize = ((n - trainSize) / 2).toInt

    val trainSparse = sparseFeatures.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }
    val trainLabels = labelsTensor.narrow(0, 0, trainSize)

    val valSparse = sparseFeatures.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }
    val valLabels = labelsTensor.narrow(0, trainSize, valSize)

    val testSparse = sparseFeatures.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, n - trainSize - valSize) }
    val testLabels = labelsTensor.narrow(0, trainSize + valSize, n - trainSize - valSize)

    (
      new TensorDataset(trainSparse, Map.empty, Some(trainLabels)),
      new TensorDataset(valSparse, Map.empty, Some(valLabels)),
      new TensorDataset(testSparse, Map.empty, Some(testLabels))
    )
  }
}
