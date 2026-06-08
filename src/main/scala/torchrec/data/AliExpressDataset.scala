package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.collection.mutable
import scala.jdk.CollectionConverters._

/**
 * AliExpress Multi-Task Dataset
 *
 * Dataset from real-world traffic logs of the search system in AliExpress.
 * Reference:
 *   https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
 *   Li, Pengcheng, et al. "Improving multi-scenario learning to rank in e-commerce
 *    by exploiting task relationships in the label space." CIKM 2020.
 *
 * Format:
 *   - 16 categorical features (search_id + categorical_1..categorical_15)
 *   - Numerical features (numerical_1..numerical_N)
 *   - 2 task labels: [click, conversion]
 *     - 0 -> [0, 0] (no interaction)
 *     - 1 -> [1, 0] (click, no conversion)
 *     - 2 -> [1, 1] (click + conversion)
 *
 * Available variants: AliExpress_NL, AliExpress_ES, AliExpress_FR, AliExpress_US
 *
 * @param datasetPath Path to the dataset directory containing train.csv and test.csv
 *                    Each CSV has columns: search_id, categorical_1..categorical_15,
 *                    numerical_1..numerical_N, click, conversion
 */
object AliExpressDataset {

  private val KaggleBaseUrl = "https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/ccorzhoudata/AliExpress/data/"
  private val DatasetVariants = List("NL", "ES", "FR", "US")

  // Column names from the preprocessing script
  private val CategoricalCols = List(
    "search_id", "categorical_1", "categorical_2", "categorical_3",
    "categorical_4", "categorical_5", "categorical_6", "categorical_7",
    "categorical_8", "categorical_9", "categorical_10", "categorical_11",
    "categorical_12", "categorical_13", "categorical_14", "categorical_15"
  )

  // Feature dimensions (max index + 1) per categorical column
  // These are determined by the preprocessing script in the old project
  private val CategoricalDims = Map(
    "search_id" -> 1000,
    "categorical_1" -> 1000,
    "categorical_2" -> 1000,
    "categorical_3" -> 1000,
    "categorical_4" -> 1000,
    "categorical_5" -> 1000,
    "categorical_6" -> 1000,
    "categorical_7" -> 1000,
    "categorical_8" -> 1000,
    "categorical_9" -> 1000,
    "categorical_10" -> 1000,
    "categorical_11" -> 1000,
    "categorical_12" -> 1000,
    "categorical_13" -> 1000,
    "categorical_14" -> 1000,
    "categorical_15" -> 1000
  )

  private val NumCategorical = CategoricalCols.size
  private val NumNumerical = 40  // inferred from the preprocessing script

  /**
   * Load AliExpress dataset from local files
   *
   * @param datasetPath Path to the dataset directory containing train.csv and test.csv
   * @param taskNames   Names for the two tasks (default: click, conversion)
   * @return (trainDataset, testDataset) as MultiTaskDataset
   */
  def load(
    datasetPath: String,
    taskNames: List[String] = List("click", "conversion")
  ): (MultiTaskDataset, MultiTaskDataset) = {
    println("=" * 60)
    println("AliExpress Multi-Task Dataset Loading")
    println("=" * 60)
    println(s"  Dataset path: $datasetPath")

    val trainFile = new java.io.File(datasetPath, "train.csv")
    val testFile = new java.io.File(datasetPath, "test.csv")

    if (!trainFile.exists()) {
      println(s"  [Warn] $trainFile not found. Generating synthetic data.")
      return generateSynthetic(taskNames)
    }

    println("  [Parse] Reading train.csv...")
    val trainRows = parseCSV(trainFile, train = true)
    println(s"  [Data] Train rows: ${trainRows.size}")

    val testRows = if (testFile.exists()) {
      println("  [Parse] Reading test.csv...")
      parseCSV(testFile, train = false)
    } else {
      println("  [Warn] test.csv not found. Using train split.")
      val split = (trainRows.size * 0.8).toInt
      trainRows.drop(split)
    }
    println(s"  [Data] Test rows: ${testRows.size}")

    if (trainRows.isEmpty) {
      println("  [Warn] No rows parsed. Generating synthetic data.")
      return generateSynthetic(taskNames)
    }

    val trainDS = buildMultiTaskDataset(trainRows, taskNames, "train")
    val testDS = buildMultiTaskDataset(testRows, taskNames, "test")

    val clickRate = trainRows.count(_._3(0) > 0.5f) * 100.0f / trainRows.size
    val convRate = trainRows.count(_._3(1) > 0.5f) * 100.0f / trainRows.size
    println(f"  [Data] Click rate: $clickRate%.2f%%")
    println(f"  [Data] Conversion rate: $convRate%.2f%%")
    println("=" * 60)

    (trainDS, testDS)
  }

  /**
   * Load from a specific AliExpress variant
   *
   * @param variant   One of NL, ES, FR, US
   * @param dataPath  Base path containing the variant subdirectory
   * @param taskNames Names for the two tasks
   */
  def loadVariant(
    variant: String,
    dataPath: String,
    taskNames: List[String] = List("click", "conversion")
  ): (MultiTaskDataset, MultiTaskDataset) = {
    require(DatasetVariants.contains(variant), s"variant must be one of $DatasetVariants, got $variant")
    val path = s"$dataPath/AliExpress_$variant"
    load(path, taskNames)
  }

  // Row format: (categorical indices, numerical values, labels [click, conversion])
  private case class AliExpressRow(
    categorical: Array[Float],
    numerical: Array[Float],
    labels: Array[Float]  // [click, conversion]
  )

  private def parseCSV(file: java.io.File, train: Boolean): Array[AliExpressRow] = {
    val builder = mutable.ArrayBuilder.make[AliExpressRow]
    val source = scala.io.Source.fromFile(file)
    val lines = source.getLines()
    var lineCount = 0

    // Skip header
    if (lines.hasNext) lines.next()

    for (line <- lines) {
      try {
        val fields = line.split(",")
        if (fields.length >= NumCategorical + 2) {
          // Parse categorical features (first 16 columns after label)
          val categorical = new Array[Float](NumCategorical)
          for (i <- 0 until NumCategorical) {
            val idx = i + 1  // after label column
            if (idx < fields.length && fields(idx) != null && fields(idx).nonEmpty) {
              categorical(i) = try fields(idx).toFloat catch { case _: Throwable => 0.0f }
            } else {
              categorical(i) = 0.0f
            }
          }

          // Parse numerical features (columns 17 to end-2)
          val numStart = NumCategorical + 1
          val numEnd = fields.length - 2
          val numCount = math.max(0, numEnd - numStart)
          val numerical = new Array[Float](NumNumerical)
          for (i <- 0 until math.min(numCount, NumNumerical)) {
            if (numStart + i < fields.length && fields(numStart + i) != null && fields(numStart + i).nonEmpty) {
              numerical(i) = try fields(numStart + i).toFloat catch { case _: Throwable => 0.0f }
            }
          }

          // Parse labels (last 2 columns: click, conversion)
          val labelRaw = fields.lastOption.map(_.toFloat).getOrElse(0.0f)
          val labels = labelRaw match {
            case 0.0f => Array(0.0f, 0.0f)  // no interaction
            case 1.0f => Array(1.0f, 0.0f)  // click, no conversion
            case 2.0f => Array(1.0f, 1.0f)  // click + conversion
            case _ => Array(labelRaw, 0.0f)  // fallback
          }

          builder += AliExpressRow(categorical, numerical, labels)
          lineCount += 1
          if (lineCount % 100000 == 0) println(s"    Parsed $lineCount rows...")
        }
      } catch {
        case _: Throwable =>
      }
    }
    source.close()
    builder.result()
  }

  private def buildMultiTaskDataset(
    rows: Seq[AliExpressRow],
    taskNames: List[String],
    name: String
  ): MultiTaskDataset = {
    val n = rows.size
    require(taskNames.size == 2, s"AliExpressDataset requires exactly 2 tasks, got ${taskNames.size}")

    // Build categorical features
    val categoricalArrays = Array.ofDim[Float](NumCategorical, n)
    rows.zipWithIndex.foreach { case (row, j) =>
      for (i <- 0 until NumCategorical) {
        categoricalArrays(i)(j) = row.categorical(i)
      }
    }

    val sparseFeatures = CategoricalCols.zipWithIndex.map { case (colName, i) =>
      val dim = CategoricalDims.getOrElse(colName, 1000)
      s"cat_$i" -> tensor(categoricalArrays(i), Array(n.toLong)).toType(ScalarType.Long)
    }.toMap

    // Build numerical features
    val numericalArrays = Array.ofDim[Float](NumNumerical, n)
    rows.zipWithIndex.foreach { case (row, j) =>
      for (i <- 0 until NumNumerical) {
        numericalArrays(i)(j) = row.numerical(i)
      }
    }

    val denseFeatures = (0 until NumNumerical).map { i =>
      s"numerical_$i" -> tensor(numericalArrays(i), Array(n.toLong))
    }.toMap

    // Combine sparse + dense into features
    val allFeatureNames = (0 until NumCategorical).map(i => s"cat_$i") ++ (0 until NumNumerical).map(i => s"numerical_$i")
    val combinedFeatures = mutable.Map[String, org.bytedeco.pytorch.Tensor]()

    // Create synthetic sparse tensors for categorical (already done)
    sparseFeatures.foreach { case (k, v) => combinedFeatures(k) = v }
    denseFeatures.foreach { case (k, v) => combinedFeatures(k) = v }

    // Build task labels
    val clickLabels = Array.ofDim[Float](n)
    val conversionLabels = Array.ofDim[Float](n)
    rows.zipWithIndex.foreach { case (row, j) =>
      clickLabels(j) = row.labels(0)
      conversionLabels(j) = row.labels(1)
    }

    val taskLabels = Map(
      taskNames(0) -> tensor(clickLabels, Array(n.toLong)),
      taskNames(1) -> tensor(conversionLabels, Array(n.toLong))
    )

    new MultiTaskDataset(combinedFeatures.toMap, taskLabels)
  }

  /** Generate realistic synthetic AliExpress data */
  private def generateSynthetic(
    taskNames: List[String] = List("click", "conversion")
  ): (MultiTaskDataset, MultiTaskDataset) = {
    val rng = new Random(42)
    val numTrain = 50000
    val numTest = 20000
    val n = numTrain + numTest

    println("  [Generate] Creating synthetic AliExpress data...")

    // Categorical features
    val categoricalArrays = Array.ofDim[Float](NumCategorical, n)
    for (j <- 0 until n) {
      for (i <- 0 until NumCategorical) {
        categoricalArrays(i)(j) = rng.nextInt(1000).toFloat
      }
    }

    // Numerical features
    val numericalArrays = Array.ofDim[Float](NumNumerical, n)
    for (j <- 0 until n) {
      for (i <- 0 until NumNumerical) {
        numericalArrays(i)(j) = rng.nextFloat() * 100
      }
    }

    // Labels: realistic multi-task labels
    val clickLabels = Array.ofDim[Float](n)
    val conversionLabels = Array.ofDim[Float](n)
    for (j <- 0 until n) {
      // Click probability based on features
      var clickScore = 0.0f
      for (i <- 0 until 5) {
        clickScore += (categoricalArrays(i)(j) / 1000) * (i + 1) * 0.1f
      }
      for (i <- 0 until 3) {
        clickScore += numericalArrays(i)(j) / 100 * 0.05f
      }
      val clickProb = 1.0f / (1.0f + math.exp(-clickScore).toFloat)
      val clicked = rng.nextFloat() < clickProb
      clickLabels(j) = if (clicked) 1.0f else 0.0f

      // Conversion given click
      if (clicked) {
        var convScore = 0.0f
        for (i <- 5 until 10) {
          convScore += (categoricalArrays(i)(j) / 1000) * (i - 4) * 0.1f
        }
        for (i <- 3 until 8) {
          convScore += numericalArrays(i)(j) / 100 * 0.05f
        }
        val convProb = 1.0f / (1.0f + math.exp(-convScore).toFloat)
        conversionLabels(j) = if (rng.nextFloat() < convProb) 1.0f else 0.0f
      } else {
        conversionLabels(j) = 0.0f
      }
    }

    // Build features
    val sparseFeatures = CategoricalCols.zipWithIndex.map { case (colName, i) =>
      s"cat_$i" -> tensor(categoricalArrays(i), Array(n.toLong)).toType(ScalarType.Long)
    }.toMap

    val denseFeatures = (0 until NumNumerical).map { i =>
      s"numerical_$i" -> tensor(numericalArrays(i), Array(n.toLong))
    }.toMap

    val combinedFeatures = (sparseFeatures.toSeq ++ denseFeatures.toSeq).toMap

    val allLabels = Map(
      taskNames(0) -> tensor(clickLabels, Array(n.toLong)),
      taskNames(1) -> tensor(conversionLabels, Array(n.toLong))
    )

    val trainFeatures = combinedFeatures.map { case (k, v) => k -> v.narrow(0, 0, numTrain) }.toMap
    val trainLabels = allLabels.map { case (k, v) => k -> v.narrow(0, 0, numTrain) }.toMap
    val testFeatures = combinedFeatures.map { case (k, v) => k -> v.narrow(0, numTrain, numTest) }.toMap
    val testLabels = allLabels.map { case (k, v) => k -> v.narrow(0, numTrain, numTest) }.toMap

    val trainDS = new MultiTaskDataset(trainFeatures, trainLabels)
    val testDS = new MultiTaskDataset(testFeatures, testLabels)

    val clickRate = clickLabels.take(numTrain).count(_ > 0.5f) * 100.0f / numTrain
    val convRate = conversionLabels.take(numTrain).count(_ > 0.5f) * 100.0f / numTrain
    println(f"  [Synthetic] Click rate: $clickRate%.2f%%")
    println(f"  [Synthetic] Conversion rate: $convRate%.2f%%")

    (trainDS, testDS)
  }
}
