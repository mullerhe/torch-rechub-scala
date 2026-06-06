package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.collection.mutable

/**
 * Census-Income (Adult) Dataset Loader.
 *
 * Downloads from UCI Machine Learning Repository:
 *   https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
 *
 * Format: 14 features + label (">50K" or "<=50K")
 *
 * Features:
 *   age, workclass, fnlwgt, education, education-num, marital-status,
 *   occupation, relationship, race, sex, capital-gain, capital-loss,
 *   hours-per-week, native-country, income
 *
 * This produces a TensorDataset for binary classification.
 */
object CensusIncomeDataset {

  private val TrainUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  private val TestUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
  private val DatasetName = "census_income"

  /**
   * Load Census-Income dataset.
   *
   * @param trainRatio Fraction for training (test file provides separate test set if None)
   * @param useOfficialTest Use official test split instead of splitting train data
   * @param seed Random seed
   * @return (trainDataset, valDataset, testDataset)
   */
  def load(
    trainRatio: Float = 0.8f,
    useOfficialTest: Boolean = false,
    seed: Int = 42
  ): (Dataset, Dataset, Dataset) = {
    println("=" * 60)
    println("Census-Income Dataset Loading")
    println("=" * 60)

    // Step 1: Download train data
    println("  [Download] Training data...")
    val trainFile = DatasetDownloader.download(TrainUrl, s"${DatasetName}_train")

    // Step 2: Download test data (if needed)
    val testFile = if (useOfficialTest) {
      println("  [Download] Official test data...")
      Some(DatasetDownloader.download(TestUrl, s"${DatasetName}_test"))
    } else None

    // Step 3: Parse
    println("  [Parse] Reading training data...")
    val trainRows = parseAdultFile(trainFile)
    println(s"  [Data] Training rows: ${trainRows.size}")

    val testRows = testFile match {
      case Some(f) =>
        println("  [Parse] Reading test data...")
        parseAdultFile(f).map { row =>
          // Fix test file label format (has trailing dot)
          row.copy(label = row.label)
        }
      case None => Seq.empty[CensusRow]
    }
    if (testRows.nonEmpty) println(s"  [Data] Test rows: ${testRows.size}")

    // Step 4: Split or use official split
    val (trainDS, valDS, testDS) = if (useOfficialTest && testRows.nonEmpty) {
      // Split train rows into train/val
      val rng = new Random(seed)
      val shuffled = rng.shuffle(trainRows)
      val trainSize = (shuffled.size * trainRatio).toInt
      val valSize = shuffled.size - trainSize
      val trainSubset = shuffled.take(trainSize)
      val valSubset = shuffled.drop(trainSize)

      println(s"  [Split] Train: ${trainSubset.size}, Val: ${valSubset.size}, Test: ${testRows.size}")

      (
        buildDataset(trainSubset, "train"),
        buildDataset(valSubset, "val"),
        buildDataset(testRows, "test")
      )
    } else {
      // Split into 3 parts
      val rng = new Random(seed)
      val shuffled = rng.shuffle(trainRows)
      val trainSize = (shuffled.size * trainRatio).toInt
      val valSize = ((shuffled.size - trainSize) / 2).toInt
      val trainPart = shuffled.take(trainSize)
      val valPart = shuffled.slice(trainSize, trainSize + valSize)
      val testPart = shuffled.drop(trainSize + valSize)

      println(s"  [Split] Train: ${trainPart.size}, Val: ${valPart.size}, Test: ${testPart.size}")

      (
        buildDataset(trainPart, "train"),
        buildDataset(valPart, "val"),
        buildDataset(testPart, "test")
      )
    }

    val posRate = trainRows.count(_.label > 0.5f) * 100.0f / trainRows.size
    println(f"  [Data] Positive rate: $posRate%.2f%%")
    println("=" * 60)

    (trainDS, valDS, testDS)
  }

  // Feature indices in the Adult dataset (14 columns before label)
  // age, workclass, fnlwgt, education, education-num, marital-status, occupation,
  // relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
  private val FeatureNames = List(
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country"
  )

  // Categorical features that need encoding
  private val CategoricalIndices = Set(1, 3, 5, 6, 7, 8, 9, 13)  // workclass, education, marital, occupation, relationship, race, sex, country
  // Numerical features
  private val NumericalIndices = Set(0, 2, 4, 10, 11, 12)  // age, fnlwgt, education_num, capital_gain, capital_loss, hours

  // Build vocabularies from training data
  private var vocabularies: Map[Int, Map[String, Int]] = Map.empty

  private[torchrec] case class CensusRow(
    features: Array[Float],   // 14 features
    label: Float              // 0 or 1
  )

  private def parseAdultFile(file: java.io.File): Seq[CensusRow] = {
    val builder = mutable.ArrayBuilder.make[CensusRow]

    val lines = DatasetDownloader.readLines(file, ", ", skipHeader = false)

    for (fields <- lines) {
      parseAdultRow(fields) match {
        case Some(row) => builder += row
        case None =>
      }
    }

    builder.result()
  }

  private def parseAdultRow(fields: Array[String]): Option[CensusRow] = {
    if (fields == null || fields.length < 15) return None

    try {
      // Clean fields: remove leading/trailing whitespace
      val cleaned = fields.map(_.trim)

      // Build feature array (14 features)
      val features = new Array[Float](14)

      for (i <- cleaned.indices if i < 14) {
        val raw = cleaned(i)
        if (raw == "?" || raw.isEmpty) {
          // Missing value → 0
          features(i) = 0.0f
        } else if (CategoricalIndices.contains(i)) {
          // Categorical → will be encoded later
          features(i) = hashCategory(i, raw)
        } else {
          // Numerical
          features(i) = try raw.toFloat catch { case _: Throwable => 0.0f }
        }
      }

      // Label: ">50K" (1) or "<=50K" (0)
      val labelRaw = cleaned(14)
      val label = if (labelRaw.contains(">50K") || labelRaw.contains(">50")) 1.0f else 0.0f

      Some(CensusRow(features, label))
    } catch {
      case _: Throwable => None
    }
  }

  /** Simple deterministic hash for categorical values */
  private def hashCategory(featureIdx: Int, value: String): Float = {
    val key = s"$featureIdx:$value"
    var h = 0
    var i = 0
    while (i < key.length) {
      h = 31 * h + key.charAt(i)
      i += 1
    }
    // Map to [0, vocabSize) where vocabSize ≈ 100 per feature
    val vocabSize = 100
    val x = h % vocabSize
    if (x < 0) x + vocabSize else x
  }

  private def buildDataset(rows: Seq[CensusRow], name: String): Dataset = {
    if (rows.isEmpty) {
      // Return empty dataset
      val emptySparse = Map("workclass" -> tensor(Array(0L), Array(1L)))
      return new TensorDataset(emptySparse, Map.empty, Some(tensor(Array(0f), Array(1L))))
    }

    val n = rows.size

    // Build feature columns
    val featureCols = Array.ofDim[Float](14, n)
    rows.zipWithIndex.foreach { case (row, j) =>
      for (i <- 0 until 14) {
        featureCols(i)(j) = row.features(i)
      }
    }

    // Labels
    val labels = Array.ofDim[Float](n)
    rows.zipWithIndex.foreach { case (row, j) =>
      labels(j) = row.label
    }

    // Build sparse features (categorical) and dense features (numerical)
    val sparseFeatures = mutable.Map[String, org.bytedeco.pytorch.Tensor]()
    val denseFeatures = mutable.Map[String, org.bytedeco.pytorch.Tensor]()

    for (i <- 0 until 14) {
      val featName = FeatureNames(i)
      if (CategoricalIndices.contains(i)) {
        sparseFeatures(featName) = tensor(featureCols(i), Array(n.toLong)).toType(ScalarType.Long)
      } else {
        denseFeatures(featName) = tensor(featureCols(i), Array(n.toLong))
      }
    }

    new TensorDataset(sparseFeatures.toMap, denseFeatures.toMap, Some(tensor(labels, Array(n.toLong))))
  }
}
