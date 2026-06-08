package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.collection.mutable

/**
 * Netflix Prize Dataset Loader.
 *
 * The Netflix Prize was a famous recommendation competition.
 * Dataset: user ratings (1-5) of movies.
 *
 * Dataset source:
 *   https://www.kaggle.com/netflix-inc/netflix-prize-data
 *
 * Format (probe set / training set):
 *   MovieID::CustomerID::Rating::Date
 *
 * This produces a MatchingDataset for two-tower retrieval.
 * Each sample is a (user, movie) pair with rating as label.
 */
object NetflixDataset {

  // Netflix Prize data files (need manual download from Kaggle)
  private val NetflixFiles = List(
    "training_set",
    "probe",
  )

  /**
   * Load Netflix Prize dataset as a MatchingDataset.
   *
   * @param trainRatio Fraction for training
   * @param maxRatings Maximum number of ratings to load
   * @param seed Random seed
   * @return (trainDataset, valDataset, testDataset) as MatchingDataset
   */
  def load(
    trainRatio: Float = 0.8f,
    maxRatings: Option[Int] = None,
    seed: Int = 42
  ): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    println("=" * 60)
    println("Netflix Prize Dataset Loading")
    println("=" * 60)

    // Try to load from local files first
    val ratings = tryLoadLocal(maxRatings.getOrElse(10000000))
    println(s"  [Data] Ratings loaded: ${ratings.size}")

    if (ratings.isEmpty) {
      println("  [Warn] No local files found. Generating synthetic data.")
      return generateSynthetic(maxRatings.getOrElse(500000), seed)
    }

    // Build user/item vocabularies
    val (userIdMap, itemIdMap) = buildVocabularies(ratings)
    val numUsers = userIdMap.size
    val numItems = itemIdMap.size
    println(s"  [Data] Users: $numUsers, Movies: $numItems")

    // Group by user
    val userRatings: Map[Int, Set[Int]] = ratings.groupBy(r => userIdMap(r.customerId)).view.mapValues {
      userRatings => userRatings.map(r => itemIdMap(r.movieId)).toSet
    }.toMap

    // Split
    val rng = new Random(seed)
    val allUsers = userIdMap.values.toSeq.sorted
    val shuffledUsers = rng.shuffle(allUsers)
    val trainSize = (shuffledUsers.size * trainRatio).toInt
    val valSize = ((shuffledUsers.size - trainSize) / 2).toInt
    val trainUsers = shuffledUsers.take(trainSize).toSet
    val valUsers = shuffledUsers.slice(trainSize, trainSize + valSize).toSet
    val testUsers = shuffledUsers.drop(trainSize + valSize).toSet

    val trainRatings = ratings.filter(r => trainUsers.contains(userIdMap(r.customerId)))
    val valRatings = ratings.filter(r => valUsers.contains(userIdMap(r.customerId)))
    val testRatings = ratings.filter(r => testUsers.contains(userIdMap(r.customerId)))

    println(s"  [Split] Train: ${trainRatings.size} / ${trainUsers.size} users")
    println(s"  [Split] Val: ${valRatings.size} / ${valUsers.size} users")
    println(s"  [Split] Test: ${testRatings.size} / ${testUsers.size} users")

    val negRatio = 4
    val trainDS = buildMatchingDataset(trainRatings, userRatings, negRatio, numUsers, numItems)
    val valDS = buildMatchingDataset(valRatings, userRatings, negRatio, numUsers, numItems)
    val testDS = buildMatchingDataset(testRatings, userRatings, negRatio, numUsers, numItems)

    println("=" * 60)
    (trainDS, valDS, testDS)
  }

  private def tryLoadLocal(maxRatings: Int): Array[NetflixRating] = {
    val baseDirs = List(
      new java.io.File("data/netflix"),
      new java.io.File("./data/netflix"),
      new java.io.File("../data/netflix"),
      new java.io.File("/data/netflix"),
    )

    for (baseDir <- baseDirs) {
      val trainingFile = new java.io.File(baseDir, "training_set")
      if (trainingFile.exists()) {
        try {
          println(s"  [Try] Loading from $trainingFile")
          return parseNetflixFile(trainingFile, maxRatings)
        } catch {
          case e: Throwable =>
            println(s"  [Fail] ${e.getMessage}")
        }
      }
    }

    // Try current directory
    val currentDir = new java.io.File(".")
    for (f <- currentDir.listFiles()) {
      if (f.isDirectory && f.getName.contains("netflix")) {
        val trainingFile = new java.io.File(f, "training_set")
        if (trainingFile.exists()) {
          try {
            println(s"  [Try] Loading from $trainingFile")
            return parseNetflixFile(trainingFile, maxRatings)
          } catch {
            case e: Throwable =>
              println(s"  [Fail] ${e.getMessage}")
          }
        }
      }
    }

    Array.empty
  }

  private[torchrec] case class NetflixRating(
    movieId: String,
    customerId: String,
    rating: Float,       // 1-5 stars
    date: String
  )

  private def parseNetflixFile(file: java.io.File, maxRatings: Int): Array[NetflixRating] = {
    val builder = mutable.ArrayBuilder.make[NetflixRating]
    var count = 0

    try {
      val source = scala.io.Source.fromFile(file)
      val lines = source.getLines()

      var currentMovieId = ""
      for (line <- lines) {
        if (count >= maxRatings) return builder.result()

        if (line.endsWith(":")) {
          // Movie ID line
          currentMovieId = line.stripSuffix(":")
        } else {
          // Rating line: CustomerID::Rating::Date
          val parts = line.split("::")
          if (parts.length >= 3) {
            try {
              val customerId = parts(0)
              val rating = parts(1).toFloat
              val date = parts(2)
              builder += NetflixRating(currentMovieId, customerId, rating, date)
              count += 1
              if (count % 500000 == 0) println(s"    Parsed $count ratings...")
            } catch {
              case _: Throwable =>
            }
          }
        }
      }
      source.close()
    } catch {
      case e: Throwable =>
        println(s"  [Parse Error] ${e.getMessage}")
    }

    builder.result()
  }

  private def buildVocabularies(ratings: Array[NetflixRating]): (Map[String, Int], Map[String, Int]) = {
    val userSet = mutable.Set[String]()
    val itemSet = mutable.Set[String]()
    ratings.foreach { r =>
      userSet += r.customerId
      itemSet += r.movieId
    }
    val sortedUsers = userSet.toList.sorted
    val sortedItems = itemSet.toList.sorted
    val userMap = sortedUsers.zipWithIndex.toMap
    val itemMap = sortedItems.zipWithIndex.toMap
    (userMap, itemMap)
  }

  private def buildMatchingDataset(
    ratings: Seq[NetflixRating],
    userRatings: Map[Int, Set[Int]],
    negRatio: Int,
    numUsers: Int,
    numItems: Int
  ): MatchingDataset = {
    val rng = new Random(42)

    val ratingMap: Map[String, Set[String]] = ratings.groupBy(_.customerId).view.mapValues {
      rs => rs.map(_.movieId).toSet
    }.toMap

    val sampleCount = ratings.size * (1 + negRatio)
    val userArr = new Array[Float](sampleCount)
    val itemArr = new Array[Float](sampleCount)
    val labelArr = new Array[Float](sampleCount)

    var idx = 0
    for (rating <- ratings) {
      val userHash = math.abs(rating.customerId.hashCode % numUsers).toFloat
      val itemHash = math.abs(rating.movieId.hashCode % numItems).toFloat

      // Positive
      userArr(idx) = userHash; itemArr(idx) = itemHash
      // Binary label: high rating (>=4) = 1, low = 0
      labelArr(idx) = if (rating.rating >= 4.0f) 1.0f else 0.0f
      idx += 1

      // Negatives
      val posSet = ratingMap.getOrElse(rating.customerId, Set.empty)
      var negCount = 0
      var attempts = 0
      while (negCount < negRatio && attempts < numItems) {
        val negItem = rng.nextInt(numItems)
        if (!posSet.map(s => math.abs(s.hashCode % numItems)).contains(negItem)) {
          userArr(idx) = userHash; itemArr(idx) = negItem.toFloat
          labelArr(idx) = 0.0f; idx += 1; negCount += 1
        }
        attempts += 1
      }
    }

    val userFeat = tensor(userArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
    val itemFeat = tensor(itemArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
    val labels = tensor(labelArr.slice(0, idx), Array(idx.toLong))

    new MatchingDataset(Map("user_id" -> userFeat), Map("item_id" -> itemFeat), Some(labels))
  }

  /** Fallback: generate realistic Netflix-style synthetic data */
  private def generateSynthetic(
    maxRatings: Int,
    seed: Int
  ): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    println("  [Synth] Generating Netflix-style synthetic data...")
    val rng = new Random(seed)
    val n = math.min(maxRatings, 1000000)

    val numUsers = 50000
    val numItems = 10000

    val interactions = mutable.ArrayBuilder.make[(Int, Int, Float)]
    for (_ <- 0 until n) {
      val user = rng.nextInt(numUsers)
      val item = rng.nextInt(numItems)
      val rating = (rng.nextFloat() * 4 + 1).toFloat
      interactions += ((user, item, rating))
    }
    val allInteractions = interactions.result()

    val trainSize = (n * 0.8).toInt
    val valSize = ((n - trainSize) / 2).toInt
    val trainInteractions = allInteractions.take(trainSize)
    val valInteractions = allInteractions.slice(trainSize, trainSize + valSize)
    val testInteractions = allInteractions.drop(trainSize + valSize)

    def build(interactions: Seq[(Int, Int, Float)], negRatio: Int): MatchingDataset = {
      val posSet = interactions.groupBy(_._1).view.mapValues(_.map(_._2).toSet).toMap

      val sampleCount = interactions.size * (1 + negRatio)
      val userArr = new Array[Float](sampleCount)
      val itemArr = new Array[Float](sampleCount)
      val labelArr = new Array[Float](sampleCount)

      var idx = 0
      for ((user, item, rating) <- interactions) {
        userArr(idx) = user.toFloat; itemArr(idx) = item.toFloat
        labelArr(idx) = if (rating >= 4.0f) 1.0f else 0.0f; idx += 1

        val ps = posSet.getOrElse(user, Set.empty)
        var neg = 0
        var att = 0
        while (neg < negRatio && att < numItems) {
          val ni = rng.nextInt(numItems)
          if (!ps.contains(ni)) {
            userArr(idx) = user.toFloat; itemArr(idx) = ni.toFloat
            labelArr(idx) = 0.0f; idx += 1; neg += 1
          }
          att += 1
        }
      }

      val userFeat = tensor(userArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
      val itemFeat = tensor(itemArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
      val labels = tensor(labelArr.slice(0, idx), Array(idx.toLong))

      new MatchingDataset(Map("user_id" -> userFeat), Map("item_id" -> itemFeat), Some(labels))
    }

    (
      build(trainInteractions, 4),
      build(valInteractions, 4),
      build(testInteractions, 4)
    )
  }
}
