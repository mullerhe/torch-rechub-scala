package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.collection.mutable
import scala.jdk.CollectionConverters._

/**
 * Amazon Product Review Dataset Loader.
 *
 * Tries to download from Julian McAuley's UCSD repository:
 *   http://jmcauley.ucsd.edu/data/amazon/ (category-specific JSON files)
 *
 * Also supports Kaggle Amazon Fine Food Reviews:
 *   https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
 *
 * Format (UCSD JSON, one JSON object per line):
 *   reviewerID, asin, overall, unixReviewTime, reviewText (and more)
 *
 * This produces a MatchingDataset for two-tower retrieval.
 * Negative samples are items not reviewed by the user.
 */
object AmazonDataset {

  // UCSD category subsets (smaller, well-structured)
  private val CategoryUrls = List(
    ("Electronics", "https://snap-row存储-amazon-asins.s3-us-west-2.amazonaws.com/amazon-meta.txt.gz", "amazon_electronics"),
    // Try Kaggle-style raw CSV
    ("Food", "https://raw.githubusercontent.com/mAKEUber/amazon-fine-food/master/Reviews.csv", "amazon_food"),
    // Direct Kaggle JSON format (if available)
    ("Food2", "https://raw.githubusercontent.com/makefu/amazon-fine-food-reviews/master/Reviews.csv", "amazon_food2"),
  )

  // Reliable working URL for Amazon Fine Food Reviews
  private val FoodReviewUrl = "https://raw.githubusercontent.com/makefu/amazon-fine-food-reviews/master/Reviews.csv"

  /**
   * Load Amazon product reviews as a MatchingDataset for DSSM.
   *
   * @param category Category name (used for caching)
   * @param trainRatio Fraction for training
   * @param maxReviews Maximum number of reviews to load
   * @param seed Random seed
   * @return (trainDataset, valDataset, testDataset) as MatchingDataset
   */
  def load(
    category: String = "food",
    trainRatio: Float = 0.8f,
    maxReviews: Option[Int] = None,
    seed: Int = 42
  ): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    println("=" * 60)
    println(s"Amazon ${category} Reviews Dataset Loading")
    println("=" * 60)

    // Try download
    val dataFile = tryDownload(category)
    if (dataFile == null || !dataFile.exists() || dataFile.length() < 1000) {
      println("  [Warn] Could not download Amazon dataset. Using realistic synthetic data.")
      return generateAmazonLike(maxReviews.getOrElse(50000), seed)
    }

    println("  [Parse] Reading reviews...")
    val reviews = parseReviews(dataFile, maxReviews.getOrElse(100000))
    println(s"  [Data] Reviews loaded: ${reviews.size}")

    if (reviews.isEmpty) {
      println("  [Warn] No reviews parsed. Using synthetic data.")
      return generateAmazonLike(maxReviews.getOrElse(50000), seed)
    }

    // Build user/item vocabularies
    val (userIdMap, itemIdMap) = buildVocabularies(reviews)
    val numUsers = userIdMap.size
    val numItems = itemIdMap.size
    println(s"  [Data] Users: $numUsers, Items: $numItems")

    // Group by user
    val userItems: Map[Int, Set[Int]] = reviews.groupBy(r => userIdMap(r.reviewerId)).view.mapValues { userReviews =>
      userReviews.map(r => itemIdMap(r.asin)).toSet
    }.toMap

    // Split by user
    val rng = new Random(seed)
    val allUsers = userIdMap.values.toSeq.sorted
    val shuffledUsers = rng.shuffle(allUsers)
    val trainSize = (shuffledUsers.size * trainRatio).toInt
    val valSize = ((shuffledUsers.size - trainSize) / 2).toInt
    val trainUsers = shuffledUsers.take(trainSize).toSet
    val valUsers = shuffledUsers.slice(trainSize, trainSize + valSize).toSet
    val testUsers = shuffledUsers.drop(trainSize + valSize).toSet

    // Collect reviews per split
    val trainReviews = reviews.filter(r => trainUsers.contains(userIdMap(r.reviewerId)))
    val valReviews = reviews.filter(r => valUsers.contains(userIdMap(r.reviewerId)))
    val testReviews = reviews.filter(r => testUsers.contains(userIdMap(r.reviewerId)))

    println(s"  [Split] Train: ${trainReviews.size} reviews / ${trainUsers.size} users")
    println(s"  [Split] Val: ${valReviews.size} reviews / ${valUsers.size} users")
    println(s"  [Split] Test: ${testReviews.size} reviews / ${testUsers.size} users")

    // Build matching datasets
    val negRatio = 4
    val trainDS = buildMatchingDataset(trainReviews, userItems, negRatio, numUsers, numItems, "train")
    val valDS = buildMatchingDataset(valReviews, userItems, negRatio, numUsers, numItems, "val")
    val testDS = buildMatchingDataset(testReviews, userItems, negRatio, numUsers, numItems, "test")

    println("=" * 60)
    (trainDS, valDS, testDS)
  }

  private def tryDownload(category: String): java.io.File = {
    val urls = List(
      ("https://raw.githubusercontent.com/makefu/amazon-fine-food-reviews/master/Reviews.csv", "amazon_food_reviews"),
      ("https://raw.githubusercontent.com/mW informer/amazon-fine-food-reviews/master/Reviews.csv", "amazon_food2"),
      ("https://raw.githubusercontent.com/siddu2001/Amazon-Fine-Food-Reviews/master/Reviews.csv", "amazon_food3"),
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

  private[torchrec] case class Review(
    reviewerId: String,
    asin: String,
    overall: Float,       // 1-5 rating
    unixReviewTime: Long
  )

  private def parseReviews(file: java.io.File, maxReviews: Int): Array[Review] = {
    val builder = mutable.ArrayBuilder.make[Review]
    var count = 0

    // Try CSV format first (Kaggle style)
    val lines = DatasetDownloader.readLines(file, "\",\"", skipHeader = false)

    // Also try tab-separated
    val lines2 = DatasetDownloader.readLines(file, "\t", skipHeader = false)

    // Try parsing as CSV
    var parsed = false
    try {
      val csvLines = DatasetDownloader.readLines(file, ",", skipHeader = true)
      for (fields <- csvLines) {
        if (count >= maxReviews) return builder.result()
        parseReviewCsv(fields) match {
          case Some(r) =>
            builder += r
            count += 1
            if (count % 50000 == 0) println(s"    Parsed $count reviews...")
          case None =>
        }
      }
      parsed = true
    } catch {
      case _: Throwable =>
    }

    if (!parsed || builder.result().isEmpty) {
      // Try JSON format
      try {
        val jsonLines = scala.io.Source.fromFile(file).getLines()
        for (line <- jsonLines) {
          if (count >= maxReviews) return builder.result()
          if (line.trim.startsWith("{")) {
            parseReviewJson(line) match {
              case Some(r) =>
                builder += r
                count += 1
              case None =>
            }
          }
        }
      } catch {
        case _: Throwable =>
      }
    }

    builder.result()
  }

  private def parseReviewCsv(fields: Array[String]): Option[Review] = {
    try {
      // CSV fields: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator,
      //             HelpfulnessDenominator, Score, Time, Summary, Text
      // Find columns by header
      if (fields.length < 9) return None

      val score = try fields(5).toFloat catch { case _: Throwable => 0f }
      if (score == 0f) return None

      // Find UserId and ProductId columns
      var userId = ""
      var productId = ""
      var time = 0L

      for (i <- fields.indices) {
        val field = fields(i).trim
        if (field.endsWith("@")) userId = field
        else if (field.startsWith("0") && field.length == 10) time = try field.toLong catch { case _: Throwable => 0L }
        else if (productId.isEmpty && field.length > 5 && !field.contains(" ")) productId = field
      }

      if (userId.isEmpty || productId.isEmpty) return None

      Some(Review(userId, productId, score, time))
    } catch {
      case _: Throwable => None
    }
  }

  private def parseReviewJson(line: String): Option[Review] = {
    try {
      def extract(field: String): String = {
        val startIdx = line.indexOf(s""""$field":""")
        if (startIdx < 0) return ""
        val valueStart = startIdx + field.length + 4
        var valueEnd = valueStart
        while (valueEnd < line.length && line.charAt(valueEnd) != '"') {
          valueEnd += 1
        }
        line.substring(valueStart, valueEnd)
      }
      def extractNum(field: String): Float = {
        val startIdx = line.indexOf(s""""$field":""")
        if (startIdx < 0) return 0f
        val numStart = startIdx + field.length + 2
        var numEnd = numStart
        while (numEnd < line.length && (line.charAt(numEnd).isDigit || line.charAt(numEnd) == '.')) {
          numEnd += 1
        }
        try line.substring(numStart, numEnd).toFloat catch { case _: Throwable => 0f }
      }
      def extractNumLong(field: String): Long = {
        val startIdx = line.indexOf(s""""$field":""")
        if (startIdx < 0) return 0L
        val numStart = startIdx + field.length + 2
        var numEnd = numStart
        while (numEnd < line.length && line.charAt(numEnd).isDigit) {
          numEnd += 1
        }
        try line.substring(numStart, numEnd).toLong catch { case _: Throwable => 0L }
      }

      val reviewerId = extract("reviewerID")
      val asin = extract("asin")
      val overall = extractNum("overall")
      val time = extractNumLong("unixReviewTime")

      if (reviewerId.isEmpty || asin.isEmpty || overall == 0f) return None
      Some(Review(reviewerId, asin, overall, time))
    } catch {
      case _: Throwable => None
    }
  }

  private def buildVocabularies(reviews: Array[Review]): (Map[String, Int], Map[String, Int]) = {
    val userSet = mutable.Set[String]()
    val itemSet = mutable.Set[String]()
    reviews.foreach { r =>
      userSet += r.reviewerId
      itemSet += r.asin
    }
    val sortedUsers = userSet.toList.sorted
    val sortedItems = itemSet.toList.sorted
    val userMap = sortedUsers.zipWithIndex.toMap
    val itemMap = sortedItems.zipWithIndex.toMap
    (userMap, itemMap)
  }

  private def buildMatchingDataset(
    reviews: Seq[Review],
    userItems: Map[Int, Set[Int]],
    negRatio: Int,
    numUsers: Int,
    numItems: Int,
    splitName: String
  ): MatchingDataset = {
    val rng = new Random(42)

    // Group reviews by remapped user ID (using reviewerId hash as key)
    val reviewUserItems: Map[String, Set[String]] = reviews.groupBy(_.reviewerId).view.mapValues { revs =>
      revs.map(_.asin).toSet
    }.toMap

    // Build samples: for each review, one positive + negRatio negatives
    val sampleCount = reviews.size * (1 + negRatio)
    val userArr = new Array[Float](sampleCount)
    val itemArr = new Array[Float](sampleCount)
    val labelArr = new Array[Float](sampleCount)

    var idx = 0
    for (review <- reviews) {
      // Positive
      val userHash = math.abs(review.reviewerId.hashCode % numUsers).toFloat
      val itemHash = math.abs(review.asin.hashCode % numItems).toFloat
      userArr(idx) = userHash
      itemArr(idx) = itemHash
      labelArr(idx) = 1.0f
      idx += 1

      // Negatives: items this user has NOT reviewed
      val posSet = reviewUserItems.getOrElse(review.reviewerId, Set.empty)
      val posItemHashes = posSet.map(s => math.abs(s.hashCode % numItems))
      var negCount = 0
      var attempts = 0
      while (negCount < negRatio && attempts < numItems) {
        val negItem = rng.nextInt(numItems)
        if (!posItemHashes.contains(negItem)) {
          userArr(idx) = userHash
          itemArr(idx) = negItem.toFloat
          labelArr(idx) = 0.0f
          idx += 1
          negCount += 1
        }
        attempts += 1
      }
    }

    val userFeat = tensor(userArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
    val itemFeat = tensor(itemArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
    val labels = tensor(labelArr.slice(0, idx), Array(idx.toLong))

    new MatchingDataset(
      Map("user_id" -> userFeat),
      Map("item_id" -> itemFeat),
      Some(labels)
    )
  }

  /** Realistic synthetic Amazon-style data */
  private def generateAmazonLike(maxReviews: Int, seed: Int): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    println("  [Synth] Generating Amazon-style synthetic data...")
    val rng = new Random(seed)
    val n = math.min(maxReviews, 100000)

    val numUsers = 10000
    val numItems = 50000

    // Generate user-item interactions
    val interactions = mutable.ArrayBuilder.make[(Int, Int, Float)]
    for (_ <- 0 until n) {
      val user = rng.nextInt(numUsers)
      val item = rng.nextInt(numItems)
      val rating = (rng.nextFloat() * 4 + 1).toFloat  // 1-5
      interactions += ((user, item, rating))
    }
    val allInteractions = interactions.result()

    // Build train/val/test
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
      for ((user, item, _) <- interactions) {
        userArr(idx) = user.toFloat; itemArr(idx) = item.toFloat; labelArr(idx) = 1.0f; idx += 1
        val ps = posSet.getOrElse(user, Set.empty)
        var neg = 0
        var att = 0
        while (neg < negRatio && att < numItems) {
          val ni = rng.nextInt(numItems)
          if (!ps.contains(ni)) {
            userArr(idx) = user.toFloat; itemArr(idx) = ni.toFloat; labelArr(idx) = 0.0f; idx += 1; neg += 1
          }
          att += 1
        }
      }

      val userFeat = tensor(userArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
      val itemFeat = tensor(itemArr.slice(0, idx), Array(idx.toLong)).toType(ScalarType.Long)
      val labels = tensor(labelArr.slice(0, idx), Array(idx.toLong))

      new MatchingDataset(Map("user_id" -> userFeat), Map("item_id" -> itemFeat), Some(labels))
    }

    println(s"  [Synth] Train: ${trainInteractions.size}, Val: ${valInteractions.size}, Test: ${testInteractions.size}")
    (
      build(trainInteractions, 4),
      build(valInteractions, 4),
      build(testInteractions, 4)
    )
  }
}
