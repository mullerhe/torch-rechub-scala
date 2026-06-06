package torchrec.data

import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.tensor

import scala.util.Random
import scala.reflect.ClassTag
import java.io.File

/**
 * MovieLens 1M Dataset Loader.
 * Downloads from: https://files.grouplens.org/datasets/movielens/ml-1m.zip
 *
 * Format:
 *   ratings.dat:  UserID::MovieID::Rating::Timestamp
 *   movies.dat:   MovieID::Title::Genres
 *   users.dat:    UserID::Gender::Age::Occupation::Zip-code
 *
 * This produces a MatchingDataset for DSSM-style two-tower retrieval.
 */
object MovieLensDataset {

  private val DatasetUrl = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
  private val DatasetName = "ml-1m"

  // Alternative mirrors
  private val Mirrors: List[String] = List(
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "https://raw.githubusercontent.com/makefu/movielens-1m/master/ml-1m.zip",
  )

  /**
   * Load MovieLens 1M as a MatchingDataset for two-tower retrieval.
   *
   * @param trainRatio Fraction of data for training (remaining split val/test equally)
   * @param negRatio Number of negative samples per positive
   * @param maxSamples Cap total samples (None = use all ~1M ratings)
   * @param seed Random seed
   * @return (trainDataset, valDataset, testDataset) as MatchingDataset
   */
  def load(
    trainRatio: Float = 0.8f,
    negRatio: Int = 4,
    maxSamples: Option[Int] = None,
    seed: Int = 42
  ): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    println("=" * 60)
    println("MovieLens 1M Dataset Loading")
    println("=" * 60)

    // Step 1: Download (try mirrors)
    val zipFile = DatasetDownloader.tryMirrors(Mirrors, DatasetName)
    val dataDir = zipFile.getParentFile

    // Step 2: Locate ratings file
    val ratingsFile = findRatingsFile(dataDir)
    if (ratingsFile == null) {
      throw new RuntimeException(s"ratings.dat not found in $dataDir. Check zip structure.")
    }
    println(s"  [Data] ratings: ${ratingsFile.getAbsolutePath}")

    // Step 3: Parse ratings
    println("  [Parse] Reading ratings...")
    val allRatings = parseRatings(ratingsFile, maxSamples)
    println(s"  [Data] Total ratings: ${allRatings.size}")

    // Step 4: Build user and item vocabularies
    val (userIds, movieIds) = buildVocabularies(allRatings)
    val numUsers = userIds.size
    val numMovies = movieIds.size
    println(s"  [Data] Unique users: $numUsers, movies: $numMovies")

    // Step 5: Build train/val/test splits
    val rng = new Random(seed)
    val shuffled = rng.shuffle(allRatings.toSeq)

    // Negative items: all items not interacted by the user
    val userPosItems: Map[Int, Set[Int]] = allRatings.groupBy(_.userId).view.mapValues(_.map(_.movieId).toSet).toMap

    val trainSize = (shuffled.size * trainRatio).toInt
    val valSize = ((shuffled.size - trainSize) / 2).toInt
    val trainRatings = shuffled.take(trainSize)
    val valRatings = shuffled.slice(trainSize, trainSize + valSize)
    val testRatings = shuffled.drop(trainSize + valSize)

    println(s"  [Split] Train: ${trainRatings.size}, Val: ${valRatings.size}, Test: ${testRatings.size}")

    // Step 6: Assemble MatchingDataset
    // For DSSM: user features = {user_id}, item features = {movie_id}
    // We use negative sampling for training
    val trainDS = buildMatchingDataset(trainRatings, userPosItems, negRatio, numUsers, numMovies, "train")
    val valDS = buildMatchingDataset(valRatings, userPosItems, negRatio, numUsers, numMovies, "val")
    val testDS = buildMatchingDataset(testRatings, userPosItems, negRatio, numUsers, numMovies, "test")

    println(s"  [Dataset] Train batches: ${trainDS.size}")
    println("=" * 60)

    (trainDS, valDS, testDS)
  }

  private def findRatingsFile(dir: File): File = {
    // Look for ml-1m/ratings.dat or just ratings.dat
    val f1 = new File(dir, "ml-1m/ratings.dat")
    val f2 = new File(dir, "ratings.dat")
    val f3p = new File(dir, "ml-1m.zip").getParentFile
    val f3 = if (f3p != null) new File(f3p, "ml-1m/ratings.dat") else null.asInstanceOf[File]

    val all: List[File] = List(f1, f2, f3)
    all.find(_.nn.exists).map(_.nn).getOrElse(null.asInstanceOf[File])
  }

  private[torchrec] case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  private def parseRatings(file: File, maxSamples: Option[Int]): Array[Rating] = {
    val lines = DatasetDownloader.readLines(file, "::", skipHeader = false)
    val builder = scala.collection.mutable.ArrayBuilder.make[Rating]
    var count = 0L

    for (fields <- lines) {
      if (fields.length >= 4) {
        try {
          val userId = fields(0).toInt
          val movieId = fields(1).toInt
          val rating = fields(2).toFloat
          val timestamp = fields(3).toLong
          builder += Rating(userId, movieId, rating, timestamp)
          count += 1
          if (maxSamples.exists(_ <= count)) {}
        } catch {
          case _: NumberFormatException =>
          // Skip malformed lines
        }
      }
      if (maxSamples.exists(_ <= count)) return builder.result()
    }

    builder.result()
  }

  private def buildVocabularies(ratings: Array[Rating]): (Map[Int, Int], Map[Int, Int]) = {
    val userIdSet = scala.collection.mutable.Set[Int]()
    val movieIdSet = scala.collection.mutable.Set[Int]()
    ratings.foreach { r =>
      userIdSet += r.userId
      movieIdSet += r.movieId
    }
    // Remap to contiguous indices starting from 0
    val sortedUsers = userIdSet.toList.sorted
    val sortedMovies = movieIdSet.toList.sorted
    val userMap = sortedUsers.zipWithIndex.toMap
    val movieMap = sortedMovies.zipWithIndex.toMap
    (userMap, movieMap)
  }

  private def buildMatchingDataset(
    ratings: Seq[Rating],
    userPosItems: Map[Int, Set[Int]],
    negRatio: Int,
    numUsers: Int,
    numMovies: Int,
    splitName: String
  ): MatchingDataset = {
    val rng = new Random(42)

    // For each rating, create positive + negRatio negatives
    val numSamples = ratings.size * (1 + negRatio)
    val userFeatArr = new Array[Float](numSamples)
    val movieFeatArr = new Array[Float](numSamples)
    val labelArr = new Array[Float](numSamples)

    var idx = 0
    for (rating <- ratings) {
      // Positive sample
      userFeatArr(idx) = rating.userId.toFloat
      movieFeatArr(idx) = rating.movieId.toFloat
      labelArr(idx) = 1.0f
      idx += 1

      // Negative samples
      val posSet = userPosItems.getOrElse(rating.userId, Set.empty)
      var negCount = 0
      var attempts = 0
      while (negCount < negRatio && attempts < numMovies) {
        val negMovie = rng.nextInt(numMovies)
        if (!posSet.contains(negMovie)) {
          userFeatArr(idx) = rating.userId.toFloat
          movieFeatArr(idx) = negMovie.toFloat
          labelArr(idx) = 0.0f
          idx += 1
          negCount += 1
        }
        attempts += 1
      }
    }

    val userFeat = tensor(userFeatArr, Array(idx.toLong)).toType(ScalarType.Long)
    val movieFeat = tensor(movieFeatArr, Array(idx.toLong)).toType(ScalarType.Long)
    val labels = tensor(labelArr.slice(0, idx), Array(idx.toLong))

    val userFeatures = Map("user_id" -> userFeat)
    val itemFeatures = Map("movie_id" -> movieFeat)
    val labelsTensor = Some(labels)

    new MatchingDataset(userFeatures, itemFeatures, labelsTensor)
  }

  private implicit class SeqOps[A: ClassTag](seq: Seq[A]) {
    def shuffle(rng: Random): Seq[A] = {
      val arr = seq.toArray[A]
      var i = arr.length - 1
      while (i > 0) {
        val j = rng.nextInt(i + 1)
        val tmp = arr(i); arr(i) = arr(j); arr(j) = tmp
        i -= 1
      }
      arr.toSeq
    }
  }
}
