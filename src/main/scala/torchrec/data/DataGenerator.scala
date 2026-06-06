package torchrec.data

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.util.Random
import scala.collection.mutable
import scala.jdk.CollectionConverters._

import torchrec.Implicits.{SeqTensorRichSeq, RichTensor, toTensorVector}
import torchrec.Implicits.tensor

/**
 * Data generator for creating synthetic datasets
 */
object DataGenerator {

  /**
   * Generate ranking dataset with sparse features
   */
  def generateRankingData(
    numSamples: Int,
    numSparseFeatures: Int,
    numDenseFeatures: Int,
    vocabSize: Int = 100,
    trainRatio: Float = 0.7f,
    valRatio: Float = 0.1f,
    seed: Int = 42
  ): (Dataset, Dataset, Dataset) = {
    val random = new Random(seed)

    // Generate sparse features
    val sparseFeatures = mutable.Map[String, Tensor]()
    val sparseFeatureNames = (0 until numSparseFeatures).map(i => s"feat_$i").toList

    sparseFeatureNames.foreach { name =>
      val data = Array.ofDim[Float](numSamples)
      for (i <- 0 until numSamples) {
        // generate indices in range [0, vocabSize-1] to match embedding table indexing
        data(i) = random.nextInt(vocabSize).toFloat
      }
      sparseFeatures(name) = tensor(data, Array(numSamples.toLong)).toType(ScalarType.Long)
    }

    // Generate dense features
    val denseFeatures = mutable.Map[String, Tensor]()
    val denseFeatureNames = (0 until numDenseFeatures).map(i => s"dense_$i").toList

    denseFeatureNames.foreach { name =>
      val data = Array.ofDim[Float](numSamples)
      for (i <- 0 until numSamples) {
        data(i) = random.nextFloat()
      }
      denseFeatures(name) = tensor(data, Array(numSamples.toLong))
    }

    // Generate labels - pre-extract arrays for efficiency
    val sparseArrays = sparseFeatureNames.map(n => sparseFeatures(n).toFloatArray).toList
    val denseArrays = denseFeatureNames.map(n => denseFeatures(n).toFloatArray).toList
    val labels = Array.ofDim[Float](numSamples)
    for (i <- 0 until numSamples) {
      var score = 0.0f
      sparseArrays.indices.foreach { j =>
        if (i < sparseArrays(j).length) {
          score += (j + 1).toFloat * sparseArrays(j)(i) / vocabSize.toFloat
        }
      }
      denseArrays.indices.foreach { j =>
        if (i < denseArrays(j).length) {
          score += (j + 1).toFloat * denseArrays(j)(i)
        }
      }
      labels(i) = if (1.0f / (1.0f + math.exp(-score.toDouble).toFloat) > 0.5f) 1.0f else 0.0f
    }
    val labelsTensor = tensor(labels, Array(numSamples.toLong))

    // Split data
    val trainSize = (numSamples * trainRatio).toInt
    val valSize = (numSamples * valRatio).toInt

    val trainSparse = sparseFeatures.map { case (k, v) =>
      k -> v.narrow(0, 0, trainSize)
    }.toMap[String, Tensor]
    val trainDense = denseFeatures.map { case (k, v) =>
      k -> v.narrow(0, 0, trainSize)
    }.toMap[String, Tensor]
    val trainLabels = labelsTensor.narrow(0, 0, trainSize)

    val valSparse = sparseFeatures.map { case (k, v) =>
      k -> v.narrow(0, trainSize, valSize)
    }.toMap[String, Tensor]
    val valDense = denseFeatures.map { case (k, v) =>
      k -> v.narrow(0, trainSize, valSize)
    }.toMap[String, Tensor]
    val valLabels = labelsTensor.narrow(0, trainSize, valSize)

    val testSparse = sparseFeatures.map { case (k, v) =>
      k -> v.narrow(0, trainSize + valSize, numSamples - trainSize - valSize)
    }.toMap[String, Tensor]
    val testDense = denseFeatures.map { case (k, v) =>
      k -> v.narrow(0, trainSize + valSize, numSamples - trainSize - valSize)
    }.toMap[String, Tensor]
    val testLabels = labelsTensor.narrow(0, trainSize + valSize, numSamples - trainSize - valSize)

    val trainDataset = new torchrec.data.TensorDataset(trainSparse, trainDense, Some(trainLabels))
    val valDataset = new torchrec.data.TensorDataset(valSparse, valDense, Some(valLabels))
    val testDataset = new torchrec.data.TensorDataset(testSparse, testDense, Some(testLabels))

    (trainDataset, valDataset, testDataset)
  }

  /**
   * Generate matching dataset
   */
  def generateMatchingData(
    numUsers: Int,
    numItems: Int,
    avgSequenceLength: Int = 10,
    numUserFeatures: Int = 3,
    numItemFeatures: Int = 2,
    vocabSize: Int = 100,
    trainRatio: Float = 0.7f,
    seed: Int = 42
  ): (MatchingDataset, MatchingDataset, MatchingDataset) = {
    val random = new Random(seed)

    // User features
    val userFeatures = mutable.Map[String, Tensor]()
    for (i <- 0 until numUserFeatures) {
      val data = Array.ofDim[Float](numUsers)
      for (j <- 0 until numUsers) {
        // generate indices in range [0, vocabSize-1]
        data(j) = random.nextInt(vocabSize).toFloat
      }
      userFeatures(s"user_feat_$i") = tensor(data, Array(numUsers.toLong)).toType(ScalarType.Long)
    }

    // Item features
    val itemFeatures = mutable.Map[String, Tensor]()
    for (i <- 0 until numItemFeatures) {
      val data = Array.ofDim[Float](numItems)
      for (j <- 0 until numItems) {
        // generate indices in range [0, vocabSize-1]
        data(j) = random.nextInt(vocabSize).toFloat
      }
      itemFeatures(s"item_feat_$i") = tensor(data, Array(numItems.toLong)).toType(ScalarType.Long)
    }

    // Sequence features (history)
    val sequenceLength = avgSequenceLength + random.nextInt(5)
    val seqFeature = Array.ofDim[Float](numUsers * sequenceLength)
    for (i <- 0 until numUsers; j <- 0 until sequenceLength) {
      // history item ids should be in [0, numItems-1]
      seqFeature(i * sequenceLength + j) = random.nextInt(numItems).toFloat
    }
    userFeatures("history") = tensor(seqFeature, Array(numUsers.toLong, sequenceLength.toLong)).toType(ScalarType.Long)

    val trainSize = (numUsers * trainRatio).toInt

    val trainUserFeats = userFeatures.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }.toMap[String, Tensor]
    val trainItemFeats = itemFeatures.toMap[String, Tensor]
    val trainLabels = Some(tensor(Array.fill(trainSize)(1.0f), Array(trainSize.toLong)))

    val testUserFeats = userFeatures.map { case (k, v) => k -> v.narrow(0, trainSize, numUsers - trainSize) }.toMap[String, Tensor]
    val testItemFeats = itemFeatures.toMap[String, Tensor]
    val testLabels = Some(tensor(Array.fill(numUsers - trainSize)(1.0f), Array((numUsers - trainSize).toLong)))

    val trainDataset = new MatchingDataset(trainUserFeats, trainItemFeats, trainLabels)
    val valDataset = new MatchingDataset(trainUserFeats, trainItemFeats, trainLabels)
    val testDataset = new MatchingDataset(testUserFeats, testItemFeats, testLabels)

    (trainDataset, valDataset, testDataset)
  }

  /**
   * Generate multi-task dataset
   */
  def generateMultiTaskData(
    numSamples: Int,
    numFeatures: Int,
    taskNames: List[String],
    vocabSize: Int = 100,
    trainRatio: Float = 0.7f,
    valRatio: Float = 0.1f,
    seed: Int = 42
  ): (MultiTaskDataset, MultiTaskDataset, MultiTaskDataset) = {
    val random = new Random(seed)

    val features = mutable.Map[String, Tensor]()
    for (i <- 0 until numFeatures) {
      val data = Array.ofDim[Float](numSamples)
      for (j <- 0 until numSamples) {
        // generate indices in range [0, vocabSize-1]
        data(j) = random.nextInt(vocabSize).toFloat
      }
      features(s"feat_$i") = tensor(data, Array(numSamples.toLong)).toType(ScalarType.Long)
    }

    val taskLabels = mutable.Map[String, Tensor]()
    taskNames.foreach { taskName =>
      val data = Array.ofDim[Float](numSamples)
      for (j <- 0 until numSamples) {
        data(j) = if (random.nextFloat() > 0.5f) 1.0f else 0.0f
      }
      taskLabels(taskName) = tensor(data, Array(numSamples.toLong))
    }

    val trainSize = (numSamples * trainRatio).toInt
    val valSize = (numSamples * valRatio).toInt

    val trainFeatures = features.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }.toMap[String, Tensor]
    val trainLabels = taskLabels.map { case (k, v) => k -> v.narrow(0, 0, trainSize) }.toMap[String, Tensor]

    val valFeatures = features.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }.toMap[String, Tensor]
    val valLabels = taskLabels.map { case (k, v) => k -> v.narrow(0, trainSize, valSize) }.toMap[String, Tensor]

    val testFeatures = features.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, numSamples - trainSize - valSize) }.toMap[String, Tensor]
    val testLabels = taskLabels.map { case (k, v) => k -> v.narrow(0, trainSize + valSize, numSamples - trainSize - valSize) }.toMap[String, Tensor]

    val trainDataset = new MultiTaskDataset(trainFeatures, trainLabels)
    val valDataset = new MultiTaskDataset(valFeatures, valLabels)
    val testDataset = new MultiTaskDataset(testFeatures, testLabels)

    (trainDataset, valDataset, testDataset)
  }
}
