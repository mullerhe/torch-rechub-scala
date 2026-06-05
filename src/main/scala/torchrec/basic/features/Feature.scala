package torchrec.basic.features

/**
 * Base trait for all feature types in TorchRec
 */
trait Feature {
  def name: String
  def embedDim: Int
  def vocabSize: Long

  /** Whether this is a sequence feature */
  def isSequence: Boolean = false
}

/**
 * Sparse (categorical) feature with embedding table
 */
case class SparseFeature(
  name: String,
  vocabSize: Long,
  embedDim: Int = 8,
  sharedWith: Option[String] = None,
  paddingIdx: Option[Long] = None
) extends Feature

/**
 * Dense (numeric) feature - passes through without embedding
 */
case class DenseFeature(
  name: String,
  embedDim: Int = 1
) extends Feature {
  val vocabSize: Long = 1
}

/**
 * Sequence feature for behavior history (e.g., clicked items)
 */
case class SequenceFeature(
  name: String,
  vocabSize: Long,
  embedDim: Int = 8,
  pooling: String = "mean",  // "mean", "sum", "concat", "last"
  sharedWith: Option[String] = None,
  maxLen: Int = 50,
  paddingIdx: Long = 0
) extends Feature {
  override def isSequence: Boolean = true
}

/**
 * Label feature for supervised learning
 */
case class LabelFeature(name: String = "label") extends Feature {
  val embedDim: Int = 1
  val vocabSize: Long = 2
}

/**
 * Helper object for feature creation and manipulation
 */
object Features {
  def sparse(name: String, vocabSize: Long, embedDim: Int = 8): SparseFeature =
    SparseFeature(name, vocabSize, embedDim)

  def dense(name: String, embedDim: Int = 1): DenseFeature =
    DenseFeature(name, embedDim)

  def sequence(name: String, vocabSize: Long, embedDim: Int = 8,
              pooling: String = "mean"): SequenceFeature =
    SequenceFeature(name, vocabSize, embedDim, pooling)

  /** Get all sparse features from a list */
  def getSparseFeatures(features: List[Feature]): List[SparseFeature] =
    features.collect { case f: SparseFeature => f }

  /** Get all dense features from a list */
  def getDenseFeatures(features: List[Feature]): List[DenseFeature] =
    features.collect { case f: DenseFeature => f }

  /** Get all sequence features from a list */
  def getSequenceFeatures(features: List[Feature]): List[SequenceFeature] =
    features.collect { case f: SequenceFeature => f }

  /** Calculate total embedding dimension for sparse features */
  def calcSparseDim(features: List[Feature], embedDim: Int): Long =
    getSparseFeatures(features).length * embedDim

  /** Calculate total embedding dimension for sequence features with pooling */
  def calcSequenceDim(features: List[SequenceFeature], embedDim: Int, pooling: String = "mean"): Long =
    pooling match {
      case "concat" => getSequenceFeatures(features).length * embedDim
      case _ => embedDim
    }
}
