package torchrec.dataframe

/**
 * Feature specification for converting DataFrame to Dataset.
 * Defines how columns map to sparse, dense, and sequence features.
 */
case class FeatureSpec(
  sparseFeatures: List[SparseSpec] = Nil,
  denseFeatures: List[DenseSpec] = Nil,
  sequenceFeatures: List[SequenceSpec] = Nil
) {
  def allColumnNames: List[String] =
    sparseFeatures.map(_.columnName) ++
    denseFeatures.map(_.columnName) ++
    sequenceFeatures.map(_.columnName)
}

/**
 * Sparse (categorical) feature specification.
 * Column values are treated as vocabulary indices.
 */
case class SparseSpec(
  columnName: String,
  vocabSize: Long,
  embedDim: Int = 8,
  paddingIdx: Option[Long] = None,
  sharedWith: Option[String] = None
)

/**
 * Dense (numeric) feature specification.
 * Column values are normalized continuous values.
 */
case class DenseSpec(
  columnName: String,
  embedDim: Int = 1,
  normalize: Boolean = true
)

/**
 * Sequence feature specification for behavior history.
 */
case class SequenceSpec(
  columnName: String,
  vocabSize: Long,
  embedDim: Int = 8,
  maxLen: Int = 50,
  pooling: String = "mean",  // "mean", "sum", "concat", "last"
  paddingIdx: Long = 0,
  sharedWith: Option[String] = None
)

/**
 * Label specification for supervised learning.
 */
case class LabelSpec(
  columnName: String,
  taskType: TaskType = TaskType.BinaryClassification,
  positiveValue: Any = 1,
  negativeValue: Any = 0
)

/**
 * Task type for label columns.
 */
enum TaskType:
  case BinaryClassification
  case MultiClassClassification
  case Regression
  case MultiLabelClassification

/**
 * Feature engineering configuration.
 */
case class TransformConfig(
  name: String,
  transformerType: String,
  params: Map[String, Any] = Map.empty
)

/**
 * Pipeline configuration for multi-stage transformations.
 */
case class PipelineConfig(
  stages: List[TransformConfig] = Nil
) {
  def addStage(config: TransformConfig): PipelineConfig =
    copy(stages = stages :+ config)
}

/**
 * DataFrame column type mapping.
 */
enum ColumnRole:
  case Sparse
  case Dense
  case Sequence
  case Label
  case UserId
  case ItemId
  case Timestamp
  case Context

/**
 * DataFrame metadata for pipeline tracking.
 */
case class DataFrameMetadata(
  sourcePath: Option[String] = None,
  numRows: Long = 0,
  numCols: Long = 0,
  schema: Option[StructType] = None,
  columnRoles: Map[String, ColumnRole] = Map.empty,
  statistics: Map[String, ColumnStats] = Map.empty
) {
  def withRole(column: String, role: ColumnRole): DataFrameMetadata =
    copy(columnRoles = columnRoles + (column -> role))
}