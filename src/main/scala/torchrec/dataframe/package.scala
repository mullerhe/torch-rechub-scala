package torchrec.dataframe

/**
 * DataFrame package for torchrec.
 *
 * Provides unified data processing pipeline connecting:
 * - DataFrame (data preprocessing/feature engineering)
 * - Dataset/DataLoader (model training)
 * - Tensor/Embedding (model inference)
 */
package object dataframe {
  // Type aliases for convenience
  type DF = DataFrame
  type Col = Column

  /** Create DataFrame from column map */
  def apply(columns: Map[String, Column]): DataFrame = {
    require(columns.nonEmpty, "DataFrame must have at least one column")
    val schema = StructType(
      columns.map { case (name, col) => StructField(name, col.dtype) }.toSeq
    )
    new DataFrame(columns, schema)
  }
}