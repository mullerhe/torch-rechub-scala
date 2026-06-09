package torchrec.dataframe

/** Schema definition for DataFrame columns */
case class StructType(fields: Seq[StructField]) {
  def this() = this(Nil) // Default constructor for empty schema

  /** Get field by name */
  def apply(name: String): StructField = fields.find(_.name == name) match
    case Some(f) => f
    case None => throw new IllegalArgumentException(s"Field '$name' not found in schema")

  /** Get field index by name */
  def indexOf(name: String): Int = fields.indexWhere(_.name == name)

  /** Check if field exists */
  def contains(name: String): Boolean = fields.exists(_.name == name)

  /** Get number of columns */
  def size: Int = fields.size

  /** Merge two schemas */
  def merge(other: StructType): StructType = StructType(fields ++ other.fields)

  /** Select subset of fields */
  def select(names: String*): StructType = StructType(
    names.flatMap(n => fields.find(_.name == n))
  )

  /** Drop fields */
  def drop(names: String*): StructType = StructType(
    fields.filter(f => !names.contains(f.name))
  )
}

/** Field in a schema */
case class StructField(
  name: String,
  dtype: DataType,
  nullable: Boolean = true,
  metadata: Map[String, String] = Map.empty
)

/** Join types for DataFrame operations */
enum JoinType:
  case Inner, Left, Right, Outer, Cross, Semi, Anti

/** Sort direction */
enum SortDirection:
  case Ascending, Descending

/** Sort specification */
case class SortColumn(name: String, direction: SortDirection = SortDirection.Ascending)

/** Aggregation function type */
enum AggregationType:
  case Sum, Mean, Avg, Count, Min, Max
  case Std, Var, Median, Quantile
  case First, Last
  case NUnique, Distinct
  case Coalesce, StringConcat

/** Aggregation specification */
case class Aggregation(
  column: String,
  aggType: AggregationType,
  alias: Option[String] = None
) {
  def outputName: String = alias.getOrElse(s"${column}_${aggType.toString.toLowerCase}")
}

object Aggregation {
  def sum(col: String) = Aggregation(col, AggregationType.Sum)
  def mean(col: String) = Aggregation(col, AggregationType.Mean)
  def avg(col: String) = Aggregation(col, AggregationType.Avg)
  def count(col: String) = Aggregation(col, AggregationType.Count)
  def min(col: String) = Aggregation(col, AggregationType.Min)
  def max(col: String) = Aggregation(col, AggregationType.Max)
  def std(col: String) = Aggregation(col, AggregationType.Std)
  def var_(col: String) = Aggregation(col, AggregationType.Var)
  def median(col: String) = Aggregation(col, AggregationType.Median)
  def first(col: String) = Aggregation(col, AggregationType.First)
  def last(col: String) = Aggregation(col, AggregationType.Last)
  def nunique(col: String) = Aggregation(col, AggregationType.NUnique)
  def distinct(col: String) = Aggregation(col, AggregationType.Distinct)
}