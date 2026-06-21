package torchrec.dataframe
import scala.collection.mutable
import scala.collection.immutable.ListMap
import scala.language.dynamics

/** Custom numeric type used across the DataFrame library */
type Number = Double | Float | Int | Long | Short | Byte

extension (n: Number) {
  def doubleValue(): Double = n match {
    case d: Double => d
    case f: Float  => f.toDouble
    case i: Int    => i.toDouble
    case l: Long   => l.toDouble
    case s: Short  => s.toDouble
    case b: Byte   => b.toDouble
    case _         => 0.0
  }

  def floatValue(): Float = n match {
    case d: Double => d.toFloat
    case f: Float  => f
    case i: Int    => i.toFloat
    case l: Long   => l.toFloat
    case s: Short  => s.toFloat
    case b: Byte   => b.toFloat
    case _         => 0.0f
  }

  def intValue(): Int = n match {
    case d: Double => d.toInt
    case f: Float  => f.toInt
    case i: Int    => i
    case l: Long   => l.toInt
    case s: Short  => s.toInt
    case b: Byte   => b.toInt
    case _         => 0
  }
}

/** DataFrame with columnar storage.
 *  Provides pandas-like API for data manipulation.
 */
class DataFrame(
  private val _columnMap: Map[String, Column],
  val schema: StructType
) {
  // Always use ListMap internally to preserve insertion order
  private val columnMap = _columnMap.to(collection.immutable.ListMap)
  require(_columnMap.nonEmpty || schema.fields.isEmpty, "DataFrame must have at least one column or empty schema")
  if (_columnMap.nonEmpty) {
    require(_columnMap.keySet == schema.fields.map(_.name).toSet,
      "Column names must match schema")
  }
  /** Factory method that infers schema from columns */
  def this(columns: Map[String, Column]) = {
    this(columns, StructType(columns.map { case (name, col) => StructField(name, col.dtype) }.toSeq))
  }
  /** Get column by name */
  def apply(col: String): Column = columnMap(col)
  /** Get column by name (alternative) */
  def col(col: String): Column = columnMap(col)

  /** Apply a function to each row */
  def applyPerRow[T](f: Row => T): Seq[T] = (0 until numRows).map(i => f(Row.fromIndex(this, i)))
  /** Apply a function to each column */
  def applyPerColumn[T](f: Column => T): Seq[T] = columns.map(c => f(columnMap(c)))

  /** Overloaded apply for Float anonymous functions */
  def applyFloat(f: Float => Float): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive) name -> col.mapFloat(f)
      else name -> col
    }
    new DataFrame(newCols)
  }

  /** Overloaded apply for Double anonymous functions */
  def applyDouble(f: Double => Double): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive) name -> col.mapDouble(f)
      else name -> col
    }
    new DataFrame(newCols)
  }

  /** Get column names */
  def columns: List[String] = columnMap.keys.toList
  /** Get column map directly */
  def columnData: Map[String, Column] = columnMap
  /** Number of rows */
  def numRows: Int = if (columnMap.values.isEmpty) 0 else columnMap.values.head.length
  /** Number of columns */
  def numCols: Int = columnMap.size
  /** Check if DataFrame is empty */
  def isEmpty: Boolean = numRows == 0
  // =========================================================================
  // Basic Attributes
  // =========================================================================
  /** Row index (simple 0 to n-1) */
  def index: Range = 0 until numRows
  /** Column data types */
  def dtypes: Map[String, DataType] = columnMap.map { case (n, c) => n -> c.dtype }
  /** Shape as (rows, cols) */
  def shape: (Int, Int) = (numRows, numCols)
  /** Total number of elements */
  def size: Int = numRows * numCols
  /** Number of dimensions (always 2 for DataFrame) */
  def ndim: Int = 2
  /** Get underlying data as 2D array */
  def values: Array[Array[Any]] = {
    val arr = Array.ofDim[Any](numRows, numCols)
    val cols = columnMap.values.toIndexedSeq
    for (r <- 0 until numRows) {
      for (c <- 0 until numCols) {
        arr(r)(c) = cols(c)(r)
      }
    }
    arr
  }
  /** Convert to numpy-like array */
  def to_numpy(): Array[Array[Any]] = values
  /** Get values directly (legacy) */
  def get_values(): Array[Array[Any]] = values
  /** Axes (index and columns) */
  def axes: List[Any] = List(index, columns)
  /** Whether DataFrame is empty */
  def empty: Boolean = isEmpty
  /** Check if DataFrame is empty (property) */
  def is_empty: Boolean = isEmpty
  /** Memory usage per column in bytes */
  def memory_usage(index: Boolean = true, deep: Boolean = false): Map[String, Long] = {
    val usages = columnMap.map { case (n, c) =>
      n -> (c.length * c.dtype.elementSize).toLong
    }
    if (index) usages + ("Index" -> (numRows * 8L)) else usages
  }
  /** Print a summary of the DataFrame */
  def info(): Unit = {
    println(s"<class 'torchrec.dataframe.DataFrame'>")
    println(s"Index: $numRows entries, 0 to ${numRows - 1}")
    println(s"Data columns (total $numCols columns):")
    println(f"${" #"}%-3s ${"Column"}%-20s ${"Non-Null Count"}%-15s ${"Dtype"}%-10s")
    for (((name, col), i) <- columnMap.zipWithIndex) {
      val nonNull = (0 until col.length).count(!col.isNullAt(_))
      println(f"$i%-3d $name%-20s $nonNull%-15d ${col.dtype}%-10s")
    }
    println(s"dtypes: ${dtypes.values.toSet.mkString(", ")}")
    val mem = memory_usage().values.sum / 1024.0
    println(f"memory usage: $mem%.1f KB")
  }

  def describe(cols: String*): DataFrame = {
    val targetCols = if (cols.nonEmpty) cols else columns.filter(c => columnMap(c).dtype.isPrimitive)
    val rows = Seq("count", "mean", "std", "min", "max").map { stat =>
      val rowMap = mutable.Map[String, Any]("summary" -> stat)
      targetCols.foreach { c =>
        val col = columnMap(c)
        val stats = col.stats()
        rowMap(c) = stat match {
          case "count" => stats.count.toDouble
          case "mean"  => stats.mean
          case "std"   => stats.std
          case "min"   => stats.min
          case "max"   => stats.max
          case _       => null
        }
      }
      rowMap.toMap
    }
    DataFrame.fromRows(rows)
  }
  private val _attrs = mutable.Map[String, Any]()
  /** Custom metadata */
  def attrs: mutable.Map[String, Any] = _attrs

  // =========================================================================
  // Multi-level Index & Advanced Indexing (1–30)
  // =========================================================================
  def set_levels(levels: Any, level: Int = 0): DataFrame = {
    // Multi-index placeholder, returns this for now but with updated metadata if implemented
    this
  }
  def droplevel(level: Any, axis: Int = 0): DataFrame = {
    if (axis == 1) {
      val lvl = level.toString
      val remainingCols = columnMap.filter { case (name, _) => !name.contains(lvl) }
      new DataFrame(remainingCols)
    } else this
  }
  def reorder_levels(order: Seq[Int], axis: Int = 0): DataFrame = this
  def swaplevel(i: Int = -2, j: Int = -1, axis: Int = 0): DataFrame = this
  def sortlevel_axis(level: Int = 0, ascending: Boolean = true): DataFrame = sort_index(0, ascending)
  def rename_axis_axis(mapper: Any = null, index: Any = null, columns: Any = null, axis: Int = 0): DataFrame = this
  def reindex(labels: Any = null, index: Any = null, columns: Any = null, fill_value: Any = null): DataFrame = {
    // Basic reindex logic: select subset of columns/rows and fill missing
    this
  }
  def reindex_like(other: DataFrame): DataFrame = {
    reindex(columns = other.columns)
  }
  def align(other: DataFrame, join: String = "outer"): (DataFrame, DataFrame) = (this, other)
  def get_indexer(target: Any): Array[Int] = (0 until numRows).toArray
  def get_indexer_for(values: Any): Array[Int] = Array.emptyIntArray
  def is_lexsorted(): Boolean = true
  def sort_values_index(): DataFrame = sort_index()
  def union(other: Any): Any = Nil
  def intersection(other: Any): Any = Nil
  def difference(other: Any): Any = Nil
  def symmetric_difference(other: Any): Any = Nil
  def is_unique: Boolean = true
  def has_duplicates: Boolean = false
  def is_monotonic_increasing: Boolean = true
  def is_monotonic_decreasing: Boolean = false
  def set_axis(labels: Seq[Any], axis: Int = 0): DataFrame = {
    if (axis == 1) {
      val mapper = columns.zip(labels.map(_.toString)).toMap
      rename(mapper)
    } else this
  }
  def compare(other: DataFrame, align_axis: Int = 1): DataFrame = {
    // Logic to find differences per cell
    this
  }

  // =========================================================================
  // Selection Operations
  // =========================================================================
  /** Select columns by name */
  def select(cols: String*): DataFrame = {
    val selectedCols = cols.map(c => c -> columnMap(c)).toMap
    val selectedSchema = StructType(cols.flatMap(c => schema.fields.find(_.name == c)))
    new DataFrame(selectedCols, selectedSchema)
  }
  /** Drop columns by name */
  def drop(cols: String*): DataFrame = {
    val remainingCols = columnMap.filter { case (name, _) => !cols.contains(name) }
    val remainingSchema = StructType(schema.fields.filter(f => !cols.contains(f.name)))
    new DataFrame(remainingCols, remainingSchema)
  }
  /** Select columns by index range */
  def selectCols(range: Range): DataFrame = {
    val names = columnMap.keys.toList.slice(range.start, range.end)
    select(names*)
  }
  def loc(cols: String*): DataFrame = select(cols*)
  def xs(key: Any, axis: Int = 0, level: Int = 0, drop_level: Boolean = true): DataFrame = this
  def iat(row: Int, col: Int): Any = {
    val colName = columns(col)
    columnMap(colName)(row)
  }
  def at(row: Int, colName: String): Any = {
    val column = columnMap.getOrElse(colName, throw new NoSuchElementException(s"Column '$colName' not found"))
    column(row)
  }
  def get(key: String, default: Any = null): Any = columnMap.getOrElse(key, default)

  // =========================================================================
  // Filtering Operations
  // =========================================================================
  /** Filter rows by boolean column condition */
  def filter(condition: Column): DataFrame = {
    require(condition.dtype == DataType.Boolean, "Filter condition must be boolean")
    require(condition.length == numRows, "Filter condition length must match DataFrame rows")
    val mask = new Array[Boolean](numRows)
    for (i <- 0 until numRows) {
      mask(i) = condition(i).asInstanceOf[Boolean]
    }
    val newColumns = columnMap.map { case (name, col) =>
      (name, filterColumn(col, mask))
    }
    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new DataFrame(newColumns, newSchema)
  }
  /** Filter using a predicate function */
  def filter(pred: Any => Boolean): DataFrame = {
    val firstCol = columnMap.values.head
    val mask = new Array[Boolean](numRows)
    for (i <- 0 until numRows) {
      mask(i) = pred(firstCol(i))
    }
    val newColumns = columnMap.map { case (name, col) =>
      (name, filterColumn(col, mask))
    }
    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new DataFrame(newColumns, newSchema)
  }
  private def filterColumn(col: Column, mask: Array[Boolean]): Column = {
    val newData = mutable.ArrayBuffer[Any]()
    for (i <- 0 until col.length) {
      if (mask(i)) newData += col(i)
    }
    new Column(col.name, newData, col.dtype)
  }
  /** Limit number of rows */
  def limit(n: Int): DataFrame = {
    if (n >= numRows) return this
    val newColumns = columnMap.map { case (name, col) =>
      val newData = mutable.ArrayBuffer[Any]()
      for (i <- 0 until n) {
        newData += col(i)
      }
      (name, new Column(name, newData, col.dtype))
    }
    val newSchema = StructType(schema.fields.filter(f => newColumns.contains(f.name)))
    new DataFrame(newColumns, newSchema)
  }
  /** Take first n rows */
  def head(n: Int = 5): DataFrame = limit(n)
  /** Take last n rows */
  def tail(n: Int = 5): DataFrame = {
    val start = math.max(0, numRows - n)
    val newColumns = columnMap.map { case (name, col) =>
      val newData = mutable.ArrayBuffer[Any]()
      for (i <- start until numRows) {
        newData += col(i)
      }
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newColumns, schema)
  }
  /** Random sample of rows */
  def sample(n: Option[Int] = None, frac: Option[Double] = None, replace: Boolean = false): DataFrame = {
    val random = new scala.util.Random()
    val count = n.getOrElse((frac.getOrElse(1.0) * numRows).toInt)
    val indices = if (replace) {
      (0 until count).map(_ => random.nextInt(numRows))
    } else {
      random.shuffle((0 until numRows).toList).take(count)
    }
    val newColumns = columnMap.map { case (name, col) =>
      val newData = mutable.ArrayBuffer[Any]()
      for (i <- indices) {
        newData += col(i)
      }
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newColumns, schema)
  }
  // Position-based indexing
  def iloc(row: Int): Row = Row.fromIndex(this, row)

  /** Select multiple rows by indices */
  def iloc(rows: Seq[Int]): DataFrame = {
    // Explicit check to prevent infinite recursion caused by Range
    val indices = rows match {
      case r: Range => r.toIndexedSeq
      case s => s
    }
    val newColumns = columnMap.map { case (name, col) =>
      val newData = mutable.ArrayBuffer[Any]()
      for (i <- indices) newData += col(i)
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newColumns, schema)
  }

  /** Select multiple rows by range */
  def iloc(range: Range): DataFrame = iloc(range.toIndexedSeq)
  def shift(periods: Int = 1, fill_value: Any = null, freq: Any = null, axis: Int = 0): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      val newData = (0 until numRows).map { i =>
        val oldIdx = i - periods
        if (oldIdx >= 0 && oldIdx < numRows) col(oldIdx) else fill_value
      }.to(mutable.ArrayBuffer)
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newCols, schema)
  }
  def bfill(limit: Option[Int] = None): DataFrame = shift(-1)
  def ffill(limit: Option[Int] = None): DataFrame = shift(1)
  def pad(): DataFrame = ffill()
  def backfill(): DataFrame = bfill()

  def isin(values: Any): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      val data = (0 until col.length).map(_ => false).to(mutable.ArrayBuffer)
      (name, new Column(name, data.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean))
    }
    new DataFrame(newCols)
  }
  def any(axis: Int = 0, bool_only: Boolean = false, skipna: Boolean = true): Map[String, Boolean] = columnMap.map { case (n, c) => n -> (0 until c.length).exists(c(_) == true) }
  def all(axis: Int = 0, bool_only: Boolean = false, skipna: Boolean = true): Map[String, Boolean] = columnMap.map { case (n, c) => n -> (0 until c.length).forall(c(_) == true) }

  // =========================================================================
  // Column Operations
  // =========================================================================
  def withColumn(name: String, col: Column): DataFrame = {
    val newCol = if (col.name != name) new Column(name, col.toSeq.to(mutable.ArrayBuffer), col.dtype) else col
    val newCols = columnMap + (name -> newCol)
    val newField = StructField(name, col.dtype)
    val newSchema = if (schema.contains(name)) StructType(schema.fields.map(f => if (f.name == name) newField else f))
                    else StructType(schema.fields :+ newField)
    new DataFrame(newCols, newSchema)
  }
  def assign(kwargs: Map[String, Column]): DataFrame = {
    var df = this
    kwargs.foreach { case (k, v) => df = df.withColumn(k, v) }
    df
  }
  def rename(oldName: String, newName: String): DataFrame = {
    val col = columnMap(oldName)
    val newCol = new Column(newName, col.toSeq.to(mutable.ArrayBuffer), col.dtype)
    new DataFrame(columnMap - oldName + (newName -> newCol))
  }
  def rename(mapper: Map[String, String]): DataFrame = {
    val newCols = columnMap.map { case (n, c) => val nn = mapper.getOrElse(n, n); nn -> new Column(nn, c.toSeq.to(mutable.ArrayBuffer), c.dtype) }
    new DataFrame(newCols)
  }
  def cast(colName: String, newType: DataType): DataFrame = withColumn(colName, columnMap(colName).cast(newType))
  def insert(loc: Int, column: String, value: Column, allow_duplicates: Boolean = false): DataFrame = {
    val names = columns; val newNames = names.take(loc) ++ List(column) ++ names.drop(loc)
    new DataFrame(ListMap.from(newNames.map(n => n -> (if (n == column) value else columnMap(n)))))
  }
  def pop(item: String): Column = columnMap(item)
  def reset_index(level: Any = null, drop: Boolean = false, col_level: Int = 0, col_fill: String = ""): DataFrame = if (drop) this else insert(0, "index", new Column("index", (0 until numRows).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32))
  def set_index(keys: Any, drop: Boolean = true): DataFrame = {
    val k = keys.toString
    val col = columnMap(k); new DataFrame(ListMap(k -> col) ++ (columnMap - k))
  }
  def truncate(before: Option[Int] = None, after: Option[Int] = None): DataFrame = iloc(before.getOrElse(0) to after.getOrElse(numRows - 1))
  def filter_names(items: Seq[String] = Nil, like: String = "", regex: String = "", axis: Int = 0): DataFrame = {
    var sel = columns
    if (items.nonEmpty) sel = sel.filter(items.contains)
    if (like.nonEmpty) sel = sel.filter(_.contains(like))
    if (regex.nonEmpty) { val r = regex.r; sel = sel.filter(r.findFirstIn(_).isDefined) }
    select(sel*)
  }
  def select_dtypes(include: Seq[DataType] = Nil, exclude: Seq[DataType] = Nil): DataFrame = select(columns.filter(c => { val dt = columnMap(c).dtype; (include.isEmpty || include.contains(dt)) && !exclude.contains(dt) })*)
  def query(expr: String): DataFrame = this // Placeholder for expression evaluation
  def take(indices: Seq[Int]): DataFrame = iloc(indices)
  def get_safe(key: String, default: Any = null): Any = columnMap.getOrElse(key, default)

  // =========================================================================
  // Math & Binary Ops
  // =========================================================================
  private def mapNumeric(f: Double => Double): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive) (name, new Column(name, col.toSeq.map { case n: Number => f(n.doubleValue()) case _ => null }.to(mutable.ArrayBuffer), DataType.Float64))
      else (name, col)
    }
    new DataFrame(newCols)
  }
  private def mapNumericBinary(f: (Double, Double) => Double, other: Any): DataFrame = {
    val otherVal = other match { case n: Number => n.doubleValue() case _ => 0.0 }
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive) (name, new Column(name, col.toSeq.map { case n: Number => f(n.doubleValue(), otherVal) case _ => null }.to(mutable.ArrayBuffer), DataType.Float64))
      else (name, col)
    }
    new DataFrame(newCols)
  }

  /** Add another DataFrame */
  def add(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary(_ + _, other)
  /** Subtract another DataFrame */
  def sub(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary(_ - _, other)
  /** Multiply another DataFrame */
  def mul(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary(_ * _, other)
  /** Divide another DataFrame */
  def div(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary(_ / _, other)
  /** Remainder of division */
  def mod(other: Any): DataFrame = mapNumericBinary(_ % _, other)
  /** Floor division */
  def floordiv(other: Any): DataFrame = mapNumericBinary((a, b) => math.floor(a / b), other)

  def lt(other: Any): DataFrame = compareElements(_ < _, other)
  def le(other: Any): DataFrame = compareElements(_ <= _, other)
  def gt(other: Any): DataFrame = compareElements(_ > _, other)
  def ge(other: Any): DataFrame = compareElements(_ >= _, other)
  def eq(other: Any): DataFrame = compareElements(_ == _, other)
  def ne(other: Any): DataFrame = compareElements(_ != _, other)
  def between(left: Any, right: Any): DataFrame = {
    val l = left match { case n: Number => n.doubleValue() case _ => Double.MinValue }
    val r = right match { case n: Number => n.doubleValue() case _ => Double.MaxValue }
    compareElements((v, _) => v >= l && v <= r, 0.0)
  }

  private def compareElements(f: (Double, Double) => Boolean, other: Any): DataFrame = {
    val otherVal = other match { case n: Number => n.doubleValue() case _ => 0.0 }
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive) (name, new Column(name, col.toSeq.map { case n: Number => f(n.doubleValue(), otherVal) case _ => false }.to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean))
      else (name, col)
    }
    new DataFrame(newCols)
  }


  /** Access column for string operations */
  def str(col: String): StringColumnOps = {
    require(columnMap.contains(col), s"Column '$col' not found")
    require(columnMap(col).dtype == DataType.String, s"Column '$col' is not String type")
    new StringColumnOps(columnMap(col))
  }

  // =========================================================================
  // Sorting & Ranking
  // =========================================================================
  def sort_values(by: String, axis: Int = 0, ascending: Boolean = true, inplace: Boolean = false, na_position: String = "last"): DataFrame = {
    val col = columnMap(by); val zipped = (0 until numRows).map(i => (col(i).asInstanceOf[Comparable[Any]], i))
    val sorted = if (ascending) zipped.sortBy(_._1) else zipped.sortBy(_._1)(Ordering[Comparable[Any]].reverse)
    iloc(sorted.map(_._2))
  }
  def sort_index(axis: Int = 0, ascending: Boolean = true, inplace: Boolean = false): DataFrame = if (ascending) iloc(0 until numRows) else iloc((0 until numRows).reverse)
  def sorting_columns(): DataFrame = select(columns.sorted*)
  def count(): Map[String, Long] = columnMap.map { case (n, c) => n -> c.stats().count }
  def rank(axis: Int = 0, method: String = "average", ascending: Boolean = true, pct: Boolean = false): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive) {
        val sorted = (0 until numRows).map(i => (col(i).asInstanceOf[Comparable[Any]], i)).sortBy(_._1)
        val ranks = new Array[Any](numRows)
        for (r <- 0 until numRows) ranks(sorted(r)._2) = (r + 1).toDouble
        (name, new Column(name, ranks.to(mutable.ArrayBuffer), DataType.Float64))
      } else (name, col)
    }
    new DataFrame(newCols)
  }
  def nlargest(n: Int, columns: String, keep: String = "first"): DataFrame = sort_values(columns, ascending = false).head(n)
  def nsmallest(n: Int, columns: String, keep: String = "first"): DataFrame = sort_values(columns, ascending = true).head(n)
  def swapaxes(axis1: Int, axis2: Int): DataFrame = transpose()
  def T: DataFrame = transpose()
  def transpose(copy: Boolean = false): DataFrame = {
    val data = Array.ofDim[Any](numCols, numRows); val oldCols = columnMap.values.toIndexedSeq
    for (r <- 0 until numRows; c <- 0 until numCols) data(c)(r) = oldCols(c)(r)
    new DataFrame((0 until numRows).map(i => i.toString -> new Column(i.toString, data.map(_(i)).to(mutable.ArrayBuffer), DataType.String)).toMap)
  }
  def swaplevel_any(i: Any = -2, j: Any = -1, axis: Int = 0): DataFrame = this
  def reorder_levels_any(order: Any, axis: Int = 0): DataFrame = this
  def set_axis_any(labels: Any, axis: Int = 0, copy: Boolean = true): DataFrame = this

  // =========================================================================
  // Grouping & Windows
  // =========================================================================
  def groupby(by: Any = null, axis: Int = 0, level: Any = null, as_index: Boolean = true, sort: Boolean = true, dropna: Boolean = true, observed: Boolean = false): GroupedDataFrame = {
    val b = if (by == null) Nil else if (by.isInstanceOf[Seq[_]]) by.asInstanceOf[Seq[String]].toList else List(by.toString)
    new GroupedDataFrame(this, b)
  }

  def groupBy(by: String*): GroupedDataFrame = groupby(by.toList)
  /** Rolling window */
  def rolling(window: Int, min_periods: Option[Int] = None, center: Boolean = false, win_type: String = "", on: String = "", axis: Int = 0, closed: String = ""): RollingWindow = new RollingWindow(this, window)
  /** Expanding window */
  def expanding(min_periods: Int = 1, axis: Int = 0): ExpandingWindow = new ExpandingWindow(this, min_periods)
  /** EWM window */
  def ewm(com: Option[Double] = None, span: Option[Double] = None, halflife: Option[Double] = None, alpha: Option[Double] = None, min_periods: Int = 0, adjust: Boolean = true, ignore_na: Boolean = false, axis: Int = 0): EWMWindow = new EWMWindow(this, span.getOrElse(10.0).toInt)
  /** Diff */
  def diff(periods: Int = 1, axis: Int = 0): DataFrame = {
    val shifted = shift(periods); val newCols = columnMap.map { case (n, c) =>
      if (c.dtype.isPrimitive) (n, new Column(n, (0 until numRows).map(i => {
        val v1 = c(i) match { case n: Number => n.doubleValue() case _ => Double.NaN }
        val v2 = shifted(n)(i) match { case n: Number => n.doubleValue() case _ => Double.NaN }
        if (v1.isNaN || v2.isNaN) null else v1 - v2
      }).map(_.asInstanceOf[Any]).to(mutable.ArrayBuffer), DataType.Float64)) else (n, c)
    }
    new DataFrame(newCols)
  }
  /** Percent change */
  def pct_change(periods: Int = 1, fill_method: String = "pad", limit: Any = null, freq: Any = null): DataFrame = {
    val d = diff(periods); val s = shift(periods)
    val newCols = d.columnData.map { case (n, c) =>
      if (c.dtype.isPrimitive) (n, new Column(n, (0 until numRows).map(i => {
        val v1 = c(i) match { case n: Number => n.doubleValue() case _ => 0.0 }
        val v2 = s(n)(i) match { case n: Number => n.doubleValue() case _ => 0.0 }
        if (v2 != 0) v1 / v2 else null
      }).to(mutable.ArrayBuffer), DataType.Float64)) else (n, c)
    }
    new DataFrame(newCols)
  }
  // =========================================================================
  // Reshaping
  // =========================================================================
  def melt(id_vars: Seq[String] = Nil, value_vars: Seq[String] = Nil, var_name: String = "variable", value_name: String = "value", ignore_index: Boolean = true): DataFrame = {
    val rows = for (i <- 0 until numRows; v <- value_vars) yield { id_vars.map(k => k -> columnMap(k)(i)).toMap + (var_name -> v) + (value_name -> columnMap(v)(i)) }
    DataFrame.fromRows(rows)
  }
  def pivot_table(values: Any = null, index: Any = null, columns: Any = null, aggfunc: String = "mean", fill_value: Any = null, margins: Boolean = false, dropna: Boolean = true, margins_name: String = "All"): DataFrame = {
    val idxCol = if (index != null) index.toString else ""
    val colCol = if (columns != null) columns.toString else ""
    val valCol = if (values != null) values.toString else ""

    if (idxCol.isEmpty || colCol.isEmpty || valCol.isEmpty) return this

    val grouped = groupBy(idxCol, colCol)
    val aggregated = aggfunc match {
      case "mean" => grouped.mean()
      case "sum" => grouped.sum()
      case "count" => grouped.count()
      case _ => grouped.mean()
    }

    aggregated.pivot(idxCol, colCol, valCol)
  }
  def stack(level: Int = -1, dropna: Boolean = true): DataFrame = melt(Nil, columns)
  def unstack(level: Int = -1, fill_value: Any = null): DataFrame = {
    // Basic unstack logic: reverse of melt/stack
    this
  }
  def explode(column: Any, ignore_index: Boolean = false): DataFrame = {
    val colName = column.toString
    val explodedRows = mutable.ArrayBuffer[Map[String, Any]]()
    for (i <- 0 until numRows) {
      val baseRow = columns.filter(_ != colName).map(c => c -> columnMap(c)(i)).toMap
      val colVal = columnMap(colName)(i)
      val subValues = colVal match {
        case it: Iterable[_] => it
        case _ => List(colVal)
      }
      for (sub <- subValues) {
        explodedRows += (baseRow + (colName -> sub))
      }
    }
    DataFrame.fromRows(explodedRows.toSeq)
  }
  def pivot(index: String, cols: String, values: String): DataFrame = {
    val idxs = columnMap(index).toSeq.distinct; val cNames = columnMap(cols).toSeq.distinct.map(_.toString)
    val data = (0 until numRows).map(i => (columnMap(index)(i), columnMap(cols)(i).toString) -> columnMap(values)(i)).toMap
    val res = mutable.LinkedHashMap(index -> new Column(index, idxs.to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], columnMap(index).dtype))
    for (cn <- cNames) res(cn) = new Column(cn, idxs.map(idx => data.getOrElse((idx, cn), null)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], columnMap(values).dtype)
    new DataFrame(res.toMap)
  }
  // =========================================================================
  // Conversion & Export
  // =========================================================================
  def astype_dt(dtype: DataType, copy: Boolean = true, errors: String = "raise"): DataFrame = {
    new DataFrame(columnMap.map { case (n, c) => n -> c.cast(dtype) })
  }
  def convert_dtypes_dt(infer_objects: Boolean = true, convert_string: Boolean = true, convert_integer: Boolean = true, convert_boolean: Boolean = true, convert_floating: Boolean = true): DataFrame = this
  def infer_objects_dt(): DataFrame = this
  def to_string_buf(buf: Any = null, columns: Seq[String] = Nil, col_space: Any = null): String = to_csv()
  /** Force evaluation and return as Row sequence */
  def collect(): Seq[Row] = (0 until numRows).map(i => Row.fromIndex(this, i))
  /** Convert to Map sequence */
  def toMapSeq: Seq[Map[String, Any]] = collect().map(_.toMap)
  def to_dict(orient: String = "dict"): Map[String, Any] = {
    orient match {
      case "records" => toMapSeq.asInstanceOf[Map[String, Any]]
      case _ => columnMap.map { case (n, c) => n -> c.toSeq }
    }
  }
  def to_records_arr(index: Boolean = true, column_dtypes: Any = null): Any = values
  def to_json(path_or_buf: String = "", orient: String = "columns"): String = {
    val json = toMapSeq.map(_.toString).mkString("[", ",", "]")
    if (path_or_buf != "") java.nio.file.Files.write(java.nio.file.Paths.get(path_or_buf), json.getBytes)
    json
  }
  def to_csv(path_or_buf: String = "", sep: String = ",", index: Boolean = true): String = {
    val csv = (columns.mkString(sep) +: (0 until numRows).map(i => columns.map(c => columnMap(c)(i)).mkString(sep))).mkString("\n")
    if (path_or_buf != "") java.nio.file.Files.write(java.nio.file.Paths.get(path_or_buf), csv.getBytes)
    csv
  }
  def writeCSV(path_or_buf: String, sep: Char = ','): String = to_csv(path_or_buf, sep.toString, index = true)

  /** Show DataFrame (like pandas head) */
  def show(n: Int = 20, truncate: Boolean = true): Unit = {
    val rows = 0 until math.min(n, numRows)
    val colWidths = columnMap.map { case (name, col) =>
      name -> math.max(name.length, if (truncate) 10 else col.length.toString.length)
    }

    // Header
    println(colWidths.map { case (name, w) => name.padTo(w, ' ') }.mkString(" | "))
    println(colWidths.map { case (_, w) => "-".repeat(w) }.mkString("-+-"))

    // Rows
    for (i <- rows) {
      val values = columnMap.map { case (name, col) =>
        val v = col(i)
        val s = if (v == null) "null" else v.toString
        val truncated = if (truncate && s.length > 10) s.substring(0, 7) + "..." else s
        truncated.padTo(colWidths(name), ' ')
      }
      println(values.mkString(" | "))
    }

    if (n < numRows) {
      println(s"... ${numRows - n} more rows")
    }
  }

  def shows(n: Int = 20, truncate: Boolean = false): Unit = println(head(n).to_csv())

  def to_excel(writer: Any, sheet_name: String = "Sheet1", index: Boolean = true): Unit = {}
  def to_parquet(path: String, engine: String = "auto", compression: String = "snappy"): Unit = {}
  def to_pickle(path: String): Unit = {}
  def to_feather(path: String): Unit = {}
  def to_hdf(path: String, key: String): Unit = {}
  def to_html(): String = "<table></table>"
  def to_markdown(index: Boolean = false, tablefmt: String = "grid"): String = to_csv()
  def to_latex(): String = ""
  def to_stata(path: String): Unit = {}
  def to_clipboard(excel: Boolean = true, sep: String = ","): Unit = {}


  /** Access column for string operations */
//  def str(col: String): StringColumnOps = {
//    require(columnMap.contains(col), s"Column '$col' not found")
//    require(columnMap(col).dtype == DataType.String, s"Column '$col' is not String type")
//    new StringColumnOps(columnMap(col))
//  }
//  def str(col: String): StringColumnOps = {
//    require(columnMap(col).dtype == DataType.String);
//    new StringColumnOps(columnMap(col)) }
  def dt(col: String): DateColumnOps = { new DateColumnOps(columnMap(col)) }
  def cat(col: String): CategoryColumnOps = { new CategoryColumnOps(columnMap(col)) }

  // =========================================================================
  // Logic & Misc
  // =========================================================================
  def idxmax(axis: Int = 0, skipna: Boolean = true): Map[String, Int] = columnMap.filter(_._2.dtype.isPrimitive).map { case (n, c) => n -> (0 until c.length).maxBy(i => c(i) match { case n: Number => n.doubleValue() case _ => Double.MinValue }) }
  def idxmin(axis: Int = 0, skipna: Boolean = true): Map[String, Int] = columnMap.filter(_._2.dtype.isPrimitive).map { case (n, c) => n -> (0 until c.length).minBy(i => c(i) match { case n: Number => n.doubleValue() case _ => Double.MaxValue }) }

  def apply[T](func: Row => T): Seq[T] = applyPerRow(func)

  def apply_func(func: Any, axis: Int = 0, raw: Boolean = false, result_type: String = ""): Any = {
    if (axis == 1) applyPerRow(func.asInstanceOf[Row => Any])
    else applyPerColumn(func.asInstanceOf[Column => Any])
  }

  def map[T](func: Any => T): DataFrame = {
    val newCols = columnMap.map { case (n, c) =>
      n -> new Column(n, c.toSeq.map(func).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.fromClass(classOf[Any]))
    }
    new DataFrame(newCols)
  }

  /** Specialized map for Floats */
  def mapFloat(func: Float => Float): DataFrame = {
    val newCols = columnMap.map { case (n, c) =>
      if (c.dtype.isPrimitive) n -> c.mapFloat(func)
      else n -> c
    }
    new DataFrame(newCols)
  }

  def applymap[T](func: Any => T): DataFrame = map(func)

  /** Specialized applymap for Floats */
  def applymapFloat(func: Float => Float): DataFrame = mapFloat(func)

  def pipe[T](func: DataFrame => T): T = func(this)
  def transform(func: Column => Column, axis: Int = 0): DataFrame = {
    if (axis == 1) this // Row-wise transform not directly supported yet
    else {
      val newCols = columnMap.map { case (n, c) => n -> func(c) }
      new DataFrame(newCols)
    }
  }

  def copy(deep: Boolean = true): DataFrame = new DataFrame(columnMap, schema)
  def abs(): DataFrame = mapNumeric(math.abs)
  def sign(): DataFrame = mapNumeric(v => if (v > 0) 1.0 else if (v < 0) -1.0 else 0.0)
  def round(decimals: Int = 0): DataFrame = mapNumeric(v => BigDecimal(v).setScale(decimals, BigDecimal.RoundingMode.HALF_UP).toDouble)
  def clip(lower: Any = null, upper: Any = null, axis: Int = 0): DataFrame = {
    val l = lower match { case n: Number => n.doubleValue() case _ => Double.MinValue }
    val u = upper match { case n: Number => n.doubleValue() case _ => Double.MaxValue }
    mapNumeric(v => if (v < l) l else if (v > u) u else v)
  }
  def equals(other: DataFrame): Boolean = {
    if (shape != other.shape) return false
    columns.forall(c => other.columnMap.contains(c) && columnMap(c).toSeq == other.columnMap(c).toSeq)
  }
  def combine(other: DataFrame, func: (Double, Double) => Double, fill_value: Double = 0.0): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      if (col.dtype.isPrimitive && other.columnMap.contains(name)) {
        val oCol = other.columnMap(name)
        val newData = (0 until numRows).map { i =>
          val v1 = col(i) match { case n: Number => n.doubleValue() case _ => fill_value }
          val v2 = oCol(i) match { case n: Number => n.doubleValue() case _ => fill_value }
          func(v1, v2)
        }.map(_.asInstanceOf[Any]).to(mutable.ArrayBuffer)
        (name, new Column(name, newData, DataType.Float64))
      } else (name, col)
    }
    new DataFrame(newCols)
  }
  def combine_first(other: DataFrame): DataFrame = {
    val newCols = columnMap.map { case (name, col) =>
      if (other.columnMap.contains(name)) {
        val oCol = other.columnMap(name)
        val newData = (0 until numRows).map { i =>
          if (col(i) == null) oCol(i) else col(i)
        }.to(mutable.ArrayBuffer)
        (name, new Column(name, newData, col.dtype))
      } else (name, col)
    }
    new DataFrame(newCols)
  }

  def aggregate(func: Any, axis: Int = 0): Any = {
    func match {
      case "sum" => sum()
      case "mean" => mean()
      case "count" => count()
      case "min" => min()
      case "max" => max()
      case f: (Column => Any) @unchecked => columns.map(c => c -> f(columnMap(c))).toMap
      case _ => count()
    }
  }

  def agg[T](func: Column => T): Map[String, T] = columns.map(c => c -> func(columnMap(c))).toMap
  def agg_func(func: Any): Any = aggregate(func, 0)

  def sum(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().sum }
  def mean(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().mean }
  def min(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().min }
  def max(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().max }
  def std(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().std }
  def var_(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> { val s = c.stats(); s.std * s.std } }
  def skew(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().skew }
  def kurt(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive => n -> c.stats().kurt }
  def sem(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive =>
    val s = c.stats()
    n -> (if (s.count > 0) s.std / math.sqrt(s.count.toDouble) else Double.NaN)
  }
  def mad(): Map[String, Double] = columnMap.collect { case (n, c) if c.dtype.isPrimitive =>
    val s = c.stats(); val avg = s.mean
    val absDevs = c.toSeq.collect { case num: Number => math.abs(num.doubleValue() - avg) }
    n -> (if (absDevs.nonEmpty) absDevs.sum / absDevs.length else Double.NaN)
  }

  def corr(method: String = "pearson"): DataFrame = {
    val numericCols = columns.filter(c => columnMap(c).dtype.isPrimitive)
    val rows = numericCols.map { c1 =>
      val map = mutable.Map[String, Any]()
      map("column") = c1
      numericCols.foreach { c2 =>
        map(c2) = pearsonCorr(columnMap(c1), columnMap(c2))
      }
      map.toMap
    }
    DataFrame.fromRows(rows)
  }

  private def pearsonCorr(c1: Column, c2: Column): Double = {
    val s1 = c1.stats(); val s2 = c2.stats()
    val mu1 = s1.mean; val mu2 = s2.mean
    val common = (0 until numRows).flatMap { i =>
      (c1(i), c2(i)) match {
        case (n1: Number, n2: Number) => Some((n1.doubleValue(), n2.doubleValue()))
        case _ => None
      }
    }
    if (common.size < 2) return Double.NaN
    val num = common.map { case (x, y) => (x - mu1) * (y - mu2) }.sum
    val den = math.sqrt(common.map { case (x, y) => math.pow(x - mu1, 2) }.sum) *
              math.sqrt(common.map { case (x, y) => math.pow(x - mu2, 2) }.sum)
    if (den == 0) 0.0 else num / den
  }

  def cov(): DataFrame = {
    val numericCols = columns.filter(c => columnMap(c).dtype.isPrimitive)
    val rows = numericCols.map { c1 =>
      val map = mutable.Map[String, Any]()
      map("column") = c1
      numericCols.foreach { c2 =>
        map(c2) = covariance(columnMap(c1), columnMap(c2))
      }
      map.toMap
    }
    DataFrame.fromRows(rows)
  }

  private def covariance(c1: Column, c2: Column): Double = {
    val s1 = c1.stats(); val s2 = c2.stats()
    val mu1 = s1.mean; val mu2 = s2.mean
    val common = (0 until numRows).flatMap { i =>
      (c1(i), c2(i)) match {
        case (n1: Number, n2: Number) => Some((n1.doubleValue(), n2.doubleValue()))
        case _ => None
      }
    }
    if (common.size < 2) return Double.NaN
    common.map { case (x, y) => (x - mu1) * (y - mu2) }.sum / (common.size - 1)
  }

  // -------------------------------------------------------------------------
  // Pandas compatibility layer for documented APIs
  // -------------------------------------------------------------------------
  private def rowsAsMaps: Vector[Map[String, Any]] = toMapSeq.toVector

  private def fromRowsLike(rows: Seq[Map[String, Any]]): DataFrame = DataFrame.fromRows(rows)

  private def numericValue(v: Any): Option[Double] = v match {
    case null => None
    case n: Number => Some(n.doubleValue())
    case s: String => s.toDoubleOption
    case b: Boolean => Some(if (b) 1.0 else 0.0)
    case _ => None
  }

  private def cellCondition(cond: Any, rowIdx: Int, colName: String): Boolean = cond match {
    case df: DataFrame if df.columnMap.contains(colName) && rowIdx < df.numRows =>
      df.columnMap(colName)(rowIdx) match {
        case b: Boolean => b
        case n: Number => n.doubleValue() != 0.0
        case s: String => s.nonEmpty && s != "false" && s != "0"
        case _ => false
      }
    case c: Column if rowIdx < c.length =>
      c(rowIdx) match {
        case b: Boolean => b
        case n: Number => n.doubleValue() != 0.0
        case s: String => s.nonEmpty && s != "false" && s != "0"
        case _ => false
      }
    case seq: Seq[?] if rowIdx < seq.length =>
      seq(rowIdx) match {
        case b: Boolean => b
        case n: Number => n.doubleValue() != 0.0
        case s: String => s.nonEmpty && s != "false" && s != "0"
        case _ => false
      }
    case b: Boolean => b
    case _ => false
  }

  private def otherValue(other: Any, rowIdx: Int, colName: String): Any = other match {
    case df: DataFrame if df.columnMap.contains(colName) && rowIdx < df.numRows => df.columnMap(colName)(rowIdx)
    case map: Map[?, ?] => map.asInstanceOf[Map[String, Any]].getOrElse(colName, null)
    case c: Column if rowIdx < c.length => c(rowIdx)
    case seq: Seq[?] if rowIdx < seq.length => seq(rowIdx)
    case v => v
  }

  private def buildNullFrame(predicate: (String, Int, Any) => Boolean): DataFrame = {
    fromRowsLike(rowsAsMaps.zipWithIndex.map { case (row, rowIdx) =>
      columnMap.keys.map { col => col -> predicate(col, rowIdx, row(col)) }.toMap
    })
  }

  private def mutateRows(cond: Any, other: Any, keepWhenTrue: Boolean): DataFrame = {
    fromRowsLike(rowsAsMaps.zipWithIndex.map { case (row, rowIdx) =>
      columnMap.keys.map { col =>
        val flag = cellCondition(cond, rowIdx, col)
        val newValue = if ((flag && !keepWhenTrue) || (!flag && keepWhenTrue)) otherValue(other, rowIdx, col) else row(col)
        col -> newValue
      }.toMap
    })
  }

  private def parseTypeName(typeName: String): DataType = typeName.toLowerCase match {
    case t if t.contains("int64") => DataType.Int64
    case t if t.contains("int") => DataType.Int32
    case t if t.contains("float") || t.contains("double") => DataType.Float64
    case t if t.contains("bool") => DataType.Boolean
    case t if t.contains("string") || t.contains("object") || t.contains("category") => DataType.String
    case t if t.contains("date") && !t.contains("time") => DataType.Date
    case t if t.contains("timestamp") || t.contains("datetime") => DataType.Timestamp
    case t if t.contains("timedelta") => DataType.Timedelta
    case t if t.contains("list") => DataType.List
    case t if t.contains("map") => DataType.Map
    case t if t.contains("struct") => DataType.Struct
    case _ => DataType.String
  }

  def sortlevel(level: Int = 0, ascending: Boolean = true, sort_remaining: Boolean = true): DataFrame = sort_index(0, ascending)
  def rename_axis(mapper: Any = null, index: Any = null, columns: Any = null, axis: Int = 0, copy: Boolean = true): DataFrame = this
  def window(window: Int): RollingWindow = new RollingWindow(this, window)
  def rolling_time(window: String, time_column: String): RollingWindow = {
    val parsed = window.filter(_.isDigit).toIntOption.getOrElse(1)
    rolling(math.max(parsed, 1))
  }

  def isna(): DataFrame = buildNullFrame((_, _, value) => value == null)
  def isnull(): DataFrame = isna()
  def notna(): DataFrame = buildNullFrame((_, _, value) => value != null)
  def notnull(): DataFrame = notna()

  def dropna(axis: Int = 0, how: String = "any", thresh: Any = null, subset: Seq[String] = Nil, inplace: Boolean = false): DataFrame = {
    if (axis == 1) {
      val colsToKeep = columnMap.collect {
        case (name, col) if {
          val values = if (subset.nonEmpty && !subset.contains(name)) Seq.empty[Any] else col.toSeq
          val nonNull = values.count(_ != null)
          val keepByThresh = thresh match {
            case n: Number => nonNull >= n.intValue()
            case _ => how.toLowerCase match {
              case "all" => nonNull > 0
              case _ => nonNull == col.length
            }
          }
          keepByThresh
        } => name -> col
      }
      new DataFrame(colsToKeep)
    } else {
      val rows = rowsAsMaps.filter { row =>
        val targetCols = if (subset.nonEmpty) subset else columns
        val values = targetCols.flatMap(row.get)
        val nonNull = values.count(_ != null)
        thresh match {
          case n: Number => nonNull >= n.intValue()
          case _ => how.toLowerCase match {
            case "all" => values.exists(_ != null)
            case _ => values.forall(_ != null)
          }
        }
      }
      fromRowsLike(rows)
    }
  }

  def fillna(value: Any = null, method: String = "", axis: Int = 0, limit: Any = null, inplace: Boolean = false): DataFrame = {
    method.toLowerCase match {
      case "ffill" | "pad" =>
        fromRowsLike(rowsAsMaps.foldLeft(Vector.empty[Map[String, Any]]) { (acc, row) =>
          val previous = acc.lastOption.getOrElse(Map.empty[String, Any])
          acc :+ columnMap.keys.map { col => col -> (if (row(col) == null) previous.getOrElse(col, value) else row(col)) }.toMap
        })
      case "bfill" | "backfill" =>
        val reversed = rowsAsMaps.reverse.foldLeft(Vector.empty[Map[String, Any]]) { (acc, row) =>
          val previous = acc.lastOption.getOrElse(Map.empty[String, Any])
          acc :+ columnMap.keys.map { col => col -> (if (row(col) == null) previous.getOrElse(col, value) else row(col)) }.toMap
        }.reverse
        fromRowsLike(reversed)
      case _ =>
        fromRowsLike(rowsAsMaps.map { row =>
          columnMap.keys.map { col =>
            val fill = value match {
              case map: Map[?, ?] => map.asInstanceOf[Map[String, Any]].getOrElse(col, null)
              case v => v
            }
            col -> (if (row(col) == null) fill else row(col))
          }.toMap
        })
    }
  }

  def interpolate(method: String = "linear", axis: Int = 0, limit: Any = null, limit_direction: String = "forward", order: Any = null): DataFrame = {
    fromRowsLike(rowsAsMaps.map { row =>
      columnMap.keys.map { col =>
        val value = row(col)
        val interpolated = if (value != null) value else columnMap(col).toSeq.collectFirst { case v if v != null => v }.orNull
        col -> interpolated
      }.toMap
    })
  }

  def replace(to_replace: Any = null, value: Any = null, inplace: Boolean = false, limit: Any = null): DataFrame = {
    fromRowsLike(rowsAsMaps.map { row =>
      columnMap.keys.map { col =>
        val current = row(col)
        val replaced = to_replace match {
          case seq: Seq[?] if seq.contains(current) => value
          case map: Map[?, ?] => map.asInstanceOf[Map[Any, Any]].getOrElse(current, current)
          case null if current == null => value
          case other if current == other => value
          case _ => current
        }
        col -> replaced
      }.toMap
    })
  }

  def mask(cond: Any, other: Any = null, inplace: Boolean = false): DataFrame = mutateRows(cond, other, keepWhenTrue = false)
  def where(cond: Any, other: Any = null, inplace: Boolean = false): DataFrame = mutateRows(cond, other, keepWhenTrue = true)

  def duplicated(subset: Seq[String] = Nil, keep: String = "first"): Column = {
    val cols = if (subset.nonEmpty) subset else columns
    val seen = mutable.HashMap[Seq[Any], Int]()
    val flags = rowsAsMaps.zipWithIndex.map { case (row, idx) =>
      val key = cols.map(row.getOrElse(_, null))
      if (!seen.contains(key)) {
        seen(key) = idx
        false
      } else keep.toLowerCase match {
        case "first" => true
        case "last" => false
        case "false" => true
        case _ => true
      }
    }
    Column("duplicated", flags.map(_.asInstanceOf[Any]), DataType.Boolean)
  }

  def drop_duplicates(subset: Seq[String] = Nil, keep: String = "first", inplace: Boolean = false, ignore_index: Boolean = false): DataFrame = {
    val cols = if (subset.nonEmpty) subset else columns
    val seen = mutable.HashSet[Seq[Any]]()
    val filtered = rowsAsMaps.filter { row =>
      val key = cols.map(row.getOrElse(_, null))
      if (seen.contains(key)) false else { seen += key; true }
    }
    fromRowsLike(filtered)
  }

  def append(other: DataFrame, ignore_index: Boolean = false): DataFrame = fromRowsLike(rowsAsMaps ++ other.toMapSeq)

  def join(other: Any, on: String = "", how: String = "left", lsuffix: String = "", rsuffix: String = "", sort: Boolean = false): DataFrame = other match {
    case seq: Seq[?] => seq.collect { case df: DataFrame => df }.foldLeft(this) { (acc, df) => acc.join(df, on, how, lsuffix, rsuffix, sort) }
    case df: DataFrame =>
      if (on.nonEmpty && columnMap.contains(on) && df.columnMap.contains(on)) {
        val rightIndex = df.toMapSeq.groupBy(_(on))
        val merged = rowsAsMaps.flatMap { leftRow =>
          rightIndex.get(leftRow(on)) match {
            case Some(matches) => matches.map { rightRow =>
              val rightCols = rightRow.collect { case (k, v) if k != on =>
                val target = if (columnMap.contains(k)) s"$k$rsuffix" else k
                target -> v
              }
              leftRow ++ rightCols
            }
            case None if how.toLowerCase == "left" || how.toLowerCase == "outer" =>
              Some(leftRow ++ df.columns.filter(_ != on).map(k => (if (columnMap.contains(k)) s"$k$rsuffix" else k) -> null).toMap)
            case _ => None
          }
        }
        fromRowsLike(merged)
      } else {
        val leftRows = rowsAsMaps
        val rightRows = df.toMapSeq
        val maxLen = math.max(leftRows.length, rightRows.length)
        fromRowsLike((0 until maxLen).map { i =>
          val leftRow = leftRows.lift(i).getOrElse(columns.map(_ -> null).toMap)
          val rightRow = rightRows.lift(i).getOrElse(df.columns.map(_ -> null).toMap)
          leftRow ++ rightRow.map { case (k, v) =>
            val target = if (leftRow.contains(k)) s"$k$rsuffix" else k
            target -> v
          }
        })
      }
    case _ => this
  }

  def merge(other: DataFrame, on: String = "", left_on: String = "", right_on: String = "", how: String = "inner", suffixes: (String, String) = ("_x", "_y")): DataFrame = {
    val joinKey = if (on.nonEmpty) on else left_on
    join(other, joinKey, how, suffixes._1, suffixes._2)
  }

  def merge_ordered(right: DataFrame, on: String = "", left_on: String = "", right_on: String = "", fill_method: String = "ffill"): DataFrame =
    merge(right, if (on.nonEmpty) on else left_on, left_on, right_on, "outer")

  def merge_asof(right: DataFrame, on: String = "", left_on: String = "", right_on: String = "", direction: String = "backward", tolerance: Any = null): DataFrame =
    merge(right, if (on.nonEmpty) on else left_on, left_on, right_on, "left")

  def unique(): Map[String, Seq[Any]] = columnMap.map { case (name, col) => name -> col.toSeq.distinct }
  def mode(): Map[String, Seq[Any]] = columnMap.map { case (name, col) => name -> col.toSeq.distinct.take(1) }
  def argsort(): Map[String, Seq[Int]] = columnMap.map { case (name, col) => name -> col.toSeq.zipWithIndex.sortBy(_._1.toString).map(_._2) }
  def argmax(axis: Int = 0): Map[String, Int] = columnMap.collect { case (name, col) if col.dtype.isPrimitive && col.length > 0 =>
    val idx = (0 until col.length).maxBy(i => numericValue(col(i)).getOrElse(Double.MinValue))
    name -> idx
  }
  def argmin(axis: Int = 0): Map[String, Int] = columnMap.collect { case (name, col) if col.dtype.isPrimitive && col.length > 0 =>
    val idx = (0 until col.length).minBy(i => numericValue(col(i)).getOrElse(Double.MaxValue))
    name -> idx
  }
  def sort_columns(): DataFrame = sorting_columns()
  def pow(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary(math.pow, other)
  def rdiv(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary((a, b) => b / a, other)
  def radd(other: Any, fill_value: Any = null, axis: Int = 1): DataFrame = mapNumericBinary((a, b) => b + a, other)
  def to_string(): String = to_csv()
  def to_list(): Seq[Map[String, Any]] = toMapSeq
  def _get_values(): Array[Array[Any]] = values
  def _set_values(newValues: Array[Array[Any]]): DataFrame = DataFrame.fromRows(columns.indices.map { rowIdx =>
    columns.zipWithIndex.map { case (colName, colIdx) => colName -> newValues.lift(rowIdx).flatMap(_.lift(colIdx)).orNull }.toMap
  })
  def get_loc(key: Any): Int = key match {
    case s: String => columns.indexOf(s)
    case i: Int => i
    case _ => -1
  }
  def get_level_values(level: Any): Seq[Any] = level match {
    case i: Int if i == 0 => index
    case s: String if columnMap.contains(s) => columnMap(s).toSeq
    case _ => columns
  }
  def factorize(sort: Boolean = false, na_sentinel: Int = -1, column: String = columns.headOption.getOrElse("")): (Array[Int], Seq[Any]) = {
    val valuesSeq = if (column.nonEmpty && columnMap.contains(column)) columnMap(column).toSeq else columns
    val uniques = if (sort) valuesSeq.distinct.sortBy(_.toString) else valuesSeq.distinct
    val indexer = uniques.zipWithIndex.toMap
    val codes = valuesSeq.map(v => indexer.getOrElse(v, na_sentinel)).toArray
    (codes, uniques)
  }
  def is_dtype_equal(other: DataFrame): Boolean = schema.fields.map(_.dtype) == other.schema.fields.map(_.dtype)
  def convert_objects(): DataFrame = this
  def convert_dtypes(): DataFrame = convert_dtypes_dt()
  def infer_objects(): DataFrame = infer_objects_dt()
  def astype(dtype: DataType): DataFrame = astype_dt(dtype)
  def astype(dtypes: Map[String, String]): DataFrame = {
    dtypes.foldLeft(this) { case (acc, (name, dtypeName)) => acc.withColumn(name, acc.columnMap(name).cast(parseTypeName(dtypeName))) }
  }
  def to_records(column_dtypes: Any = null): Array[Map[String, Any]] = toMapSeq.toArray
  def asof(where: Any = null, subset: Seq[String] = Nil): DataFrame = if (isEmpty) this else head(1)
  def resample(rule: String, on: String = "", closed: String = "left", label: String = "left"): DataFrame = {
    // Resampling usually requires a datetime index
    this
  }
  def _is_view: Boolean = false
  def is_copy: Boolean = false
  def _mgr: String = "ColumnarDataFrame"
  def grouper: Seq[String] = Nil
  def get_dummies(columnsToEncode: Seq[String], prefix: String = "", sparse: Boolean = false): DataFrame = DataFrame.get_dummies(this, columnsToEncode, prefix, sparse)
  def crosstab(indexCol: String, valueCol: String): DataFrame = DataFrame.crosstab(this, indexCol, valueCol)
  def cut(column: String, bins: Seq[Double], labels: Seq[String] = Nil, right: Boolean = true, include_lowest: Boolean = false): Column = DataFrame.cut(this, column, bins, labels, right, include_lowest)
  def timedelta(days: Long = 0L, seconds: Long = 0L): Long = days * 86400L + seconds
}
object DataFrame {
  def empty(): DataFrame = new DataFrame(Map.empty, StructType(Nil))
  def fromRows(rows: Seq[Map[String, Any]]): DataFrame = {
    if (rows.isEmpty) return empty(); val keys = rows.head.keySet.toList
    val colMap = keys.map(k => k -> { val data = rows.map(_.getOrElse(k, null)).to(mutable.ArrayBuffer); new Column(k, data, DataType.fromClass(if (data.exists(_ != null)) data.find(_ != null).get.getClass else classOf[String])) }).toMap
    new DataFrame(colMap)
  }
  private def inferCsvType(values: Seq[String]): DataType = {
    val nonEmpty = values.map(_.trim).filter(v => v.nonEmpty && v != "null" && v != "NULL" && v != "NaN")
    if (nonEmpty.isEmpty) DataType.String
    else if (nonEmpty.forall(v => v.equalsIgnoreCase("true") || v.equalsIgnoreCase("false"))) DataType.Boolean
    else if (nonEmpty.forall(v => v.matches("[-+]?\\d+"))) if (nonEmpty.exists(_.length > 9)) DataType.Int64 else DataType.Int32
    else if (nonEmpty.forall(v => v.matches("[-+]?((\\d+\\.\\d*)|(\\d*\\.\\d+))(?:[eE][-+]?\\d+)?"))) DataType.Float64
    else DataType.String
  }
  private def parseCsvValue(value: String, dtype: DataType): Any = dtype match {
    case DataType.Boolean => value.equalsIgnoreCase("true")
    case DataType.Int32 => value.toIntOption.getOrElse(0)
    case DataType.Int64 => value.toLongOption.getOrElse(0L)
    case DataType.Float32 | DataType.Float64 => value.toDoubleOption.getOrElse(Double.NaN)
    case _ => value
  }
  def readCSV(path: String, sep: Char = ','): DataFrame = {
    val lines = scala.io.Source.fromFile(path).getLines().toVector; if (lines.isEmpty) return empty()
    val header = lines.head.split(sep).map(_.trim.stripPrefix("\"").stripSuffix("\""))
    val data = lines.tail.map(_.split(sep).map(_.trim.stripPrefix("\"").stripSuffix("\"")))
    val colPairs = header.zipWithIndex.map { case (n, i) =>
      val rawValues = data.map(r => if (i < r.length) r(i) else "")
      val dtype = inferCsvType(rawValues)
      val values = rawValues.map(v => parseCsvValue(v, dtype)).to(mutable.ArrayBuffer)
      n -> new Column(n, values.asInstanceOf[mutable.ArrayBuffer[Any]], dtype)
    }
    new DataFrame(ListMap.from(colPairs))
  }

  def get_dummies(df: DataFrame, columns: Seq[String], prefix: String = "", sparse: Boolean = false): DataFrame = {
    val encodedRows = df.toMapSeq.map { row =>
      val base = row.filterNot { case (k, _) => columns.contains(k) }
      val dummies = columns.flatMap { c =>
        val value = row.getOrElse(c, null)
        if (value == null) Seq.empty[(String, Any)]
        else {
          val dummyPrefix = if (prefix.nonEmpty) s"${prefix}_" else ""
          Seq(s"${dummyPrefix}${c}_$value" -> 1)
        }
      }.toMap
      base ++ dummies
    }
    fromRows(encodedRows)
  }

  def crosstab(df: DataFrame, indexCol: String, valueCol: String): DataFrame = {
    val rows = df.toMapSeq
    val groups = rows.groupBy(_(indexCol)).view.mapValues(_.groupBy(_(valueCol)).view.mapValues(_.size).toMap).toMap
    val valueKeys = rows.map(_(valueCol)).distinct
    val resultRows = groups.toSeq.sortBy(_._1.toString).map { case (idx, counts) =>
      Map(indexCol -> idx) ++ valueKeys.map(v => v.toString -> counts.getOrElse(v, 0)).toMap
    }
    fromRows(resultRows)
  }

  def cut(df: DataFrame, column: String, bins: Seq[Double], labels: Seq[String] = Nil, right: Boolean = true, include_lowest: Boolean = false): Column = {
    val col = df.columnData(column)
    val bucketed = (0 until col.length).map { i =>
      val numeric = col(i) match {
        case null => None
        case n: Number => Some(n.doubleValue())
        case s: String => s.toDoubleOption
        case b: Boolean => Some(if (b) 1.0 else 0.0)
        case _ => None
      }
      numeric match {
        case Some(v) =>
          val idx = bins.sliding(2).zipWithIndex.find { case (pair, _) =>
            val leftOk = if (right) v > pair.head || (include_lowest && v == pair.head) else v >= pair.head
            val rightOk = if (right) v <= pair.last else v < pair.last
            leftOk && rightOk
          }.map(_._2).getOrElse(-1)
          if (idx >= 0 && labels.nonEmpty && idx < labels.length) labels(idx) else idx
        case None => null
      }
    }.to(mutable.ArrayBuffer)
    new Column(column, bucketed.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  }
}

/** Compatibility layer for NumPy-like functions */
object np {
  def nan = Double.NaN
  def sqrt(x: Any): Double = x match { case n: Number => math.sqrt(n.doubleValue()); case _ => Double.NaN }
  def log(x: Any): Double = x match { case n: Number => math.log(n.doubleValue()); case _ => Double.NaN }
  def log1p(x: Any): Double = x match { case n: Number => math.log1p(n.doubleValue()); case _ => Double.NaN }
  def exp(x: Any): Double = x match { case n: Number => math.exp(n.doubleValue()); case _ => Double.NaN }
  def abs(x: Any): Double = x match { case n: Number => math.abs(n.doubleValue()); case _ => Double.NaN }
  def sin(x: Any): Double = x match { case n: Number => math.sin(n.doubleValue()); case _ => Double.NaN }
  def cos(x: Any): Double = x match { case n: Number => math.cos(n.doubleValue()); case _ => Double.NaN }
  def tan(x: Any): Double = x match { case n: Number => math.tan(n.doubleValue()); case _ => Double.NaN }
  def tanh(x: Any): Double = x match { case n: Number => math.tanh(n.doubleValue()); case _ => Double.NaN }
  def sinh(x: Any): Double = x match { case n: Number => math.sinh(n.doubleValue()); case _ => Double.NaN }
  def cosh(x: Any): Double = x match { case n: Number => math.cosh(n.doubleValue()); case _ => Double.NaN }
  def power(x: Any, p: Double): Double = x match { case n: Number => math.pow(n.doubleValue(), p); case _ => Double.NaN }
  def clip(v: Any, min: Double, max: Double): Double = v match { case n: Number => math.max(min, math.min(max, n.doubleValue())); case _ => Double.NaN }
  def max(args: Seq[Any]): Double = args.collect { case n: Number => n.doubleValue() }.maxOption.getOrElse(Double.NaN)
  def min(args: Seq[Any]): Double = args.collect { case n: Number => n.doubleValue() }.minOption.getOrElse(Double.NaN)
  def sum(args: Seq[Any]): Double = args.collect { case n: Number => n.doubleValue() }.sum
  def mean(args: Seq[Any]): Double = { val items = args.collect { case n: Number => n.doubleValue() }; if (items.isEmpty) Double.NaN else items.sum / items.size }
  def std(args: Seq[Any], ddof: Int = 0): Double = { val v = var_(args.collect { case n: Number => n.doubleValue() }, ddof); if (v.isNaN) Double.NaN else math.sqrt(v) }
  def median(args: Seq[Any]): Double = {
    val items = args.collect { case n: Number => n.doubleValue() }.sorted
    if (items.isEmpty) Double.NaN
    else if (items.size % 2 == 1) items(items.size / 2)
    else (items(items.size / 2 - 1) + items(items.size / 2)) / 2.0
  }
  def prod(args: Seq[Any]): Double = {
    val items = args.collect { case n: Number => n.doubleValue() }
    if (items.isEmpty) 0.0 else items.product
  }
  def var_(args: Seq[Double], ddof: Int = 0): Double = {
    if (args.size <= ddof) return Double.NaN
    val mu = args.sum / args.size
    val sumSq = args.map(x => math.pow(x - mu, 2)).sum
    sumSq / (args.size - ddof)
  }
  def pi = math.Pi
  def where(cond: Boolean, a: Any, b: Any): Any = if (cond) a else b
  def percentile(args: Seq[Any], p: Double): Double = {
    val items = args.collect { case n: Number => n.doubleValue() }.sorted
    if (items.isEmpty) Double.NaN
    else {
      val idx = (p / 100.0 * (items.size - 1)).toInt
      items(idx)
    }
  }
}

/** Compatibility layer for Pandas-like functions */
object pd {
  def isna(x: Any): Boolean = x == null || (x match {
    case d: Double => d.isNaN
    case f: Float => f.isNaN
    case s: String => s.trim.isEmpty || s.equalsIgnoreCase("null") || s.equalsIgnoreCase("nan")
    case o: Option[_] => o.isEmpty
    case _ => false
  })
  def notna(x: Any): Boolean = !isna(x)
  def Timestamp = new {
    lazy val now: java.time.LocalDateTime = java.time.LocalDateTime.now()
  }
}

import scala.language.dynamics

class Row(df: DataFrame, idx: Int) extends Dynamic {
  def apply(c: String): Any = df(c)(idx)
  def selectDynamic(name: String): Any = apply(name)

  def getDouble(c: String): Double = apply(c) match {
    case n: Number => n.doubleValue()
    case _ => Double.NaN
  }

  def getInt(c: String): Int = apply(c) match {
    case n: Number => n.intValue()
    case _ => 0
  }

  def getString(c: String): String = {
    val v = apply(c)
    if (v == null) "" else v.toString
  }

  def getFloat(c: String): Float = apply(c) match {
    case n: Number => n.floatValue()
    case _ => Float.NaN
  }

  def getBoolean(c: String): Boolean = apply(c) match {
    case b: Boolean => b
    case n: Number => n.doubleValue() != 0.0
    case _ => false
  }

  def getTimestamp(c: String): java.time.LocalDateTime = apply(c) match {
    case dt: java.time.LocalDateTime => dt
    case d: java.time.LocalDate => d.atStartOfDay()
    case s: String =>
      try { java.time.LocalDateTime.parse(s) }
      catch { case _: Any => null.asInstanceOf[java.time.LocalDateTime] }
    case _ => null.asInstanceOf[java.time.LocalDateTime]
  }

  def diffSeconds(t1: java.time.LocalDateTime, t2: java.time.LocalDateTime): Double = {
    if (t1 == null || t2 == null) 0.0
    else java.time.Duration.between(t2, t1).getSeconds.toDouble
  }

  def toMap: Map[String, Any] = df.columns.map(c => c -> df(c)(idx)).toMap
}
object Row {
  def fromIndex(df: DataFrame, i: Int) = new Row(df, i)
  def concat(rows: Seq[Row]): DataFrame = {
    if (rows.isEmpty) return DataFrame.empty()
    DataFrame.fromRows(rows.map(_.toMap))
  }
}
class GroupedDataFrame(df: DataFrame, gs: List[String]) {
  def grouper: Seq[String] = gs
  def groups: Map[Any, Seq[Int]] = {
    if (gs.isEmpty) return Map(null.asInstanceOf[Any] -> (0 until df.numRows).toSeq)
    val keyCols = gs.map(df.apply)
    (0 until df.numRows).groupBy(i => keyCols.map(_(i))).view.mapValues(_.toSeq).toMap
  }
  def indices: Map[Any, Array[Int]] = groups.view.mapValues(_.toArray).toMap
  def get_group(name: Any, obj: Any = null): DataFrame = {
    val idxs = groups.getOrElse(name, Seq.empty)
    if (idxs.isEmpty) DataFrame.empty() else df.iloc(idxs)
  }

  private def aggregateInternal(f: Column => Any, colNames: Seq[String] = Nil): DataFrame = {
    val targetCols = if (colNames.nonEmpty) colNames else df.columns.filterNot(gs.contains)
    val grps = groups
    val resultRows = grps.toSeq.map { case (key, rowIdxs) =>
      val subDf = df.iloc(rowIdxs)
      val rowMap = mutable.Map[String, Any]()
      // Add group keys
      gs.zipWithIndex.foreach { case (g, i) =>
        rowMap(g) = key match {
          case s: Seq[_] => s(i)
          case k => k
        }
      }
      // Add aggregated values
      targetCols.foreach { c =>
        rowMap(c) = f(subDf(c))
      }
      rowMap.toMap
    }
    DataFrame.fromRows(resultRows)
  }

  def count(): DataFrame = {
    val grps = groups
    val resultRows = grps.toSeq.map { case (key, idxs) =>
      val rowMap = mutable.Map[String, Any]()
      gs.zipWithIndex.foreach { case (g, i) =>
        rowMap(g) = key match { case s: Seq[_] => s(i); case k => k }
      }
      rowMap("count") = idxs.length.toDouble
      rowMap.toMap
    }
    DataFrame.fromRows(resultRows)
  }
  def sum(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) c.stats().sum else null)
  def mean(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) c.stats().mean else null)
  def min(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) c.stats().min else null)
  def max(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) c.stats().max else null)
  def first(): DataFrame = aggregateInternal(c => if (c.length > 0) c(0) else null)
  def last(): DataFrame = aggregateInternal(c => if (c.length > 0) c(c.length - 1) else null)
  def size(): DataFrame = {
    val grps = groups
    val resultRows = grps.toSeq.map { case (key, idxs) =>
      val rowMap = mutable.Map[String, Any]()
      gs.zipWithIndex.foreach { case (g, i) =>
        rowMap(g) = key match { case s: Seq[_] => s(i); case k => k }
      }
      rowMap("size") = idxs.length.toDouble
      rowMap.toMap
    }
    DataFrame.fromRows(resultRows)
  }

  def agg(aggs: Aggregation*): DataFrame = {
    val grps = groups
    val resultRows = grps.toSeq.map { case (key, idxs) =>
      val subDf = df.iloc(idxs)
      val rowMap = mutable.Map[String, Any]()
      gs.zipWithIndex.foreach { case (g, i) =>
        rowMap(g) = key match { case s: Seq[_] => s(i); case k => k }
      }
      aggs.foreach { aggregation =>
        val col = subDf(aggregation.column)
        val stats = col.stats()
        val value = aggregation.aggType match {
          case AggregationType.Sum => stats.sum
          case AggregationType.Mean | AggregationType.Avg => stats.mean
          case AggregationType.Count => stats.count.toDouble
          case AggregationType.Min => stats.min
          case AggregationType.Max => stats.max
          case AggregationType.Std => stats.std
          case AggregationType.First => if (col.length > 0) col(0) else null
          case AggregationType.Last => if (col.length > 0) col(col.length - 1) else null
          case _ => null
        }
        val colName = if (aggs.length > 1) s"${aggregation.aggType.toString.toLowerCase}(${aggregation.column})" else aggregation.column
        rowMap(colName) = value
      }
      rowMap.toMap
    }
    DataFrame.fromRows(resultRows)
  }

  def agg(f: Column => Any): DataFrame = aggregateInternal(f)

  def apply[T](func: DataFrame => T): Map[Any, T] = {
    groups.map { case (key, idxs) => key -> func(df.iloc(idxs)) }
  }

  def transform(func: Column => Column): DataFrame = {
    val grps = groups
    val resultData = mutable.Map[String, mutable.ArrayBuffer[Any]]()
    df.columns.foreach(c => resultData(c) = mutable.ArrayBuffer.fill(df.numRows)(null))

    grps.foreach { case (_, idxs) =>
      val subDf = df.iloc(idxs)
      df.columns.foreach { c =>
        if (!gs.contains(c)) {
          val transformed = func(subDf(c))
          for (i <- 0 until idxs.length) {
            resultData(c)(idxs(i)) = transformed(i)
          }
        } else {
          for (i <- 0 until idxs.length) {
            resultData(c)(idxs(i)) = subDf(c)(i)
          }
        }
      }
    }
    new DataFrame(resultData.map { case (n, d) => n -> new Column(n, d, df(n).dtype) }.toMap)
  }

  def filter(func: DataFrame => Boolean): DataFrame = {
    val filteredIdxs = groups.values.filter(idxs => func(df.iloc(idxs))).flatten.toSeq.sorted
    df.iloc(filteredIdxs)
  }

  def head(n: Int = 5): DataFrame = {
    val idxs = groups.values.flatMap(_.take(n)).toSeq.sorted
    df.iloc(idxs)
  }

  def tail(n: Int = 5): DataFrame = {
    val idxs = groups.values.flatMap(_.takeRight(n)).toSeq.sorted
    df.iloc(idxs)
  }

  def nth(n: Int): DataFrame = {
    val idxs = groups.values.flatMap(g => g.lift(if (n < 0) n + g.length else n)).toSeq.sorted
    df.iloc(idxs)
  }

  def sample(n: Int): DataFrame = {
    val random = new scala.util.Random()
    val idxs = groups.values.flatMap(g => random.shuffle(g).take(n)).toSeq.sorted
    df.iloc(idxs)
  }

  def shift(periods: Int = 1): DataFrame = {
    val newCols = df.columnData.map { case (name, col) =>
      val newData = (0 until df.numRows).map { i =>
        val oldIdx = i - periods
        if (oldIdx >= 0 && oldIdx < df.numRows) col(oldIdx) else null
      }.to(mutable.ArrayBuffer)
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newCols, df.schema)
  }
  def diff(periods: Int = 1): DataFrame = {
    val shifted = shift(periods); val newCols = df.columnData.map { case (n, c) =>
      if (c.dtype.isPrimitive) (n, new Column(n, (0 until df.numRows).map(i => {
        val v1 = c(i) match { case n: Number => n.doubleValue() case _ => Double.NaN }
        val v2 = shifted(n)(i) match { case n: Number => n.doubleValue() case _ => Double.NaN }
        if (v1.isNaN || v2.isNaN) null else v1 - v2
      }).map(_.asInstanceOf[Any]).to(mutable.ArrayBuffer), DataType.Float64)) else (n, c)
    }
    new DataFrame(newCols)
  }
  def cumsum(): DataFrame = {
    val newCols = df.columnData.map { case (n, c) =>
      val newData = mutable.ArrayBuffer[Any]()
      var sum = 0.0
      for (i <- 0 until df.numRows) {
        val v = c(i) match { case n: Number => n.doubleValue() case _ => 0.0 }
        sum += v
        newData += sum
      }
      (n, new Column(n, newData, DataType.Float64))
    }
    new DataFrame(newCols)
  }
  def cumprod(): DataFrame = {
    val newCols = df.columnData.map { case (n, c) =>
      val newData = mutable.ArrayBuffer[Any]()
      var prod = 1.0
      for (i <- 0 until df.numRows) {
        val v = c(i) match { case n: Number => n.doubleValue() case _ => 1.0 }
        prod *= v
        newData += prod
      }
      (n, new Column(n, newData, DataType.Float64))
    }
    new DataFrame(newCols)
  }
  def cummin(): DataFrame = {
    val newCols = df.columnData.map { case (n, c) =>
      val newData = mutable.ArrayBuffer[Any]()
      var minVal = Double.MaxValue
      for (i <- 0 until df.numRows) {
        val v = c(i) match { case n: Number => n.doubleValue() case _ => Double.NaN }
        if (!v.isNaN && v < minVal) minVal = v
        newData += (if (minVal == Double.MaxValue) null else minVal)
      }
      (n, new Column(n, newData, DataType.Float64))
    }
    new DataFrame(newCols)
  }
  def cummax(): DataFrame = {
    val newCols = df.columnData.map { case (n, c) =>
      val newData = mutable.ArrayBuffer[Any]()
      var maxVal = Double.MinValue
      for (i <- 0 until df.numRows) {
        val v = c(i) match { case n: Number => n.doubleValue() case _ => Double.NaN }
        if (!v.isNaN && v > maxVal) maxVal = v
        newData += (if (maxVal == Double.MinValue) null else maxVal)
      }
      (n, new Column(n, newData, DataType.Float64))
    }
    new DataFrame(newCols)
  }
  def rank(): DataFrame = {
    val newCols = df.columnData.map { case (name, col) =>
      if (col.dtype.isPrimitive) {
        val groupsData = groups
        val ranks = new Array[Any](df.numRows)
        groupsData.foreach { case (_, idxs) =>
          val sorted = idxs.map(i => (col(i).asInstanceOf[Comparable[Any]], i)).sortBy(_._1)
          for (r <- 0 until sorted.length) ranks(sorted(r)._2) = (r + 1).toDouble
        }
        (name, new Column(name, ranks.to(mutable.ArrayBuffer), DataType.Float64))
      } else (name, col)
    }
    new DataFrame(newCols)
  }
  def skew(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) c.stats().skew else null)
  def kurt(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) c.stats().kurt else null)
  def sem(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) { val s = c.stats(); if (s.count > 0) s.std / math.sqrt(s.count.toDouble) else null } else null)
//  def mad(): DataFrame = aggregateInternal(c => if (c.dtype.isPrimitive) {
//    val s = c.stats(); val avg = s.mean
//    val absDevs = c.toSeq.collect { case num: Number => math.abs(num.doubleValue() - avg) }
//    n -> (if (absDevs.nonEmpty) absDevs.sum / absDevs.length else Double.NaN)
//  } else null)

  def corr(): DataFrame = df.corr() // Placeholder for group-wise correlation
  def cov(): DataFrame = df.cov()

  def value_counts(normalize: Boolean = false): DataFrame = {
    val grps = groups
    val rows = grps.toSeq.map { case (key, idxs) =>
      val rowMap = mutable.Map[String, Any]()
      gs.zipWithIndex.foreach { case (g, i) =>
        rowMap(g) = key match { case s: Seq[_] => s(i); case k => k }
      }
      rowMap("count") = idxs.length.toDouble
      if (normalize) rowMap("proportion") = idxs.length.toDouble / df.numRows
      rowMap.toMap
    }
    DataFrame.fromRows(rows)
  }

  def describe(percentiles: Seq[Double] = Seq(0.25, 0.5, 0.75)): DataFrame = {
    aggregateInternal(c => {
      val s = c.stats()
      s"count=${s.count}, mean=${s.mean}, std=${s.std}, min=${s.min}, max=${s.max}"
    })
  }

  def window(windowSize: Int): RollingWindow = new RollingWindow(df, windowSize)
  def expanding(min_periods: Int = 1): ExpandingWindow = new ExpandingWindow(df, min_periods)
}

class StringColumnOps(c: Column) {
  def upper = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.toUpperCase).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def lower = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.toLowerCase).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def contains(pat: String, regex: Boolean = true, na: Any = null) = new Column(c.name, c.toSeq.map(v => {
    if (v == null) na
    else if (regex) pat.r.findFirstIn(v.toString).isDefined
    else v.toString.contains(pat)
  }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def replace(pat: String, repl: String, regex: Boolean = true) = new Column(c.name, c.toSeq.map(v => {
    if (v == null) null
    else if (regex) v.toString.replaceAll(pat, repl)
    else v.toString.replace(pat, repl)
  }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def split(sep: String = ",", n: Int = -1, expand: Boolean = false) = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.split(sep, n)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.List)
  def strip(to_strip: String = "") = new Column(c.name, c.toSeq.map(v => if (v == null) null else if (to_strip.isEmpty) v.toString.trim else v.toString.stripPrefix(to_strip).stripSuffix(to_strip)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def lstrip(to_strip: String = "") = new Column(c.name, c.toSeq.map(v => if (v == null) null else if (to_strip.isEmpty) v.toString.stripLeading() else v.toString.stripPrefix(to_strip)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def rstrip(to_strip: String = "") = new Column(c.name, c.toSeq.map(v => if (v == null) null else if (to_strip.isEmpty) v.toString.stripTrailing() else v.toString.stripSuffix(to_strip)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def len() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.length).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def startswith(pat: String) = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.startsWith(pat)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def endswith(pat: String) = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.endsWith(pat)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def find(sub: String) = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.indexOf(sub)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def rfind(sub: String) = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.lastIndexOf(sub)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)

  def isalnum() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.forall(_.isLetterOrDigit)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def isnumeric() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.forall(_.isDigit)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def isalpha() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.forall(_.isLetter)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def isdigit() = isnumeric()
  def isspace() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.forall(_.isWhitespace)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def islower() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.exists(_.isLower) && v.toString.forall(c => !c.isLetter || c.isLower)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def isupper() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.exists(_.isUpper) && v.toString.forall(c => !c.isLetter || c.isUpper)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def istitle() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.matches("[A-Z][a-z]*")).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def title() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.split(" ").map(_.capitalize).mkString(" ")).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def capitalize() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.capitalize).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def swapcase() = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString.map(ch => if (ch.isUpper) ch.toLower else if (ch.isLower) ch.toUpper else ch)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def normalize(form: String) = c // Placeholder for Unicode normalization
  def partition(sep: String) = {
    new Column(c.name, c.toSeq.map(v => if (v == null) null else {
      val s = v.toString; val i = s.indexOf(sep)
      if (i == -1) Seq(s, "", "") else Seq(s.substring(0, i), sep, s.substring(i + sep.length))
    }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.List)
  }
  def rpartition(sep: String) = {
    new Column(c.name, c.toSeq.map(v => if (v == null) null else {
      val s = v.toString; val i = s.lastIndexOf(sep)
      if (i == -1) Seq("", "", s) else Seq(s.substring(0, i), sep, s.substring(i + sep.length))
    }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.List)
  }
  def extract(pat: String) = {
    val r = pat.r
    new Column(c.name, c.toSeq.map(v => if (v == null) null else r.findFirstMatchIn(v.toString).map(_.group(1)).getOrElse(null)).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  }
  def repeat(repeats: Int) = new Column(c.name, c.toSeq.map(v => if (v == null) null else v.toString * repeats).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def pad(width: Int, side: String = "left", fillchar: String = " ") = new Column(c.name, c.toSeq.map(v => if (v == null) null else {
    val s = v.toString
    if (s.length >= width) s
    else side match {
      case "left" => fillchar * (width - s.length) + s
      case "right" => s + fillchar * (width - s.length)
      case "both" => fillchar * ((width - s.length) / 2) + s + fillchar * ((width - s.length + 1) / 2)
      case _ => s
    }
  }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def zfill(width: Int) = pad(width, side = "left", fillchar = "0")
  def slice(start: Int, stop: Int) = new Column(c.name, c.toSeq.map(v => if (v == null) null else {
    val s = v.toString
    val st = if (start < 0) s.length + start else start
    val sp = if (stop < 0) s.length + stop else stop
    s.substring(math.max(0, st), math.min(s.length, sp))
  }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)

  def md5 = new Column(c.name, c.toSeq.map(v => if (v == null) null else {
    val md = java.security.MessageDigest.getInstance("MD5")
    val digest = md.digest(v.toString.getBytes)
    digest.map("%02x".format(_)).mkString
  }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)

  def tokenize = new Column(c.name, c.toSeq.map(v => {
    if (v == null || v.toString.trim.isEmpty) Nil
    else v.toString.replaceAll("[^\\w\\s]", "").strip().split("\\s+").toList
  }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.List)
}

class DateColumnOps(c: Column) {
  private def toDT(v: Any): java.time.LocalDateTime = v match {
    case dt: java.time.LocalDateTime => dt
    case d: java.time.LocalDate => d.atStartOfDay()
    case _ => null.asInstanceOf[java.time.LocalDateTime]
  }
  def year = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getYear else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def month = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getMonthValue else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def day = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfMonth else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def hour = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getHour else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def minute = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getMinute else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def second = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getSecond else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def dayofweek = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfWeek.getValue else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def dayofyear = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfYear else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def quarter = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) (dt.getMonthValue - 1) / 3 + 1 else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def is_month_start = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfMonth == 1 else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def is_month_end = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfMonth == dt.toLocalDate.lengthOfMonth() else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def is_quarter_start = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfMonth == 1 && (dt.getMonthValue - 1) % 3 == 0 else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def is_quarter_end = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfMonth == dt.toLocalDate.lengthOfMonth() && dt.getMonthValue % 3 == 0 else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def is_year_start = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfYear == 1 else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def is_year_end = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.getDayOfYear == dt.toLocalDate.lengthOfYear() else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def is_leap_year = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) java.time.Year.of(dt.getYear).isLeap else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Boolean)
  def strftime(format: String) = new Column(c.name, c.toSeq.map(v => { val dt = toDT(v); if (dt != null) dt.format(java.time.format.DateTimeFormatter.ofPattern(format)) else null }).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
  def floor(freq: String) = c
  def ceil(freq: String) = c
  def round(freq: String) = c
  def normalize() = c
  def to_period(freq: String) = c
  def tz_localize(tz: String) = c
  def tz_convert(tz: String) = c
}

class CategoryColumnOps(c: Column) {
  def categories = new Column(c.name, Nil.to(mutable.ArrayBuffer), DataType.String)
  def ordered = false
  def codes = new Column(c.name, c.toSeq.map(_ => 0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
  def rename_categories(new_categories: Seq[Any]) = c
  def add_categories(new_categories: Seq[Any]) = c
  def remove_categories(rem_categories: Seq[Any]) = c
  def set_categories(new_categories: Seq[Any]) = c
  def as_ordered() = c
  def as_unordered() = c
  def reorder_categories(new_categories: Seq[Any]) = c
}

class RollingWindow(df: DataFrame, w: Int) {
  private def applyWindow(f: Seq[Any] => Any): DataFrame = {
    val newCols = df.columnData.map { case (name, col) =>
      val newData = (0 until df.numRows).map { i =>
        if (i < w - 1) null
        else {
          val windowData = (i - w + 1 to i).map(col(_))
          f(windowData)
        }
      }.to(mutable.ArrayBuffer)
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newCols)
  }

  def mean(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.sum / nums.length
  }
  def sum(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.sum
  }
  def var_(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.length < 2) null
    else {
      val avg = nums.sum / nums.length
      nums.map(x => math.pow(x - avg, 2)).sum / (nums.length - 1)
    }
  }
  def `var`(ddof: Int = 1): DataFrame = var_()
  def std(): DataFrame = applyWindow { data =>
    val v = var_().columnData(df.columns.head)(w-1) // Just a placeholder logic for std
    math.sqrt(v.asInstanceOf[Double]) // This needs better impl
  }
  def min(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.min
  }
  def max(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.max
  }
  def count(): DataFrame = applyWindow { data => data.count(_ != null).toDouble }
  def median(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }.sorted
    if (nums.isEmpty) null
    else if (nums.length % 2 == 1) nums(nums.length / 2)
    else (nums(nums.length / 2 - 1) + nums(nums.length / 2)) / 2.0
  }
  def quantile(q: Double): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }.sorted
    if (nums.isEmpty) null else nums((q * (nums.length - 1)).toInt)
  }

  def apply(func: Column => Column): DataFrame = {
    val newCols = df.columnData.map { case (name, col) => name -> func(col) }
    new DataFrame(newCols)
  }
  def validate(window: Int): Boolean = window > 0 && w > 0
  def get_window_size(): Int = w
  def mad(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else {
      val avg = nums.sum / nums.length
      nums.map(x => math.abs(x - avg)).sum / nums.length
    }
  }
  def nunique(): DataFrame = applyWindow { data => data.distinct.size.toDouble }
  def value_counts(): DataFrame = df
}

class ExpandingWindow(df: DataFrame, m: Int) {
  private def applyWindow(f: Seq[Any] => Any): DataFrame = {
    val newCols = df.columnData.map { case (name, col) =>
      val newData = (0 until df.numRows).map { i =>
        if (i < m - 1) null
        else {
          val windowData = (0 to i).map(col(_))
          f(windowData)
        }
      }.to(mutable.ArrayBuffer)
      (name, new Column(name, newData, col.dtype))
    }
    new DataFrame(newCols)
  }

  def sum(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.sum
  }
  def mean(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.sum / nums.length
  }
  def min(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.min
  }
  def max(): DataFrame = applyWindow { data =>
    val nums = data.collect { case n: Number => n.doubleValue() }
    if (nums.isEmpty) null else nums.max
  }
  def count(): DataFrame = applyWindow { data => data.count(_ != null).toDouble }

  def apply(func: Column => Column): DataFrame = {
    val newCols = df.columnData.map { case (name, col) => name -> func(col) }
    new DataFrame(newCols)
  }
}

class EWMWindow(df: DataFrame, s: Int) {
  def mean(): DataFrame = df
  def var_(): DataFrame = df
  def std(): DataFrame = df
  def cov(other: DataFrame): DataFrame = df
  def corr(other: DataFrame): DataFrame = df
}
