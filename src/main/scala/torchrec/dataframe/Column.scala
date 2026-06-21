package torchrec.dataframe

import scala.collection.mutable

/** A single column in a DataFrame.
 *  Stores data in a mutable array with associated type information.
 */
class Column(
  val name: String,
  private val data: mutable.ArrayBuffer[Any],
  val dtype: DataType
) {
  require(name != null && name.nonEmpty, "Column name cannot be empty")
  require(data != null, "Column data cannot be null")

  /** Number of elements in this column */
  def length: Int = data.length

  /** Check if column is empty */
  def isEmpty: Boolean = length == 0

  /** Get element at index */
  def apply(i: Int): Any = {
    require(i >= 0 && i < length, s"Index $i out of bounds [0, $length)")
    data(i)
  }

  /** Update element at index */
  def update(i: Int, value: Any): Unit = {
    require(i >= 0 && i < length, s"Index $i out of bounds [0, $length)")
    data(i) = value
  }

  /** Get all values as Scala sequence */
  def toSeq: Seq[Any] = data.toSeq

  /** Check if element at index is null */
  def isNullAt(i: Int): Boolean = data(i) == null

  /** Create a null-mask column */
  def isNull: Column = {
    val nulls = mutable.ArrayBuffer[Boolean]()
    for (i <- 0 until length) {
      nulls += isNullAt(i)
    }
    new Column(name + "_is_null", nulls.map(_.asInstanceOf[Any]), DataType.Boolean)
  }

  /** Fill null values with specified value */
  def fillNull(value: Any): Column = {
    val filled = mutable.ArrayBuffer[Any]()
    for (i <- 0 until length) {
      if (isNullAt(i)) filled += value else filled += data(i)
    }
    new Column(name, filled, dtype)
  }

  /** Drop null values */
  def dropNull: (Column, Column) = {
    val notNull = mutable.ArrayBuffer[Boolean]()
    val notNullData = mutable.ArrayBuffer[Any]()
    val nullMask = mutable.ArrayBuffer[Boolean]()

    for (i <- 0 until length) {
      val notNullVal = !isNullAt(i)
      notNull += notNullVal
      if (notNullVal) notNullData += data(i)
      nullMask += !notNullVal
    }

    val resultCol = new Column(name, notNullData, dtype)
    val nullMaskCol = new Column(name + "_is_null", nullMask.map(_.asInstanceOf[Any]), DataType.Boolean)
    (resultCol, nullMaskCol)
  }

  /** Cast to different type */
  def cast(newType: DataType): Column = {
    if (dtype == newType) return this

    val newData = mutable.ArrayBuffer[Any]()

    (dtype, newType) match
      case (DataType.Int32, DataType.Float32) =>
        for (i <- 0 until length) {
          data(i) match
            case v: Int => newData += v.toFloat
            case v: Float => newData += v
            case _ => newData += 0.0f
        }
        new Column(name, newData, newType)

      case (DataType.Float32, DataType.Int32) =>
        for (i <- 0 until length) {
          data(i) match
            case v: Float => newData += v.toInt
            case v: Int => newData += v
            case _ => newData += 0
        }
        new Column(name, newData, newType)

      case (DataType.String, DataType.Int32) =>
        for (i <- 0 until length) {
          data(i) match
            case s: String => newData += s.toIntOption.getOrElse(0)
            case v: Int => newData += v
            case _ => newData += 0
        }
        new Column(name, newData, newType)

      case _ =>
        throw new UnsupportedOperationException(s"Cast from $dtype to $newType not supported")
  }

  /** Get basic statistics */
  def stats(): ColumnStats = {
    dtype match
      case DataType.Float32 | DataType.Int32 | DataType.Int64 | DataType.Float64 =>
        var minVal = Double.MaxValue
        var maxVal = Double.MinValue
        var sum = 0.0
        var count = 0L

        for (i <- 0 until length) {
          if (!isNullAt(i)) {
            val v = data(i) match
              case f: Float => f.toDouble
              case d: Double => d
              case l: Long => l.toDouble
              case i: Int => i.toDouble
              case _ => Double.NaN
            if (!v.isNaN) {
              if (v < minVal) minVal = v
              if (v > maxVal) maxVal = v
              sum += v
              count += 1
            }
          }
        }

        val mean = if (count > 0) sum / count else Double.NaN
        var variance = 0.0
        var skew = 0.0
        var kurt = 0.0

        if (count > 1) {
          var sqDiff = 0.0
          var cubeDiff = 0.0
          var fourthDiff = 0.0
          for (i <- 0 until length) {
            if (!isNullAt(i)) {
              val v = data(i) match {
                case f: Float => f.toDouble
                case d: Double => d
                case l: Long => l.toDouble
                case i: Int => i.toDouble
                case _ => Double.NaN
              }
              if (!v.isNaN) {
                val diff = v - mean
                val d2 = diff * diff
                sqDiff += d2
                cubeDiff += d2 * diff
                fourthDiff += d2 * d2
              }
            }
          }
          variance = sqDiff / count
          val std = math.sqrt(variance)
          if (std > 0) {
            val m3 = cubeDiff / count
            val m4 = fourthDiff / count
            skew = m3 / math.pow(std, 3)
            kurt = m4 / math.pow(std, 4) - 3.0 // Excess kurtosis
          }
        }

        ColumnStats(
          count = count,
          nullCount = length - count,
          min = if (minVal == Double.MaxValue) Double.NaN else minVal,
          max = if (maxVal == Double.MinValue) Double.NaN else maxVal,
          mean = mean,
          std = math.sqrt(variance),
          sum = sum,
          skew = skew,
          kurt = kurt
        )

      case _ =>
        ColumnStats(length, 0, Double.NaN, Double.NaN, Double.NaN, 0.0)
  }

  override def toString: String = s"Column($name, $dtype, $length elements)"

  /** Comparison operators returning a Boolean Column */
  def eq(value: Any): Column = {
    val res = data.map(v => (v == value).asInstanceOf[Any])
    new Column(s"$name == $value", res, DataType.Boolean)
  }

  def ne(value: Any): Column = {
    val res = data.map(v => (v != value).asInstanceOf[Any])
    new Column(s"$name != $value", res, DataType.Boolean)
  }

  def gt(value: Double): Column = {
    val res = data.map(v => (v.toString.toDouble > value).asInstanceOf[Any])
    new Column(s"$name > $value", res, DataType.Boolean)
  }

  def lt(value: Double): Column = {
    val res = data.map(v => (v.toString.toDouble < value).asInstanceOf[Any])
    new Column(s"$name < $value", res, DataType.Boolean)
  }

  def ge(value: Double): Column = {
    val res = data.map(v => (v.toString.toDouble >= value).asInstanceOf[Any])
    new Column(s"$name >= $value", res, DataType.Boolean)
  }

  def le(value: Double): Column = {
    val res = data.map(v => (v.toString.toDouble <= value).asInstanceOf[Any])
    new Column(s"$name <= $value", res, DataType.Boolean)
  }

  /** Map over elements with a function */
  def map(f: Any => Any): Column = {
    val newData = data.map(f)
    new Column(name, newData, DataType.fromClass(classOf[Any]))
  }

  /** Specialization for Float anonymous functions */
  def mapFloat(f: Float => Float): Column = {
    val newData = data.map {
      case n: Number => f(n.floatValue())
      case _ => Float.NaN
    }.map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Float32)
  }

  /** Specialization for Double anonymous functions */
  def mapDouble(f: Double => Double): Column = {
    val newData = data.map {
      case n: Number => f(n.doubleValue())
      case _ => Double.NaN
    }.map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Float64)
  }

  // --- Numeric Operations ---
  def +(other: Column): Column = {
    val newData = (0 until length).map(i => this (i).asInstanceOf[Number].doubleValue() + other(i).asInstanceOf[Number].doubleValue()).to(mutable.ArrayBuffer)
    new Column(s"$name+${other.name}", newData.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
  }
  def +(other: Number): Column = mapDouble(_ + other.doubleValue())
  def -(other: Column): Column = {
    val newData = (0 until length).map(i => this (i).asInstanceOf[Number].doubleValue() - other(i).asInstanceOf[Number].doubleValue()).to(mutable.ArrayBuffer)
    new Column(s"$name-${other.name}", newData.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
  }
  def -(other: Number): Column = mapDouble(_ - other.doubleValue())
  def *(other: Column): Column = {
    val newData = (0 until length).map(i => this (i).asInstanceOf[Number].doubleValue() * other(i).asInstanceOf[Number].doubleValue()).to(mutable.ArrayBuffer)
    new Column(s"$name*${other.name}", newData.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
  }
  def *(other: Number): Column = mapDouble(_ * other.doubleValue())
  def /(other: Column): Column = {
    val newData = (0 until length).map(i => this (i).asInstanceOf[Number].doubleValue() / other(i).asInstanceOf[Number].doubleValue()).to(mutable.ArrayBuffer)
    new Column(s"$name/${other.name}", newData.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
  }
  def /(other: Number): Column = mapDouble(_ / other.doubleValue())
  def floorDiv(other: Number): Column = mapDouble(v => math.floor(v / other.doubleValue()))
  def %(other: Number): Column = mapDouble(_ % other.doubleValue())

  def pow(p: Double): Column = mapDouble(math.pow(_, p))
  def sqrt: Column = mapDouble(math.sqrt)
  def abs: Column = mapDouble(math.abs)
  def log: Column = mapDouble(math.log)
  def exp: Column = mapDouble(math.exp)
  def sin: Column = mapDouble(math.sin)
  def cos: Column = mapDouble(math.cos)
  def tan: Column = mapDouble(math.tan)

  def gt(other: Number): Column = {
    val newData = data.map { case n: Number => n.doubleValue() > other.doubleValue(); case _ => false }.map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Boolean)
  }
  def lt(other: Number): Column = {
    val newData = data.map { case n: Number => n.doubleValue() < other.doubleValue(); case _ => false }.map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Boolean)
  }
  def ge(other: Number): Column = {
    val newData = data.map { case n: Number => n.doubleValue() >= other.doubleValue(); case _ => false }.map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Boolean)
  }
  def le(other: Number): Column = {
    val newData = data.map { case n: Number => n.doubleValue() <= other.doubleValue(); case _ => false }.map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Boolean)
  }

  def notna: Column = {
    val newData = data.map(_ != null).map(_.asInstanceOf[Any])
    new Column(name, newData, DataType.Boolean)
  }
  def isna: Column = isNull
}

/** Statistics for a numeric column */
case class ColumnStats(
  count: Long,
  nullCount: Long,
  min: Double,
  max: Double,
  mean: Double,
  std: Double = 0.0,
  sum: Double = 0.0,
  skew: Double = 0.0,
  kurt: Double = 0.0
)

/** Factory methods for creating columns */
object Column {
  /** Create column from Scala sequence */
  def apply(name: String, values: Seq[Any], dtype: DataType): Column = {
    val data = mutable.ArrayBuffer[Any]()
    dtype match
      case DataType.Float32 =>
        for (v <- values) {
          data += (v match
            case f: Float => f
            case i: Int => i.toFloat
            case d: Double => d.toFloat
            case l: Long => l.toFloat
            case null | _ => Float.NaN
          )
        }
      case DataType.Int32 =>
        for (v <- values) {
          data += (v match
            case i: Int => i
            case f: Float => f.toInt
            case d: Double => d.toInt
            case l: Long => l.toInt
            case null | _ => 0
          )
        }
      case DataType.String =>
        for (v <- values) {
          data += (v match
            case s: String => s
            case null => ""
            case _ => v.toString
          )
        }
      case _ =>
        for (v <- values) {
          data += v
        }

    new Column(name, data, dtype)
  }

  /** Create column from array of floats */
  def apply(name: String, data: Array[Float]): Column = {
    val buffer = mutable.ArrayBuffer[Any]()
    for (f <- data) buffer += f
    new Column(name, buffer, DataType.Float32)
  }

  /** Create column from array of ints */
  def int(name: String, data: Array[Int]): Column = {
    val buffer = mutable.ArrayBuffer[Any]()
    for (i <- data) buffer += i
    new Column(name, buffer, DataType.Int32)
  }

  /** Create column from array of strings */
  def string(name: String, data: Array[String]): Column = {
    val buffer = mutable.ArrayBuffer[Any]()
    for (s <- data) buffer += s
    new Column(name, buffer, DataType.String)
  }
}