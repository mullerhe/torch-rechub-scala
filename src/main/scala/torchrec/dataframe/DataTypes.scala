package torchrec.dataframe

/** Data types supported by the DataFrame */
enum DataType:
  case Int8, Int16, Int32, Int64
  case UInt8, UInt16, UInt32, UInt64
  case Float32, Float64
  case Boolean
  case String
  case Binary
  case Date
  case Timestamp
  case Timedelta
  case List
  case Map
  case Struct

  /** Element size in bytes */
  def elementSize: Int = this match
    case Int8 | UInt8 => 1
    case Int16 | UInt16 => 2
    case Int32 | UInt32 | Float32 => 4
    case Int64 | UInt64 | Float64 => 8
    case Boolean => 1
    case String | Binary | Date | Timestamp | Timedelta | List | Map | Struct => 0 // variable size

  /** Whether this is a primitive type with fixed size */
  def isPrimitive: Boolean = this match
    case Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 | Float32 | Float64 | Boolean => true
    case _ => false

  /** Convert to pandas/cuDF dtype string */
  def toPandas: String = this match
    case Int8 => "int8"
    case Int16 => "int16"
    case Int32 => "int32"
    case Int64 => "int64"
    case UInt8 => "uint8"
    case UInt16 => "uint16"
    case UInt32 => "uint32"
    case UInt64 => "uint64"
    case Float32 => "float32"
    case Float64 => "float64"
    case Boolean => "bool"
    case String => "object"
    case Binary => "bytes"
    case Date => "datetime64"
    case Timestamp => "datetime64"
    case Timedelta => "timedelta64"
    case List => "object"
    case Map => "object"
    case Struct => "object"

object DataType:
  /** Infer DataType from Scala class */
  def fromClass(cls: Class[_]): DataType = cls match
    case _ if cls == classOf[Int] || cls == classOf[Integer] => Int32
    case _ if cls == classOf[Long] => Int64
    case _ if cls == classOf[Float] => Float32
    case _ if cls == classOf[Double] => Float64
    case _ if cls == classOf[Boolean] => Boolean
    case _ if cls == classOf[String] => String
    case _ if cls == classOf[Array[Byte]] => Binary
    case _ => String

  /** Default value for this type */
  def defaultValue: Any = this match
    case Int8 | Int16 | Int32 | Int64 => 0
    case UInt8 | UInt16 | UInt32 | UInt64 => 0
    case Float32 | Float64 => 0.0
    case Boolean => false
    case String => ""
    case _ => null