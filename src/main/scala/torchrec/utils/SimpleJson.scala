package torchrec.utils

import org.bytedeco.pytorch.global.torch
/** Simple JSON parser for HuggingFace config files */
object SimpleJson {
  def parse(content: String): Map[String, Any] = {
    val trimmed = content.trim
    parseValue(trimmed, 0)._1.asInstanceOf[Map[String, Any]]
  }

  private def skipWhitespace(s: String, i: Int): Int = {
    var j = i
    while (j < s.length && (s(j) == ' ' || s(j) == '\n' || s(j) == '\r' || s(j) == '\t')) j += 1
    j
  }

  private def parseValue(s: String, i: Int): (Any, Int) = {
    val j = skipWhitespace(s, i)
    s(j) match {
      case '{' => parseObject(s, j)
      case '[' => parseArray(s, j)
      case '"' => parseString(s, j)
      case 't' | 'f' => parseBool(s, j)
      case 'n' => parseNull(s, j)
      case _ => parseNumber(s, j)
    }
  }

  private def parseObject(s: String, i: Int): (Map[String, Any], Int) = {
    val result = collection.mutable.Map[String, Any]()
    var j = i + 1
    j = skipWhitespace(s, j)
    if (s(j) == '}') return (result.toMap, j + 1)

    while (j < s.length) {
      j = skipWhitespace(s, j)
      if (s(j) != '"') throw new IllegalArgumentException(s"Expected string at $j: ${s.substring(j)}")
      val (key, afterKey) = parseString(s, j)
      j = skipWhitespace(s, afterKey)
      if (s(j) != ':') throw new IllegalArgumentException(s"Expected ':' at $j")
      j += 1
      j = skipWhitespace(s, j)
      val (value, afterValue) = parseValue(s, j)
      result(key.asInstanceOf[String]) = value
      j = skipWhitespace(s, afterValue)
      if (s(j) == '}') return (result.toMap, j + 1)
      if (s(j) != ',') throw new IllegalArgumentException(s"Expected ',' or '}' at $j")
      j += 1
    }
    throw new IllegalArgumentException("Unterminated object")
  }

  private def parseArray(s: String, i: Int): (List[Any], Int) = {
    val result = collection.mutable.ListBuffer[Any]()
    var j = i + 1
    j = skipWhitespace(s, j)
    if (s(j) == ']') return (result.toList, j + 1)

    while (j < s.length) {
      val (value, afterValue) = parseValue(s, j)
      result += value
      j = skipWhitespace(s, afterValue)
      if (s(j) == ']') return (result.toList, j + 1)
      if (s(j) != ',') throw new IllegalArgumentException(s"Expected ',' at $j")
      j += 1
    }
    throw new IllegalArgumentException("Unterminated array")
  }

  private def parseString(s: String, i: Int): (String, Int) = {
    var j = i + 1
    val sb = new StringBuilder
    while (j < s.length && s(j) != '"') {
      if (s(j) == '\\' && j + 1 < s.length) {
        j += 1
        s(j) match {
          case '"' => sb += '"'
          case '\\' => sb += '\\'
          case '/' => sb += '/'
          case 'n' => sb += '\n'
          case 'r' => sb += '\r'
          case 't' => sb += '\t'
          case 'u' if j + 4 < s.length =>
            val hex = s.substring(j + 1, j + 5)
            sb += Integer.parseInt(hex, 16).toChar
            j += 4
          case c => sb += c
        }
      } else sb += s(j)
      j += 1
    }
    (sb.toString, j + 1)
  }

  private def parseNumber(s: String, i: Int): (Number, Int) = {
    var j = i
    if (s(j) == '-') j += 1
    while (j < s.length && s(j).isDigit) j += 1
    val hasDecimal = j < s.length && s(j) == '.'
    if (hasDecimal) {
      j += 1
      while (j < s.length && s(j).isDigit) j += 1
    }
    val isFloat = hasDecimal || (j < s.length && (s(j) == 'e' || s(j) == 'E'))
    if (isFloat) {
      // Advance past exponent sign and digits
      if (j < s.length && (s(j) == 'e' || s(j) == 'E')) {
        j += 1
        if (j < s.length && (s(j) == '+' || s(j) == '-')) j += 1
        while (j < s.length && s(j).isDigit) j += 1
      }
      val num = java.lang.Double.parseDouble(s.substring(i, j))
      (num, j)
    } else {
      val num = java.lang.Long.parseLong(s.substring(i, j))
      (num, j)
    }
  }

  private def parseBool(s: String, i: Int): (Boolean, Int) = {
    if (s.substring(i).startsWith("true")) (true, i + 4) else (false, i + 5)
  }

  private def parseNull(s: String, i: Int): (Null, Int) = {
    if (s.substring(i).startsWith("null")) (null, i + 4) else throw new IllegalArgumentException(s"Invalid null at $i")
  }
}
