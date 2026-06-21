package torchrec.dataframe

import scala.util.matching.Regex
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.collection.mutable

object SpacySim {
  def tokenize(text: Any): List[String] = {
    if (pd.isna(text) || text.toString.trim.isEmpty) Nil
    else text.toString.replaceAll("[^\\w\\s]", "").trim.split("\\s+").toList
  }

  def posTag(tokens: List[String]): List[String] = {
    tokens.map { word =>
      if (word.forall(_.isDigit)) "NUM"
      else if (word.nonEmpty && word.head.isUpper && word.tail.forall(_.isLower)) "PROPN"
      else if (List("is", "are", "was", "be").contains(word.toLowerCase)) "VERB"
      else if (word.length <= 3) "ADP"
      else "NOUN"
    }
  }

  def entityExtract(text: Any): List[(String, String)] = {
    if (pd.isna(text)) return Nil
    val s = text.toString
    val numPat = "\\d{4}".r
    val namePat = "[A-Z][a-z]+".r

    val nums = numPat.findAllIn(s).map(n => (n, "DATE_NUM")).toList
    val names = namePat.findAllIn(s).map(n => (n, "PERSON")).toList
    nums ++ names
  }

  def sentimentScore(text: Any): Double = {
    val posWords = Set("good", "best", "perfect", "excellent", "happy", "awesome")
    val negWords = Set("bad", "worst", "terrible", "sad", "awful", "ugly")
    val tks = tokenize(text)
    if (tks.isEmpty) return 0.0
    val posCnt = tks.count(w => posWords.contains(w.toLowerCase))
    val negCnt = tks.count(w => negWords.contains(w.toLowerCase))
    BigDecimal((posCnt - negCnt).toDouble / tks.size).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble
  }
}

