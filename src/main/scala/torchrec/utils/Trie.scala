package torchrec.utils

import scala.collection.mutable

/**
 * Trie (prefix tree) for efficient prefix-based token generation.
 * Used for constrained decoding in generative models.
 */
class Trie(
  sequences: List[List[Int]] = List.empty
) {
  private val trieDict = mutable.Map[Int, mutable.Map[Int, Any]]()

  // For append mode
  private var appendTrie: Option[Trie] = None
  private var bosTokenId: Option[Int] = None

  // Initialize
  if (sequences.nonEmpty) {
    sequences.foreach(add)
    _len = sequences.length
  }
  private var _len: Int = 0

  def length: Int = _len

  /**
   * Add a sequence to the trie.
   */
  def add(sequence: List[Int]): Unit = {
    addToTrie(sequence, trieDict)
    _len += 1
  }

  /**
   * Add a sequence to a specific trie dict (recursive).
   */
  private def addToTrie(sequence: List[Int], dict: mutable.Map[Int, mutable.Map[Int, Any]]): Unit = {
    if (sequence.isEmpty) return
    val first = sequence.head
    if (!dict.contains(first)) {
      dict(first) = mutable.Map[Int, Any]()
    }
    addToTrie(sequence.tail, dict(first).asInstanceOf[mutable.Map[Int, mutable.Map[Int, Any]]])
  }

  /**
   * Get all tokens that can follow a given prefix sequence.
   */
  def get(prefixSequence: List[Int]): List[Int] = {
    getFromTrie(prefixSequence, trieDict, appendTrie, bosTokenId)
  }

  /**
   * Recursive get implementation.
   */
  private def getFromTrie(
    prefix: List[Int],
    dict: mutable.Map[Int, mutable.Map[Int, Any]],
    appendT: Option[Trie],
    bosId: Option[Int]
  ): List[Int] = {
    if (prefix.isEmpty) {
      val keys = dict.keys.toList
      val filtered = bosId match {
        case Some(bos) if keys.contains(bos) =>
          val withoutBos = keys.filter(_ != bos)
          val appended = appendT.map(_.trieDict.keys.toList).getOrElse(List.empty)
          withoutBos ++ appended
        case _ =>
          val appended = appendT.map(_.trieDict.keys.toList).getOrElse(List.empty)
          keys ++ appended
      }
      filtered
    } else if (dict.contains(prefix.head)) {
      getFromTrie(
        prefix.tail,
        dict(prefix.head).asInstanceOf[mutable.Map[Int, mutable.Map[Int, Any]]],
        appendT,
        bosId
      )
    } else {
      appendT match {
        case Some(at) => at.get(prefix)
        case None => List.empty
      }
    }
  }

  /**
   * Enable append mode: after reaching end of trie, switch to another trie.
   */
  def append(trie: Trie, bosTokenId: Int): Unit = {
    appendTrie = Some(trie)
    this.bosTokenId = Some(bosTokenId)
  }

  /**
   * Load trie from a dictionary representation.
   */
  def loadFromDict(trieDictMap: Map[Int, Map[Int, Any]]): Unit = {
    def build(trieDict: mutable.Map[Int, mutable.Map[Int, Any]], source: Map[Int, Map[Int, Any]]): Unit = {
      source.foreach { case (k, v) =>
        val inner = mutable.Map[Int, mutable.Map[Int, Any]]()
        trieDict(k) = inner.asInstanceOf[mutable.Map[Int, Any]]
        build(inner, v.asInstanceOf[Map[Int, Map[Int, Any]]])
      }
    }
    build(trieDict, trieDictMap)
    _len = trieDictMap.size
  }

  /**
   * Iterator over all sequences in the trie.
   */
  def iterator: Iterator[List[Int]] = {
    traverse(List.empty, trieDict)
  }

  private def traverse(
    prefix: List[Int],
    dict: mutable.Map[Int, mutable.Map[Int, Any]]
  ): Iterator[List[Int]] = {
    if (dict.isEmpty) {
      Iterator(prefix)
    } else {
      dict.keys.flatMap { key =>
        traverse(prefix :+ key, dict(key).asInstanceOf[mutable.Map[Int, mutable.Map[Int, Any]]])
      }.iterator
    }
  }

  /**
   * Check if trie is empty.
   */
  def isEmpty: Boolean = trieDict.isEmpty
}

/**
 * Factory and utility methods for Trie.
 */
object Trie {
  def empty: Trie = new Trie()

  def apply(sequences: List[List[Int]]): Trie = new Trie(sequences)

  /**
   * Load from a nested map representation.
   */
  def loadFromDict(dict: Map[Int, Any]): Trie = {
    val trie = new Trie()
    trie.loadFromDict(dict.asInstanceOf[Map[Int, Map[Int, Any]]])
    trie
  }
}
