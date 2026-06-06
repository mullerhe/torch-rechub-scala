package torchrec.data

import org.bytedeco.pytorch._
import scala.util.Random
import torchrec.Implicits._

/**
 * JavaBackedDataLoader - iterate over a JavaCPP JavaDataset/JavaTensorDataset
 * and return Scala `Batch` objects. This class does not depend on a specific
 * Java DataLoader implementation; instead it queries the provided JavaDataset
 * for individual examples and groups them into Scala Batches. This makes it
 * robust across JavaCPP versions while still using JavaDataset as the source.
 */
class JavaBackedDataLoader(
  javaDs: JavaDataset,
  featureOrder: Seq[String] = Seq.empty,
  batchSize: Int = 256,
  shuffle: Boolean = true,
  dropLast: Boolean = false
) extends Iterable[Batch] {

  private val dsSize: Int = try { javaDs.size().get().toInt } catch { case _: Throwable => 0 }

  private val indices: Array[Int] = if (shuffle) Random.shuffle((0 until dsSize).toList).toArray else (0 until dsSize).toArray

  override def iterator: Iterator[Batch] = new Iterator[Batch] {
    private var pos = 0

    override def hasNext: Boolean = pos < indices.length

    override def next(): Batch = {
      val end = math.min(pos + batchSize, indices.length)
      val slice = indices.slice(pos, end)
      pos = end
      if (slice.isEmpty) throw new NoSuchElementException("No more elements")

      // fetch individual Examples from javaDs and convert
      val samples = slice.map { idx =>
        try {
          val ex = javaDs.get(idx.toLong)
          Batch.fromExample(ex, featureOrder)
        } catch {
          case _: Throwable => Batch(Map.empty)
        }
      }

      stackSamples(samples)
    }
  }

  private def stackSamples(samples: Seq[Batch]): Batch = {
    import scala.collection.mutable
    val sparseBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    val denseBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    val seqBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    var labelsBuf = mutable.ListBuffer[Tensor]()
    var hasLabels = false
    var tokensBuf = mutable.ListBuffer[Tensor]()
    var hasTokens = false
    var positionsBuf = mutable.ListBuffer[Tensor]()
    var hasPositions = false
    var timeDiffsBuf = mutable.ListBuffer[Tensor]()
    var hasTimeDiffs = false
    var targetsBuf = mutable.ListBuffer[Tensor]()
    var hasTargets = false
    val itemBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()

    samples.foreach { b =>
      b.sparseFeatures.foreach { case (k, t) => sparseBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.denseFeatures.foreach { case (k, t) => denseBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.sequenceFeatures.foreach { case (k, t) => seqBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.itemFeatures.foreach { case (k, t) => itemBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.labels.foreach { l => labelsBuf += l; hasLabels = true }
      b.tokens.foreach { t => tokensBuf += t; hasTokens = true }
      b.positions.foreach { p => positionsBuf += p; hasPositions = true }
      b.timeDiffs.foreach { td => timeDiffsBuf += td; hasTimeDiffs = true }
      b.targets.foreach { tgt => targetsBuf += tgt; hasTargets = true }
    }

    def moveStack(lst: mutable.ListBuffer[Tensor]): Tensor = lst.toSeq.stack(0)

    val sparse = sparseBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap
    val dense = denseBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap
    val seqs = seqBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap
    val items = itemBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap

    val labels = if (hasLabels) Some(moveStack(labelsBuf)) else None
    val tokens = if (hasTokens) Some(moveStack(tokensBuf)) else None
    val positions = if (hasPositions) Some(moveStack(positionsBuf)) else None
    val timeDiffs = if (hasTimeDiffs) Some(moveStack(timeDiffsBuf)) else None
    val targets = if (hasTargets) Some(moveStack(targetsBuf)) else None

    Batch(sparse, dense, seqs, labels, tokens, positions, timeDiffs, targets, items)
  }
}

/** Tensor-backed Java dataset loader variant */
class JavaBackedTensorDataLoader(
  javaDs: JavaTensorDataset,
  featureOrder: Seq[String] = Seq.empty,
  batchSize: Int = 256,
  shuffle: Boolean = true,
  dropLast: Boolean = false
) extends Iterable[Batch] {

  private val dsSize: Int = try { javaDs.size().get().toInt } catch { case _: Throwable => 0 }
  private val indices: Array[Int] = if (shuffle) Random.shuffle((0 until dsSize).toList).toArray else (0 until dsSize).toArray

  override def iterator: Iterator[Batch] = new Iterator[Batch] {
    private var pos = 0
    override def hasNext: Boolean = pos < indices.length
    override def next(): Batch = {
      val end = math.min(pos + batchSize, indices.length)
      val slice = indices.slice(pos, end)
      pos = end
      if (slice.isEmpty) throw new NoSuchElementException("No more elements")

      val samples = slice.map { idx =>
        try {
          val te = javaDs.get(idx.toLong)
          Batch.fromTensorExample(te, featureOrder)
        } catch {
          case _: Throwable => Batch(Map.empty)
        }
      }

      stackSamples(samples)
    }
  }

  private def stackSamples(samples: Seq[Batch]): Batch = {
    import scala.collection.mutable
    val sparseBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    val denseBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    val seqBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()
    var labelsBuf = mutable.ListBuffer[Tensor]()
    var hasLabels = false
    var tokensBuf = mutable.ListBuffer[Tensor]()
    var hasTokens = false
    var positionsBuf = mutable.ListBuffer[Tensor]()
    var hasPositions = false
    var timeDiffsBuf = mutable.ListBuffer[Tensor]()
    var hasTimeDiffs = false
    var targetsBuf = mutable.ListBuffer[Tensor]()
    var hasTargets = false
    val itemBuilders = mutable.Map[String, mutable.ListBuffer[Tensor]]()

    samples.foreach { b =>
      b.sparseFeatures.foreach { case (k, t) => sparseBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.denseFeatures.foreach { case (k, t) => denseBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.sequenceFeatures.foreach { case (k, t) => seqBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.itemFeatures.foreach { case (k, t) => itemBuilders.getOrElseUpdate(k, mutable.ListBuffer()) += t }
      b.labels.foreach { l => labelsBuf += l; hasLabels = true }
      b.tokens.foreach { t => tokensBuf += t; hasTokens = true }
      b.positions.foreach { p => positionsBuf += p; hasPositions = true }
      b.timeDiffs.foreach { td => timeDiffsBuf += td; hasTimeDiffs = true }
      b.targets.foreach { tgt => targetsBuf += tgt; hasTargets = true }
    }

    def moveStack(lst: mutable.ListBuffer[Tensor]): Tensor = lst.toSeq.stack(0)

    val sparse = sparseBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap
    val dense = denseBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap
    val seqs = seqBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap
    val items = itemBuilders.map { case (k, buf) => k -> moveStack(buf) }.toMap

    val labels = if (hasLabels) Some(moveStack(labelsBuf)) else None
    val tokens = if (hasTokens) Some(moveStack(tokensBuf)) else None
    val positions = if (hasPositions) Some(moveStack(positionsBuf)) else None
    val timeDiffs = if (hasTimeDiffs) Some(moveStack(timeDiffsBuf)) else None
    val targets = if (hasTargets) Some(moveStack(targetsBuf)) else None

    Batch(sparse, dense, seqs, labels, tokens, positions, timeDiffs, targets, items)
  }
}

