package torchrec.basic.layers

import torchrec.basic.features._
import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import torchrec.utils.DeviceSupport
import scala.jdk.CollectionConverters._
import scala.collection.mutable

/**
 * Embedding layer for sparse and sequence features
 */
class EmbeddingLayer(
  val features: List[Feature],
  val embedDim: Int = 8,
  val device: String = DeviceSupport.backend,
  val paddingIdx: Option[Long] = None,
  val sparse: Boolean = false
) extends Module {

  // Map from feature name to embedding module
  private val embeddingTables = mutable.Map[String, EmbeddingImpl]()
  private val sharedEmbeddingTables = mutable.Map[String, EmbeddingImpl]()

  // Build embedding tables and ensure they're on the target device
  features.foreach {
    case f: SparseFeature =>
      if (!f.sharedWith.exists(sharedEmbeddingTables.contains)) {
        val options = new EmbeddingOptions(f.vocabSize, f.embedDim)
        f.paddingIdx.foreach { idx =>
          options.padding_idx().put(idx)
        }
        var embedding = new EmbeddingImpl(options)
        if (device != "cpu") {
          val dev = new org.bytedeco.pytorch.Device(device)
          embedding = new EmbeddingImpl(options)
          embedding.to(dev, false)
        }
        val registeredName = s"embed_${f.name}"
        register_module(registeredName, embedding)
        embeddingTables(f.name) = embedding
        f.sharedWith.foreach(sn => sharedEmbeddingTables(sn) = embedding)
      } else {
        embeddingTables(f.name) = sharedEmbeddingTables(f.sharedWith.get)
      }

    case f: SequenceFeature =>
      if (!f.sharedWith.exists(sharedEmbeddingTables.contains)) {
        val options = new EmbeddingOptions(f.vocabSize, f.embedDim)
        if (f.paddingIdx != 0) {
          options.padding_idx().put(f.paddingIdx)
        }
        var embedding = new EmbeddingImpl(options)
        if (device != "cpu") {
          val dev = new org.bytedeco.pytorch.Device(device)
          embedding = new EmbeddingImpl(options)
          embedding.to(dev, false)
        }
        val registeredName = s"embed_seq_${f.name}"
        register_module(registeredName, embedding)
        embeddingTables(s"${f.name}_seq") = embedding
        f.sharedWith.foreach(sn => sharedEmbeddingTables(sn) = embedding)
      } else {
        embeddingTables(s"${f.name}_seq") = sharedEmbeddingTables(f.sharedWith.get)
      }

    case _ => // DenseFeature - no embedding table
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    sequenceFeats: Map[String, Tensor] = Map.empty,
    squeeze: Boolean = true
  ): Tensor = {
    val embeddingList = mutable.ListBuffer[Tensor]()

    // Process sparse features
    sparseFeats.foreach { case (name, indices) =>
      // First try exact match, then try with _seq suffix (for sequence features passed as sparse)
      val embed = embeddingTables.get(name).orElse(embeddingTables.get(s"${name}_seq"))
      embed.foreach { e =>
        val embedDev = e.weight().device()
        // Ensure indices is 1D for embedding lookup (squeeze if 2D with last dim 1)
        val idx1d = if (indices.dim() == 2L && indices.size(1) == 1L) {
          indices.squeeze(1)
        } else {
          indices
        }
        val idxOnDev = if (idx1d.device().equals(embedDev)) {
          idx1d.toType(ScalarType.Long)
        } else {
          idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        val emb = e.forward(idxOnDev)
        // Add dimension if squeezed (batch,) -> (batch, 1, embedDim)
        val emb3d = if (emb.dim() == 2L) emb.unsqueeze(1) else emb
        println(s"[DEBUG sparse] $name: idx1d shape=${idx1d.size(0)}, emb shape=${emb.size(0)}x${emb.size(1)}, emb3d=${emb3d.size(0)}x${emb3d.size(1)}x${emb3d.size(2)}")
        embeddingList += emb3d
      }
    }

    // Process sequence features
    sequenceFeats.foreach { case (name, indices) =>
      embeddingTables.get(s"${name}_seq").foreach { embed =>
        val embedDev = embed.weight().device()
        val idxOnDev = if (indices.device().equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        val emb = embed.forward(idxOnDev)

        // Apply pooling
        val pooled = emb match {
          case _ if name.endsWith("_mean") || name.endsWith("_sum") || name.endsWith("_concat") =>
            println(s"[DEBUG] $name skipping pool")
            emb
          case _ =>
            println(s"[DEBUG] $name before pool: ${emb.size(0)}x${emb.size(1)}")
            val pooledEmb = poolSequence(emb, idxOnDev, "mean")
            println(s"[DEBUG] $name after pool: ${pooledEmb.size(0)}x${pooledEmb.size(1)}")
            pooledEmb
        }
        embeddingList += pooled
      }
    }

    if (embeddingList.isEmpty) {
      // Debug: print what keys are available
      println(s"[DEBUG] embeddingTables keys: ${embeddingTables.keys.mkString(", ")}")
      println(s"[DEBUG] sparseFeats keys: ${sparseFeats.keys.mkString(", ")}")
      println(s"[DEBUG] sequenceFeats keys: ${sequenceFeats.keys.mkString(", ")}")
      throw new IllegalArgumentException("No embeddings found for given features")
    }

    val batchSize = embeddingList.head.size(0).toInt
    val numFields = embeddingList.size

    // Calculate actual dimensions for each embedding (handle both 2D and 3D tensors)
    // 2D: (batch, embedDim), 3D: (batch, 1, embedDim)
    val embedDims = embeddingList.map(e => if (e.dim() == 3L) e.size(2).toInt else e.size(1).toInt)
    val totalDim = embedDims.sum

    // Debug output
    println(s"[DEBUG EmbeddingLayer] batchSize=$batchSize, numFields=$numFields, embedDims=$embedDims, totalDim=$totalDim")

    // Manually concatenate embeddings into a single tensor to avoid JNI device issues
    // Move embeddings to CPU first to avoid CUDA device assert errors when calling toFloatArray
    val resultArr = new Array[Float](batchSize * totalDim)
    var offset = 0
    for ((emb, f) <- embeddingList.zipWithIndex) {
      // Move to CPU before converting to FloatArray
      val embCpu = try {
        if (emb.is_cuda()) emb.cpu() else emb
      } catch {
        case _: Exception => emb.to(new Device("cpu"), ScalarType.Float)
      }
      val arr = embCpu.toFloatArray
      val thisDim = embedDims(f)
      var i = 0
      while (i < batchSize) {
        var k = 0
        while (k < thisDim) {
          resultArr(i * totalDim + offset + k) = arr(i * thisDim + k)
          k += 1
        }
        i += 1
      }
      offset += thisDim
    }
    val embedDev = embeddingList.head.device()
    val flattened = torchrec.TorchRec.tensor(resultArr, batchSize.toLong, totalDim.toLong).to(embedDev, ScalarType.Float)
    if (squeeze && embeddingList.size == 1) {
      flattened.squeeze(1)
    } else {
      flattened
    }
  }

  private def poolSequence(emb: Tensor, indices: Tensor, pooling: String): Tensor = {
    val padIdx = paddingIdx.getOrElse(0L)
    // Handle 1D indices (single item per batch): return embedding directly
    if (indices.dim() == 1L) {
      return emb
    }
    pooling match {
      case "mean" =>
        val padTensor = torch.full(Array(1L), new Scalar(padIdx)).to(indices.device(), ScalarType.Long)
        val mask = indices.ne(padTensor).toType(ScalarType.Float)
        val sum = emb.mul(mask.unsqueeze(2)).sum(1)
        val count = mask.sum(1).unsqueeze(1)
        sum.div(count)

      case "sum" =>
        emb.sum(1)

      case "max" =>
        // Use dimension 1 for max
        val maxPair = torch.max(emb, 1.toLong)
        maxPair.get0

      case "last" =>
        val padTensor = torch.full(Array(1L), new Scalar(padIdx)).to(indices.device(), ScalarType.Long)
        val mask = indices.ne(padTensor)
        val lastIdx = mask.sum(1).toType(ScalarType.Long)
        emb.mean(1)

      case _ => emb.mean(1)
    }
  }

  /** Get embedding for a single feature */
  def getEmbedding(name: String, indices: Tensor): Tensor = {
    embeddingTables.get(name) match {
      case Some(embed) =>
        // Squeeze 2D (batch, 1) indices to 1D (batch,) for standard embedding lookup
        val idx1d = if (indices.dim() == 2L && indices.size(1) == 1L) {
          indices.squeeze(1)
        } else if (indices.dim() == 1L) {
          indices
        } else {
          indices
        }
        // Move indices to the SAME device as the embedding table (not just "device" field)
        val embedDev = embed.weight().device()
        val idxOnDev = if (idx1d.device().equals(embedDev)) {
          idx1d.toType(ScalarType.Long)
        } else {
          idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        embed.forward(idxOnDev)
      case None => throw new IllegalArgumentException(s"No embedding table for: $name")
    }
  }

  def to(device: String): this.type = {
    // Embeddings are already on the target device from the constructor.
    // This method is kept for API compatibility but is a no-op.
    this
  }
}
