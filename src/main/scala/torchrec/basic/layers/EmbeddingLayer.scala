package torchrec.basic.layers

import torchrec.basic.features._
import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.jdk.CollectionConverters._
import scala.collection.mutable

/**
 * Embedding layer for sparse and sequence features
 */
class EmbeddingLayer(
  val features: List[Feature],
  val embedDim: Int = 8,
  val device: String = "cpu",
  val paddingIdx: Option[Long] = None,
  val sparse: Boolean = false
) extends Module {

  // Map from feature name to embedding module
  private val embeddingTables = mutable.Map[String, EmbeddingImpl]()
  private val sharedEmbeddingTables = mutable.Map[String, EmbeddingImpl]()

  // Build embedding tables
  features.foreach {
    case f: SparseFeature =>
      if (!f.sharedWith.exists(sharedEmbeddingTables.contains)) {
        val options = new EmbeddingOptions(f.vocabSize, embedDim)
        f.paddingIdx.foreach { idx =>
          options.padding_idx().put(idx)
        }
        val embedding = new EmbeddingImpl(options)
        try {
          embedding.to(new org.bytedeco.pytorch.Device(device), false)
        } catch {
          case _: Throwable =>
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
        val options = new EmbeddingOptions(f.vocabSize, embedDim)
        if (f.paddingIdx != 0) {
          options.padding_idx().put(f.paddingIdx)
        }
        val embedding = new EmbeddingImpl(options)
        try {
          embedding.to(new org.bytedeco.pytorch.Device(device), false)
        } catch {
          case _: Throwable =>
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
      embeddingTables.get(name).foreach { embed =>
        val embedDev = embed.weight().device()
        val indicesOnDev = if (indices.device().equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        val emb = embed.forward(indicesOnDev)
        indicesOnDev.close()
        embeddingList += emb.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
      }
    }

    // Process sequence features
    sequenceFeats.foreach { case (name, indices) =>
      embeddingTables.get(s"${name}_seq").foreach { embed =>
        val embedDev = embed.weight().device()
        val indicesOnDev = if (indices.device().equals(embedDev)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long)
        }
        val emb = embed.forward(indicesOnDev)
        indicesOnDev.close()

        // Apply pooling
        val pooled = emb match {
          case _ if name.endsWith("_mean") || name.endsWith("_sum") || name.endsWith("_concat") =>
            emb
          case _ =>
            poolSequence(emb, indicesOnDev, "mean")
        }
        embeddingList += pooled.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
      }
    }

    if (embeddingList.isEmpty) {
      throw new IllegalArgumentException("No embeddings found for given features")
    }

    val batchSize = embeddingList.head.size(0).toInt
    val numFields = embeddingList.size
    val embedDims = embeddingList.map(e => (e.numel().toInt / batchSize))
    val embedDim = if (embedDims.nonEmpty) embedDims.head else 0

    // Manually concatenate embeddings into a single tensor to avoid JNI device issues
    val resultArr = new Array[Float](batchSize * numFields * embedDim)
    for ((emb, f) <- embeddingList.zipWithIndex) {
      val arr = emb.toFloatArray
      var i = 0
      while (i < batchSize) {
        var k = 0
        while (k < embedDim) {
          resultArr((i * numFields + f) * embedDim + k) = arr(i * embedDim + k)
          k += 1
        }
        i += 1
      }
    }
    val concatenated = torchrec.TorchRec.tensor(resultArr, batchSize.toLong, numFields.toLong, embedDim.toLong).to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
    val flattened = concatenated.view(batchSize.toLong, (numFields * embedDim).toLong)
    if (squeeze && embeddingList.size == 1) {
      concatenated.squeeze(1)
    } else {
      flattened
    }
  }

  private def poolSequence(emb: Tensor, indices: Tensor, pooling: String): Tensor = {
    val padIdx = paddingIdx.getOrElse(0L)
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
        val embedDevice = embed.weight().device()
        val indicesOnDevice = if (indices.device().equals(embedDevice)) {
          indices.toType(ScalarType.Long)
        } else {
          indices.toType(ScalarType.Long).to(embedDevice, ScalarType.Long)
        }
        val emb = embed.forward(indicesOnDevice)
        indicesOnDevice.close()
        emb
      case None => throw new IllegalArgumentException(s"No embedding table for: $name")
    }
  }

  def to(device: String): this.type = {
    // Device transfer not supported in JavaCPP - just return this
    this
  }
}
