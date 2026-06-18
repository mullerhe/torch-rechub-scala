package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits._
import torchrec.Implicits.SeqTensorRichSeq

import scala.collection.mutable

/**
 * Deep Interest Evolution Network (DIEN)
 * Reference: DIEN paper, AAAI 2019
 *
 * Simplified implementation for benchmark compatibility.
 * Uses Interest Extractor GRU and Interest Evolving AUGRU.
 */
class DIEN(
  features: List[Feature],
  sequenceFeatures: List[SequenceFeature],
  embedDim: Int = 8,
  hiddenDim: Int = 8,
  mlpDims: List[Long] = List(64L, 32L),
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")

  // Total dimension for MLP
  private val sparseDim: Long = Features.calcSparseDim(features)
  private val seqDim: Long = sequenceFeatures.map(_.embedDim).sum
  private val totalDim: Long = sparseDim + seqDim

  // Embedding layer
  private val embedding = new EmbeddingLayer(
    features ++ sequenceFeatures,
    embedDim,
    device
  )
  register_module("embedding", embedding)

  // Interest Extractor GRUs - one per sequence feature
  private val interestGRUs = sequenceFeatures.map { fea =>
    val opts = new GRUOptions(fea.embedDim, fea.embedDim)
    opts.batch_first().put(true)
    val gru = new GRUImpl(opts)
    if (device != "cpu") {
      gru.to(new org.bytedeco.pytorch.Device(device), false)
    }
    gru
  }
  interestGRUs.zipWithIndex.foreach { case (gru, i) =>
    register_module(s"interest_gru_$i", gru)
  }

  // Interest Evolving AUGRUs - one per sequence feature
  private val augrus = sequenceFeatures.map { fea =>
    new AUGRU(fea.embedDim, device)
  }
  augrus.zipWithIndex.foreach { case (augru, i) =>
    register_module(s"augru_$i", augru)
  }

  // MLP for final prediction with dice activation
  private val mlp = new MLP(
    inputDim = totalDim,
    hiddenDims = mlpDims,
    outputDim = 1,
    activation = "dice",
    dropout = dropout,
    device = device
  )
  register_module("mlp", mlp)

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    this.to(dev, false)
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    seqFeats: Map[String, Tensor]
  ): Tensor = {
    // Feature embeddings
    val featEmb = embedding.forward(sparseFeats, Map.empty, squeeze = true)

    // Sequence embeddings
    val seqEmb = embedding.forward(Map.empty, seqFeats, squeeze = false)

    // Interest Extractor + Evolving
    val interestOut = mutable.ListBuffer[Tensor]()

    for (i <- sequenceFeatures.indices) {
      val seq = seqEmb.select(1, i)  // (B, T, D)
      val mask = getMask(seqFeats, sequenceFeatures(i))  // (B, T)

      // Interest Extractor: GRU
      val gruOut = interestGRUs(i).forward(seq).get1()  // (B, T, D)

      // Use last valid hidden state as target
      val seqLens = mask.sum(1).toType(ScalarType.Long)
      val clampedLens = seqLens.clamp(new ScalarOptional(new Scalar(1L)))
      val lastIdx = (clampedLens.sub(new Scalar(1l)) ).unsqueeze(1).expand(-1, seq.size(2))
      val targetEmb = gruOut.gather(1, lastIdx).squeeze(1)  // (B, D)

      // Interest Evolving: AUGRU
      val hasHistAny = mask.any(1)
      val embedDimAU = augrus(i).embed_dim

      val h = if (hasHistAny.count_nonzero().item().toInt() > 0) {
        val validMask = hasHistAny
        val validSeq = seq.masked_select(validMask.unsqueeze(2)).view(-1, seq.size(1), seq.size(2))
        val validTarget = targetEmb.masked_select(validMask.unsqueeze(1)).view(-1, targetEmb.size(1))
        val validMaskFlat = mask.masked_select(validMask)

        val (_, hValid) = augrus(i).forward(validSeq, validTarget, validMaskFlat)

        // Scatter back
        val hFull = torch.zeros(hasHistAny.size(0).toInt, embedDimAU).to(seq.device(), ScalarType.Float)
        val validIdx = torch.where(validMask).get(1).toType(ScalarType.Long)
        hFull.index_copy(0, validIdx, hValid)
        hFull
      } else {
        torch.zeros(hasHistAny.size(0).toInt, embedDimAU).to(seq.device(), ScalarType.Float)
      }

      interestOut += h
    }

    // Concatenate all interest outputs
    val interestCat = interestOut.toSeq.cat(1)

    // Combine with features
    val combined = torch.cat(new TensorVector(interestCat, featEmb), 1)

    // Final MLP
    val y = mlp.forward(combined)

    torch.sigmoid(y.squeeze(1))
  }

  private def getMask(seqFeats: Map[String, Tensor], feature: SequenceFeature): Tensor = {
    val indices = seqFeats(feature.name)
    val paddingIdx = feature.paddingIdx
    if (paddingIdx >= 0) {
      indices.ne(new Scalar(paddingIdx)).toType(ScalarType.Bool)
    } else {
      indices.ne(new Scalar(-1L)).toType(ScalarType.Bool)
    }
  }
}