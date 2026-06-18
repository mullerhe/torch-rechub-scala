package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.Implicits._
import torchrec.TensorImplicits.RichTensor
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch

/**
 * FiBiNet: Feature Importance and Bilinear feature Interaction
 * Reference: RecSys 2019
 */
class FiBiNet(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L),
  reduction: Int = 3,
  bilinearType: String = "field_interaction",  // "field_all", "field_each", "field_interaction"
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  private val numFields = features.collect { case f: SparseFeature => 1 }.size

  // SENET layer: takes (num_fields, reduction_ratio) like Python
  private val senet = new SENETLayer(numFields, reduction, device)
  register_module("senet", senet)

  // Bilinear interaction layer
  private val bilinear = bilinearType match {
    case "field_all" => new BilinearInteraction(embedDim, numFields, "field_all", device)
    case "field_each" => new BilinearInteraction(embedDim, numFields, "field_each", device)
    case _ => new BilinearInteraction(embedDim, numFields, "field_interaction", device)
  }
  register_module("bilinear", bilinear)

  // MLP input dim: num_fields * (num_fields - 1) * embed_dim (for both bi1 and bi2 concatenated)
  private val numInteractions = numFields * (numFields - 1) / 2
  private val mlpInputDim = numInteractions * embedDim * 2  // both bi1 and bi2
  private val mlp = new MLP(mlpInputDim, mlpDims.map(_.toLong), 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0)
    val reshaped = embeddings.view(batchSize, numFields, embedDim)

    // SENET-enhanced features
    val senetFeatures = senet.forward(reshaped)

    // Bilinear interactions on original embeddings: (batch, num_interactions, embed_dim)
    val biOut1 = bilinear.forward(reshaped, reshaped)

    // Bilinear interactions on SENET-enhanced embeddings: (batch, num_interactions, embed_dim)
    val biOut2 = bilinear.forward(senetFeatures, reshaped)

    // Concatenate both bilinear outputs along dim=1
    // biOut1 and biOut2 are Seq[Tensor], need to stack then concatenate
    val stacked1 = torch.stack(new TensorVector(biOut1.toSeq: _*), 1L)
    val stacked2 = torch.stack(new TensorVector(biOut2.toSeq: _*), 1L)
    val combined = torch.cat(new TensorVector(stacked1, stacked2), 1L)

    // Flatten and pass through MLP: (batch, num_interactions * 2 * embed_dim)
    val flattened = combined.view(batchSize, -1)
    val logits = mlp.forward(flattened)
    logits
  }
}

/**
 * Bilinear Feature Interaction
 */
class BilinearInteraction(
  embedDim: Int,
  numFields: Int,
  bilinearType: String,
  device: String = DeviceSupport.backend
) extends Module {

  private val weight = bilinearType match {
    case "field_all" =>
      val w = new LinearImpl(embedDim, embedDim)
      register_module("bilinear_weight", w)
      w
    case _ =>
      val w = Array.fill(numFields)(new LinearImpl(embedDim, embedDim))
      w.zipWithIndex.foreach { case (lin, i) =>
        register_module(s"bilinear_weight_$i", lin)
      }
      w
  }

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    bilinearType match {
      case "field_all" =>
        weight.asInstanceOf[LinearImpl].to(dev, false)
      case _ =>
        weight.asInstanceOf[Array[LinearImpl]].foreach(_.to(dev, false))
    }
  }

  def forward(f1: Tensor, f2: Tensor): Seq[Tensor] = {
    // f1, f2: (batch, num_fields, embed_dim)
    bilinearType match {
      case "field_all" =>
        val w = weight.asInstanceOf[LinearImpl]
        (0 until numFields).flatMap { i =>
          (0 until numFields).map { j =>
            val vi = f1.select(1, i)
            val vj = w.forward(f2.select(1, j))
            vi.mul(vj).sum(1).unsqueeze(1)
          }
        }

      case _ =>
        val w = weight.asInstanceOf[Array[LinearImpl]]
        (0 until numFields).flatMap { i =>
          (0 until numFields).map { j =>
            val vi = f1.select(1, i)
            val vj = w(i).forward(f2.select(1, j))
            vi.mul(vj).sum(1).unsqueeze(1)
          }
        }
    }
  }
}