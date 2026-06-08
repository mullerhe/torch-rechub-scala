package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

/**
 * Product-based Neural Network (PNN)
 *
 * Captures both first-order (linear) and second-order (product) feature interactions.
 * Uses an embedding layer + product layer (inner or outer product) + MLP.
 *
 * Two product types:
 *   - "inner": Inner product between field embeddings (learns scalar interactions)
 *   - "outer": Outer product via learnable kernel (learns vector interactions)
 *
 * Architecture:
 *   Sparse Input → EmbeddingLayer → Flat Embeddings + Product Features → MLP → Output
 *
 * Reference: "Product-based Neural Networks for User Response Prediction" (Song et al., 2016)
 *
 * @param features     List of sparse features
 * @param embedDim    Embedding dimension
 * @param mlpDims     Hidden layer dimensions for the MLP
 * @param productType One of "inner" or "outer"
 * @param dropout     Dropout rate
 * @param device      Device to run on
 */
class PNN(
  features: List[Feature],
  embedDim: Int = 8,
  mlpDims: List[Long] = List(256L, 128L, 64L),
  productType: String = "inner",
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(productType == "inner" || productType == "outer",
    s"productType must be 'inner' or 'outer', got $productType")

  private val numFields = features.collect { case f: SparseFeature => 1 }.size
  require(numFields >= 2, "PNN requires at least 2 sparse features")

  // Embedding layer
  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Number of field pairs
  private val numPairs = (numFields * (numFields - 1)) / 2

  // Product layer
  productType match {
    case "inner" =>
      // InnerProductNetwork is stateless, can be created here
    case "outer" =>
      // OuterProductNetwork has learnable parameters, needs registration
  }

  // MLP input dimension
  // - Flat embeddings: num_fields * embed_dim
  // - Product: num_pairs (inner) or num_pairs * embed_dim (outer)
  private val embedFlatDim = numFields * embedDim
  private val productDim = productType match {
    case "inner" => numPairs
    case "outer" => numPairs * embedDim
  }
  private val mlpInputDim = embedFlatDim + productDim

  // Outer product network (registered here in constructor)
  private val outerNet = productType match {
    case "outer" =>
      val opn = new OuterProductNetwork(numFields, embedDim, "vec", device)
      register_module("opn", opn)
      Some(opn)
    case _ => None
  }

  // MLP
  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, device = device)
  register_module("mlp", mlp)

  if (device != "cpu") {
    mlp.to(new org.bytedeco.pytorch.Device(device), false)
    ()
  }

  def forward(
    sparseFeats: Map[String, Tensor],
    denseFeats: Map[String, Tensor] = Map.empty
  ): Tensor = {
    // Get embeddings: (batch, num_fields, embed_dim)
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0).toInt

    // Flatten embeddings: (batch, num_fields * embed_dim)
    val flatEmbed = embeddings.view(batchSize, embedFlatDim.toLong)

    // Compute product interactions
    val productFeatures: Tensor = productType match {
      case "inner" =>
        // Inner product between all field pairs: scalar per pair
        val ip = new InnerProductNetwork()
        ip.forward(embeddings)  // (batch, num_pairs)

      case "outer" =>
        // Outer product via learnable kernel: vector per pair
        outerNet match {
          case Some(opn) => opn.forward(embeddings)  // (batch, num_pairs)
          case None => throw new IllegalStateException("outerNet not initialized")
        }
    }

    // Concatenate flat embeddings and product features
    val mlpInput = torch.cat(new TensorVector().push_back(flatEmbed).push_back(productFeatures), 1)

    // MLP
    mlp.forward(mlpInput).squeeze(1)
  }
}
