package torchrec.models.knowledge_tracing

import org.bytedeco.pytorch.*
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.Implicits.*
import torchrec.models.knowledge_tracing.layers.{LearnablePositionalEmbedding, SAINTDecoderBlock, SAINTEncoderBlock}
import torchrec.utils.DeviceSupport

/**
 * SAINT: Self-Attentive Interpretable Knowledge Tracing
 *
 * Reference: "SAINT: An Interpretable Self-Attentive Knowledge Tracing" (Shin et al., 2021)
 *
 * Architecture:
 *   Encoder: exercise + category + position -> Transformer encoder
 *   Decoder: response + position (with start token) -> Transformer decoder
 *   Output: sigmoid(decoder output)
 *
 * @param numExercises  Number of unique exercises/questions
 * @param numCategories  Number of categories/concepts
 * @param numResponses   Number of response types (typically 2)
 * @param embedDim      Embedding dimension
 * @param numHeads       Number of attention heads
 * @param numEncoderBlocks Number of encoder blocks
 * @param numDecoderBlocks Number of decoder blocks
 * @param ffnDim         FFN hidden dimension
 * @param dropout        Dropout rate
 * @param device         Device
 */
class SAINT(
  numExercises: Long,
  numCategories: Long,
  numResponses: Int = 2,
  embedDim: Int = 64,
  numHeads: Int = 8,
  numEncoderBlocks: Int = 3,
  numDecoderBlocks: Int = 3,
  ffnDim: Int = 256,
  dropout: Float = 0.2f,
  device: String = DeviceSupport.backend
) extends Module {

  require(embedDim % numHeads == 0)

  // Exercise embedding
  private val exEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 1, embedDim))
  register_module("ex_emb", exEmb)

  // Category/concept embedding
  private val catEmb = new EmbeddingImpl(new EmbeddingOptions(numCategories + 1, embedDim))
  register_module("cat_emb", catEmb)

  // Response embedding (0, 1, ... numResponses-1, start_token, pad_token)
  private val resEmb = new EmbeddingImpl(new EmbeddingOptions(numResponses + 2, embedDim))
  register_module("res_emb", resEmb)

  // Positional embedding
  private val posEmb = new LearnablePositionalEmbedding(200, embedDim, dropout, device)
  register_module("pos_emb", posEmb)

  // Encoder blocks
  private val encoderBlocks = (0 until numEncoderBlocks).map { i =>
    val block = new SAINTEncoderBlock(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"encoder_$i", block)
    block
  }

  // Decoder blocks
  private val decoderBlocks = (0 until numDecoderBlocks).map { i =>
    val block = new SAINTDecoderBlock(embedDim, numHeads, ffnDim, dropout, device)
    register_module(s"decoder_$i", block)
    block
  }

  // Output layer
  private val outLayer = new LinearImpl(embedDim, 1)
  register_module("out", outLayer)

  // Start token ID
  private val startTokenId = numResponses + 1  // After 0, 1 response values

  if (device != "cpu") {
    val dev = new org.bytedeco.pytorch.Device(device)
    exEmb.to(dev, false); catEmb.to(dev, false); resEmb.to(dev, false)
    outLayer.to(dev, false)
  }

  /**
   * Forward pass for SAINT.
   * @param exerciseIds  Exercise IDs (batch, seqLen)
   * @param categoryIds  Category/concept IDs (batch, seqLen)
   * @param responseIds  Response IDs 0/1 (batch, seqLen)
   * @return Predictions (batch, seqLen) - probability of correct response
   */
  def forward(exerciseIds: Tensor, categoryIds: Tensor, responseIds: Tensor): Tensor = {
    val batchSize = exerciseIds.size(0).toInt
    val seqLen = exerciseIds.size(1).toInt

    // Encoder: exercise + category + position
    val exEmbOut = exEmb.forward(exerciseIds)
    val catEmbOut = catEmb.forward(categoryIds)
    val posEnc = posEmb.forward(seqLen.toLong)
    val posEx = posEnc.expand(batchSize.toLong, seqLen.toLong, embedDim.toLong)

    var enOut = exEmbOut.add(catEmbOut)
    encoderBlocks.foreach { block =>
      enOut = block.forward(enOut, catEmbOut, posEx)
    }

    // Decoder: prepend start token to responses
    val startTokens = torch.full(Array(batchSize.toLong, 1L), new Scalar(startTokenId.toDouble),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
    if (device != "cpu") {
      startTokens.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
    }
    val paddedResponses = torch.cat(new TensorVector(startTokens, responseIds), 1)

    val resEmbOut = resEmb.forward(paddedResponses)
    val posDec = posEmb.forward((seqLen + 1).toLong)
    val posDecEx = posDec.expand(batchSize.toLong, (seqLen + 1).toLong, embedDim.toLong)

    var decOut = resEmbOut
    decoderBlocks.foreach { block =>
      decOut = block.forward(decOut, posDecEx, enOut)
    }

    // Predict from decoder output (skip the start token position)
    val decOutShifted = decOut.narrow(1, 1, seqLen)
    val logits = outLayer.forward(decOutShifted)

    logits.sigmoid().squeeze(2)
  }

  def predict(exerciseIds: Tensor, categoryIds: Tensor, responseIds: Tensor): Tensor =
    forward(exerciseIds, categoryIds, responseIds)
}
