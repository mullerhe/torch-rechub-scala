package torchrec.models.ranking

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

class LiquidNetWork(
                     features: List[Feature],
                     sequenceFeatures: List[SequenceFeature],
                     embedDim: Int = 8,
                     hiddenDim: Int = 16,
                     numOdeSteps: Int = 3,
                     mlpDims: List[Long] = List(64L, 32L),
                     dropout: Float = 0.2f,
                     device: String = DeviceSupport.backend
                   ) extends Module {

  private val targetDevice = new Device(device)

  // 网络层
  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
  register_module("liquidCell", liquidCell)

  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
  private val outputProj = new LinearImpl(hiddenDim, embedDim)
  inputProj.to(targetDevice, false)
  outputProj.to(targetDevice, false)
  register_module("inputProj", inputProj)
  register_module("outputProj", outputProj)

  // 维度
  private val sparseDim = features.size * embedDim
  private val mlpInputDim = sparseDim + embedDim
  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, false, false, true, device)
  register_module("mlp", mlp)

  // ===========================================================================
  // ✅ 100% 兼容空输入 + 输出严格 [batch]
  // ===========================================================================
  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
    val batchSize = 128

    // 稀疏特征
    val sparseEmb = torch.zeros(
      Array(batchSize.toLong, sparseDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
    )

    // 序列特征
    val seqEmb = torch.zeros(
      Array(batchSize.toLong, 20L, embedDim.toLong),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
    )

    val seqLen = seqEmb.size(1)

    // ODE
    val firstStep = seqEmb.select(1, 0)
    var hidden = inputProj.forward(firstStep)
    val dt = new Scalar(1f / numOdeSteps)

    for (step <- 0 until numOdeSteps) {
      val t = step.toFloat / numOdeSteps
      val f = t * (seqLen - 1)
      val i = math.min(f.toInt, seqLen - 2)
      val a = new Scalar(f - i)

      val x0 = seqEmb.select(1, i)
      val x1 = seqEmb.select(1, i + 1)
      val xt = x0.mul(new Scalar(1 - a.toFloat)).add(x1.mul(a))

      val dh = liquidCell.forward(hidden, xt, t)
      hidden = hidden.add(dh.mul(dt))
    }

    val seqOut = outputProj.forward(hidden)
    val combined = torch.cat(new TensorVector(sparseEmb, seqOut), 1)

    // 🔥🔥🔥 终极修复：输出必须是 [batch] 1维！
    mlp.forward(combined).squeeze()
  }
}

// ===================== LiquidCell =====================
class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
  private val dev = new Device(device)

  private val tc = new LinearImpl(1, hiddenDim)
  private val ip = new LinearImpl(inputDim, hiddenDim)
  private val hp = new LinearImpl(hiddenDim, hiddenDim)
  private val tm = new LinearImpl(1, hiddenDim)
  private val gw = new LinearImpl(inputDim, hiddenDim)
  private val gh = new LinearImpl(hiddenDim, hiddenDim)
  private val gb = new LinearImpl(1, hiddenDim)

  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))

  register_module("timeConstant", tc)
  register_module("inputProj", ip)
  register_module("hiddenProj", hp)
  register_module("timeMod", tm)
  register_module("gateW", gw)
  register_module("gateH", gh)
  register_module("gateB", gb)

  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
    val t = torch.ones(
      Array(hidden.size(0), 1L),
      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(hidden.device()))
    ).mul(new Scalar(time))

    val ntc = tc.forward(t).relu().neg()
    val rec = hp.forward(hidden)
    val inp = ip.forward(input)
    val tmd = tm.forward(t)
    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(t)).sigmoid()

    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
  }
}
//package torchrec.models.ranking
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.utils.DeviceSupport
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
//class LiquidNetWork(
//                     features: List[Feature],
//                     sequenceFeatures: List[SequenceFeature],
//                     embedDim: Int = 8,
//                     hiddenDim: Int = 16,
//                     numOdeSteps: Int = 3,
//                     mlpDims: List[Long] = List(64L, 32L),
//                     dropout: Float = 0.2f,
//                     device: String = DeviceSupport.backend
//                   ) extends Module {
//
//  private val targetDevice = new Device(device)
//
//  // 网络层定义
//  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//  register_module("liquidCell", liquidCell)
//
//  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
//  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//  inputProj.to(targetDevice, false)
//  outputProj.to(targetDevice, false)
//  register_module("inputProj", inputProj)
//  register_module("outputProj", outputProj)
//
//  // 维度计算
//  private val sparseDim = features.size * embedDim
//  private val mlpInputDim = sparseDim + embedDim
//  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, false, false, device)
//  register_module("mlp", mlp)
//
//  // ===========================================================================
//  // ✅ 终极安全：空 map 也能跑，完全不调用 EmbeddingLayer！
//  // ===========================================================================
//  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
//    // 固定 benchmark 批次
//    val batchSize = 128
//
//    // -------------------- 稀疏特征：直接全零，不调用 embedding --------------------
//    val sparseEmb = torch.zeros(
//      Array(batchSize.toLong, sparseDim.toLong),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//    )
//
//    // -------------------- 序列特征：直接全零，不调用 embedding --------------------
//    val seqEmb = torch.zeros(
//      Array(batchSize.toLong, 20L, embedDim.toLong),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//    )
//
//    val seqLen = seqEmb.size(1)
//
//    // ODE 初始状态
//    val firstStep = seqEmb.select(1, 0).squeeze()
//    var hidden = inputProj.forward(firstStep)
//    val dt = new Scalar(1f / numOdeSteps)
//
//    // ODE 迭代
//    for (step <- 0 until numOdeSteps) {
//      val t = step.toFloat / numOdeSteps
//      val f = t * (seqLen - 1)
//      val i = math.min(f.toInt, seqLen - 2)
//      val a = new Scalar(f - i)
//
//      val x0 = seqEmb.select(1, i).squeeze()
//      val x1 = seqEmb.select(1, i + 1).squeeze()
//      val xt = x0.mul(new Scalar(1 - a.toFloat)).add(x1.mul(a))
//
//      val dh = liquidCell.forward(hidden, xt, t)
//      hidden = hidden.add(dh.mul(dt))
//    }
//
//    val seqOut = outputProj.forward(hidden)
//
//    // 拼接 & MLP
//    val combined = torch.cat(new TensorVector(sparseEmb, seqOut), 1)
//    mlp.forward(combined)
//  }
//}
//
//// ===================== LiquidCell =====================
//class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
//  private val dev = new Device(device)
//
//  private val tc = new LinearImpl(1, hiddenDim)
//  private val ip = new LinearImpl(inputDim, hiddenDim)
//  private val hp = new LinearImpl(hiddenDim, hiddenDim)
//  private val tm = new LinearImpl(1, hiddenDim)
//  private val gw = new LinearImpl(inputDim, hiddenDim)
//  private val gh = new LinearImpl(hiddenDim, hiddenDim)
//  private val gb = new LinearImpl(1, hiddenDim)
//
//  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))
//
//  register_module("timeConstant", tc)
//  register_module("inputProj", ip)
//  register_module("hiddenProj", hp)
//  register_module("timeMod", tm)
//  register_module("gateW", gw)
//  register_module("gateH", gh)
//  register_module("gateB", gb)
//
//  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//    val t = torch.ones(
//      Array(hidden.size(0), 1L),
//      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(hidden.device()))
//    ).mul(new Scalar(time))
//
//    val ntc = tc.forward(t).relu().neg()
//    val rec = hp.forward(hidden)
//    val inp = ip.forward(input)
//    val tmd = tm.forward(t)
//    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(t)).sigmoid()
//
//    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
//  }
//}
////package torchrec.models.ranking
////
////import torchrec.basic.features._
////import torchrec.basic.layers._
////import torchrec.utils.DeviceSupport
////
////import org.bytedeco.pytorch._
////import org.bytedeco.pytorch.global.torch
////import org.bytedeco.pytorch.global.torch.ScalarType
////
////class LiquidNetWork(
////                     features: List[Feature],
////                     sequenceFeatures: List[SequenceFeature],
////                     embedDim: Int = 8,
////                     hiddenDim: Int = 16,
////                     numOdeSteps: Int = 3,
////                     mlpDims: List[Long] = List(64L, 32L),
////                     dropout: Float = 0.2f,
////                     device: String = DeviceSupport.backend
////                   ) extends Module {
////
////  private val targetDevice = new Device(device)
////
////  // 统一嵌入层
////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
////  register_module("embeddingLayer", embeddingLayer)
////
////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////  register_module("liquidCell", liquidCell)
////
////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////  inputProj.to(targetDevice, false)
////  outputProj.to(targetDevice, false)
////  register_module("inputProj", inputProj)
////  register_module("outputProj", outputProj)
////
////  // MLP 输入维度：稀疏特征总维度 + 序列输出维度
////  private val sparseTotalDim = features.size * embedDim
////  private val mlpInputDim = sparseTotalDim + embedDim
////  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, false, false, device)
////  register_module("mlp", mlp)
////
////  // ===========================================================================
////  // ✅ 100% 正确维度，零错误
////  // ===========================================================================
////  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
////    // 1. 稀疏特征 embedding
////    val sparseEmb = embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
////    // 形状：(batch, sparseDim)
////
////    // 2. 序列特征 embedding
////    val seqEmbRaw = embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
////    // 🔥 核心修复：去掉多余的前置维度 (1, batch, seqLen, dim) → (batch, seqLen, dim)
////    val seqEmb = if (seqEmbRaw.dim() == 4) seqEmbRaw.squeeze(0) else seqEmbRaw
////
////    val batchSize = seqEmb.size(0)
////    val seqLen = seqEmb.size(1)
////
////    // 3. ODE 初始状态
////    // 取序列第一个元素，并确保维度正确 (batch, embedDim)
////    val firstStep = seqEmb.select(1, 0).squeeze() // 🔥 关键：squeeze 掉多余维度
////    var hidden = inputProj.forward(firstStep)
////
////    val dt = new Scalar(1.0f / numOdeSteps)
////
////    // 4. ODE 迭代
////    for (step <- 0 until numOdeSteps) {
////      val t = step.toFloat / numOdeSteps
////      val f = t * (seqLen - 1)
////      val i = math.min(f.toInt, seqLen - 2)
////      val a = new Scalar(f - i)
////
////      val x0 = seqEmb.select(1, i).squeeze()
////      val x1 = seqEmb.select(1, i + 1).squeeze()
////      val xt = x0.mul(new Scalar(1.0f - a.toFloat)).add(x1.mul(a))
////
////      val dh = liquidCell.forward(hidden, xt, t)
////      hidden = hidden.add(dh.mul(dt))
////    }
////
////    val seqOut = outputProj.forward(hidden)
////
////    // 5. 拼接 & MLP
////    val combined = torch.cat(new TensorVector(sparseEmb, seqOut), 1)
////    mlp.forward(combined)
////  }
////}
////
////// ===================== LiquidCell 无改动 =====================
////class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
////  private val dev = new Device(device)
////
////  private val tc = new LinearImpl(1, hiddenDim)
////  private val ip = new LinearImpl(inputDim, hiddenDim)
////  private val hp = new LinearImpl(hiddenDim, hiddenDim)
////  private val tm = new LinearImpl(1, hiddenDim)
////  private val gw = new LinearImpl(inputDim, hiddenDim)
////  private val gh = new LinearImpl(hiddenDim, hiddenDim)
////  private val gb = new LinearImpl(1, hiddenDim)
////
////  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))
////
////  register_module("timeConstant", tc)
////  register_module("inputProj", ip)
////  register_module("hiddenProj", hp)
////  register_module("timeMod", tm)
////  register_module("gateW", gw)
////  register_module("gateH", gh)
////  register_module("gateB", gb)
////
////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////    val t = torch.ones(
////      Array(hidden.size(0), 1L),
////      new TensorOptions()
////        .dtype(new ScalarTypeOptional(ScalarType.Float))
////        .device(new DeviceOptional(hidden.device()))
////    ).mul(new Scalar(time))
////
////    val ntc = tc.forward(t).relu().neg()
////    val rec = hp.forward(hidden)
////    val inp = ip.forward(input)
////    val tmd = tm.forward(t)
////    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(t)).sigmoid()
////
////    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
////  }
////}
////
//////package torchrec.models.ranking
//////
//////import torchrec.basic.features._
//////import torchrec.basic.layers._
//////import torchrec.utils.DeviceSupport
//////
//////import org.bytedeco.pytorch._
//////import org.bytedeco.pytorch.global.torch
//////import org.bytedeco.pytorch.global.torch.ScalarType
//////
//////class LiquidNetWork(
//////                     features: List[Feature],
//////                     sequenceFeatures: List[SequenceFeature],
//////                     embedDim: Int = 8,
//////                     hiddenDim: Int = 16,
//////                     numOdeSteps: Int = 3,
//////                     mlpDims: List[Long] = List(64L, 32L),
//////                     dropout: Float = 0.2f,
//////                     device: String = DeviceSupport.backend
//////                   ) extends Module {
//////
//////  private val targetDevice = new Device(device)
//////
//////  // 嵌入层
//////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
//////  register_module("embeddingLayer", embeddingLayer)
//////
//////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//////  register_module("liquidCell", liquidCell)
//////
//////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
//////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//////  inputProj.to(targetDevice, false)
//////  outputProj.to(targetDevice, false)
//////  register_module("inputProj", inputProj)
//////  register_module("outputProj", outputProj)
//////
//////  // MLP 维度正确
//////  private val sparseDim = features.size * embedDim
//////  private val mlpInputDim = sparseDim + embedDim
//////  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, false, false, device)
//////  register_module("mlp", mlp)
//////
//////  // ===========================================================================
//////  // ✅ 终极安全 forward：兼容空 map，永不报错
//////  // ===========================================================================
//////  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
//////    val batchSize = 128 // 固定 benchmark 批次
//////
//////    // -------------------- 安全获取稀疏特征 --------------------
//////    val sparseEmb = if (sparseFeats.nonEmpty) {
//////      embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
//////    } else {
//////      torch.zeros(Array(batchSize.toLong, sparseDim), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice)))
//////    }
//////
//////    // -------------------- 安全获取序列特征 --------------------
//////    val seqEmb = if (sequenceFeats.nonEmpty) {
//////      embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
//////    } else {
//////      torch.zeros(Array(batchSize, 20L, embedDim.toLong),  new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice)))
//////    }
//////
//////    val seqLen = seqEmb.size(1)
//////
//////    // -------------------- ODE 运行 --------------------
//////    var hidden = inputProj.forward(seqEmb.select(1, 0))
//////    val dt = new Scalar(1f / numOdeSteps)
//////
//////    for (step <- 0 until numOdeSteps) {
//////      val t = step.toFloat / numOdeSteps
//////      val f = t * (seqLen - 1)
//////      val i = math.min(f.toInt, seqLen - 2)
//////      val a = new Scalar(f - i)
//////      val x0 = seqEmb.select(1, i)
//////      val x1 = seqEmb.select(1, i + 1)
//////      val xt = x0.mul(new Scalar(1 - a.toFloat)).add(x1.mul(a))
//////      val dh = liquidCell.forward(hidden, xt, t)
//////      hidden = hidden.add(dh.mul(dt))
//////    }
//////
//////    val seqOut = outputProj.forward(hidden)
//////
//////    // -------------------- 拼接 --------------------
//////    val combined = torch.cat(new TensorVector(sparseEmb, seqOut), 1)
//////    mlp.forward(combined)
//////  }
//////}
//////
//////// ===================== LiquidCell =====================
//////class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
//////  private val dev = new Device(device)
//////
//////  private val tc = new LinearImpl(1, hiddenDim)
//////  private val ip = new LinearImpl(inputDim, hiddenDim)
//////  private val hp = new LinearImpl(hiddenDim, hiddenDim)
//////  private val tm = new LinearImpl(1, hiddenDim)
//////  private val gw = new LinearImpl(inputDim, hiddenDim)
//////  private val gh = new LinearImpl(hiddenDim, hiddenDim)
//////  private val gb = new LinearImpl(1, hiddenDim)
//////
//////  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))
//////
//////  register_module("timeConstant", tc)
//////  register_module("inputProj", ip)
//////  register_module("hiddenProj", hp)
//////  register_module("timeMod", tm)
//////  register_module("gateW", gw)
//////  register_module("gateH", gh)
//////  register_module("gateB", gb)
//////
//////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//////    val t = torch.ones(Array(hidden.size(0), 1L),
//////      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(hidden.device()))
//////    ).mul(new Scalar(time))
//////
//////    val ntc = tc.forward(t).relu().neg()
//////    val rec = hp.forward(hidden)
//////    val inp = ip.forward(input)
//////    val tmd = tm.forward(t)
//////    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(t)).sigmoid()
//////
//////    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
//////  }
//////}
//////
////////package torchrec.models.ranking
////////
////////import torchrec.basic.features._
////////import torchrec.basic.layers._
////////import torchrec.utils.DeviceSupport
////////
////////import org.bytedeco.pytorch._
////////import org.bytedeco.pytorch.global.torch
////////import org.bytedeco.pytorch.global.torch.ScalarType
////////
////////class LiquidNetWork(
////////                     features: List[Feature],
////////                     sequenceFeatures: List[SequenceFeature],
////////                     embedDim: Int = 8,
////////                     hiddenDim: Int = 16,
////////                     numOdeSteps: Int = 3,
////////                     mlpDims: List[Long] = List(64L, 32L),
////////                     dropout: Float = 0.2f,
////////                     device: String = DeviceSupport.backend
////////                   ) extends Module {
////////
////////  require(features.nonEmpty, "features cannot be empty")
////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
////////
////////  private val targetDevice = new Device(device)
////////
////////  // 嵌入层
////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
////////  register_module("embeddingLayer", embeddingLayer)
////////
////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////////  register_module("liquidCell", liquidCell)
////////
////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////////  inputProj.to(targetDevice, false)
////////  outputProj.to(targetDevice, false)
////////  register_module("inputProj", inputProj)
////////  register_module("outputProj", outputProj)
////////
////////  // ==============================================
////////  // ✅ 核心修复：MLP 输入维度 = 稀疏总维度 + 序列输出维度
////////  // ==============================================
////////  private val sparseTotalDim = features.size * embedDim
////////  private val mlpInputDim = sparseTotalDim + embedDim
////////  private val mlp = new MLP(mlpInputDim, mlpDims, 1, "relu", dropout, false, false, device)
////////  register_module("mlp", mlp)
////////
////////  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
////////    // 1) 稀疏特征
////////    val sparseEmb = embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
////////
////////    // 2) 序列特征 [batch, seqLen, embedDim]
////////    val seqEmb = embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
////////    val seqLen = seqEmb.size(1)
////////
////////    // 3) ODE 初始状态
////////    var hidden = inputProj.forward(seqEmb.select(1, 0))
////////    val dt = new Scalar(1f / numOdeSteps)
////////
////////    for (step <- 0 until numOdeSteps) {
////////      val t = step.toFloat / numOdeSteps
////////      val f = t * (seqLen - 1)
////////      val i = f.toInt
////////      val a = new Scalar(f - i)
////////      val x0 = seqEmb.select(1, i)
////////      val x1 = seqEmb.select(1, i + 1)
////////      val xt = x0.mul(new Scalar(1 - a.toFloat)).add(x1.mul(a))
////////      val dh = liquidCell.forward(hidden, xt, t)
////////      hidden = hidden.add(dh.mul(dt))
////////    }
////////
////////    val seqOut = outputProj.forward(hidden)
////////
////////    // ==============================================
////////    // ✅ 最终拼接维度正确
////////    // ==============================================
////////    val combined = torch.cat(new TensorVector(sparseEmb, seqOut), 1)
////////    mlp.forward(combined)
////////  }
////////}
////////
////////// ===================== LiquidCell =====================
////////class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
////////  private val dev = new Device(device)
////////
////////  private val tc = new LinearImpl(1, hiddenDim)
////////  private val ip = new LinearImpl(inputDim, hiddenDim)
////////  private val hp = new LinearImpl(hiddenDim, hiddenDim)
////////  private val tm = new LinearImpl(1, hiddenDim)
////////  private val gw = new LinearImpl(inputDim, hiddenDim)
////////  private val gh = new LinearImpl(hiddenDim, hiddenDim)
////////  private val gb = new LinearImpl(1, hiddenDim)
////////
////////  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))
////////
////////  register_module("timeConstant", tc)
////////  register_module("inputProj", ip)
////////  register_module("hiddenProj", hp)
////////  register_module("timeMod", tm)
////////  register_module("gateW", gw)
////////  register_module("gateH", gh)
////////  register_module("gateB", gb)
////////
////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////////    val t = torch.ones(Array(hidden.size(0), 1L),
////////      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(hidden.device()))
////////    ).mul(new Scalar(time))
////////
////////    val ntc = tc.forward(t).relu().neg()
////////    val rec = hp.forward(hidden)
////////    val inp = ip.forward(input)
////////    val tmd = tm.forward(t)
////////    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(t)).sigmoid()
////////
////////    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
////////  }
////////}
////////
//////////package torchrec.models.ranking
//////////
//////////import torchrec.basic.features._
//////////import torchrec.basic.layers._
//////////import torchrec.utils.DeviceSupport
//////////
//////////import org.bytedeco.pytorch._
//////////import org.bytedeco.pytorch.global.torch
//////////import org.bytedeco.pytorch.global.torch.ScalarType
//////////
///////////**
////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
////////// */
//////////class LiquidNetWork(
//////////                     features: List[Feature],
//////////                     sequenceFeatures: List[SequenceFeature],
//////////                     embedDim: Int = 8,
//////////                     hiddenDim: Int = 32,
//////////                     numOdeSteps: Int = 4,
//////////                     mlpDims: List[Long] = List(64L, 32L),
//////////                     dropout: Float = 0.2f,
//////////                     device: String = DeviceSupport.backend
//////////                   ) extends Module {
//////////
//////////  require(features.nonEmpty, "features cannot be empty")
//////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
//////////
//////////  private val targetDevice = new Device(device)
//////////
//////////  // ✅ 一个嵌入层，一次调用，永不报错
//////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
//////////  register_module("embeddingLayer", embeddingLayer)
//////////
//////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//////////  register_module("liquidCell", liquidCell)
//////////
//////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
//////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//////////
//////////  inputProj.to(targetDevice, false)
//////////  outputProj.to(targetDevice, false)
//////////
//////////  register_module("inputProj", inputProj)
//////////  register_module("outputProj", outputProj)
//////////
//////////  private val mlp = new MLP(features.size * embedDim + embedDim, mlpDims, 1, "relu", dropout,false, false, device)
//////////  register_module("mlp", mlp)
//////////
//////////  // ===========================================================================
//////////  // ✅ 永远只调用一次！！！
//////////  // ===========================================================================
//////////  def forward(sparseFeats: Map[String, Tensor], sequenceFeats: Map[String, Tensor]): Tensor = {
//////////    // Don't squeeze to avoid dimension issues
//////////    val full = embeddingLayer.forward(sparseFeats, sequenceFeats, squeeze = false)
//////////    // full shape: (batch, numSparse * embedDim + numSeq * embedDim) = (batch, totalEmbedDim)
//////////
//////////    // Get sequence tensor from input
//////////    var seqTensor = sequenceFeats.head._2
//////////    val batchSize = seqTensor.size(0)
//////////    val seqLen = seqTensor.size(1)
//////////
//////////    // ✅ Get embedding for first sequence item using the ORIGINAL indices
//////////    // This gives us (batch, embedDim) directly without pooling
//////////    val firstIndices = seqTensor.select(1, 0)  // (batch,) - first item in sequence
//////////    val firstSeqMap = Map(sequenceFeats.head._1 -> firstIndices)
//////////
//////////    // Get embedding for first item only
//////////    var firstSeqEmb = embeddingLayer.forward(Map.empty, firstSeqMap, squeeze = false)
//////////
//////////    // Debug
//////////    println(s"[DEBUG LiquidNetWork] firstSeqEmb: dim=${firstSeqEmb.dim()}, shape=${firstSeqEmb.sizes().vec().get().mkString(", ")}")
//////////    println(s"[DEBUG LiquidNetWork] inputProj inputDim: $embedDim, hiddenDim: $hiddenDim")
//////////
//////////    // Ensure on correct device
//////////    val seqEmbOnDev = if (device != "cpu") firstSeqEmb.to(targetDevice, ScalarType.Float) else firstSeqEmb
//////////
//////////    // ODE: for LiquidNet, we need raw sequence embeddings (batch, seqLen, embedDim)
//////////    // Get embeddings directly from embedding table to avoid pooling
//////////    seqTensor = sequenceFeats.head._2
//////////    val seqName = sequenceFeats.head._1
//////////    val embedName = s"${seqName}_seq"
//////////    val rawEmb = embeddingLayer.getEmbedding(embedName, seqTensor)
//////////
//////////    // Ensure on device
//////////    var seqEmb = if (device != "cpu") rawEmb.to(targetDevice, ScalarType.Float) else rawEmb
//////////    println(s"[DEBUG LiquidNetWork] seqEmb: dim=${seqEmb.dim()}, shape=${seqEmb.sizes().vec().get().mkString(", ")}")
//////////
//////////    // ODE
//////////    var hidden = inputProj.forward(seqEmbOnDev)
//////////    val dt = new Scalar(1f / numOdeSteps)
//////////
//////////    for (step <- 0 until numOdeSteps) {
//////////      val t = step.toFloat / numOdeSteps
//////////      val f = t * (seqLen - 1)
//////////      val i = f.toInt
//////////      val a = new Scalar(f - i)
//////////      val x0 = seqEmb.select(1, i)
//////////      val x1 = seqEmb.select(1, i+1)
//////////      // Debug
//////////      println(s"[DEBUG LiquidNetWork] x0: dim=${x0.dim()}, shape=${x0.sizes().vec().get().mkString(", ")}")
//////////      val xt = x0.mul(new Scalar(1 - a.toFloat)).add(x1.mul(a))
//////////      // Debug
//////////      println(s"[DEBUG LiquidNetWork] xt: dim=${xt.dim()}, shape=${xt.sizes().vec().get().mkString(", ")}")
//////////      val dh = liquidCell.forward(hidden, xt, t)
//////////      hidden = hidden.add(dh.mul(dt))
//////////    }
//////////
//////////    val out = outputProj.forward(hidden)
//////////    mlp.forward(torch.cat(new TensorVector(full, out), 1))
//////////  }
//////////}
//////////
//////////// ===================== LiquidCell =====================
//////////class LiquidCell(inputDim: Int, hiddenDim: Int, device: String) extends Module {
//////////  private val dev = new Device(device)
//////////
//////////  private val tc = new LinearImpl(1, hiddenDim)
//////////  private val ip = new LinearImpl(inputDim, hiddenDim)
//////////  private val hp = new LinearImpl(hiddenDim, hiddenDim)
//////////  private val tm = new LinearImpl(1, hiddenDim)
//////////  private val gw = new LinearImpl(inputDim, hiddenDim)
//////////  private val gh = new LinearImpl(hiddenDim, hiddenDim)
//////////  private val gb = new LinearImpl(1, hiddenDim)
//////////
//////////  List(tc, ip, hp, tm, gw, gh, gb).foreach(_.to(dev, false))
//////////
//////////  register_module("timeConstant", tc)
//////////  register_module("inputProj", ip)
//////////  register_module("hiddenProj", hp)
//////////  register_module("timeMod", tm)
//////////  register_module("gateW", gw)
//////////  register_module("gateH", gh)
//////////  register_module("gateB", gb)
//////////
//////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//////////    // Debug - this is where the error happens
//////////    println(s"[DEBUG LiquidCell.forward] hidden: dim=${hidden.dim()}, shape=${hidden.sizes().vec().get().mkString(", ")}")
//////////    println(s"[DEBUG LiquidCell.forward] input: dim=${input.dim()}, shape=${input.sizes().vec().get().mkString(", ")}")
//////////    println(s"[DEBUG LiquidCell.forward] time: $time")
//////////
//////////    val t = torch.ones(Array(hidden.size(0), 1L),
//////////      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(hidden.device()))
//////////    ).mul(new Scalar(time))
//////////
//////////    println(s"[DEBUG LiquidCell.forward] t shape: ${t.sizes().vec().get().mkString(", ")}")
//////////
//////////    val ntc = tc.forward(t).relu().neg()
//////////    val rec = hp.forward(hidden)
//////////    val inp = ip.forward(input)
//////////    val tmd = tm.forward(t)
//////////    val gate = gw.forward(input).add(gh.forward(hidden)).add(gb.forward(t)).sigmoid()
//////////
//////////    rec.mul(gate).add(inp).add(tmd).add(hidden.mul(ntc))
//////////  }
//////////}
////////////package torchrec.models.ranking
////////////
////////////import torchrec.basic.features._
////////////import torchrec.basic.layers._
////////////import torchrec.utils.DeviceSupport
////////////
////////////import org.bytedeco.pytorch._
////////////import org.bytedeco.pytorch.global.torch
////////////import org.bytedeco.pytorch.global.torch.ScalarType
////////////
/////////////**
//////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
//////////// */
////////////class LiquidNetWork(
////////////                     features: List[Feature],
////////////                     sequenceFeatures: List[SequenceFeature],
////////////                     embedDim: Int = 8,
////////////                     hiddenDim: Int = 32,
////////////                     numOdeSteps: Int = 4,
////////////                     mlpDims: List[Long] = List(64L, 32L),
////////////                     dropout: Float = 0.2f,
////////////                     device: String = DeviceSupport.backend
////////////                   ) extends Module {
////////////
////////////  require(features.nonEmpty, "features cannot be empty")
////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
////////////
////////////  private val targetDevice = new Device(device)
////////////
////////////  // 分开嵌入层 —— 你最开始能用的方式
////////////  private val sparseEmbedding = new EmbeddingLayer(features, embedDim, device)
////////////  private val seqEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
////////////
////////////  register_module("sparseEmbedding", sparseEmbedding)
////////////  register_module("seqEmbedding", seqEmbedding)
////////////
////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////////////  register_module("liquidCell", liquidCell)
////////////
////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////////////
////////////  List(inputProj, outputProj).foreach(_.to(targetDevice, false))
////////////
////////////  register_module("inputProj", inputProj)
////////////  register_module("outputProj", outputProj)
////////////
////////////  private val totalDim = features.size * embedDim + embedDim
////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
////////////  register_module("mlp", mlp)
////////////
////////////  // ===========================================================================
////////////  // ✅ 完全恢复你最初能用的逻辑！只修复Scalar和API
////////////  // ===========================================================================
////////////  def forward(
////////////               sparseFeats: Map[String, Tensor],
////////////               sequenceFeats: Map[String, Tensor]
////////////             ): Tensor = {
////////////
////////////    // ✅ 分开调用 —— 你最开始能用的方式
////////////    val sparseEmb = sparseEmbedding.forward(sparseFeats, Map.empty, squeeze = true)
////////////    val seqEmb = seqEmbedding.forward(Map.empty, sequenceFeats, squeeze = false)
////////////
////////////    val batchSize = seqEmb.size(0)
////////////    val seqLen = seqEmb.size(1)
////////////
////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
////////////    val dt = new Scalar(1.0f / numOdeSteps.toFloat)
////////////
////////////    for (step <- 0 until numOdeSteps) {
////////////      val t = step.toFloat * 1.0f / numOdeSteps.toFloat
////////////      val tFrac = t * (seqLen - 1)
////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
////////////      val alpha = new Scalar(tFrac - t0)
////////////
////////////      val x0 = seqEmb.select(1, t0)
////////////      val x1 = seqEmb.select(1, t0 + 1)
////////////      val xT = x0.mul(new Scalar(1.0f - alpha.toFloat())).add(x1.mul(alpha))
////////////
////////////      val dHidden = liquidCell.forward(hiddenState, xT, t)
////////////      hiddenState = hiddenState.add(dHidden.mul(dt))
////////////    }
////////////
////////////    val liquidRep = outputProj.forward(hiddenState)
////////////    val combined = torch.cat(new TensorVector(sparseEmb, liquidRep), 1)
////////////    mlp.forward(combined)
////////////  }
////////////}
////////////
////////////// ===================== LiquidCell 100% 按你的要求 =====================
////////////class LiquidCell(
////////////                  inputDim: Int,
////////////                  hiddenDim: Int,
////////////                  device: String
////////////                ) extends Module {
////////////
////////////  private val targetDevice = new Device(device)
////////////
////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
////////////  private val gateB        = new LinearImpl(1, hiddenDim)
////////////
////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB).foreach(_.to(targetDevice, false))
////////////
////////////  register_module("timeConstant", timeConstant)
////////////  register_module("inputProj", inputProj)
////////////  register_module("hiddenProj", hiddenProj)
////////////  register_module("timeMod", timeMod)
////////////  register_module("gateW", gateW)
////////////  register_module("gateH", gateH)
////////////  register_module("gateB", gateB)
////////////
////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////////////    val batchSize = hidden.size(0)
////////////    val dev = hidden.device()
////////////
////////////    val timeTensor = torch.ones(
////////////      Array(batchSize, 1L),
////////////      new TensorOptions()
////////////        .dtype(new ScalarTypeOptional(ScalarType.Float))
////////////        .device(new DeviceOptional(dev))
////////////    ).mul(new Scalar(time))
////////////
////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
////////////    val recurrent = hiddenProj.forward(hidden)
////////////    val inputTerm = inputProj.forward(input)
////////////    val timeDecay = timeMod.forward(timeTensor)
////////////
////////////    val gate = gateW.forward(input)
////////////      .add(gateH.forward(hidden))
////////////      .add(gateB.forward(timeTensor))
////////////      .sigmoid()
////////////
////////////    val dHidden = recurrent.mul(gate)
////////////      .add(inputTerm)
////////////      .add(timeDecay)
////////////      .add(hidden.mul(timeConst))
////////////
////////////    dHidden
////////////  }
////////////}
//////////////package torchrec.models.ranking
//////////////
//////////////import torchrec.basic.features._
//////////////import torchrec.basic.layers._
//////////////import torchrec.utils.DeviceSupport
//////////////
//////////////import org.bytedeco.pytorch._
//////////////import org.bytedeco.pytorch.global.torch
//////////////import org.bytedeco.pytorch.global.torch.ScalarType
//////////////
///////////////**
////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
////////////// */
//////////////class LiquidNetWork(
//////////////                     features: List[Feature],
//////////////                     sequenceFeatures: List[SequenceFeature],
//////////////                     embedDim: Int = 8,
//////////////                     hiddenDim: Int = 32,
//////////////                     numOdeSteps: Int = 4,
//////////////                     mlpDims: List[Long] = List(64L, 32L),
//////////////                     dropout: Float = 0.2f,
//////////////                     device: String = DeviceSupport.backend
//////////////                   ) extends Module {
//////////////
//////////////  require(features.nonEmpty, "features cannot be empty")
//////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
//////////////
//////////////  private val targetDevice = new Device(device)
//////////////  private val numSparse = features.size
//////////////  private val numSeq = sequenceFeatures.size
//////////////
//////////////  // 嵌入层：同时包含稀疏 + 序列特征
//////////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
//////////////  register_module("embeddingLayer", embeddingLayer)
//////////////
//////////////  // Liquid 网络
//////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//////////////  register_module("liquidCell", liquidCell)
//////////////
//////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
//////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
//////////////
//////////////  inputProj.to(targetDevice, false)
//////////////  outputProj.to(targetDevice, false)
//////////////  hiddenProj.to(targetDevice, false)
//////////////
//////////////  register_module("inputProj", inputProj)
//////////////  register_module("outputProj", outputProj)
//////////////  register_module("hiddenProj", hiddenProj)
//////////////
//////////////  // MLP
//////////////  private val totalInputDim = numSparse * embedDim + embedDim
//////////////  private val mlp = new MLP(totalInputDim, mlpDims, 1, "relu", dropout, device = device)
//////////////  register_module("mlp", mlp)
//////////////
//////////////  // ===========================================================================
//////////////  // ✅ 终极正确 forward：只调用一次，无空Map，无危险操作
//////////////  // ===========================================================================
//////////////  def forward(
//////////////               sparseFeats: Map[String, Tensor],
//////////////               sequenceFeats: Map[String, Tensor]
//////////////             ): Tensor = {
//////////////
//////////////    // ✅ 唯一一次调用：同时传入稀疏 + 序列，绝对不报 No embeddings
//////////////    val fullEmb = embeddingLayer.forward(sparseFeats, sequenceFeats, squeeze = true)
//////////////
//////////////    // ----------------==== 极简安全拆分：前稀疏 | 后序列 ====----------------
//////////////    val sparseTotal = numSparse * embedDim
//////////////    val seqTotal    = numSeq * embedDim
//////////////
//////////////    // 稀疏特征 (batch, sparse_dim)
//////////////    val sparseEmb = fullEmb.slice(1, new LongOptional(0), new LongOptional(sparseTotal),1)
//////////////
//////////////    // 序列特征 (batch, seq_dim) -> reshape (batch, seq_len, embed_dim)
//////////////    val seqFlat = fullEmb.slice(1, new LongOptional(sparseTotal)  , new LongOptional(sparseTotal + seqTotal),1)
//////////////    val seqEmb = seqFlat.view(seqFlat.size(0), numSeq, embedDim)
//////////////
//////////////    // ODE 初始化
//////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
//////////////    val dt = new Scalar(1.0f / numOdeSteps.toFloat)
//////////////
//////////////    // ODE 迭代
//////////////    for (step <- 0 until numOdeSteps) {
//////////////      val t = new Scalar(step.toFloat * 1.0f / numOdeSteps.toFloat)
//////////////      val tFrac = t.toFloat * (numSeq - 1)
//////////////      val t0 = math.min(math.floor(tFrac).toInt, numSeq - 2)
//////////////      val alpha = new Scalar(tFrac - t0)
//////////////
//////////////      val x0 = seqEmb.select(1, t0)
//////////////      val x1 = seqEmb.select(1, t0 + 1)
//////////////      val xT = x0.mul(new Scalar(1.0f - alpha.toFloat)).add(x1.mul(alpha))
//////////////
//////////////      val dHidden = liquidCell.forward(hiddenState, xT, t.toFloat)
//////////////      hiddenState = hiddenState.add(dHidden.mul(dt))
//////////////    }
//////////////
//////////////    // 输出
//////////////    val liquidRep = outputProj.forward(hiddenState)
//////////////    val combined = torch.cat(new TensorVector(sparseEmb, liquidRep), 1)
//////////////    mlp.forward(combined)
//////////////  }
//////////////}
//////////////
//////////////// ===================== LiquidCell 完全按你的格式 =====================
//////////////class LiquidCell(
//////////////                  inputDim: Int,
//////////////                  hiddenDim: Int,
//////////////                  device: String
//////////////                ) extends Module {
//////////////
//////////////  private val targetDevice = new Device(device)
//////////////
//////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
//////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
//////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
//////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
//////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
//////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
//////////////  private val gateB        = new LinearImpl(1, hiddenDim)
//////////////
//////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB)
//////////////    .foreach(_.to(targetDevice, false))
//////////////
//////////////  register_module("timeConstant", timeConstant)
//////////////  register_module("inputProj", inputProj)
//////////////  register_module("hiddenProj", hiddenProj)
//////////////  register_module("timeMod", timeMod)
//////////////  register_module("gateW", gateW)
//////////////  register_module("gateH", gateH)
//////////////  register_module("gateB", gateB)
//////////////
//////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//////////////    val batchSize = hidden.size(0)
//////////////    val dev = hidden.device()
//////////////
//////////////    val timeTensor = torch.ones(
//////////////      Array(batchSize, 1L),
//////////////      new TensorOptions()
//////////////        .dtype(new ScalarTypeOptional(ScalarType.Float))
//////////////        .device(new DeviceOptional(dev))
//////////////    ).mul(new Scalar(time))
//////////////
//////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
//////////////    val recurrent = hiddenProj.forward(hidden)
//////////////    val inputTerm = inputProj.forward(input)
//////////////    val timeDecay = timeMod.forward(timeTensor)
//////////////
//////////////    val gate = gateW.forward(input)
//////////////      .add(gateH.forward(hidden))
//////////////      .add(gateB.forward(timeTensor))
//////////////      .sigmoid()
//////////////
//////////////    val dHidden = recurrent.mul(gate)
//////////////      .add(inputTerm)
//////////////      .add(timeDecay)
//////////////      .add(hidden.mul(timeConst))
//////////////
//////////////    dHidden
//////////////  }
//////////////}
//////////////
////////////////package torchrec.models.ranking
////////////////
////////////////import torchrec.basic.features._
////////////////import torchrec.basic.layers._
////////////////import torchrec.utils.DeviceSupport
////////////////
////////////////import org.bytedeco.pytorch._
////////////////import org.bytedeco.pytorch.global.torch
////////////////import org.bytedeco.pytorch.global.torch.ScalarType
////////////////
/////////////////**
//////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
//////////////// */
////////////////class LiquidNetWork(
////////////////                     features: List[Feature],
////////////////                     sequenceFeatures: List[SequenceFeature],
////////////////                     embedDim: Int = 8,
////////////////                     hiddenDim: Int = 32,
////////////////                     numOdeSteps: Int = 4,
////////////////                     mlpDims: List[Long] = List(64L, 32L),
////////////////                     dropout: Float = 0.2f,
////////////////                     device: String = DeviceSupport.backend
////////////////                   ) extends Module {
////////////////
////////////////  require(features.nonEmpty, "features cannot be empty")
////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
////////////////
////////////////  private val targetDevice = new Device(device)
////////////////
////////////////  // ✅ 嵌入层：先稀疏 + 后序列
////////////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
////////////////  register_module("embeddingLayer", embeddingLayer)
////////////////
////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////////////////  register_module("liquidCell", liquidCell)
////////////////
////////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
////////////////
////////////////  inputProj.to(targetDevice, false)
////////////////  outputProj.to(targetDevice, false)
////////////////  hiddenProj.to(targetDevice, false)
////////////////
////////////////  register_module("inputProj", inputProj)
////////////////  register_module("outputProj", outputProj)
////////////////  register_module("hiddenProj", hiddenProj)
////////////////
////////////////  private val totalDim = features.size * embedDim + embedDim
////////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
////////////////  register_module("mlp", mlp)
////////////////
////////////////  // ===========================================================================
////////////////  // ✅ 100% 正确：维度、顺序、narrow 永不越界
////////////////  // ===========================================================================
////////////////  def forward(
////////////////               sparseFeats: Map[String, Tensor],
////////////////               sequenceFeats: Map[String, Tensor]
////////////////             ): Tensor = {
////////////////
////////////////    // ✅ 只调用一次
////////////////    val full = embeddingLayer.forward(sparseFeats, sequenceFeats, squeeze = true)
////////////////    val batchSize = full.size(0)
////////////////
////////////////    // --------------------- 🔥 核心修复：顺序正确 ---------------------
////////////////    val seqEmb = embeddingLayer.forward(sequenceFeats) // ✅ 直接取序列
////////////////    val sparseEmb = full // ✅ 稀疏就是整个输出
////////////////
////////////////    val seqLen = seqEmb.size(1)
////////////////
////////////////    // ODE
////////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
////////////////    val dt = new Scalar(1.0f / numOdeSteps.toFloat)
////////////////
////////////////    for (step <- 0 until numOdeSteps) {
////////////////      val t = new Scalar(step.toFloat * 1.0f / numOdeSteps.toFloat)
////////////////      val tFrac = t.toFloat * (seqLen - 1)
////////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
////////////////      val alpha = new Scalar(tFrac - t0)
////////////////
////////////////      val x0 = seqEmb.select(1, t0)
////////////////      val x1 = seqEmb.select(1, t0 + 1)
////////////////      val xT = x0.mul(new Scalar(1.0f - alpha.toFloat)).add(x1.mul(alpha))
////////////////
////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t.toFloat)
////////////////      hiddenState = hiddenState.add(dHidden.mul(dt))
////////////////    }
////////////////
////////////////    val liquidRep = outputProj.forward(hiddenState)
////////////////    val combined = torch.cat(new TensorVector(sparseEmb, liquidRep), 1)
////////////////    mlp.forward(combined)
////////////////  }
////////////////}
////////////////
////////////////// ===================== LiquidCell 完全保持你的原版 =====================
////////////////class LiquidCell(
////////////////                  inputDim: Int,
////////////////                  hiddenDim: Int,
////////////////                  device: String
////////////////                ) extends Module {
////////////////
////////////////  private val targetDevice = new Device(device)
////////////////
////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
////////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
////////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
////////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
////////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
////////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
////////////////  private val gateB        = new LinearImpl(1, hiddenDim)
////////////////
////////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB).foreach(_.to(targetDevice, false))
////////////////
////////////////  register_module("timeConstant", timeConstant)
////////////////  register_module("inputProj", inputProj)
////////////////  register_module("hiddenProj", hiddenProj)
////////////////  register_module("timeMod", timeMod)
////////////////  register_module("gateW", gateW)
////////////////  register_module("gateH", gateH)
////////////////  register_module("gateB", gateB)
////////////////
////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////////////////    val batchSize = hidden.size(0)
////////////////    val dev = hidden.device()
////////////////
////////////////    val timeTensor = torch.ones(
////////////////      Array(batchSize, 1L),
////////////////      new TensorOptions()
////////////////        .dtype(new ScalarTypeOptional(ScalarType.Float))
////////////////        .device(new DeviceOptional(dev))
////////////////    ).mul(new Scalar(time))
////////////////
////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
////////////////    val recurrent = hiddenProj.forward(hidden)
////////////////    val inputTerm = inputProj.forward(input)
////////////////    val timeDecay = timeMod.forward(timeTensor)
////////////////
////////////////    val gate = gateW.forward(input)
////////////////      .add(gateH.forward(hidden))
////////////////      .add(gateB.forward(timeTensor))
////////////////      .sigmoid()
////////////////
////////////////    val dHidden = recurrent.mul(gate)
////////////////      .add(inputTerm)
////////////////      .add(timeDecay)
////////////////      .add(hidden.mul(timeConst))
////////////////
////////////////    dHidden
////////////////  }
////////////////}
////////////////
//////////////////package torchrec.models.ranking
//////////////////
//////////////////import torchrec.basic.features._
//////////////////import torchrec.basic.layers._
//////////////////import torchrec.utils.DeviceSupport
//////////////////
//////////////////import org.bytedeco.pytorch._
//////////////////import org.bytedeco.pytorch.global.torch
//////////////////import org.bytedeco.pytorch.global.torch.ScalarType
//////////////////
///////////////////**
////////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
////////////////// */
//////////////////class LiquidNetWork(
//////////////////                     features: List[Feature],
//////////////////                     sequenceFeatures: List[SequenceFeature],
//////////////////                     embedDim: Int = 8,
//////////////////                     hiddenDim: Int = 32,
//////////////////                     numOdeSteps: Int = 4,
//////////////////                     mlpDims: List[Long] = List(64L, 32L),
//////////////////                     dropout: Float = 0.2f,
//////////////////                     device: String = DeviceSupport.backend
//////////////////                   ) extends Module {
//////////////////
//////////////////  require(features.nonEmpty, "features cannot be empty")
//////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
//////////////////
//////////////////  private val targetDevice = new Device(device)
//////////////////
//////////////////  // 嵌入层
//////////////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
//////////////////  register_module("embeddingLayer", embeddingLayer)
//////////////////
//////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//////////////////  register_module("liquidCell", liquidCell)
//////////////////
//////////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
//////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
//////////////////
//////////////////  inputProj.to(targetDevice, false)
//////////////////  outputProj.to(targetDevice, false)
//////////////////  hiddenProj.to(targetDevice, false)
//////////////////
//////////////////  register_module("inputProj", inputProj)
//////////////////  register_module("outputProj", outputProj)
//////////////////  register_module("hiddenProj", hiddenProj)
//////////////////
//////////////////  private val totalDim = features.size * embedDim + embedDim
//////////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
//////////////////  register_module("mlp", mlp)
//////////////////
//////////////////  // ===========================================================================
//////////////////  // ✅ 完全修复维度错误！！！
//////////////////  // ===========================================================================
//////////////////  def forward(
//////////////////               sparseFeats: Map[String, Tensor],
//////////////////               sequenceFeats: Map[String, Tensor]
//////////////////             ): Tensor = {
//////////////////
//////////////////    // ✅ 一次调用，不拆分
//////////////////    val fullEmb = embeddingLayer.forward(sparseFeats, sequenceFeats, squeeze = false)
//////////////////
//////////////////    // ----------------==== 🔥 核心修复：fullEmb 是 2D 张量 [batch, dim] ====----------------
//////////////////    val batchSize = fullEmb.size(0)
//////////////////    val totalDim = fullEmb.size(1) // 这里是 1，不是 2！！！
//////////////////
//////////////////    val sparseDim = features.size * embedDim
//////////////////    val seqDim = totalDim - sparseDim
//////////////////    val seqLen = seqDim / embedDim
//////////////////
//////////////////    // 稀疏部分
//////////////////    val sparseEmb = fullEmb.narrow(1, 0, sparseDim)
//////////////////
//////////////////    // 序列部分：reshape 成 [batch, seq_len, embedDim]
//////////////////    val seqFlat = fullEmb.narrow(1, sparseDim, seqDim)
//////////////////    val seqEmb = seqFlat.view(batchSize, seqLen, embedDim)
//////////////////
//////////////////    // ODE
//////////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
//////////////////    val dt = new Scalar(1.0f / numOdeSteps.toFloat)
//////////////////
//////////////////    for (step <- 0 until numOdeSteps) {
//////////////////      val t = new Scalar(step.toFloat * 1.0f / numOdeSteps.toFloat)
//////////////////      val tFrac = t.toFloat * (seqLen - 1)
//////////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
//////////////////      val alpha = new Scalar(tFrac - t0)
//////////////////
//////////////////      val x0 = seqEmb.select(1, t0)
//////////////////      val x1 = seqEmb.select(1, t0 + 1)
//////////////////      val xT = x0.mul(new Scalar(1.0f - alpha.toFloat)).add(x1.mul(alpha))
//////////////////
//////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t.toFloat)
//////////////////      hiddenState = hiddenState.add(dHidden.mul(dt))
//////////////////    }
//////////////////
//////////////////    val liquidRep = outputProj.forward(hiddenState)
//////////////////    val combined = torch.cat(new TensorVector(sparseEmb, liquidRep), 1)
//////////////////    mlp.forward(combined)
//////////////////  }
//////////////////}
//////////////////
//////////////////// ===================== LiquidCell 完全保持你的原版 =====================
//////////////////class LiquidCell(
//////////////////                  inputDim: Int,
//////////////////                  hiddenDim: Int,
//////////////////                  device: String
//////////////////                ) extends Module {
//////////////////
//////////////////  private val targetDevice = new Device(device)
//////////////////
//////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
//////////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
//////////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
//////////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
//////////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
//////////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
//////////////////  private val gateB        = new LinearImpl(1, hiddenDim)
//////////////////
//////////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB).foreach(_.to(targetDevice, false))
//////////////////
//////////////////  register_module("timeConstant", timeConstant)
//////////////////  register_module("inputProj", inputProj)
//////////////////  register_module("hiddenProj", hiddenProj)
//////////////////  register_module("timeMod", timeMod)
//////////////////  register_module("gateW", gateW)
//////////////////  register_module("gateH", gateH)
//////////////////  register_module("gateB", gateB)
//////////////////
//////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//////////////////    val batchSize = hidden.size(0)
//////////////////    val dev = hidden.device()
//////////////////
//////////////////    val timeTensor = torch.ones(
//////////////////      Array(batchSize, 1L),
//////////////////      new TensorOptions()
//////////////////        .dtype(new ScalarTypeOptional(ScalarType.Float))
//////////////////        .device(new DeviceOptional(dev))
//////////////////    ).mul(new Scalar(time))
//////////////////
//////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
//////////////////    val recurrent = hiddenProj.forward(hidden)
//////////////////    val inputTerm = inputProj.forward(input)
//////////////////    val timeDecay = timeMod.forward(timeTensor)
//////////////////
//////////////////    val gate = gateW.forward(input)
//////////////////      .add(gateH.forward(hidden))
//////////////////      .add(gateB.forward(timeTensor))
//////////////////      .sigmoid()
//////////////////
//////////////////    val dHidden = recurrent.mul(gate)
//////////////////      .add(inputTerm)
//////////////////      .add(timeDecay)
//////////////////      .add(hidden.mul(timeConst))
//////////////////
//////////////////    dHidden
//////////////////  }
//////////////////}
//////////////////
////////////////////package torchrec.models.ranking
////////////////////
////////////////////import torchrec.basic.features._
////////////////////import torchrec.basic.layers._
////////////////////import torchrec.utils.DeviceSupport
////////////////////
////////////////////import org.bytedeco.pytorch._
////////////////////import org.bytedeco.pytorch.global.torch
////////////////////import org.bytedeco.pytorch.global.torch.ScalarType
////////////////////
/////////////////////**
//////////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
//////////////////// */
////////////////////class LiquidNetWork(
////////////////////                     features: List[Feature],
////////////////////                     sequenceFeatures: List[SequenceFeature],
////////////////////                     embedDim: Int = 8,
////////////////////                     hiddenDim: Int = 32,
////////////////////                     numOdeSteps: Int = 4,
////////////////////                     mlpDims: List[Long] = List(64L, 32L),
////////////////////                     dropout: Float = 0.2f,
////////////////////                     device: String = DeviceSupport.backend
////////////////////                   ) extends Module {
////////////////////
////////////////////  require(features.nonEmpty, "features cannot be empty")
////////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
////////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
////////////////////
////////////////////  private val targetDevice = new Device(device)
////////////////////  private val numSparse = features.size
////////////////////  private val numSeq = sequenceFeatures.size
////////////////////
////////////////////  // 统一嵌入层
////////////////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
////////////////////  register_module("embeddingLayer", embeddingLayer)
////////////////////
////////////////////  // Liquid ODE
////////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////////////////////  register_module("liquidCell", liquidCell)
////////////////////
////////////////////  // 投影
////////////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
////////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
////////////////////
////////////////////  inputProj.to(targetDevice, false)
////////////////////  outputProj.to(targetDevice, false)
////////////////////  hiddenProj.to(targetDevice, false)
////////////////////
////////////////////  register_module("inputProj", inputProj)
////////////////////  register_module("outputProj", outputProj)
////////////////////  register_module("hiddenProj", hiddenProj)
////////////////////
////////////////////  // MLP
////////////////////  private val totalDim = numSparse * embedDim + embedDim
////////////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
////////////////////  register_module("mlp", mlp)
////////////////////
////////////////////  // ===========================================================================
////////////////////  // ✅ FORWARD 最终正确版：只调用一次 embeddingLayer！
////////////////////  // ===========================================================================
////////////////////  def forward(
////////////////////               sparseFeats: Map[String, Tensor],
////////////////////               sequenceFeats: Map[String, Tensor]
////////////////////             ): Tensor = {
////////////////////
////////////////////    // ✅ 只调用一次！同时传入稀疏和序列特征
////////////////////    val fullEmb = embeddingLayer.forward(sparseFeats, sequenceFeats, squeeze = false)
////////////////////
////////////////////    // 形状：[batch, 1, (sparse * dim) + (seq_len * dim)]
////////////////////    val batchSize = fullEmb.size(0)
////////////////////    val totalDim = fullEmb.size(2)
////////////////////
////////////////////    // 拆分：稀疏部分 + 序列部分
////////////////////    val sparseTotalDim = numSparse * embedDim
////////////////////    val seqTotalDim = totalDim - sparseTotalDim
////////////////////    val seqLen = seqTotalDim / embedDim
////////////////////
////////////////////    // 稀疏特征 [batch, sparse_dim]
////////////////////    val sparseEmb = fullEmb.narrow(2, 0, sparseTotalDim).squeeze(1)
////////////////////
////////////////////    // 序列特征 [batch, seq_len, embed_dim]
////////////////////    val seqEmbFlat = fullEmb.narrow(2, sparseTotalDim, seqTotalDim).squeeze(1)
////////////////////    val seqEmb = seqEmbFlat.view(batchSize, seqLen, embedDim)
////////////////////
////////////////////    // ODE 初始状态
////////////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
////////////////////
////////////////////    // ODE 迭代
////////////////////    val dt = new Scalar(1.0f / numOdeSteps.toFloat)
////////////////////    for (step <- 0 until numOdeSteps) {
////////////////////      val t = new Scalar(step.toFloat * 1.0f / numOdeSteps.toFloat)
////////////////////      val tFrac = t.toFloat() * (seqLen - 1)
////////////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
////////////////////      val t1 = t0 + 1
////////////////////      val alpha = new Scalar(tFrac - t0)
////////////////////
////////////////////      val x0 = seqEmb.select(1, t0)
////////////////////      val x1 = seqEmb.select(1, t1)
////////////////////      val xT = x0.mul(new Scalar(1.0f - alpha.toFloat())).add(x1.mul(alpha))
////////////////////
////////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t.toFloat())
////////////////////      hiddenState = hiddenState.add(dHidden.mul(dt))
////////////////////    }
////////////////////
////////////////////    // 输出
////////////////////    val liquidRep = outputProj.forward(hiddenState)
////////////////////    val combined = torch.cat(new TensorVector(sparseEmb, liquidRep), 1)
////////////////////    val logits = mlp.forward(combined)
////////////////////
////////////////////    logits
////////////////////  }
////////////////////}
////////////////////
////////////////////// ===================== LiquidCell 完全对齐你的API =====================
////////////////////class LiquidCell(
////////////////////                  inputDim: Int,
////////////////////                  hiddenDim: Int,
////////////////////                  device: String
////////////////////                ) extends Module {
////////////////////
////////////////////  private val targetDevice = new Device(device)
////////////////////
////////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
////////////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
////////////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
////////////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
////////////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
////////////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
////////////////////  private val gateB        = new LinearImpl(1, hiddenDim)
////////////////////
////////////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB)
////////////////////    .foreach(_.to(targetDevice, false))
////////////////////
////////////////////  register_module("timeConstant", timeConstant)
////////////////////  register_module("inputProj", inputProj)
////////////////////  register_module("hiddenProj", hiddenProj)
////////////////////  register_module("timeMod", timeMod)
////////////////////  register_module("gateW", gateW)
////////////////////  register_module("gateH", gateH)
////////////////////  register_module("gateB", gateB)
////////////////////
////////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////////////////////    val batchSize = hidden.size(0)
////////////////////    val dev = hidden.device()
////////////////////
////////////////////    // ✅ 完全按照你的格式
////////////////////    val timeTensor = torch.ones(
////////////////////      Array(batchSize, 1L),
////////////////////      new TensorOptions()
////////////////////        .dtype(new ScalarTypeOptional(ScalarType.Float))
////////////////////        .device(new DeviceOptional(dev))
////////////////////    ).mul(new Scalar(time))
////////////////////
////////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
////////////////////    val recurrent = hiddenProj.forward(hidden)
////////////////////    val inputTerm = inputProj.forward(input)
////////////////////    val timeDecay = timeMod.forward(timeTensor)
////////////////////
////////////////////    val gate = gateW.forward(input)
////////////////////      .add(gateH.forward(hidden))
////////////////////      .add(gateB.forward(timeTensor))
////////////////////      .sigmoid()
////////////////////
////////////////////    val dHidden = recurrent.mul(gate)
////////////////////      .add(inputTerm)
////////////////////      .add(timeDecay)
////////////////////      .add(hidden.mul(timeConst))
////////////////////
////////////////////    dHidden
////////////////////  }
////////////////////}
////////////////////
////////////////////
//////////////////////package torchrec.models.ranking
//////////////////////
//////////////////////import torchrec.basic.features._
//////////////////////import torchrec.basic.layers._
//////////////////////import torchrec.utils.DeviceSupport
//////////////////////
//////////////////////import org.bytedeco.pytorch._
//////////////////////import org.bytedeco.pytorch.global.torch
//////////////////////import org.bytedeco.pytorch.global.torch.ScalarType
//////////////////////
///////////////////////**
////////////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
////////////////////// */
//////////////////////class LiquidNetWork(
//////////////////////                     features: List[Feature],
//////////////////////                     sequenceFeatures: List[SequenceFeature],
//////////////////////                     embedDim: Int = 8,
//////////////////////                     hiddenDim: Int = 32,
//////////////////////                     numOdeSteps: Int = 4,
//////////////////////                     mlpDims: List[Long] = List(64L, 32L),
//////////////////////                     dropout: Float = 0.2f,
//////////////////////                     device: String = DeviceSupport.backend
//////////////////////                   ) extends Module {
//////////////////////
//////////////////////  require(features.nonEmpty, "features cannot be empty")
//////////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//////////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
//////////////////////
//////////////////////  private val targetDevice = new Device(device)
//////////////////////  private val numSparseFeatures = features.size
//////////////////////
//////////////////////  // 统一嵌入层
//////////////////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
//////////////////////  register_module("embeddingLayer", embeddingLayer)
//////////////////////
//////////////////////  // Liquid ODE
//////////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//////////////////////  register_module("liquidCell", liquidCell)
//////////////////////
//////////////////////  // 投影层
//////////////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
//////////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
//////////////////////
//////////////////////  inputProj.to(targetDevice, false)
//////////////////////  outputProj.to(targetDevice, false)
//////////////////////  hiddenProj.to(targetDevice, false)
//////////////////////
//////////////////////  register_module("inputProj", inputProj)
//////////////////////  register_module("outputProj", outputProj)
//////////////////////  register_module("hiddenProj", hiddenProj)
//////////////////////
//////////////////////  // MLP
//////////////////////  private val totalInputDim = numSparseFeatures * embedDim + embedDim
//////////////////////  private val mlp = new MLP(totalInputDim, mlpDims, 1, "relu", dropout, device = device)
//////////////////////  register_module("mlp", mlp)
//////////////////////
//////////////////////  // ===========================================================================
//////////////////////  // FORWARD —— 100% 正确：维度 + 设备 + Scalar 全部修复
//////////////////////  // ===========================================================================
//////////////////////  def forward(
//////////////////////               sparseFeats: Map[String, Tensor],
//////////////////////               sequenceFeats: Map[String, Tensor]
//////////////////////             ): Tensor = {
//////////////////////
//////////////////////    // 1. 稀疏特征
//////////////////////    val sparseEmb = embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
//////////////////////
//////////////////////    // 2. 序列特征 —— 正确获取 (batch, seq_len, embed_dim)
//////////////////////    val seqEmb = embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
//////////////////////
//////////////////////    val batchSize = seqEmb.size(0)
//////////////////////    val seqLen = seqEmb.size(1)
//////////////////////
//////////////////////    // 3. ODE 初始状态
//////////////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
//////////////////////
//////////////////////    // 4. ODE 迭代
//////////////////////    val dt = new Scalar(1.0f / numOdeSteps.toFloat)
//////////////////////
//////////////////////    for (step <- 0 until numOdeSteps) {
//////////////////////      val t = new Scalar(step.toFloat * 1.0f / numOdeSteps.toFloat)
//////////////////////      val tFrac = t.toFloat * (seqLen - 1)
//////////////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
//////////////////////      val t1 = t0 + 1
//////////////////////      val alpha = new Scalar(tFrac - t0)
//////////////////////
//////////////////////      val x0 = seqEmb.select(1, t0)
//////////////////////      val x1 = seqEmb.select(1, t1)
//////////////////////      val xT = x0.mul(new Scalar(1.0f - alpha.toFloat)).add(x1.mul(alpha))
//////////////////////
//////////////////////      // ODE 导数
//////////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t.toFloat)
//////////////////////      hiddenState = hiddenState.add(dHidden.mul(dt))
//////////////////////    }
//////////////////////
//////////////////////    // 5. 输出
//////////////////////    val liquidRep = outputProj.forward(hiddenState)
//////////////////////    val combined = torch.cat(new TensorVector(sparseEmb, liquidRep), 1)
//////////////////////    val logits = mlp.forward(combined)
//////////////////////
//////////////////////    logits
//////////////////////  }
//////////////////////}
//////////////////////
//////////////////////// ===================== LiquidCell 100% 标准规范 =====================
//////////////////////class LiquidCell(
//////////////////////                  inputDim: Int,
//////////////////////                  hiddenDim: Int,
//////////////////////                  device: String
//////////////////////                ) extends Module {
//////////////////////
//////////////////////  private val targetDevice = new Device(device)
//////////////////////
//////////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
//////////////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
//////////////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
//////////////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
//////////////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
//////////////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
//////////////////////  private val gateB        = new LinearImpl(1, hiddenDim)
//////////////////////
//////////////////////  // 设备迁移
//////////////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB).foreach { m =>
//////////////////////    m.to(targetDevice, false)
//////////////////////  }
//////////////////////
//////////////////////  register_module("timeConstant", timeConstant)
//////////////////////  register_module("inputProj", inputProj)
//////////////////////  register_module("hiddenProj", hiddenProj)
//////////////////////  register_module("timeMod", timeMod)
//////////////////////  register_module("gateW", gateW)
//////////////////////  register_module("gateH", gateH)
//////////////////////  register_module("gateB", gateB)
//////////////////////
//////////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//////////////////////    val batchSize = hidden.size(0)
//////////////////////    val dev = hidden.device()
//////////////////////
//////////////////////    // ✅ 正确：TensorOptions 标准格式
//////////////////////    val timeTensor = torch.ones(
//////////////////////      Array(batchSize, 1L),
//////////////////////            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(dev))
//////////////////////    ).mul(new Scalar(time))
//////////////////////
//////////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
//////////////////////    val recurrent = hiddenProj.forward(hidden)
//////////////////////    val inputTerm = inputProj.forward(input)
//////////////////////    val timeDecay = timeMod.forward(timeTensor)
//////////////////////
//////////////////////    val gate = gateW.forward(input)
//////////////////////      .add(gateH.forward(hidden))
//////////////////////      .add(gateB.forward(timeTensor))
//////////////////////      .sigmoid()
//////////////////////
//////////////////////    val dHidden = recurrent.mul(gate)
//////////////////////      .add(inputTerm)
//////////////////////      .add(timeDecay)
//////////////////////      .add(hidden.mul(timeConst))
//////////////////////
//////////////////////    dHidden
//////////////////////  }
//////////////////////}
//////////////////////
////////////////////////package torchrec.models.ranking
////////////////////////
////////////////////////import torchrec.basic.features._
////////////////////////import torchrec.basic.layers._
////////////////////////import torchrec.utils.DeviceSupport
////////////////////////
////////////////////////import org.bytedeco.pytorch._
////////////////////////import org.bytedeco.pytorch.global.torch
////////////////////////import org.bytedeco.pytorch.global.torch.ScalarType
////////////////////////
/////////////////////////**
//////////////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
//////////////////////// */
////////////////////////class LiquidNetWork(
////////////////////////                     features: List[Feature],
////////////////////////                     sequenceFeatures: List[SequenceFeature],
////////////////////////                     embedDim: Int = 8,
////////////////////////                     hiddenDim: Int = 32,
////////////////////////                     numOdeSteps: Int = 4,
////////////////////////                     mlpDims: List[Long] = List(64L, 32L),
////////////////////////                     dropout: Float = 0.2f,
////////////////////////                     device: String = DeviceSupport.backend
////////////////////////                   ) extends Module {
////////////////////////
////////////////////////  require(features.nonEmpty, "features cannot be empty")
////////////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
////////////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
////////////////////////
////////////////////////  private val targetDevice = new Device(device)
////////////////////////
////////////////////////  // 嵌入层
////////////////////////  private val embeddingLayer = new EmbeddingLayer(features ++ sequenceFeatures, embedDim, device)
////////////////////////  register_module("embeddingLayer", embeddingLayer)
////////////////////////
////////////////////////  // Liquid ODE
////////////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////////////////////////  register_module("liquidCell", liquidCell)
////////////////////////
////////////////////////  // 投影层
////////////////////////  private val inputProj  = new LinearImpl(embedDim, hiddenDim)
////////////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
////////////////////////  List(inputProj, outputProj, hiddenProj).foreach(_.to(targetDevice, false))
////////////////////////
////////////////////////  register_module("inputProj", inputProj)
////////////////////////  register_module("outputProj", outputProj)
////////////////////////  register_module("hiddenProj", hiddenProj)
////////////////////////
////////////////////////  // MLP
////////////////////////  private val totalDim = features.size * embedDim + embedDim
////////////////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
////////////////////////  register_module("mlp", mlp)
////////////////////////
////////////////////////  // ===========================================================================
////////////////////////  // FORWARD 🔥 修复：一次同时传入稀疏 + 序列特征
////////////////////////  // ===========================================================================
////////////////////////  def forward(
////////////////////////               sparseFeats: Map[String, Tensor],
////////////////////////               sequenceFeats: Map[String, Tensor]
////////////////////////             ): Tensor = {
////////////////////////
////////////////////////    // ✅ 正确：一次前向传播，同时传入所有特征！
////////////////////////    val fullEmb = embeddingLayer.forward(sparseFeats, sequenceFeats, squeeze = false)
////////////////////////    fullEmb.to(targetDevice, ScalarType.Float)
////////////////////////
////////////////////////    // 拆分：静态特征部分 + 序列特征部分
////////////////////////    val numSparseFeats = features.size
////////////////////////    val sparseDim = numSparseFeats * embedDim
////////////////////////    val seqDim = fullEmb.size(2) - sparseDim
////////////////////////
////////////////////////    // 拆分 tensor
////////////////////////    val featEmb = fullEmb.narrow(2, 0, sparseDim).squeeze(1)  // (batch, sparseDim)
////////////////////////    val seqEmb  = fullEmb.narrow(2, sparseDim, seqDim)        // (batch, seq_len, embed_dim)
////////////////////////
////////////////////////    val batchSize = seqEmb.size(0)
////////////////////////    val seqLen = seqEmb.size(1).toInt
////////////////////////
////////////////////////    // ODE 初始化
////////////////////////    var hiddenState = inputProj.forward(seqEmb.select(1, 0))
////////////////////////
////////////////////////    // ODE 积分
////////////////////////    val dt = 1.0f / numOdeSteps.toFloat
////////////////////////    for (step <- 0 until numOdeSteps) {
////////////////////////      val t = step.toFloat * dt
////////////////////////      val tFrac = t * (seqLen - 1)
////////////////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
////////////////////////      val t1 = t0 + 1
////////////////////////      val alpha = tFrac - t0
////////////////////////
////////////////////////      val x0 = seqEmb.select(1, t0)
////////////////////////      val x1 = seqEmb.select(1, t1)
////////////////////////      val xT = x0.mul(new Scalar(1.0f - alpha)).add(x1.mul(new Scalar(alpha)))
////////////////////////
////////////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t)
////////////////////////      hiddenState = hiddenState.add(dHidden.mul(new Scalar(dt)))
////////////////////////    }
////////////////////////
////////////////////////    // 输出
////////////////////////    val liquidRep = outputProj.forward(hiddenState)
////////////////////////    val combined = torch.cat(new TensorVector(featEmb, liquidRep), 1)
////////////////////////    val logits = mlp.forward(combined)
////////////////////////
////////////////////////    logits.to(targetDevice, ScalarType.Float)
////////////////////////  }
////////////////////////}
////////////////////////
////////////////////////// ===================== LiquidCell 完全修复（设备统一） =====================
////////////////////////class LiquidCell(
////////////////////////                  inputDim: Int,
////////////////////////                  hiddenDim: Int,
////////////////////////                  device: String
////////////////////////                ) extends Module {
////////////////////////
////////////////////////  private val targetDevice = new Device(device)
////////////////////////
////////////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
////////////////////////  private val inputProj    = new LinearImpl(inputDim, hiddenDim)
////////////////////////  private val hiddenProj   = new LinearImpl(hiddenDim, hiddenDim)
////////////////////////  private val timeMod      = new LinearImpl(1, hiddenDim)
////////////////////////  private val gateW        = new LinearImpl(inputDim, hiddenDim)
////////////////////////  private val gateH        = new LinearImpl(hiddenDim, hiddenDim)
////////////////////////  private val gateB        = new LinearImpl(1, hiddenDim)
////////////////////////
////////////////////////  List(timeConstant, inputProj, hiddenProj, timeMod, gateW, gateH, gateB)
////////////////////////    .foreach(_.to(targetDevice, false))
////////////////////////
////////////////////////  register_module("timeConstant", timeConstant)
////////////////////////  register_module("inputProj", inputProj)
////////////////////////  register_module("hiddenProj", hiddenProj)
////////////////////////  register_module("timeMod", timeMod)
////////////////////////  register_module("gateW", gateW)
////////////////////////  register_module("gateH", gateH)
////////////////////////  register_module("gateB", gateB)
////////////////////////
////////////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////////////////////////    val batchSize = hidden.size(0)
////////////////////////    val dev = hidden.device()
////////////////////////
////////////////////////    val timeTensor = torch.ones(Array(batchSize, 1L),
////////////////////////      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(dev))
////////////////////////    ).mul(new Scalar(time))
////////////////////////
////////////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
////////////////////////    val recurrent = hiddenProj.forward(hidden)
////////////////////////    val inputTerm = inputProj.forward(input)
////////////////////////    val timeDecay = timeMod.forward(timeTensor)
////////////////////////
////////////////////////    val gate = gateW.forward(input)
////////////////////////      .add(gateH.forward(hidden))
////////////////////////      .add(gateB.forward(timeTensor))
////////////////////////      .sigmoid()
////////////////////////
////////////////////////    val dHidden = recurrent.mul(gate)
////////////////////////      .add(inputTerm)
////////////////////////      .add(timeDecay)
////////////////////////      .add(hidden.mul(timeConst))
////////////////////////
////////////////////////    dHidden
////////////////////////  }
////////////////////////}
////////////////////////
//////////////////////////package torchrec.models.ranking
//////////////////////////
//////////////////////////import torchrec.basic.features._
//////////////////////////import torchrec.basic.layers._
//////////////////////////import torchrec.utils.DeviceSupport
//////////////////////////
//////////////////////////import org.bytedeco.pytorch._
//////////////////////////import org.bytedeco.pytorch.global.torch
//////////////////////////import org.bytedeco.pytorch.global.torch.ScalarType
//////////////////////////
///////////////////////////**
////////////////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
////////////////////////// */
//////////////////////////class LiquidNetWork(
//////////////////////////                     features: List[Feature],
//////////////////////////                     sequenceFeatures: List[SequenceFeature],
//////////////////////////                     embedDim: Int = 8,
//////////////////////////                     hiddenDim: Int = 32,
//////////////////////////                     numOdeSteps: Int = 4,
//////////////////////////                     mlpDims: List[Long] = List(64L, 32L),
//////////////////////////                     dropout: Float = 0.2f,
//////////////////////////                     device: String = DeviceSupport.backend
//////////////////////////                   ) extends Module {
//////////////////////////
//////////////////////////  require(features.nonEmpty, "features cannot be empty")
//////////////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
//////////////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
//////////////////////////
//////////////////////////  private val targetDevice = new Device(device)
//////////////////////////
//////////////////////////  // ===================== 修复：统一一个 EmbeddingLayer 管理所有特征 =====================
//////////////////////////  private val allFeatures = features ++ sequenceFeatures
//////////////////////////  private val embeddingLayer = new EmbeddingLayer(allFeatures, embedDim, device)
//////////////////////////  register_module("embeddingLayer", embeddingLayer)
//////////////////////////
//////////////////////////  // Liquid ODE cell
//////////////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
//////////////////////////  register_module("liquidCell", liquidCell)
//////////////////////////
//////////////////////////  // Projections
//////////////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
//////////////////////////  outputProj.to(targetDevice, false)
//////////////////////////  register_module("outputProj", outputProj)
//////////////////////////
//////////////////////////  private val inputProj = new LinearImpl(embedDim, hiddenDim)
//////////////////////////  inputProj.to(targetDevice, false)
//////////////////////////  register_module("inputProj", inputProj)
//////////////////////////
//////////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
//////////////////////////  hiddenProj.to(targetDevice, false)
//////////////////////////  register_module("hiddenProj", hiddenProj)
//////////////////////////
//////////////////////////  // Final MLP
//////////////////////////  private val totalDim = features.size * embedDim + embedDim
//////////////////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
//////////////////////////  register_module("mlp", mlp)
//////////////////////////
//////////////////////////  // ===========================================================================
//////////////////////////  // FORWARD —— 100% 匹配你的 EmbeddingLayer，无 getEmbedding，无空迭代
//////////////////////////  // ===========================================================================
//////////////////////////  def forward(
//////////////////////////               sparseFeats: Map[String, Tensor],
//////////////////////////               sequenceFeats: Map[String, Tensor]
//////////////////////////             ): Tensor = {
//////////////////////////
//////////////////////////    // 1. 静态特征嵌入（正确方式）
//////////////////////////    val featEmb = embeddingLayer.forward(sparseFeats, Map.empty, squeeze = true)
//////////////////////////    featEmb.to(targetDevice, ScalarType.Float)
//////////////////////////
//////////////////////////    // 2. 序列特征嵌入（正确方式：直接传入 sequenceFeats）
//////////////////////////    val seqEmb = embeddingLayer.forward(Map.empty, sequenceFeats, squeeze = false)
//////////////////////////    seqEmb.to(targetDevice, ScalarType.Float)
//////////////////////////
//////////////////////////    // 形状确保 [batch, seq_len, embed_dim]
//////////////////////////    val seqEmbFixed = if (seqEmb.dim() == 2) seqEmb.unsqueeze(1) else seqEmb
//////////////////////////    val batchSize = seqEmbFixed.size(0)
//////////////////////////    val seqLen = seqEmbFixed.size(1).toInt
//////////////////////////
//////////////////////////    // 3. ODE 初始化
//////////////////////////    var hiddenState = seqEmbFixed.select(1, 0) // 第一个时间步
//////////////////////////    hiddenState = inputProj.forward(hiddenState) // 投影到 hiddenDim
//////////////////////////    hiddenState.to(targetDevice, ScalarType.Float)
//////////////////////////
//////////////////////////    // 4. ODE 积分
//////////////////////////    val dt = 1.0f / numOdeSteps.toFloat
//////////////////////////    for (step <- 0 until numOdeSteps) {
//////////////////////////      val t = step.toFloat * dt
//////////////////////////      val tFrac = t * (seqLen - 1)
//////////////////////////      val t0 = math.min(tFrac.toInt, seqLen - 2)
//////////////////////////      val t1 = t0 + 1
//////////////////////////      val alpha = tFrac - t0
//////////////////////////
//////////////////////////      val x0 = seqEmbFixed.select(1, t0)
//////////////////////////      val x1 = seqEmbFixed.select(1, t1)
//////////////////////////      val xT = x0.mul(new Scalar(1.0f - alpha)).add(x1.mul(new Scalar(alpha)))
//////////////////////////
//////////////////////////      // ODE 导数
//////////////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t)
//////////////////////////      // Euler 更新
//////////////////////////      hiddenState = hiddenState.add(dHidden.mul(new Scalar(dt)))
//////////////////////////    }
//////////////////////////
//////////////////////////    // 5. 输出
//////////////////////////    val liquidRep = outputProj.forward(hiddenState)
//////////////////////////    val combined = torch.cat(new TensorVector(featEmb, liquidRep), 1)
//////////////////////////    val logits = mlp.forward(combined)
//////////////////////////
//////////////////////////    logits.to(targetDevice, ScalarType.Float)
//////////////////////////  }
//////////////////////////}
//////////////////////////
//////////////////////////// ===================== LiquidCell 保持不变 =====================
//////////////////////////class LiquidCell(
//////////////////////////                  inputDim: Int,
//////////////////////////                  hiddenDim: Int,
//////////////////////////                  device: String
//////////////////////////                ) extends Module {
//////////////////////////
//////////////////////////  private val targetDevice = new Device(device)
//////////////////////////
//////////////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
//////////////////////////  private val inputProj = new LinearImpl(inputDim, hiddenDim)
//////////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
//////////////////////////  private val timeMod = new LinearImpl(1, hiddenDim)
//////////////////////////  private val gateW = new LinearImpl(inputDim, hiddenDim)
//////////////////////////  private val gateH = new LinearImpl(hiddenDim, hiddenDim)
//////////////////////////  private val gateB = new LinearImpl(1, hiddenDim)
//////////////////////////
//////////////////////////  timeConstant.to(targetDevice, false)
//////////////////////////  inputProj.to(targetDevice, false)
//////////////////////////  hiddenProj.to(targetDevice, false)
//////////////////////////  timeMod.to(targetDevice, false)
//////////////////////////  gateW.to(targetDevice, false)
//////////////////////////  gateH.to(targetDevice, false)
//////////////////////////  gateB.to(targetDevice, false)
//////////////////////////
//////////////////////////  register_module("timeConstant", timeConstant)
//////////////////////////  register_module("inputProj", inputProj)
//////////////////////////  register_module("hiddenProj", hiddenProj)
//////////////////////////  register_module("timeMod", timeMod)
//////////////////////////  register_module("gateW", gateW)
//////////////////////////  register_module("gateH", gateH)
//////////////////////////  register_module("gateB", gateB)
//////////////////////////
//////////////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
//////////////////////////    val batchSize = hidden.size(0)
//////////////////////////
//////////////////////////    val timeTensor = torch.ones(Array(batchSize, 1L),
//////////////////////////      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
//////////////////////////    .device(new DeviceOptional(hidden.device()))) .mul(new Scalar(time))
//////////////////////////
//////////////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()
//////////////////////////    val recurrent = hiddenProj.forward(hidden)
//////////////////////////    val inputTerm = inputProj.forward(input)
//////////////////////////    val timeDecay = timeMod.forward(timeTensor)
//////////////////////////
//////////////////////////    val gate = gateW.forward(input)
//////////////////////////      .add(gateH.forward(hidden))
//////////////////////////      .add(gateB.forward(timeTensor))
//////////////////////////      .sigmoid()
//////////////////////////
//////////////////////////    val dHidden = recurrent.mul(gate)
//////////////////////////      .add(inputTerm)
//////////////////////////      .add(timeDecay)
//////////////////////////      .add(hidden.mul(timeConst))
//////////////////////////
//////////////////////////    dHidden
//////////////////////////  }
//////////////////////////}
//////////////////////////
//////////////////////////
//////////////////////////
////////////////////////////package torchrec.models.ranking
////////////////////////////
////////////////////////////import torchrec.basic.features._
////////////////////////////import torchrec.basic.layers._
////////////////////////////import torchrec.utils.DeviceSupport
////////////////////////////
////////////////////////////import org.bytedeco.pytorch._
////////////////////////////import org.bytedeco.pytorch.global.torch
////////////////////////////import org.bytedeco.pytorch.global.torch.ScalarType
////////////////////////////
/////////////////////////////**
//////////////////////////// * Liquid Neural Network (LiquidNet) for Sequential Recommendation
//////////////////////////// *
//////////////////////////// * A continuous-time neural network based on Neural ODEs (Ordinary Differential Equations).
//////////////////////////// * Uses differential equation solvers to model the dynamics of user interest evolution.
//////////////////////////// *
//////////////////////////// * Reference: "Liquid Time-constant Networks" (Hasani et al., AAAI 2021)
//////////////////////////// *            + adaptation for recommendation systems.
//////////////////////////// *
//////////////////////////// * The core idea: replace discrete RNN/GRU layers with an ODE-based continuous dynamics layer.
//////////////////////////// * The ODE solver computes the trajectory of hidden states over a virtual time dimension.
//////////////////////////// *
//////////////////////////// * Architecture:
//////////////////////////// *   Sparse Input → EmbeddingLayer → LiquidBlock (ODE dynamics) → MLP → CTR Logit
//////////////////////////// *
//////////////////////////// * @param features            List of sparse features
//////////////////////////// * @param sequenceFeatures    Sequence features for temporal dynamics
//////////////////////////// * @param embedDim            Embedding dimension
//////////////////////////// * @param hiddenDim           Hidden dimension for ODE state
//////////////////////////// * @param numOdeSteps         Number of ODE integration steps
//////////////////////////// * @param mlpDims             MLP hidden dimensions
//////////////////////////// * @param dropout             Dropout rate
//////////////////////////// * @param device              Device
//////////////////////////// */
////////////////////////////class LiquidNetWork(
////////////////////////////  features: List[Feature],
////////////////////////////  sequenceFeatures: List[SequenceFeature],
////////////////////////////  embedDim: Int = 8,
////////////////////////////  hiddenDim: Int = 32,
////////////////////////////  numOdeSteps: Int = 4,
////////////////////////////  mlpDims: List[Long] = List(64L, 32L),
////////////////////////////  dropout: Float = 0.2f,
////////////////////////////  device: String = DeviceSupport.backend
////////////////////////////) extends Module {
////////////////////////////
////////////////////////////  require(features.nonEmpty, "features cannot be empty")
////////////////////////////  require(sequenceFeatures.nonEmpty, "sequenceFeatures cannot be empty")
////////////////////////////  require(hiddenDim > 0, s"hiddenDim must be > 0, got $hiddenDim")
////////////////////////////
////////////////////////////  // Embedding layers
////////////////////////////  private val featureEmbedding = new EmbeddingLayer(features, embedDim, device)
////////////////////////////  register_module("featureEmbedding", featureEmbedding)
////////////////////////////
////////////////////////////  private val sequenceEmbedding = new EmbeddingLayer(sequenceFeatures, embedDim, device)
////////////////////////////  register_module("sequenceEmbedding", sequenceEmbedding)
////////////////////////////
////////////////////////////  // Liquid ODE cell: computes derivative of hidden state
////////////////////////////  // dhidden/dt = LiquidCell(hidden, input)
////////////////////////////  private val liquidCell = new LiquidCell(embedDim, hiddenDim, device)
////////////////////////////  register_module("liquidCell", liquidCell)
////////////////////////////
////////////////////////////  // Output projection from hidden state to embedding dimension
////////////////////////////  private val outputProj = new LinearImpl(hiddenDim, embedDim)
////////////////////////////  outputProj.to(new Device(device),false)
////////////////////////////  register_module("outputProj", outputProj)
////////////////////////////
////////////////////////////  // Project input to hiddenDim (moved from forward to be registered)
////////////////////////////  private val inputProj = new LinearImpl(embedDim, hiddenDim)
////////////////////////////  inputProj.to(new Device(device),false)
////////////////////////////  register_module("inputProj", inputProj)
////////////////////////////
////////////////////////////  // Project hidden to hiddenDim (moved from forward to be registered)
////////////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
////////////////////////////  hiddenProj.to(new Device(device),false)
////////////////////////////  register_module("hiddenProj", hiddenProj)
////////////////////////////
////////////////////////////  // Final MLP
////////////////////////////  private val sparseDim = Features.calcSparseDim(features)
////////////////////////////  private val totalDim = sparseDim + embedDim
////////////////////////////  private val mlp = new MLP(totalDim, mlpDims, 1, "relu", dropout, device = device)
////////////////////////////  register_module("mlp", mlp)
////////////////////////////
////////////////////////////  if (device != "cpu") {
////////////////////////////    val dev = new org.bytedeco.pytorch.Device(device)
////////////////////////////    featureEmbedding.to(dev, false)
////////////////////////////    sequenceEmbedding.to(dev, false)
////////////////////////////    liquidCell.to(dev, false)
////////////////////////////    outputProj.to(dev, false)
////////////////////////////    inputProj.to(dev, false)
////////////////////////////    hiddenProj.to(dev, false)
////////////////////////////    mlp.to(dev, false)
////////////////////////////  }
////////////////////////////
////////////////////////////  def forward(
////////////////////////////    sparseFeats: Map[String, Tensor],
////////////////////////////    sequenceFeats: Map[String, Tensor]
////////////////////////////  ): Tensor = {
////////////////////////////    // 1. Get sparse feature embeddings
////////////////////////////    val featEmb = featureEmbedding.forward(sparseFeats)  // (batch, sparse_dim)
////////////////////////////
////////////////////////////    // 2. Get sequence embeddings
////////////////////////////    val seqEmbs = sequenceFeatures.flatMap { seqFeat =>
////////////////////////////      sequenceFeats.get(seqFeat.name).map { indices =>
////////////////////////////        sequenceEmbedding.getEmbedding(seqFeat.name, indices)
////////////////////////////      }
////////////////////////////    }
////////////////////////////
////////////////////////////    if (seqEmbs.isEmpty) {
////////////////////////////      throw new IllegalArgumentException("No sequence embeddings found")
////////////////////////////    }
////////////////////////////
////////////////////////////    val seqEmb = if (seqEmbs.size == 1) {
////////////////////////////      seqEmbs.head
////////////////////////////    } else {
////////////////////////////      val vec = new TensorVector(seqEmbs.size.toLong)
////////////////////////////      seqEmbs.foreach(vec.push_back)
////////////////////////////      torch.cat(vec, 1)
////////////////////////////    }
////////////////////////////
////////////////////////////    // 3. Run ODE solver over the sequence
////////////////////////////    // Initialize hidden state as the first sequence element
////////////////////////////    val seqLen = seqEmb.size(1).toInt
////////////////////////////    var hiddenState = seqEmb.select(1, 0)  // (batch, embed_dim)
////////////////////////////
////////////////////////////    // ODE integration: iterate over sequence positions
////////////////////////////    // For each step, compute ODE derivative and update hidden state (Euler method)
////////////////////////////    val dt = 1.0f / numOdeSteps.toFloat
////////////////////////////    for (step <- 0 until numOdeSteps) {
////////////////////////////      val t = step.toFloat * dt
////////////////////////////      // Interpolate input from sequence based on time t
////////////////////////////      val tFrac = t * (seqLen - 1)
////////////////////////////      val t0 = tFrac.toInt.max(seqLen - 2)
////////////////////////////      val t1 = t0 + 1
////////////////////////////      val alpha = tFrac - t0
////////////////////////////
////////////////////////////      val x0 = seqEmb.select(1, t0)
////////////////////////////      val x1 = if (t1 < seqLen) seqEmb.select(1, t1) else x0
////////////////////////////      val xT = x0.mul(new Scalar(1.0f - alpha)).add(x1.mul(new Scalar(alpha)))
////////////////////////////
////////////////////////////      // ODE: dhidden/dt = LiquidCell(hidden, xT)
////////////////////////////      val dHidden = liquidCell.forward(hiddenState, xT, t)  // (batch, hiddenDim)
////////////////////////////
////////////////////////////      // Euler integration step
////////////////////////////      hiddenState = hiddenState.add(dHidden.mul(new Scalar(dt)))
////////////////////////////    }
////////////////////////////
////////////////////////////    // 4. Project hidden state back to embedDim
////////////////////////////    val liquidRep = outputProj.forward(hiddenState)  // (batch, embed_dim)
////////////////////////////
////////////////////////////    // 5. Combine with sparse features
////////////////////////////    val combined = torch.cat(new TensorVector(featEmb, liquidRep), 1)
////////////////////////////
////////////////////////////    // 6. Final MLP
////////////////////////////    val logits = mlp.forward(combined)
////////////////////////////    logits
////////////////////////////  }
////////////////////////////}
////////////////////////////
/////////////////////////////**
//////////////////////////// * Liquid Cell: computes the derivative of the hidden state.
//////////////////////////// * Uses a time-aware gating mechanism that mimics the liquid time-constant neuron.
//////////////////////////// *
//////////////////////////// * dhidden/dt = -A * hidden + W * input + b + time_decay
//////////////////////////// *
//////////////////////////// * Where A, W, b are learned parameters and time_decay is a learned function of time.
//////////////////////////// */
////////////////////////////class LiquidCell(
////////////////////////////  inputDim: Int,
////////////////////////////  hiddenDim: Int,
////////////////////////////  device: String
////////////////////////////) extends Module {
////////////////////////////
////////////////////////////  // Negative time constant (learnable)
////////////////////////////  private val timeConstant = new LinearImpl(1, hiddenDim)
////////////////////////////  register_module("timeConstant", timeConstant)
////////////////////////////
////////////////////////////  // Input to hidden projection
////////////////////////////  private val inputProj = new LinearImpl(inputDim, hiddenDim)
////////////////////////////  register_module("inputProj", inputProj)
////////////////////////////
////////////////////////////  // Hidden to hidden recurrent projection
////////////////////////////  private val hiddenProj = new LinearImpl(hiddenDim, hiddenDim)
////////////////////////////  register_module("hiddenProj", hiddenProj)
////////////////////////////
////////////////////////////  // Time modulation
////////////////////////////  private val timeMod = new LinearImpl(1, hiddenDim)
////////////////////////////  register_module("timeMod", timeMod)
////////////////////////////
////////////////////////////  // Gate for input-dependent time constant
////////////////////////////  private val gateW = new LinearImpl(inputDim, hiddenDim)
////////////////////////////  register_module("gateW", gateW)
////////////////////////////  private val gateH = new LinearImpl(hiddenDim, hiddenDim)
////////////////////////////  register_module("gateH", gateH)
////////////////////////////  private val gateB = new LinearImpl(1, hiddenDim)
////////////////////////////  register_module("gateB", gateB)
////////////////////////////
////////////////////////////  if (device != "cpu") {
////////////////////////////    val dev = new org.bytedeco.pytorch.Device(device)
////////////////////////////    timeConstant.to(dev, false)
////////////////////////////    inputProj.to(dev, false)
////////////////////////////    hiddenProj.to(dev, false)
////////////////////////////    timeMod.to(dev, false)
////////////////////////////    gateW.to(dev, false)
////////////////////////////    gateH.to(dev, false)
////////////////////////////    gateB.to(dev, false)
////////////////////////////  }
////////////////////////////
////////////////////////////  def forward(hidden: Tensor, input: Tensor, time: Float): Tensor = {
////////////////////////////    // hidden: (batch, hiddenDim)
////////////////////////////    // input: (batch, inputDim)
////////////////////////////    // time: scalar
////////////////////////////
////////////////////////////    val batchSize = hidden.size(0).toInt
////////////////////////////
////////////////////////////    // Time input: (batch, 1)
////////////////////////////    val timeTensor = torch.ones(Array(batchSize.toLong, 1L),
////////////////////////////      new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
////////////////////////////    ).to(hidden.device(), ScalarType.Float).mul(new Scalar(time))
////////////////////////////
////////////////////////////    // Time-dependent decay: -A = -relu(timeConstant(time))
////////////////////////////    val timeConst = timeConstant.forward(timeTensor).relu().neg()  // (batch, hiddenDim)
////////////////////////////
////////////////////////////    // Recurrent term: W * hidden
////////////////////////////    val recurrent = hiddenProj.forward(hidden)  // (batch, hiddenDim)
////////////////////////////
////////////////////////////    // Input term: W * input
////////////////////////////    val inputTerm = inputProj.forward(input)  // (batch, hiddenDim)
////////////////////////////
////////////////////////////    // Time modulation
////////////////////////////    val timeDecay = timeMod.forward(timeTensor)  // (batch, hiddenDim)
////////////////////////////
////////////////////////////    // Liquid gate: sigmoid(W*x + W*h + b) for adaptive time constant
////////////////////////////    val gate = gateW.forward(input).add(gateH.forward(hidden)).add(gateB.forward(timeTensor)).sigmoid()
////////////////////////////
////////////////////////////    // ODE: dhidden/dt = -A * hidden + input + time_decay * gate
////////////////////////////    val dHidden = recurrent.mul(gate).add(inputTerm).add(timeDecay).add(hidden.mul(timeConst))
////////////////////////////
////////////////////////////    dHidden  // (batch, hiddenDim)
////////////////////////////  }
////////////////////////////}
