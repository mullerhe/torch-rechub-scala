package torchrec.basic.layers

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
import torchrec.utils.DeviceSupport
import torchrec.TensorImplicits.RichTensor
import torchrec.utils.DeviceSupport.deviceOf

/**
 * Capsule network for multi-interest (MIND/Comirec).
 *
 * Parameters
 * ----------
 * embeddingDim : int
 *   Item embedding dimension.
 * seqLen : int
 *   Sequence length.
 * bilinearType : {0, 1, 2}, default 2
 *   0 for MIND, 2 for ComirecDR.
 * interestNum : int, default 4
 *   Number of interests.
 * routingTimes : int, default 3
 *   Routing iterations.
 * reluLayer : bool, default False
 *   Whether to apply ReLU after routing.
 *
 * Shape
 * -----
 * Input
 *   itemEb : ``(B, L, D)``
 *   mask : ``(B, L, 1)``
 * Output
 *   ``(B, interest_num, D)``
 */



class CapsuleNetwork(
                      embeddingDim: Int,
                      seqLen: Int,
                      bilinearType: Int = 2,
                      interestNum: Int = 4,
                      routingTimes: Int = 3,
                      reluLayer: Boolean = false,
                      device: String = DeviceSupport.backend
                    ) extends Module {
  private val h = embeddingDim
  private val s = seqLen
  private val k = interestNum
  private val dev = deviceOf(device)
  private val tensorOptsCPU = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))
  private val tensorOptsDev = tensorOptsCPU.device(new DeviceOptional(dev))

  // 1. ReLU 仅在开启时创建，bias=False 对齐Python
  private val relu: Option[SequentialImpl] = if (reluLayer) {
    val seq = new SequentialImpl()
    val lin = new LinearImpl(embeddingDim, embeddingDim)
    seq.push_back("linear", lin)
    seq.push_back("relu", new ReLUImpl())
    register_module("relu", seq)
    Some(seq)
  } else None

  private var linear: Option[LinearImpl] = None
  private var w: Option[Tensor] = None

  bilinearType match {
    case 0 => // MIND: bias=False
      val l = new LinearImpl(embeddingDim, embeddingDim)
      register_module("linear", l)
      linear = Some(l)
    case 1 =>
      val l = new LinearImpl(embeddingDim, embeddingDim * interestNum)
      register_module("linear", l)
      linear = Some(l)
    case _ => // ComirecDR w参数，绑定device
      val wt = torch.rand(
        Array[Long](1L, seqLen.toLong, interestNum.toLong * embeddingDim.toLong, embeddingDim.toLong),
        tensorOptsDev
      )
      register_parameter("w", wt)
      w = Some(wt)
  }

  def forward(itemEb: Tensor, mask: Tensor): Tensor = {
    val batchSize = itemEb.size(0)
    println(s"[CAP DEBUG] 输入itemEb shape: [${itemEb.size(0)},${itemEb.size(1)},${itemEb.size(2)}]")
    println(s"[CAP DEBUG] 输入mask shape: [${mask.size(0)},${mask.size(1)},${mask.size(2)}]")
    require(itemEb.device().equals(dev), s"itemEb device mismatch, layer=$dev, input=${itemEb.device()}")

    // Step1 Bilinear transform
    var itemEbHat = bilinearType match {
      case 0 =>
        val out = linear.get.forward(itemEb)
        out.repeat(Array(1L, 1L, interestNum.toLong) *)
      case 1 =>
        linear.get.forward(itemEb)
      case _ =>
        val u = itemEb.unsqueeze(2)
        val wt = w.get
        torch.sum(wt.narrow(1, 0, seqLen) * u, 3)
    }
    println(s"[CAP DEBUG] bilinear itemEbHat raw shape: [${itemEbHat.size(0)},${itemEbHat.size(1)},${itemEbHat.size(2)}]")

    // reshape & transpose 对齐Python
    val reshaped = itemEbHat.reshape(Array(batchSize, seqLen, interestNum, embeddingDim) *)
    val transposed = reshaped.transpose(1, 2).contiguous()
    itemEbHat = transposed.reshape(Array(batchSize, interestNum, seqLen, embeddingDim) *)
    println(s"[CAP DEBUG] itemEbHat after transpose shape: [${itemEbHat.size(0)},${itemEbHat.size(1)},${itemEbHat.size(2)},${itemEbHat.size(3)}]")

    val itemEbHatIter = itemEbHat.detach() // stop_grad = True
    println(s"[CAP DEBUG] itemEbHatIter detach shape same as above")

    // capsule_weight requires_grad=False
    var capsuleWeight = if (bilinearType > 0) {
      torch.zeros(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    } else {
      torch.randn(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    }
    println(s"[CAP DEBUG] capsuleWeight init shape: [${capsuleWeight.size(0)},${capsuleWeight.size(1)},${capsuleWeight.size(2)}]")

    var interestCapsule: Tensor = torch.empty()
    for (i <- 0 until routingTimes) {
      println(s"\n===== Route Iter $i =====")
      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L) *)
      println(s"[CAP DEBUG] attenMask shape: [${attenMask.size(0)},${attenMask.size(1)},${attenMask.size(2)},${attenMask.size(3)}]")
      val paddings = torch.zeros_like(attenMask)
      println(s"[CAP DEBUG] paddings shape same as attenMask")

      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
      println(s"[CAP DEBUG] capsuleSoftmaxWeight raw shape: [${capsuleSoftmaxWeight.size(0)},${capsuleSoftmaxWeight.size(1)},${capsuleSoftmaxWeight.size(2)}]")

      val capsuleSoftmaxWeight4d = capsuleSoftmaxWeight.unsqueeze(-1)
      println(s"[CAP DEBUG] capsuleSoftmaxWeight4d shape: [${capsuleSoftmaxWeight4d.size(0)},${capsuleSoftmaxWeight4d.size(1)},${capsuleSoftmaxWeight4d.size(2)},${capsuleSoftmaxWeight4d.size(3)}]")

      println("[CAP DEBUG] 执行 torch.where 开始")
      val maskedWeight4d = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight4d)
      println(s"[CAP DEBUG] maskedWeight4d shape: [${maskedWeight4d.size(0)},${maskedWeight4d.size(1)},${maskedWeight4d.size(2)},${maskedWeight4d.size(3)}]")

      val maskedWeight3d = maskedWeight4d.squeeze(-1)
      val unsqueezedWeight = maskedWeight3d.unsqueeze(2) // [B,K,1,L] 四维，无五维
      println(s"[CAP DEBUG] unsqueezedWeight shape: [${unsqueezedWeight.size(0)},${unsqueezedWeight.size(1)},${unsqueezedWeight.size(2)},${unsqueezedWeight.size(3)}]")

      if (i < 2) {
        println("[CAP DEBUG] 前两轮路由，使用detach itemEbHatIter做matmul")
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
        println(s"[CAP DEBUG] interestCapsule shape after matmul: [${interestCapsule.size(0)},${interestCapsule.size(1)},${interestCapsule.size(2)},${interestCapsule.size(3)}]")

        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)

        val capT = interestCapsule.transpose(2, 3).contiguous()
        var deltaWeight = torch.matmul(itemEbHatIter, capT)
        println(s"[CAP DEBUG] deltaWeight raw matmul shape: [${deltaWeight.size(0)},${deltaWeight.size(1)},${deltaWeight.size(2)},${deltaWeight.size(3)}]")
        // 修复：用squeeze替代reshape，规避view尺寸报错
        deltaWeight = deltaWeight.squeeze(-1)
        println(s"[CAP DEBUG] deltaWeight after squeeze shape: [${deltaWeight.size(0)},${deltaWeight.size(1)},${deltaWeight.size(2)}]")
        capsuleWeight = capsuleWeight.add(deltaWeight)
        println(s"[CAP DEBUG] 更新后capsuleWeight shape不变")
      } else {
        println("[CAP DEBUG] 第三轮路由，使用原始itemEbHat做matmul")
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHat)
        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)
        println(s"[CAP DEBUG] final interestCapsule shape: [${interestCapsule.size(0)},${interestCapsule.size(1)},${interestCapsule.size(2)},${interestCapsule.size(3)}]")
      }
    }

    // 输出reshape [B,K,1,D] -> [B,K,D]
    val result = interestCapsule.reshape(Array(batchSize, interestNum, embeddingDim) *)
    println(s"\n[CAP DEBUG] 最终输出result shape: [${result.size(0)},${result.size(1)},${result.size(2)}]")
    if (reluLayer && relu.isDefined) relu.get.forward(result) else result
  }
  def forward3(itemEb: Tensor, mask: Tensor): Tensor = {
    val batchSize = itemEb.size(0)
    println(s"[CAP DEBUG] 输入itemEb shape: [${itemEb.size(0)},${itemEb.size(1)},${itemEb.size(2)}]")
    println(s"[CAP DEBUG] 输入mask shape: [${mask.size(0)},${mask.size(1)},${mask.size(2)}]")
    require(itemEb.device().equals(dev), s"itemEb device mismatch, layer=$dev, input=${itemEb.device()}")

    // Step1 Bilinear transform
    var itemEbHat = bilinearType match {
      case 0 =>
        val out = linear.get.forward(itemEb)
        out.repeat(Array(1L, 1L, interestNum.toLong) *)
      case 1 =>
        linear.get.forward(itemEb)
      case _ =>
        val u = itemEb.unsqueeze(2)
        val wt = w.get
        torch.sum(wt.narrow(1, 0, seqLen) * u, 3)
    }
    println(s"[CAP DEBUG] bilinear itemEbHat raw shape: [${itemEbHat.size(0)},${itemEbHat.size(1)},${itemEbHat.size(2)}]")

    // reshape & transpose 对齐Python
    val reshaped = itemEbHat.reshape(Array(batchSize, seqLen, interestNum, embeddingDim) *)
    val transposed = reshaped.transpose(1, 2).contiguous()
    itemEbHat = transposed.reshape(Array(batchSize, interestNum, seqLen, embeddingDim) *)
    println(s"[CAP DEBUG] itemEbHat after transpose shape: [${itemEbHat.size(0)},${itemEbHat.size(1)},${itemEbHat.size(2)},${itemEbHat.size(3)}]")

    val itemEbHatIter = itemEbHat.detach() // stop_grad = True
    println(s"[CAP DEBUG] itemEbHatIter detach shape same as above")

    // capsule_weight requires_grad=False
    var capsuleWeight = if (bilinearType > 0) {
      torch.zeros(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    } else {
      torch.randn(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    }
    println(s"[CAP DEBUG] capsuleWeight init shape: [${capsuleWeight.size(0)},${capsuleWeight.size(1)},${capsuleWeight.size(2)}]")

    var interestCapsule: Tensor = torch.empty()
    for (i <- 0 until routingTimes) {
      println(s"\n===== Route Iter $i =====")
      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L) *)
      println(s"[CAP DEBUG] attenMask shape: [${attenMask.size(0)},${attenMask.size(1)},${attenMask.size(2)},${attenMask.size(3)}]")
      val paddings = torch.zeros_like(attenMask)
      println(s"[CAP DEBUG] paddings shape same as attenMask")

      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
      println(s"[CAP DEBUG] capsuleSoftmaxWeight raw shape: [${capsuleSoftmaxWeight.size(0)},${capsuleSoftmaxWeight.size(1)},${capsuleSoftmaxWeight.size(2)}]")

      val capsuleSoftmaxWeight4d = capsuleSoftmaxWeight.unsqueeze(-1)
      println(s"[CAP DEBUG] capsuleSoftmaxWeight4d shape: [${capsuleSoftmaxWeight4d.size(0)},${capsuleSoftmaxWeight4d.size(1)},${capsuleSoftmaxWeight4d.size(2)},${capsuleSoftmaxWeight4d.size(3)}]")

      println("[CAP DEBUG] 执行 torch.where 开始")
      val maskedWeight4d = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight4d)
      println(s"[CAP DEBUG] maskedWeight4d shape: [${maskedWeight4d.size(0)},${maskedWeight4d.size(1)},${maskedWeight4d.size(2)},${maskedWeight4d.size(3)}]")

      // 关键修复：squeeze最后一维，把 [B,K,L,1] → [B,K,L]，再unsqueeze(2)得到 [B,K,1,L] 四维，无第五维
      val maskedWeight3d = maskedWeight4d.squeeze(-1)
      val unsqueezedWeight = maskedWeight3d.unsqueeze(2) // [B,K,1,L] 四维，和Python完全一致
      println(s"[CAP DEBUG] unsqueezedWeight shape: [${unsqueezedWeight.size(0)},${unsqueezedWeight.size(1)},${unsqueezedWeight.size(2)},${unsqueezedWeight.size(3)}]")

      if (i < 2) {
        println("[CAP DEBUG] 前两轮路由，使用detach itemEbHatIter做matmul")
        // [B,K,1,L] @ [B,K,L,D] = [B,K,1,D] 维度完美匹配
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
        println(s"[CAP DEBUG] interestCapsule shape after matmul: [${interestCapsule.size(0)},${interestCapsule.size(1)},${interestCapsule.size(2)},${interestCapsule.size(3)}]")

        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)

        val capT = interestCapsule.transpose(2, 3).contiguous()
        var deltaWeight = torch.matmul(itemEbHatIter, capT)
        deltaWeight = deltaWeight.reshape(Array(batchSize, interestNum, seqLen) *)
        capsuleWeight = capsuleWeight.add(deltaWeight)
        println(s"[CAP DEBUG] 更新后capsuleWeight shape不变")
      } else {
        println("[CAP DEBUG] 第三轮路由，使用原始itemEbHat做matmul")
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHat)
        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)
        println(s"[CAP DEBUG] final interestCapsule shape: [${interestCapsule.size(0)},${interestCapsule.size(1)},${interestCapsule.size(2)},${interestCapsule.size(3)}]")
      }
    }
    for (i <- 0 until 0) {
      println(s"\n===== Route Iter $i =====")
      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L) *)
      println(s"[CAP DEBUG] attenMask shape: [${attenMask.size(0)},${attenMask.size(1)},${attenMask.size(2)},${attenMask.size(3)}]")
      val paddings = torch.zeros_like(attenMask)
      println(s"[CAP DEBUG] paddings shape same as attenMask")

      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
      println(s"[CAP DEBUG] capsuleSoftmaxWeight raw shape: [${capsuleSoftmaxWeight.size(0)},${capsuleSoftmaxWeight.size(1)},${capsuleSoftmaxWeight.size(2)}]")

      // 核心修复：手动升维到4维 [B,K,L] -> [B,K,L,1]，和mask完全同维度
      val capsuleSoftmaxWeight4d = capsuleSoftmaxWeight.unsqueeze(-1)
      println(s"[CAP DEBUG] capsuleSoftmaxWeight4d shape: [${capsuleSoftmaxWeight4d.size(0)},${capsuleSoftmaxWeight4d.size(1)},${capsuleSoftmaxWeight4d.size(2)},${capsuleSoftmaxWeight4d.size(3)}]")

      // 三个张量全部4维，无广播冲突，彻底解决where报错
      println("[CAP DEBUG] 执行 torch.where 开始")
      val maskedWeight = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight4d)
      println(s"[CAP DEBUG] maskedWeight shape: [${maskedWeight.size(0)},${maskedWeight.size(1)},${maskedWeight.size(2)},${maskedWeight.size(3)}]")

      val unsqueezedWeight = maskedWeight.unsqueeze(2) // [B,K,1,L,1]
      println(s"[CAP DEBUG] unsqueezedWeight shape: [${unsqueezedWeight.size(0)},${unsqueezedWeight.size(1)},${unsqueezedWeight.size(2)},${unsqueezedWeight.size(3)},${unsqueezedWeight.size(4)}]")

      if (i < 2) {
        println("[CAP DEBUG] 前两轮路由，使用detach itemEbHatIter做matmul")
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
        println(s"[CAP DEBUG] interestCapsule shape after matmul: [${interestCapsule.size(0)},${interestCapsule.size(1)},${interestCapsule.size(2)},${interestCapsule.size(3)}]")

        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)

        val capT = interestCapsule.transpose(2, 3).contiguous()
        var deltaWeight = torch.matmul(itemEbHatIter, capT)
        deltaWeight = deltaWeight.reshape(Array(batchSize, interestNum, seqLen) *)
        capsuleWeight = capsuleWeight.add(deltaWeight)
        println(s"[CAP DEBUG] 更新后capsuleWeight shape不变")
      } else {
        println("[CAP DEBUG] 第三轮路由，使用原始itemEbHat做matmul")
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHat)
        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)
        println(s"[CAP DEBUG] final interestCapsule shape: [${interestCapsule.size(0)},${interestCapsule.size(1)},${interestCapsule.size(2)},${interestCapsule.size(3)}]")
      }
    }

    val result = interestCapsule.reshape(Array(batchSize, interestNum, embeddingDim) *)
    println(s"\n[CAP DEBUG] 最终输出result shape: [${result.size(0)},${result.size(1)},${result.size(2)}]")
    if (reluLayer && relu.isDefined) relu.get.forward(result) else result
  }
  def forward2(itemEb: Tensor, mask: Tensor): Tensor = {
    val batchSize = itemEb.size(0)
    require(itemEb.device().equals(dev), s"itemEb device mismatch, layer=$dev, input=${itemEb.device()}")

    // Step1 Bilinear transform
    var itemEbHat = bilinearType match {
      case 0 =>
        val out = linear.get.forward(itemEb)
        out.repeat(1, 1, interestNum)
      case 1 =>
        linear.get.forward(itemEb)
      case _ =>
        val u = itemEb.unsqueeze(2)
        val wt = w.get
        torch.sum(wt.narrow(1, 0, seqLen) * u, 3)
    }

    // reshape & transpose 对齐Python
    val reshaped = itemEbHat.reshape(Array(batchSize, seqLen, interestNum, embeddingDim)*)
    val transposed = reshaped.transpose(1, 2).contiguous()
    itemEbHat = transposed.reshape(Array(batchSize, interestNum, seqLen, embeddingDim)*)

    val itemEbHatIter = itemEbHat.detach() // stop_grad = True

    // capsule_weight requires_grad=False
    var capsuleWeight = if (bilinearType > 0) {
      torch.zeros(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    } else {
      torch.randn(
        Array[Long](batchSize.toLong, interestNum, seqLen),
        tensorOptsDev.requires_grad(new BoolOptional(false))
      )
    }

    var interestCapsule: Tensor = torch.empty()
    for (i <- 0 until routingTimes) {
      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L) *)
      val paddings = torch.zeros_like(attenMask)

      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
      // 移除unsqueeze(-1)，不再4维weight，保持3维 [B,K,L]
      val maskedWeight = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight)
      // 只在维度2插入1，得到 [B,K,1,L] 四维，和Python完全一致
      val unsqueezedWeight = maskedWeight.unsqueeze(2) // [B,K,1,L]

      if (i < 2) {
        // [B,K,1,L] @ [B,K,L,D] = [B,K,1,D] 合法矩阵乘
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)

        val capT = interestCapsule.transpose(2, 3).contiguous() // [B,K,D,1]
        var deltaWeight = torch.matmul(itemEbHatIter, capT) // [B,K,L,D] @ [B,K,D,1] = [B,K,L,1]
        deltaWeight = deltaWeight.reshape(Array(batchSize, interestNum, seqLen) *)
        capsuleWeight = capsuleWeight.add(deltaWeight)
      } else {
        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHat)
        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
        interestCapsule = scalarFactor.mul(interestCapsule)
      }
    }
//    for (i <- 0 until -1) {
//      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L) *)
//      val paddings = torch.zeros_like(attenMask)
//
//      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
//      // 关键修复：给softmax权重增加最后一维，对齐mask 4维 [B,K,L] → [B,K,L,1]
//      val capsuleSoftmaxWeight4d = capsuleSoftmaxWeight.unsqueeze(-1)
//
//      // where 三个张量全部4维，维度完全匹配
//      val maskedWeight = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight4d)
//      val unsqueezedWeight = maskedWeight.unsqueeze(2) // [B,K,1,L,1]
//
//      if (i < 2) {
//        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
//          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        interestCapsule = scalarFactor.mul(interestCapsule)
//
//        val capT = interestCapsule.transpose(2, 3).contiguous()
//        var deltaWeight = torch.matmul(itemEbHatIter, capT)
//        deltaWeight = deltaWeight.reshape(Array(batchSize, interestNum, seqLen) *)
//        capsuleWeight = capsuleWeight.add(deltaWeight)
//      } else {
//        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHat)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
//          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        interestCapsule = scalarFactor.mul(interestCapsule)
//      }
//    }
//    for (i <- 0 until 0) {
////      val attenMask = mask.unsqueeze(1).repeat(1, interestNum, 1)
//      val attenMask = mask.unsqueeze(1).repeat(Array(1L, interestNum.toLong, 1L, 1L)*)
//
//      val paddings = torch.zeros_like(attenMask)
//
//      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
//      val maskedWeight = torch.where(torch.eq(attenMask, new Scalar(0f)), paddings, capsuleSoftmaxWeight)
//      val unsqueezedWeight = maskedWeight.unsqueeze(2) // [B,K,1,L]
//
//      if (i < 2) {
//        // 前两次迭代：用detach版本更新权重
//        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter) // [B,K,1,D]
//
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
//          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        interestCapsule = scalarFactor.mul(interestCapsule)
//
//        // 修复：矩阵乘法顺序与Python完全一致
//        val capT = interestCapsule.transpose(2, 3).contiguous() // [B,K,D,1]
//        var deltaWeight = torch.matmul(itemEbHatIter, capT) // [B,K,L,D] @ [B,K,D,1] = [B,K,L,1]
//        deltaWeight = deltaWeight.reshape(Array(batchSize, interestNum, seqLen)*)
//        capsuleWeight = capsuleWeight.add(deltaWeight)
//      } else {
//        // 第三次迭代：使用原始不带detach的itemEbHat（修复梯度阻塞）
//        interestCapsule = torch.matmul(unsqueezedWeight, itemEbHat)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1f)))
//          .div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        interestCapsule = scalarFactor.mul(interestCapsule)
//      }
//    }

    // flatten输出
    val result = interestCapsule.reshape(Array(batchSize, interestNum, embeddingDim)*)
    if (reluLayer && relu.isDefined) relu.get.forward(result) else result
  }
}

object CapsuleNetwork {
  def apply(
             embeddingDim: Int,
             seqLen: Int,
             bilinearType: Int = 2,
             interestNum: Int = 4,
             routingTimes: Int = 3,
             reluLayer: Boolean = false,
             device: String = DeviceSupport.backend
           ): CapsuleNetwork = new CapsuleNetwork(embeddingDim, seqLen, bilinearType, interestNum, routingTimes, reluLayer, device)
}

//class CapsuleNetwork(
//  embeddingDim: Int,
//  seqLen: Int,
//  bilinearType: Int = 2,
//  interestNum: Int = 4,
//  routingTimes: Int = 3,
//  reluLayer: Boolean = false,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  private val h = embeddingDim
//  private val s = seqLen
//  private val k = interestNum
//
//  private var linear: Option[LinearImpl] = None
//  private var w: Option[Tensor] = None
//  private val relu = if (reluLayer) {
//    val r = new SequentialImpl()
//    r.push_back("linear", new LinearImpl(embeddingDim, embeddingDim))
//    r.push_back("relu", new ReLUImpl())
//    register_module("relu", r)
//    Some(r)
//  } else None
//
//  bilinearType match {
//    case 0 => // MIND
//      linear = Some {
//        val l = new LinearImpl(embeddingDim, embeddingDim)
//        register_module("linear", l)
//        l
//      }
//    case 1 =>
//      linear = Some {
//        val l = new LinearImpl(embeddingDim, embeddingDim * interestNum)
//        register_module("linear", l)
//        l
//      }
//    case _ =>
//      w = Some {
//        val wt = torch.rand(Array[Long](1L, seqLen.toLong, interestNum.toLong * embeddingDim.toLong, embeddingDim.toLong),
//          new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//        register_parameter("w", wt)
//        wt
//      }
//  }
//
//  def forward(itemEb: Tensor, mask: Tensor): Tensor = {
//    val batchSize = itemEb.size(0)
//
//    val itemEbHat = bilinearType match {
//      case 0 =>
//        val out = linear.get.forward(itemEb)
//        out.repeat(1, 1, interestNum)
//      case 1 =>
//        linear.get.forward(itemEb)
//      case _ =>
//        val u = itemEb.unsqueeze(2)
//        val wt = w.get
//        torch.sum(wt.narrow(1, 0, seqLen) * u, 3)
//    }
//
//    val reshaped = itemEbHat.reshape(batchSize, seqLen, interestNum, embeddingDim)
//    val transposed = reshaped.transpose(1, 2).contiguous()
//    val finalReshaped = transposed.reshape(Array[Long](batchSize, interestNum, seqLen, embeddingDim))
//
//    val itemEbHatIter = if (true) finalReshaped.detach() else finalReshaped
//
//    var capsuleWeight = if (bilinearType > 0) {
//      torch.zeros(Array[Long](batchSize.toLong, interestNum, seqLen),
//        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//    } else {
//      torch.randn(Array[Long](batchSize.toLong, interestNum, seqLen),
//        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
//    }.to(itemEb.device(), ScalarType.Float)
//
//    for (iter <- 0 until routingTimes) {
//      val attenMask = mask.unsqueeze(1).repeat(1, interestNum, 1)
//      val paddings = torch.zeros_like(attenMask)
//
//      val capsuleSoftmaxWeight = torch.softmax(capsuleWeight, -1)
//      val maskedWeight = torch.where(torch.eq(attenMask, new Scalar(0)), paddings, capsuleSoftmaxWeight)
//      val unsqueezedWeight = maskedWeight.unsqueeze(2)
//
//      if (iter < 2) {
//        val interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1.0f))).div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        val scaledCapsule = scalarFactor * interestCapsule
//
//        val deltaWeight = torch.matmul(itemEbHatIter.transpose(2, 3).contiguous(), scaledCapsule)
//        val reshapedDelta = deltaWeight.reshape(Array[Long](batchSize, interestNum, seqLen))
//        capsuleWeight = capsuleWeight.add(reshapedDelta)
//      } else {
//        val interestCapsule = torch.matmul(unsqueezedWeight, itemEbHatIter)
//        val capNorm = torch.sum(torch.square(interestCapsule), -1, 1L)
//        val scalarFactor = capNorm.div(capNorm.add(new Scalar(1.0f))).div(torch.sqrt(capNorm.add(new Scalar(1e-9f))))
//        val scaledCapsule = scalarFactor * interestCapsule
//      }
//    }
//
//    val result = itemEbHatIter.reshape(Array[Long](batchSize, interestNum, embeddingDim))
//
//    if (reluLayer && relu.isDefined) {
//      relu.get.forward(result)
//    } else {
//      result
//    }
//  }
//}
//
///**
// * CapsuleNetwork companion object with factory methods.
// */
//object CapsuleNetwork {
//  def apply(embeddingDim: Int, seqLen: Int, bilinearType: Int = 2, interestNum: Int = 4,
//            routingTimes: Int = 3, reluLayer: Boolean = false, device: String = DeviceSupport.backend): CapsuleNetwork = {
//    new CapsuleNetwork(embeddingDim, seqLen, bilinearType, interestNum, routingTimes, reluLayer, device)
//  }
//}