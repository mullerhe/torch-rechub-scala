package torchrec.models.multi_task

import torchrec.basic.features._
import torchrec.basic.layers._
import torchrec.utils.DeviceSupport
import torchrec.Implicits._
import torchrec.TensorImplicits.RichTensor

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType

import scala.collection.mutable

/**
 * Progressive Layered Extraction
 * Reference: Tencent, RecSys 2020
 */
class PLE(
           features: List[Feature],
           taskNames: List[String],
           embedDim: Int = 8,
           numSharedExperts: Int = 2,
           numTaskExperts: Int = 2,
           numLayers: Int = 3,
           expertDims: List[Long] = List(128L),
           towerDims: List[Long] = List(64L),
           dropout: Float = 0.2f,
           device: String = DeviceSupport.backend
         ) extends Module {
  private val targetDevice = new Device(device)
  private val floatOptions = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))

  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embeddingLayer)

  // Expert layers
  private val expertLayers = (0 until numLayers).map { layerIdx =>
    val numExperts = if (layerIdx == numLayers - 1) numTaskExperts * taskNames.size else numSharedExperts + numTaskExperts * taskNames.size

    val experts = (0 until numExperts).map { i =>
      val expert = new MLP(embedDim, expertDims, embedDim, "relu", dropout, device = device)
      register_module(s"expert_layer${layerIdx}_$i", expert)
      expert
    }
    experts
  }

  // Task-specific gates
  private val taskGates = taskNames.map { name =>
    val gate = new LinearImpl(embedDim, numSharedExperts + numTaskExperts)
    gate.to(targetDevice, true)
    register_module(s"gate_$name", gate)
    (name, gate)
  }.toMap

  // Task towers
  private val taskTowers = taskNames.map { name =>
    val tower = new MLP(embedDim, towerDims, 1, "relu", dropout, device = device)
    register_module(s"tower_$name", tower)
    (name, tower)
  }.toMap

  def forward(
               sparseFeats: Map[String, Tensor],
               denseFeats: Map[String, Tensor] = Map.empty
             ): Map[String, Tensor] = {
    val embeddings = embeddingLayer.forward(sparseFeats)
    val batchSize = embeddings.size(0)

    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val reshaped = embeddings.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
    var currentInput = reshaped.sum(1).to(targetDevice, ScalarType.Float)
    var result: Map[String, Tensor] = Map.empty

    var layerIdx = 0
    while (layerIdx < numLayers) {
      val experts = expertLayers(layerIdx)
      val isLastLayer = layerIdx == numLayers - 1
      val expertOutputs = experts.map(_.forward(currentInput).to(targetDevice, ScalarType.Float))

      if (isLastLayer) {
        val taskExpertOutputs = taskNames.map { name =>
          val startIdx = taskNames.indexOf(name) * numTaskExperts
          expertOutputs.slice(startIdx, startIdx + numTaskExperts)
        }

        result = taskNames.map { name =>
          val taskExperts = taskExpertOutputs(taskNames.indexOf(name))
          val gate = taskGates(name)
          val gateWeights = gate.forward(currentInput).softmax(1)

          val paddedGateWeights = if (gateWeights.size(1) < taskExperts.size) {
            val padSize = taskExperts.size - gateWeights.size(1).toInt
            val pad = torch.zeros(Array(gateWeights.size(0), padSize.toLong), floatOptions)
            torch.cat(new TensorVector(gateWeights, pad), 1)
          } else {
            gateWeights.narrow(1, 0, taskExperts.size)
          }

          val weightedExperts = taskExperts.zipWithIndex.map { case (out, i) =>
            paddedGateWeights.select(1, i).unsqueeze(1).mul(out)
          }

          val combined = weightedExperts.reduce(_.add(_))
          val taskOut = taskTowers(name).forward(combined)
          (name, taskOut)
        }.toMap
        layerIdx = numLayers

      } else {
        val allExpertOuts = expertOutputs
        // ===================== 修复核心：不要拼接！保持维度不变 =====================
        val nextInputs = taskNames.map { name =>
          val gateWeights = taskGates(name).forward(currentInput).softmax(1)
          val paddedWeights = if (gateWeights.size(1) < allExpertOuts.size) {
            val padSize = allExpertOuts.size - gateWeights.size(1).toInt
            val pad = torch.zeros(Array(gateWeights.size(0), padSize.toLong), floatOptions)
            torch.cat(new TensorVector(gateWeights, pad), 1)
          } else {
            gateWeights.narrow(1, 0, allExpertOuts.size)
          }

          allExpertOuts.zipWithIndex.map { case (out, i) =>
            paddedWeights.select(1, i).unsqueeze(1).mul(out)
          }.reduce(_.add(_))
        }

        // 关键修复：对各个任务输出取平均 → 保持维度依然是 embedDim！
        val tensorVec = new TensorVector()
        nextInputs.foreach(t => tensorVec.push_back(t.to(targetDevice, ScalarType.Float)))
        val stacked = torch.stack(tensorVec, 0)
        currentInput = stacked.mean(0).to(targetDevice, ScalarType.Float)

        layerIdx += 1
      }
    }
    result
  }
}


//package torchrec.models.multi_task
//
//import torchrec.basic.features._
//import torchrec.basic.layers._
//import torchrec.utils.DeviceSupport
//import torchrec.Implicits._
//import torchrec.TensorImplicits.RichTensor
//
//import org.bytedeco.pytorch._
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
//import scala.collection.mutable
//
///**
// * Progressive Layered Extraction
// * Reference: Tencent, RecSys 2020
// */
//class PLE(
//           features: List[Feature],
//           taskNames: List[String],
//           embedDim: Int = 8,
//           numSharedExperts: Int = 2,
//           numTaskExperts: Int = 2,
//           numLayers: Int = 3,
//           expertDims: List[Long] = List(128L),
//           towerDims: List[Long] = List(64L),
//           dropout: Float = 0.2f,
//           device: String = DeviceSupport.backend
//         ) extends Module {
//  private val targetDevice = new Device(device)
//  private val floatOptions = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).device(new DeviceOptional(targetDevice))
//
//  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
//  register_module("embedding", embeddingLayer)
//
//  // Expert layers
//  private val expertLayers = (0 until numLayers).map { layerIdx =>
//    val numExperts = if (layerIdx == numLayers - 1) numTaskExperts * taskNames.size else numSharedExperts + numTaskExperts * taskNames.size
//
//    val experts = (0 until numExperts).map { i =>
//      val expert = new MLP(embedDim, expertDims, embedDim, "relu", dropout, device = device)
//      register_module(s"expert_layer${layerIdx}_$i", expert)
//      expert
//    }
//    experts
//  }
//
//  // Task-specific gates (修复：正确移到GPU + 注册)
//  private val taskGates = taskNames.map { name =>
//    val gate = new LinearImpl(embedDim, numSharedExperts + numTaskExperts)
//    gate.to(targetDevice, true)
//    register_module(s"gate_$name", gate)
//    (name, gate)
//  }.toMap
//
//  // Task towers
//  private val taskTowers = taskNames.map { name =>
//    val tower = new MLP(embedDim, towerDims, 1, "relu", dropout, device = device)
//    register_module(s"tower_$name", tower)
//    (name, tower)
//  }.toMap
//
//  def forward(
//               sparseFeats: Map[String, Tensor],
//               denseFeats: Map[String, Tensor] = Map.empty
//             ): Map[String, Tensor] = {
//    val embeddings = embeddingLayer.forward(sparseFeats)
//    val batchSize = embeddings.size(0)
//
//    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
//    val reshaped = embeddings.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
//    var currentInput = reshaped.sum(1).to(targetDevice, ScalarType.Float)
//    var result: Map[String, Tensor] = Map.empty
//
//    var layerIdx = 0
//    while (layerIdx < numLayers) {
//      val experts = expertLayers(layerIdx)
//      val isLastLayer = layerIdx == numLayers - 1
//      val expertOutputs = experts.map(_.forward(currentInput).to(targetDevice,ScalarType.Float))
//
//      if (isLastLayer) {
//        val taskExpertOutputs = taskNames.map { name =>
//          val startIdx = taskNames.indexOf(name) * numTaskExperts
//          expertOutputs.slice(startIdx, startIdx + numTaskExperts)
//        }
//
//        result = taskNames.map { name =>
//          val taskExperts = taskExpertOutputs(taskNames.indexOf(name))
//          val gate = taskGates(name)
//          val gateWeights = gate.forward(currentInput).softmax(1)
//
//          // 修复：统一设备 + 正确创建张量
//          val paddedGateWeights = if (gateWeights.size(1) < taskExperts.size) {
//            val padSize = taskExperts.size - gateWeights.size(1).toInt
//            val pad = torch.zeros(Array(gateWeights.size(0), padSize.toLong), floatOptions)
//            torch.cat(new TensorVector(gateWeights, pad), 1)
//          } else {
//            gateWeights.narrow(1, 0, taskExperts.size)
//          }
//
//          val weightedExperts = taskExperts.zipWithIndex.map { case (out, i) =>
//            paddedGateWeights.select(1, i).unsqueeze(1).mul(out)
//          }
//
//          val combined = weightedExperts.reduce(_.add(_))
//          val taskOut = taskTowers(name).forward(combined)
//          (name, taskOut)
//        }.toMap
//        layerIdx = numLayers
//
//      } else {
//        val allExpertOuts = expertOutputs
//        val nextInputs = taskNames.map { name =>
//          val gateWeights = taskGates(name).forward(currentInput).softmax(1)
//          val paddedWeights = if (gateWeights.size(1) < allExpertOuts.size) {
//            val padSize = allExpertOuts.size - gateWeights.size(1).toInt
//            val pad = torch.zeros(Array(gateWeights.size(0), padSize.toLong), floatOptions)
//            torch.cat(new TensorVector(gateWeights, pad), 1)
//          } else {
//            gateWeights.narrow(1, 0, allExpertOuts.size)
//          }
//
//          allExpertOuts.zipWithIndex.map { case (out, i) =>
//            paddedWeights.select(1, i).unsqueeze(1).mul(out)
//          }.reduce(_.add(_))
//        }
//
//        // 修复：cat 前全部移到正确设备
//        val tensorVec = new TensorVector()
//        nextInputs.foreach(t => tensorVec.push_back(t.to(targetDevice, ScalarType.Float)))
//        val scale = new Scalar(taskNames.size.toFloat)
//        currentInput = torch.cat(tensorVec, 1L).div(scale).to(targetDevice, ScalarType.Float)
//        layerIdx += 1
//      }
//    }
//    result
//  }
//}
//
////package torchrec.models.multi_task
////
////import torchrec.basic.features._
////import torchrec.basic.layers._
////import torchrec.utils.DeviceSupport
////import torchrec.Implicits._
////import torchrec.TensorImplicits.RichTensor
////
////import org.bytedeco.pytorch._
////import org.bytedeco.pytorch.global.torch
////import org.bytedeco.pytorch.global.torch.ScalarType
////
////import scala.collection.mutable
////
/////**
//// * Progressive Layered Extraction
//// * Reference: Tencent, RecSys 2020
//// */
////class PLE(
////  features: List[Feature],
////  taskNames: List[String],
////  embedDim: Int = 8,
////  numSharedExperts: Int = 2,
////  numTaskExperts: Int = 2,
////  numLayers: Int = 3,
////  expertDims: List[Long] = List(128L),
////  towerDims: List[Long] = List(64L),
////  dropout: Float = 0.2f,
////  device: String = DeviceSupport.backend
////) extends Module {
////
////  private val embeddingLayer = new EmbeddingLayer(features, embedDim, device)
////  register_module("embedding", embeddingLayer)
////
////  // Expert layers
////  private val expertLayers = (0 until numLayers).map { layerIdx =>
////    val numExperts = if (layerIdx == numLayers - 1) numTaskExperts * taskNames.size else numSharedExperts + numTaskExperts * taskNames.size
////
////    val experts = (0 until numExperts).map { i =>
////      val expert = new MLP(embedDim, expertDims, embedDim, "relu", dropout, device = device)
////      register_module(s"expert_layer${layerIdx}_$i", expert)
////      expert
////    }
////    experts
////  }
////
////  // Task-specific gates
////  private val taskGates = taskNames.map { name =>
////    val gate = new LinearImpl(embedDim, numSharedExperts + numTaskExperts)
////    gate.to(new Device(device),false)
////    register_module(s"gate_$name", gate)
////    (name, gate)
////  }.toMap
////
////  // Task towers
////  private val taskTowers = taskNames.map { name =>
////    val tower = new MLP(embedDim, towerDims, 1, "relu", dropout, device = device)
////    register_module(s"tower_$name", tower)
////    (name, tower)
////  }.toMap
////
////  def forward(
////    sparseFeats: Map[String, Tensor],
////    denseFeats: Map[String, Tensor] = Map.empty
////  ): Map[String, Tensor] = {
////    val embeddings = embeddingLayer.forward(sparseFeats)
////    val batchSize = embeddings.size(0)
////
////    // Reshape to (batch, numFields, embedDim) then sum-pool over fields
////    // so gates and experts receive (batch, embedDim) regardless of numFields
////    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
////    val reshaped = embeddings.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
////    var currentInput = reshaped.sum(1) // (batch, embedDim)
////    var result: Map[String, Tensor] = Map.empty
////
////    // Progressive extraction layers
////    var layerIdx = 0
////    while (layerIdx < numLayers) {
////      val experts = expertLayers(layerIdx)
////      val isLastLayer = layerIdx == numLayers - 1
////
////      val numExperts = experts.size
////      val expertOutputs = experts.map(_.forward(currentInput))
////
////      if (isLastLayer) {
////        // Split experts by task
////        val taskExpertOutputs = taskNames.map { name =>
////          val startIdx = taskNames.indexOf(name) * numTaskExperts
////          expertOutputs.slice(startIdx, startIdx + numTaskExperts)
////        }
////
////        result = taskNames.map { name =>
////          val taskExperts = taskExpertOutputs(taskNames.indexOf(name))
////          val numTaskExpertsCurrent = taskExperts.size
////
////          // Gate
////          val gateWeights = taskGates(name).forward(currentInput).softmax(1)
////          // Ensure gateWeights matches number of task experts
////          val paddedGateWeights = if (gateWeights.size(1) < numTaskExpertsCurrent) {
////            val padSize = numTaskExpertsCurrent - gateWeights.size(1)
////            val dev = gateWeights.device()
////            val pad = torch.zeros(Array(gateWeights.size(0).toLong, padSize.toLong),
////              new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
////              .to(dev, ScalarType.Float)
////            // Make sure gateWeights is on the same device
////            val gwOnDev = if (dev.`type`().toString != "cpu") gateWeights.to(dev, ScalarType.Float) else gateWeights
////            torch.cat(new TensorVector(gwOnDev, pad), 1)
////          } else gateWeights
////          val weightedExperts = taskExperts.zipWithIndex.map { case (expertOut, i) =>
////            paddedGateWeights.select(1, i).unsqueeze(1).mul(expertOut)
////          }
////          val combined = weightedExperts.reduce(_.add(_))
////          val taskOut = taskTowers(name).forward(combined)
////          (name, taskOut)
////        }.toMap
////        layerIdx = numLayers // Exit loop
////
////      } else {
////        // Shared extraction
////        val allExpertOuts = expertOutputs
////        val numTotalExperts = allExpertOuts.size
////
////        // Update input for next layer - use all experts
////        val nextInputs = taskNames.map { name =>
////          val gateWeights = taskGates(name).forward(currentInput).softmax(1)
////          // Ensure gateWeights has same size as expert outputs
////          val paddedWeights = if (gateWeights.size(1) < numTotalExperts) {
////            val padSize = numTotalExperts - gateWeights.size(1)
////            val dev = gateWeights.device()
////            val pad = torch.zeros(Array(gateWeights.size(0).toLong, padSize.toLong),
////              new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
////              .to(dev, ScalarType.Float)
////            // Make sure gateWeights is also on the same device
////            val gateOnDev = if (dev.`type`().toString != "cpu") gateWeights.to(dev, ScalarType.Float) else gateWeights
////            torch.cat(new TensorVector(gateOnDev, pad), 1)
////          } else gateWeights
////          val weightedExperts = allExpertOuts.zipWithIndex.map { case (expertOut, i) =>
////            val w = paddedWeights.select(1, i).unsqueeze(1)
////            w.mul(expertOut)
////          }
////          val combined = weightedExperts.reduce(_.add(_))
////          combined
////        }
////
////        // Move all inputs to device before cat
////        val tensorVec = new TensorVector(nextInputs.size.toLong)
////        val targetDev = new Device(device)
////        nextInputs.foreach { t =>
////          tensorVec.push_back(t.to(targetDev, ScalarType.Float))
////        }
////        val scale = new Scalar(taskNames.size.toFloat)
////        currentInput = torch.cat(tensorVec, 1L).div(scale)
////        layerIdx += 1
////      }
////    }
////    result
////  }
////}
