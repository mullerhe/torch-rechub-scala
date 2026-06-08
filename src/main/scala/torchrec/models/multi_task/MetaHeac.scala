package torchrec.models.multi_task

import torchrec.basic.features.*
import torchrec.basic.layers.*
import torchrec.utils.DeviceSupport
import org.bytedeco.pytorch.{Device, *}
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.ScalarType
/**
 * Meta Hybrid Experts and Critics (MetaHeac)
 * Reference: "Learning to Expand Audience via Meta Hybrid Experts and Critics
 *             for Recommendation and Advertising" - Zhu et al., KDD 2021
 *
 * Architecture:
 *   - EmbeddingLayer: shared categorical + numerical embeddings
 *   - Expert gates: route input to multiple expert MLPs
 *   - Task embeddings (Meta_Embedding): learnable per-task embeddings
 *   - Critics: per-task output MLPs weighted by critic gates
 *
 * Note: This implementation includes the forward pass. For MAML training,
 * local/global update logic from the original Python is needed, but for
 * inference benchmarking, the forward pass suffices.
 *
 * @param features         List of sparse features
 * @param taskNames        Names of tasks
 * @param embedDim         Embedding dimension
 * @param bottomDims       Expert MLP hidden dimensions
 * @param towerDims        Critic MLP hidden dimensions
 * @param expertNum        Number of expert networks
 * @param criticNum        Number of critic networks per task
 * @param dropout          Dropout rate
 * @param device           Device
 */
class MetaHeac(
                features: List[Feature],
                taskNames: List[String],
                embedDim: Int = 8,
                bottomDims: List[Long] = List(128L, 64L),
                towerDims: List[Long] = List(32L, 16L),
                expertNum: Int = 4,
                criticNum: Int = 5,
                dropout: Float = 0.2f,
                device: String = DeviceSupport.backend
              ) extends Module {

  require(features.nonEmpty, "features cannot be empty")
  require(taskNames.nonEmpty, "taskNames cannot be empty")
  require(expertNum >= 1, s"expertNum must be >= 1, got $expertNum")
  require(criticNum >= 1, s"criticNum must be >= 1, got $criticNum")

  private val taskNum = taskNames.size
  private val sparseDim = Features.calcSparseDim(features)
  private val targetDevice = new Device(device)

  // Embedding layer
  private val embedding = new EmbeddingLayer(features, embedDim, device)
  register_module("embedding", embedding)

  // Task embeddings
  private val taskEmbeddings = taskNames.zipWithIndex.map { case (name, idx) =>
    val emb = new MetaEmbedding(taskNum, embedDim, device)
    emb.to(targetDevice, false)
    register_module(s"taskEmbedding_$name", emb)
    (name, (emb, idx))
  }.toMap

  // Experts
  private val experts = (0 until expertNum).map { i =>
    val expert = new MLP(sparseDim, bottomDims, bottomDims.last, "relu", dropout, device = device)
    register_module(s"expert_$i", expert)
    expert
  }

  // Expert gate
  private val expertGate = new MetaLinear(embedDim * 2, expertNum, device)
  expertGate.to(targetDevice, false)
  register_module("expertGate", expertGate)

  // Critics
  private val critics = taskNames.map { name =>
    val taskCritics = (0 until criticNum).map { i =>
      val critic = new MLP(bottomDims.last, towerDims, 1, "relu", dropout, device = device)
      register_module(s"critic_${name}_$i", critic)
      critic
    }
    (name, taskCritics)
  }.toMap

  // Critic gates
  private val criticGates = taskNames.map { name =>
    val gate = new MetaLinear(embedDim * 2, criticNum, device)
    gate.to(targetDevice, false)
    register_module(s"criticGate_$name", gate)
    (name, gate)
  }.toMap

  def forward(
               sparseFeats: Map[String, Tensor],
               taskIdx: Tensor
             ): Map[String, Tensor] = {
    val emb = embedding.forward(sparseFeats)
    emb.to(targetDevice, ScalarType.Float)
    val batchSize = emb.size(0)

    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val reshaped = emb.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
    val pooled = reshaped.mean(1)
    pooled.to(targetDevice, ScalarType.Float)

    val taskIdxCuda = taskIdx.to(targetDevice, ScalarType.Long)

    val expertGateInputsList = taskNames.map { name =>
      val (taskEmb, idx) = taskEmbeddings(name)
      val taskEmbBatched = taskEmb.forward(taskIdxCuda)
      taskEmbBatched.to(targetDevice, ScalarType.Float)

      val vec = new TensorVector()
      vec.push_back(taskEmbBatched)
      vec.push_back(pooled)
      val combined = torch.cat(vec, 1)
      combined.to(targetDevice, ScalarType.Float)

      (name, combined)
    }

    val expertOutputs = experts.map(_.forward(emb))
    expertOutputs.foreach(_.to(targetDevice, ScalarType.Float))

    val expertGateOut = expertGate.forward(expertGateInputsList.head._2)
    val expertWeights = expertGateOut.softmax(1)

    val expertFused = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
      expertWeights.select(1, i).unsqueeze(1).mul(expertOut)
    }.reduce(_.add(_))
    expertFused.to(targetDevice, ScalarType.Float)

    taskNames.map { name =>
      val (taskEmb, _) = taskEmbeddings(name)
      val taskEmbBatched = taskEmb.forward(taskIdxCuda)
      taskEmbBatched.to(targetDevice, ScalarType.Float)

      val vec = new TensorVector()
      vec.push_back(taskEmbBatched)
      vec.push_back(pooled)
      val combinedCritic = torch.cat(vec, 1)
      combinedCritic.to(targetDevice, ScalarType.Float)

      val criticGateOut = criticGates(name).forward(combinedCritic)
      val criticWeights = criticGateOut.softmax(1)

      val taskCritics = critics(name)
      val criticOutputs = taskCritics.map(_.forward(expertFused))
      criticOutputs.foreach(_.to(targetDevice, ScalarType.Float))

      val combinedOut = criticOutputs.zipWithIndex.map { case (criticOut, i) =>
        criticWeights.select(1, i).unsqueeze(1).mul(criticOut)
      }.reduce(_.add(_))

      val sigmoidOut = combinedOut.sigmoid()
      sigmoidOut.to(targetDevice, ScalarType.Float)

      (name, sigmoidOut)
    }.toMap
  }

  def forwardByName(
                     sparseFeats: Map[String, Tensor]
                   ): Map[String, Tensor] = {
    val emb = embedding.forward(sparseFeats)
    emb.to(targetDevice, ScalarType.Float)
    val batchSize = emb.size(0)

    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
    val reshaped = emb.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
    val pooled = reshaped.mean(1)
    pooled.to(targetDevice, ScalarType.Float)

    val expertOutputs = experts.map(_.forward(emb))
    expertOutputs.foreach(_.to(targetDevice, ScalarType.Float))

    taskNames.map { name =>
      val (taskEmb, idx) = taskEmbeddings(name)

      // 完全按你的写法创建张量 + 迁移设备
      var taskIdxTensor = torch.zeros(Array(batchSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
      taskIdxTensor = taskIdxTensor.to(targetDevice, ScalarType.Long)

      val taskEmbBatched = taskEmb.forward(taskIdxTensor)
      taskEmbBatched.to(targetDevice, ScalarType.Float)

      // Expert cat
      val vecExpert = new TensorVector()
      vecExpert.push_back(taskEmbBatched)
      vecExpert.push_back(pooled)
      val combinedExpert = torch.cat(vecExpert, 1)
      combinedExpert.to(targetDevice, ScalarType.Float)

      val expertGateOut = expertGate.forward(combinedExpert)
      val expertWeights = expertGateOut.softmax(1)

      val expertFused = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
        expertWeights.select(1, i).unsqueeze(1).mul(expertOut)
      }.reduce(_.add(_))
      expertFused.to(targetDevice, ScalarType.Float)

      // Critic cat
      val vecCritic = new TensorVector()
      vecCritic.push_back(taskEmbBatched)
      vecCritic.push_back(pooled)
      val combinedCritic = torch.cat(vecCritic, 1)
      combinedCritic.to(targetDevice, ScalarType.Float)

      val criticGateOut = criticGates(name).forward(combinedCritic)
      val criticWeights = criticGateOut.softmax(1)

      val taskCritics = critics(name)
      val criticOutputs = taskCritics.map(_.forward(expertFused))
      criticOutputs.foreach(_.to(targetDevice, ScalarType.Float))

      val combinedOut = criticOutputs.zipWithIndex.map { case (criticOut, i) =>
        criticWeights.select(1, i).unsqueeze(1).mul(criticOut)
      }.reduce(_.add(_))

      val out = combinedOut.sigmoid()
      out.to(targetDevice, ScalarType.Float)

      (name, out)
    }.toMap
  }
}

class MetaEmbedding(
                     numEmbeddings: Int,
                     embeddingDim: Int,
                     device: String = DeviceSupport.backend
                   ) extends Module {
  private val targetDevice = new Device(device)
  private val embedding = new EmbeddingImpl(new EmbeddingOptions(numEmbeddings, embeddingDim))
  embedding.to(targetDevice, false)
  register_module("embedding", embedding)

  def forward(x: Tensor): Tensor = {
    val res = embedding.forward(x)
    res.to(targetDevice, ScalarType.Float)
  }

  def forwardFast(x: Tensor, fastWeight: Tensor): Tensor = {
    forward(x)
  }
}

class MetaLinear(
                  inFeatures: Long,
                  outFeatures: Long,
                  device: String = DeviceSupport.backend
                ) extends Module {
  private val targetDevice = new Device(device)
  private val linear = new LinearImpl(inFeatures, outFeatures)
  linear.to(targetDevice, false)
  register_module("linear", linear)

  def forward(x: Tensor): Tensor = {
    val res = linear.forward(x)
    res.to(targetDevice, ScalarType.Float)
  }

  def forwardFast(x: Tensor, fastWeight: Tensor, fastBias: Tensor): Tensor = {
    forward(x)
  }
}


//package torchrec.models.multi_task
//
//import torchrec.basic.features.*
//import torchrec.basic.layers.*
//import torchrec.utils.DeviceSupport
//import org.bytedeco.pytorch.{Device, *}
//import org.bytedeco.pytorch.global.torch
//import org.bytedeco.pytorch.global.torch.ScalarType
//
///**
// * Meta Hybrid Experts and Critics (MetaHeac)
// * Reference: "Learning to Expand Audience via Meta Hybrid Experts and Critics
// *             for Recommendation and Advertising" - Zhu et al., KDD 2021
// *
// * Architecture:
// *   - EmbeddingLayer: shared categorical + numerical embeddings
// *   - Expert gates: route input to multiple expert MLPs
// *   - Task embeddings (Meta_Embedding): learnable per-task embeddings
// *   - Critics: per-task output MLPs weighted by critic gates
// *
// * Note: This implementation includes the forward pass. For MAML training,
// * local/global update logic from the original Python is needed, but for
// * inference benchmarking, the forward pass suffices.
// *
// * @param features         List of sparse features
// * @param taskNames        Names of tasks
// * @param embedDim         Embedding dimension
// * @param bottomDims       Expert MLP hidden dimensions
// * @param towerDims        Critic MLP hidden dimensions
// * @param expertNum        Number of expert networks
// * @param criticNum        Number of critic networks per task
// * @param dropout          Dropout rate
// * @param device           Device
// */
//class MetaHeac(
//  features: List[Feature],
//  taskNames: List[String],
//  embedDim: Int = 8,
//  bottomDims: List[Long] = List(128L, 64L),
//  towerDims: List[Long] = List(32L, 16L),
//  expertNum: Int = 4,
//  criticNum: Int = 5,
//  dropout: Float = 0.2f,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  require(features.nonEmpty, "features cannot be empty")
//  require(taskNames.nonEmpty, "taskNames cannot be empty")
//  require(expertNum >= 1, s"expertNum must be >= 1, got $expertNum")
//  require(criticNum >= 1, s"criticNum must be >= 1, got $criticNum")
//
//  private val taskNum = taskNames.size
//  private val sparseDim = Features.calcSparseDim(features)
//
//  // Embedding layer for categorical features
//  private val embedding = new EmbeddingLayer(features, embedDim, device)
//  register_module("embedding", embedding)
//
//  // Task embeddings (Meta-style fast-weight embedding layer)
//  private val taskEmbeddings = taskNames.zipWithIndex.map { case (name, idx) =>
//    val emb = new MetaEmbedding(taskNum, embedDim)
//    emb.to(new Device(device),false)
//    register_module(s"taskEmbedding_$name", emb)
//    (name, (emb, idx))
//  }.toMap
//
//  // Expert networks
//  private val experts = (0 until expertNum).map { i =>
//    val expert = new MLP(sparseDim, bottomDims, bottomDims.last, "relu", dropout, device = device)
//    register_module(s"expert_$i", expert)
//    expert
//  }
//
//  // Expert gate: routes based on (task_embedding, mean_embed)
//  private val expertGate = new MetaLinear(embedDim * 2, expertNum)
//  expertGate.to(new Device(device),false)
//  register_module("expertGate", expertGate)
//
//  // Critics (multiple per task, weighted by critic gates)
//  private val critics = taskNames.map { name =>
//    val taskCritics = (0 until criticNum).map { i =>
//      val critic = new MLP(bottomDims.last, towerDims, 1, "relu", dropout, device = device)
//      register_module(s"critic_${name}_$i", critic)
//      critic
//    }
//    (name, taskCritics)
//  }.toMap
//
//  // Critic gate: per task, routes across critics
//  private val criticGates = taskNames.map { name =>
//    val gate = new MetaLinear(embedDim * 2, criticNum)
//    gate.to(new Device(device),false)
//    register_module(s"criticGate_$name", gate)
//    (name, gate)
//  }.toMap
//
//  def forward(
//    sparseFeats: Map[String, Tensor],
//    taskIdx: Tensor  // (batch_size,) tensor of task indices [0, taskNum)
//  ): Map[String, Tensor] = {
//    val emb = embedding.forward(sparseFeats)
//    val batchSize = emb.size(0)
//
//    // Reshape to (batch, numFields, embedDim) then mean pool
//    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
//    val reshaped = emb.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
//    val pooled = reshaped.mean(1)  // (batch, embedDim)
//
//    // Ensure taskIdx is on correct device
//    val taskIdxDevice = taskIdx.device()
//    val taskIdxCuda = if (device != "cpu" && taskIdxDevice.`type`().toString != device) {
//      taskIdx.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
//    } else taskIdx
//
//    // Ensure pooled is on correct device
//    val pooledDevice = pooled.device()
//    val pooledCuda = if (device != "cpu" && pooledDevice.`type`().toString != device) {
//      pooled.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
//    } else pooled
//
//    // Expert routing: get task embeddings for each sample
//    val expertGateInputsList = taskNames.map { name =>
//      val (taskEmb, idx) = taskEmbeddings(name)
//      // Get the task embedding for the given taskIdx
//      val taskEmbBatched = taskEmb.forward(taskIdxCuda)  // (batch, embedDim)
//      // Combine task embedding with pooled embedding
//      val combined = {
//        val vec = new TensorVector(2L)
//        vec.push_back(taskEmbBatched)
//        vec.push_back(pooledCuda)
//        torch.cat(vec, 1)  // (batch, embedDim * 2)
//      }
//      (name, combined)
//    }
//
//    // Compute expert outputs
//    val expertOutputs = experts.map(_.forward(emb))  // List[(batch, bottomDim)]
//
//    // Expert gating (use first task's gate as shared expert gate)
//    val expertGateOut = expertGate.forward(expertGateInputsList.head._2)  // (batch, expertNum)
//    val expertWeights = expertGateOut.softmax(1)  // (batch, expertNum)
//
//    // Weighted sum of expert outputs
//    val expertFused = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
//      expertWeights.select(1, i).unsqueeze(1).mul(expertOut)
//    }.reduce(_.add(_))  // (batch, bottomDim)
//
//    // Per-task critics with task-specific critic gates
//    taskNames.map { name =>
//      val (taskEmb, _) = taskEmbeddings(name)
//      val taskEmbBatched = taskEmb.forward(taskIdxCuda)
//
//      // Critic gate
//      val combinedCritic = {
//        val vec = new TensorVector(2L)
//        vec.push_back(taskEmbBatched)
//        vec.push_back(pooledCuda)
//        torch.cat(vec, 1)
//      }
//      val criticGateOut = criticGates(name).forward(combinedCritic)  // (batch, criticNum)
//      val criticWeights = criticGateOut.softmax(1)  // (batch, criticNum)
//
//      // Apply critics
//      val taskCritics = critics(name)
//      val criticOutputs = taskCritics.map(_.forward(expertFused))  // List[(batch, 1)]
//
//      // Weighted combination of critic outputs
//      val combinedOut = criticOutputs.zipWithIndex.map { case (criticOut, i) =>
//        criticWeights.select(1, i).unsqueeze(1).mul(criticOut)
//      }.reduce(_.add(_))  // (batch, 1)
//
//      val sigmoidOut = combinedOut.sigmoid()
//      (name, sigmoidOut)
//    }.toMap
//  }
//
//  /**
//   * Forward with explicit task name (no task index tensor needed)
//   * Returns predictions for all tasks
//   */
//  def forwardByName(
//    sparseFeats: Map[String, Tensor]
//  ): Map[String, Tensor] = {
//    val emb = embedding.forward(sparseFeats)
//    val batchSize = emb.size(0)
//
//    // Reshape and mean pool
//    val numSparseFeatures = features.collect { case f: SparseFeature => 1 }.size
//    val reshaped = emb.view(batchSize, numSparseFeatures.toLong, embedDim.toLong)
//    val pooled = reshaped.mean(1)  // (batch, embedDim)
//
//    // Expert outputs
//    val expertOutputs = experts.map(_.forward(emb))
//
//    // Per-task computation
//    taskNames.map { name =>
//      val (taskEmb, idx) = taskEmbeddings(name)
//      // Create task index tensor on correct device
//      var taskIdxTensor = torch.zeros(Array(batchSize.toLong), new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
//      if (device != "cpu") {
//        taskIdxTensor = taskIdxTensor.to(new org.bytedeco.pytorch.Device(device), ScalarType.Long)
//      } else {
//        taskIdxTensor = taskIdxTensor.to(new org.bytedeco.pytorch.Device("cpu"), ScalarType.Long)
//      }
//      val taskEmbBatched = taskEmb.forward(taskIdxTensor)
//
//      // Ensure pooled is on the correct device
//      val pooledDevice = pooled.device()
//      val pooledCuda = if (device != "cpu" && pooledDevice.`type`().toString != device) {
//        pooled.to(new org.bytedeco.pytorch.Device(device), ScalarType.Float)
//      } else pooled
//
//      // Expert gate
//      val combinedCombined = {
//        val vec = new TensorVector(2L)
//        vec.push_back(taskEmbBatched)
//        vec.push_back(pooledCuda)
//        torch.cat(vec, 1)
//      }
//      val expertGateOut = expertGate.forward(combinedCombined)
//      val expertWeights = expertGateOut.softmax(1)
//
//      val expertFused = expertOutputs.zipWithIndex.map { case (expertOut, i) =>
//        expertWeights.select(1, i).unsqueeze(1).mul(expertOut)
//      }.reduce(_.add(_))
//
//      // Critic gate
//      val combinedCritic2 = {
//        val vec = new TensorVector(2L)
//        vec.push_back(taskEmbBatched)
//        vec.push_back(pooledCuda)
//        torch.cat(vec, 1)
//      }
//      val criticGateOut = criticGates(name).forward(combinedCritic2)
//      val criticWeights = criticGateOut.softmax(1)
//
//      val taskCritics = critics(name)
//      val criticOutputs = taskCritics.map(_.forward(expertFused))
//
//      val combinedOut = criticOutputs.zipWithIndex.map { case (criticOut, i) =>
//        criticWeights.select(1, i).unsqueeze(1).mul(criticOut)
//      }.reduce(_.add(_))
//
//      (name, combinedOut.sigmoid())
//    }.toMap
//  }
//}
//
///**
// * Meta Embedding - supports fast weight updates (MAML-style)
// * Reference: "Learning to Expand Audience" - KDD 2021
// */
//class MetaEmbedding(
//  numEmbeddings: Int,
//  embeddingDim: Int,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  private val embedding = new EmbeddingImpl(
//    new EmbeddingOptions(numEmbeddings, embeddingDim)
//  )
//  embedding.to(new Device(device),false)
//  register_module("embedding", embedding)
//
//  def forward(x: Tensor): Tensor = {
//    // x: (batch_size,) task indices
//    embedding.forward(x)
//  }
//
//  def forwardFast(x: Tensor, fastWeight: Tensor): Tensor = {
//    // Use fast (adapted) weight instead of base weight
//    // For inference, just use regular forward
//    forward(x)
//  }
//}
//
///**
// * Meta Linear - supports fast weight updates (MAML-style)
// * Reference: "Learning to Expand Audience" - KDD 2021
// */
//class MetaLinear(
//  inFeatures: Long,
//  outFeatures: Long,
//  device: String = DeviceSupport.backend
//) extends Module {
//
//  private val linear = new LinearImpl(inFeatures, outFeatures)
//  linear.to(new Device(device),false)
//  register_module("linear", linear)
//
//  def forward(x: Tensor): Tensor = {
//    linear.forward(x)
//  }
//
//  def forwardFast(x: Tensor, fastWeight: Tensor, fastBias: Tensor): Tensor = {
//    // Use fast (adapted) weight and bias
//    // For inference, just use regular forward
//    forward(x)
//  }
//}
