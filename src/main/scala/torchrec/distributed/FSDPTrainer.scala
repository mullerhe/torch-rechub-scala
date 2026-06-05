package torchrec.distributed

import org.bytedeco.pytorch._
import org.bytedeco.pytorch.global.torch
import torchrec.data.Batch

import scala.collection.mutable

class FSDPTrainer(
  model: Module,
  config: FSDPConfig,
  learningRate: Float = 1e-3f,
  weightDecay: Float = 1e-6f
) {
  private val context = new DistributedContext(DDPConfig(
    rank = config.rank,
    worldSize = config.worldSize,
    masterAddr = config.masterAddr,
    masterPort = config.masterPort
  ))

  private val optimizer = {
    val params = model.parameters()
    val vec = new TensorVector(params.size())
    var i = 0L
    while (i < params.size()) { vec.push_back(params.get(i)); i += 1 }
    new Adam(vec, new AdamOptions(learningRate))
  }

  def step(batch: Batch, lossFn: Batch => Tensor): Tensor = {
    optimizer.zero_grad()
    val loss = lossFn(batch)
    loss.backward()
    optimizer.step()
    loss
  }

  def getContext: DistributedContext = context

  def cleanup(): Unit = {}
}
