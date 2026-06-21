package tutorials

import benchmarks.rec.RunGradNormReplicaBenchmark
import benchmarks.rec.RunMetaBalanceReplicaBenchmark

object TutorialMultiTaskNotebookReplica {
  def main(args: Array[String]): Unit = {
    RunGradNormReplicaBenchmark.main(Array("--model_name", "SharedBottom", "--epoch", "1", "--batch_size", "1024", "--device", "cpu"))
    RunMetaBalanceReplicaBenchmark.main(Array("--model_name", "SharedBottom", "--epoch", "1", "--batch_size", "1024", "--device", "cpu"))
  }
}

