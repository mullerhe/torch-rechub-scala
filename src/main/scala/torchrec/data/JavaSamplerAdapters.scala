package torchrec.data

import org.bytedeco.pytorch._

/**
 * JavaCPP Sampler adapters and factory methods.
 */
object JavaSamplerAdapters {

  /** Create a RandomSampler with specified size */
  def createRandomSampler(size: Long): RandomSampler = {
    new RandomSampler(size)
  }

  /** Create a SequentialSampler with specified size */
  def createSequentialSampler(size: Long): SequentialSampler = {
    new SequentialSampler(size)
  }

  /** Create a StreamSampler with specified size */
  def createStreamSampler(size: Long): StreamSampler = {
    new StreamSampler(size)
  }

  /** Create a DistributedRandomSampler for distributed training */
  def createDistributedRandomSampler(
    size: Long,
    rank: Int,
    worldSize: Int
  ): DistributedRandomSampler = {
    new DistributedRandomSampler(size, worldSize.toLong, rank.toLong, false)
  }

  /** Create a DistributedSequentialSampler for distributed training */
  def createDistributedSequentialSampler(
    size: Long,
    rank: Int,
    worldSize: Int,
    allowDuplicates: Boolean = false
  ): DistributedSequentialSampler = {
    new DistributedSequentialSampler(size, worldSize.toLong, rank.toLong, allowDuplicates)
  }

  /** Create a BatchSizeSampler for dynamic batch sizing.
   *  Note: BatchSizeSampler is typically created via C++ factory methods.
   *  This creates a basic sampler that can be configured.
   */
  def createBatchSizeSampler(
    sizes: Array[Long],
    maxBatchSize: Long = 256L,
    dropLast: Boolean = false
  ): BatchSizeSampler = {
    // BatchSizeSampler requires allocate pattern
    val ptr = new org.bytedeco.javacpp.Pointer()
    val sampler = new BatchSizeSampler(ptr)
    sampler
  }
}

/**
 * Unified DataLoader factory object.
 */
object JavaDataLoaderFactory {

  import JavaDatasetAdapters._
  import JavaTensorDatasetAdapters._
  import JavaDistributedAdapters._

  // ============= Non-Tensor DataLoaders =============

  /** Create a Random DataLoader (Example-based) */
  def random(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaRandomDataLoader = {
    createRandomDataLoader(backing, batchSize, numWorkers, dropLast)
  }

  /** Create a Sequential DataLoader (Example-based) */
  def sequential(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaSequentialDataLoader = {
    createSequentialDataLoader(backing, batchSize, numWorkers, dropLast)
  }

  /** Create a Stateful DataLoader (Example-based) */
  def stateful(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStatefulDataLoader = {
    createStatefulDataLoader(backing, batchSize, numWorkers)
  }

  /** Create a Stream DataLoader (Example-based) */
  def stream(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStreamDataLoader = {
    createStreamDataLoader(backing, batchSize, numWorkers)
  }

  // ============= Tensor DataLoaders =============

  /** Create a Random Tensor DataLoader */
  def randomTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaRandomTensorDataLoader = {
    createRandomTensorDataLoader(backing, batchSize, numWorkers, dropLast)
  }

  /** Create a Sequential Tensor DataLoader */
  def sequentialTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaSequentialTensorDataLoader = {
    createSequentialTensorDataLoader(backing, batchSize, numWorkers, dropLast)
  }

  /** Create a Stateful Tensor DataLoader */
  def statefulTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStatefulTensorDataLoader = {
    createStatefulTensorDataLoader(backing, batchSize, numWorkers)
  }

  /** Create a Stream Tensor DataLoader */
  def streamTensor(
    backing: Dataset,
    batchSize: Long = 256L,
    numWorkers: Long = 0L
  ): JavaStreamTensorDataLoader = {
    createStreamTensorDataLoader(backing, batchSize, numWorkers)
  }

  // ============= Distributed DataLoaders =============

  /** Create a Distributed Random DataLoader */
  def distributedRandom(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedRandomDataLoader = {
    createDistributedRandomDataLoader(backing, rank, worldSize, batchSize, numWorkers, dropLast)
  }

  /** Create a Distributed Sequential DataLoader */
  def distributedSequential(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedSequentialDataLoader = {
    createDistributedSequentialDataLoader(backing, rank, worldSize, batchSize, numWorkers, dropLast)
  }

  /** Create a Distributed Random Tensor DataLoader */
  def distributedRandomTensor(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedRandomTensorDataLoader = {
    createDistributedRandomTensorDataLoader(backing, rank, worldSize, batchSize, numWorkers, dropLast)
  }

  /** Create a Distributed Sequential Tensor DataLoader */
  def distributedSequentialTensor(
    backing: Dataset,
    rank: Int,
    worldSize: Int,
    batchSize: Long = 256L,
    numWorkers: Long = 0L,
    dropLast: Boolean = false
  ): JavaDistributedSequentialTensorDataLoader = {
    createDistributedSequentialTensorDataLoader(backing, rank, worldSize, batchSize, numWorkers, dropLast)
  }
}

/**
 * Configuration case classes.
 */
case class DataLoaderConfig(
  batchSize: Long = 256L,
  numWorkers: Long = 0L,
  dropLast: Boolean = false
)

case class DistributedDataLoaderConfig(
  rank: Int = 0,
  worldSize: Int = 1,
  batchSize: Long = 256L,
  numWorkers: Long = 0L,
  dropLast: Boolean = false
)

/**
 * Extension methods for Dataset to create DataLoaders more conveniently.
 */
object DatasetDataLoaderExtensions {

  implicit class DatasetOps(backing: Dataset) {

    def randomLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaRandomDataLoader = {
      JavaDataLoaderFactory.random(backing, batchSize, numWorkers, dropLast)
    }

    def sequentialLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaSequentialDataLoader = {
      JavaDataLoaderFactory.sequential(backing, batchSize, numWorkers, dropLast)
    }

    def streamLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L
    ): JavaStreamDataLoader = {
      JavaDataLoaderFactory.stream(backing, batchSize, numWorkers)
    }

    def statefulLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L
    ): JavaStatefulDataLoader = {
      JavaDataLoaderFactory.stateful(backing, batchSize, numWorkers)
    }
  }

  implicit class TensorDatasetOps(backing: Dataset) {

    def randomTensorLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaRandomTensorDataLoader = {
      JavaDataLoaderFactory.randomTensor(backing, batchSize, numWorkers, dropLast)
    }

    def sequentialTensorLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaSequentialTensorDataLoader = {
      JavaDataLoaderFactory.sequentialTensor(backing, batchSize, numWorkers, dropLast)
    }

    def streamTensorLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L
    ): JavaStreamTensorDataLoader = {
      JavaDataLoaderFactory.streamTensor(backing, batchSize, numWorkers)
    }

    def statefulTensorLoader(
      batchSize: Long = 256L,
      numWorkers: Long = 0L
    ): JavaStatefulTensorDataLoader = {
      JavaDataLoaderFactory.statefulTensor(backing, batchSize, numWorkers)
    }
  }

  implicit class DistributedDatasetOps(backing: Dataset) {

    def distributedRandomLoader(
      rank: Int,
      worldSize: Int,
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaDistributedRandomDataLoader = {
      JavaDataLoaderFactory.distributedRandom(backing, rank, worldSize, batchSize, numWorkers, dropLast)
    }

    def distributedRandomTensorLoader(
      rank: Int,
      worldSize: Int,
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaDistributedRandomTensorDataLoader = {
      JavaDataLoaderFactory.distributedRandomTensor(backing, rank, worldSize, batchSize, numWorkers, dropLast)
    }

    def distributedSequentialLoader(
      rank: Int,
      worldSize: Int,
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaDistributedSequentialDataLoader = {
      JavaDataLoaderFactory.distributedSequential(backing, rank, worldSize, batchSize, numWorkers, dropLast)
    }

    def distributedSequentialTensorLoader(
      rank: Int,
      worldSize: Int,
      batchSize: Long = 256L,
      numWorkers: Long = 0L,
      dropLast: Boolean = false
    ): JavaDistributedSequentialTensorDataLoader = {
      JavaDataLoaderFactory.distributedSequentialTensor(backing, rank, worldSize, batchSize, numWorkers, dropLast)
    }
  }
}