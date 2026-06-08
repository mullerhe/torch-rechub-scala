package torchrec.utils

/**
 * Thread-local device context for DataLoader.
 *
 * When CTRTrainer/MatchTrainer runs with a specific device (e.g. "cpu"),
 * the DataLoader used inside the training loop should also use that device
 * to avoid cross-device tensor mismatches.
 *
 * Usage:
 *   DataLoaderDevice.set("cpu")  // in trainer before iteration
 *   val loader = new DataLoader(dataset, batchSize, ...)  // uses "cpu"
 *   DataLoaderDevice.clear()     // after iteration
 */
object DataLoaderDevice {
  private val threadLocal = new ThreadLocal[String] {
    override def initialValue(): String = null.asInstanceOf[String]
  }

  /** Set the device for DataLoader creation in the current thread. */
  def set(device: String): Unit = threadLocal.set(device)

  /** Get the device override for DataLoader, or null if not set. */
  def get: String = threadLocal.get()

  /** Clear the device override. */
  def clear(): Unit = threadLocal.remove()
}
