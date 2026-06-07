package torchrec.utils

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.*
import org.bytedeco.pytorch.{Device, DeviceOptional, ScalarTypeOptional, TensorOptions}

import java.lang.management.ManagementFactory
import scala.util.Try

/**
 * Central device selector for the whole project.
 *
 * The active backend ("cuda" | "mps" | "cpu") is resolved exactly once on first use and then
 * cached. It can be chosen, in priority order, via:
 *   1. programmatic override: `DeviceSupport.setDevice(DeviceType.CUDA)` (call BEFORE building the model)
 *   2. system property:       `-Dnanovllm.device=cuda`
 *   3. environment variable:  `NANOVLLM_DEVICE=cuda`
 *   4. AUTO (default): MPS on macOS, else CUDA if available, else CPU.
 *
 * Every tensor-allocation site goes through `deviceOf()` / `*Opts()` / `backend`, so flipping this
 * one flag switches the entire project between CUDA and MPS (and CPU).
 */
object DeviceSupport {

  /** Device selection flag. AUTO performs platform-aware auto-detection. */
  enum DeviceType {
    case AUTO, CUDA, MPS, CPU, XPU, NPU
  }

  private val osName = System.getProperty("os.name", "").toLowerCase

  // ---- selection state ----------------------------------------------------

  @volatile private var requested: DeviceType = readInitialRequest()
  @volatile private var resolvedBackend: Option[String] = None

  private def readInitialRequest(): DeviceType = {
    sys.props.get("nanovllm.device")
      .orElse(sys.env.get("NANOVLLM_DEVICE"))
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(parseDeviceType)
      .getOrElse(DeviceType.AUTO)
  }

  /** Parse a free-form device string into a [[DeviceType]]. Unknown values map to AUTO. */
  def parseDeviceType(name: String): DeviceType = name.trim.toLowerCase match {
    case "cuda" | "gpu" | "nvidia" => DeviceType.CUDA
    case "mps" | "metal"           => DeviceType.MPS
    case "cpu"                     => DeviceType.CPU
    case "auto" | ""               => DeviceType.AUTO
    case "xpu"                     => DeviceType.XPU
    case "npu"                     => DeviceType.NPU
    case other =>
      println(s"[DeviceSupport] Unknown device '$other', falling back to AUTO")
      DeviceType.AUTO
  }

  /** Programmatically choose the backend. Must be called before the device is first used. */
  def setDevice(dt: DeviceType): Unit = synchronized {
    resolvedBackend match {
      case Some(current) =>
        val target = resolveBackend(dt)
        if (current != target)
          println(s"[DeviceSupport] WARNING: device already initialized as '$current'; request to " +
            s"switch to '$target' is ignored. Choose the device before creating the model " +
            s"(use -Dnanovllm.device / NANOVLLM_DEVICE, or call setDevice earlier).")
      case None =>
        requested = dt
    }
  }

  /** Convenience overload accepting a string flag ("cuda" | "mps" | "cpu" | "auto"). */
  def setDevice(name: String): Unit = setDevice(parseDeviceType(name))

  /** The currently requested (possibly unresolved) selection. */
  def requestedDevice: DeviceType = requested

  // ---- availability -------------------------------------------------------

  private def invokeTorchBoolean(methodName: String): Boolean =
    Try {
      val method = Class.forName("org.bytedeco.pytorch.global.torch").getMethod(methodName)
      method.invoke(null).asInstanceOf[Boolean]
    }.getOrElse(false)

  lazy val cudaAvailable: Boolean = Try(torch.cuda_is_available()).getOrElse(false)

  lazy val mpsAvailable: Boolean =
    invokeTorchBoolean("mps_is_available") || invokeTorchBoolean("hasMPS") ||
      Try {
        val probe = torch.zeros(Array(1L), floatOptsFor("mps"))
        probe.close()
        true
      }.getOrElse(false)

  // ---- resolution ---------------------------------------------------------

  private def resolveBackend(dt: DeviceType): String = dt match {
    case DeviceType.CPU  => "cpu"
    case DeviceType.CUDA =>
      if (cudaAvailable) "cuda"
      else { println("[DeviceSupport] CUDA requested but not available; falling back to AUTO"); autoBackend() }
    case DeviceType.MPS =>
      if (mpsAvailable) "mps"
      else { println("[DeviceSupport] MPS requested but not available; falling back to AUTO"); autoBackend() }
    case DeviceType.AUTO => autoBackend()
  }

  private def autoBackend(): String = {
    if (osName.contains("mac") && mpsAvailable) "mps"
    else if (cudaAvailable) "cuda"
    else "cpu"
  }

  /** Alias for [[backend]]: the device string to use as the project-wide default. */
  def defaultDevice: String = backend

  /** The resolved backend string: "cuda" | "mps" | "cpu". Resolved once, then cached. */
  lazy val backend: String = synchronized {
    resolvedBackend.getOrElse {
      val r = resolveBackend(requested)
      resolvedBackend = Some(r)
      println(s"[DeviceSupport] active device backend: $r (requested=$requested, " +
        s"cuda=$cudaAvailable, mps=$mpsAvailable, os=$osName)")
      r
    }
  }

  lazy val acceleratorAvailable: Boolean = backend != "cpu"

  // ---- tensor option helpers ---------------------------------------------

  def deviceOf(deviceType: String = backend): Device = new Device(deviceType)

  def opts(dtype: ScalarType, deviceType: String = backend): TensorOptions =
    new TensorOptions()
      .dtype(new ScalarTypeOptional(dtype))
      .device(new DeviceOptional(deviceOf(deviceType)))

  def floatOpts(deviceType: String = backend): TensorOptions = opts(ScalarType.Float, deviceType)
  def longOpts(deviceType: String = backend): TensorOptions = opts(ScalarType.Long, deviceType)

  // internal variant that does not trigger `backend` resolution (used by the mps probe)
  private def floatOptsFor(deviceType: String): TensorOptions =
    new TensorOptions()
      .dtype(new ScalarTypeOptional(ScalarType.Float))
      .device(new DeviceOptional(new Device(deviceType)))

  // ---- memory estimation --------------------------------------------------

  /** Free VRAM in bytes for the active CUDA device, if queryable. */
  private def cudaFreeMemoryBytes(): Option[Long] =
    Try {
      val info = torch.getMemoryInfo(0.toByte) // (free, total)
      val free = info.first()
      if (free > 0L) Some(free) else None
    }.getOrElse(None)

  /** Total VRAM in bytes for the active CUDA device, if queryable. */
  private def cudaTotalMemoryBytes(): Option[Long] =
    Try {
      val info = torch.getMemoryInfo(0.toByte) // (free, total)
      val total = info.second()
      if (total > 0L) Some(total) else None
    }.getOrElse(None)

  private def physicalMemoryBytes(): Long =
    Try {
      val bean = ManagementFactory.getOperatingSystemMXBean
      val method = bean.getClass.getMethod("getTotalMemorySize")
      method.invoke(bean).asInstanceOf[Long]
    }.getOrElse(16L << 30)

  /**
   * Estimate the memory budget available for the KV cache, scaled by `utilization`.
   *
   * On CUDA: uses `total_GPU_memory * utilization`, which correctly reserves
   * space for model weights, activations, and KV cache proportionally.
   * Falls back to `free_GPU_memory * utilization` if total is unavailable.
   * On MPS/CPU: uses a fraction of system RAM.
   */
  def estimateAvailableMemoryBytes(utilization: Float): Long = {
    val budget = backend match {
      case "cuda" =>
        // Use total memory * utilization as the budget.
        // The caller (ModelRunner) must ensure model weights fit in the remaining space.
        cudaTotalMemoryBytes()
          .map(total => (total * utilization).toLong)
          .orElse {
            // Fallback to free memory * utilization if total unavailable
            cudaFreeMemoryBytes().map(free => (free * utilization).toLong)
          }
          .getOrElse((physicalMemoryBytes() * math.min(utilization, 0.5f)).toLong)
      case _ =>
        (physicalMemoryBytes() * utilization).toLong
    }
    math.max(512L << 20, budget)
  }

  /**
   * Returns the actual free GPU memory in bytes (after model weights are loaded).
   * This is what should be used for KV cache allocation when model is already on GPU.
   */
  def getActualFreeMemoryBytes(): Long = {
    backend match {
      case "cuda" =>
        cudaFreeMemoryBytes()
          .getOrElse(physicalMemoryBytes() * 0.3.toLong)
      case _ =>
        (physicalMemoryBytes() * 0.5).toLong
    }
  }

  /** Returns estimated allocated GPU memory in bytes (approximation). */
  def getAllocatedMemory(): Long = {
    backend match {
      case "cuda" =>
        // Approximate based on total VRAM minus free
        val free = cudaFreeMemoryBytes().getOrElse(0L)
        physicalMemoryBytes() - free
      case _ =>
        physicalMemoryBytes() / 2
    }
  }
}
