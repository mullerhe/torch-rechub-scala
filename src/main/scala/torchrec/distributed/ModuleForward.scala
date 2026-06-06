package torchrec.distributed

import org.bytedeco.pytorch.{Module, Tensor}

import java.lang.reflect.Method

/**
 * Utility for invoking Module.forward(Tensor) via reflection.
 */
object ModuleForward {
  /**
   * Find and cache the forward(Tensor) method on the module class hierarchy.
   */
    def of(module: Module): ModuleForward = {
      var cls: Class[_] = module.getClass
      while (cls != null && cls != classOf[java.lang.Object]) try {
        val m = cls.getDeclaredMethod("forward", classOf[Tensor])
        m.setAccessible(true)
        System.out.printf("[ModuleForward] Found forward method on %s%n", cls.getName)
        return new ModuleForward(m)
      } catch {
        case e: NoSuchMethodException =>
          cls = cls.getSuperclass
      }
      throw new UnsupportedOperationException("Module " + module.getClass.getName + " has no forward(Tensor) method")
    }
}

class ModuleForward private(private val forwardMethod: Method) {
  /**
   * Invoke forward via reflection.
   */
  def apply(module: Module, input: Tensor): Tensor = try forwardMethod.invoke(module, input).asInstanceOf[Tensor]
  catch {
    case e: Exception =>
      throw new RuntimeException("Failed to invoke forward: " + e.getMessage, e)
  }

  def apply(module: Module, inputMap: Map[String, Tensor]): Tensor = try forwardMethod.invoke(module, inputMap).asInstanceOf[Tensor]
  catch {
    case e: Exception =>
      throw new RuntimeException("Failed to invoke forward: " + e.getMessage, e)
  }
}