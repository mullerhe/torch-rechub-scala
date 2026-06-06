//package torchrec.utils
//
//import org.bytedeco.pytorch.Tensor
//import org.bytedeco.pytorch.global.torch
//
///** Global context for passing prefill/decode metadata */
//class Context(
//  var isPrefill: Boolean = false,
//  var cuSeqlensQ: Tensor = null,
//  var cuSeqlensK: Tensor = null,
//  var maxSeqlenQ: Int = 0,
//  var maxSeqlenK: Int = 0,
//  var slotMapping: Tensor = null,
//  var contextLens: Tensor = null,
//  var blockTables: Tensor = null,
//)
//
//object Context {
//  private var _context: Context = new Context()
//
//  def get: Context = _context
//
//  def set(
//    isPrefill: Boolean,
//    cuSeqlensQ: Tensor = null, cuSeqlensK: Tensor = null,
//    maxSeqlenQ: Int = 0, maxSeqlenK: Int = 0,
//    slotMapping: Tensor = null, contextLens: Tensor = null, blockTables: Tensor = null,
//  ): Unit = {
//    _context = new Context(isPrefill, cuSeqlensQ, cuSeqlensK, maxSeqlenQ, maxSeqlenK, slotMapping, contextLens, blockTables)
//  }
//
//  def reset(): Unit = { _context = new Context() }
//}