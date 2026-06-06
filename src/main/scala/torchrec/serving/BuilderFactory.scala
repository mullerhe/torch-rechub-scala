package torchrec.serving

/**
 * Factory for creating vector index builders.
 *
 * @example
 * {{{
 * val builder = BuilderFactory("vector", Map("metric" -> IndexMetric.L2))
 * val builder = BuilderFactory("annoy", Map("d" -> 128, "nTrees" -> 10))
 * val builder = BuilderFactory("faiss", Map("indexType" -> IndexType.HNSW))
 * }}}
 */
object BuilderFactory {

  /**
   * Supported retrieval backend models.
   */
  enum RetrievalModel:
    case annoy, faiss, milvus, vector

  /**
   * Create a builder by model name string with config.
   */
  def apply(model: String, config: Map[String, Any]): BaseBuilder =
    createBuilder(model, config)

  /**
   * Create a builder by model name string with default empty config.
   */
  def apply(model: String): BaseBuilder =
    createBuilder(model, Map.empty)

  private def createBuilder(model: String, config: Map[String, Any]): BaseBuilder = {
    model.toLowerCase(java.util.Locale.ROOT) match {
      case "annoy" =>
        val d = config.getOrElse("d", 128).asInstanceOf[Int]
        val metric = config.get("metric") match {
          case Some(m: IndexMetric) => m
          case Some(s: String) =>
            s match
              case "angular" | "cosine" | "cos" => IndexMetric.COSINE
              case "ip" | "inner_product" => IndexMetric.IP
              case _ => IndexMetric.L2
          case _ => IndexMetric.L2
        }
        val nTrees = config.getOrElse("nTrees", 10).asInstanceOf[Int]
        val searchK = config.getOrElse("searchK", -1).asInstanceOf[Int]
        new AnnoyBuilder(d, metric, nTrees, searchK)

      case "faiss" =>
        val indexType = config.get("indexType") match {
          case Some(t: IndexType) => t
          case Some(s: String) =>
            s match
              case "HNSW" | "hnsw" => IndexType.HNSW
              case "IVF" | "ivf" => IndexType.IVF
              case _ => IndexType.FLAT
          case _ => IndexType.FLAT
        }
        val metric = config.get("metric") match {
          case Some(m: IndexMetric) => m
          case Some(s: String) =>
            s match
              case "IP" | "ip" | "inner_product" => IndexMetric.IP
              case "COSINE" | "cosine" | "cos" => IndexMetric.COSINE
              case _ => IndexMetric.L2
          case _ => IndexMetric.L2
        }
        val m = config.getOrElse("m", 32).asInstanceOf[Int]
        val nlist = config.getOrElse("nlist", 100).asInstanceOf[Int]
        val efSearch = config.get("efSearch").asInstanceOf[Option[Int]]
        val nprobe = config.get("nprobe").asInstanceOf[Option[Int]]
        new FaissBuilder(indexType, metric, m, nlist, efSearch, nprobe)

      case "milvus" =>
        val d = config.getOrElse("d", 128).asInstanceOf[Int]
        val indexType = config.get("indexType") match {
          case Some(t: IndexType) => t
          case Some(s: String) =>
            s match
              case "HNSW" | "hnsw" => IndexType.HNSW
              case "IVF" | "ivf" => IndexType.IVF
              case _ => IndexType.FLAT
          case _ => IndexType.FLAT
        }
        val metric = config.get("metric") match {
          case Some(m: IndexMetric) => m
          case Some(s: String) =>
            s match
              case "COSINE" | "cosine" | "cos" => IndexMetric.COSINE
              case "IP" | "ip" | "inner_product" => IndexMetric.IP
              case _ => IndexMetric.COSINE
          case _ => IndexMetric.COSINE
        }
        val m = config.getOrElse("m", 30).asInstanceOf[Int]
        val nlist = config.getOrElse("nlist", 128).asInstanceOf[Int]
        val ef = config.get("ef").asInstanceOf[Option[Int]]
        val nprobe = config.get("nprobe").asInstanceOf[Option[Int]]
        new MilvusBuilder(d, indexType, metric, m, nlist, ef, nprobe)

      case "vector" | "default" =>
        val metric = config.get("metric") match {
          case Some(m: IndexMetric) => m
          case Some(s: String) =>
            s match
              case "COSINE" | "cosine" | "cos" => IndexMetric.COSINE
              case "IP" | "ip" | "inner_product" => IndexMetric.IP
              case _ => IndexMetric.L2
          case _ => IndexMetric.L2
        }
        new VectorIndexerBuilder(metric)

      case _ => throw new NotImplementedError(s"Model '$model' is not implemented yet!")
    }
  }
}
