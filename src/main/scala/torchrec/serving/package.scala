package torchrec.serving

/**
 * Serving module for vector search and ANN indexing.
 * Supports multiple backend implementations: VectorIndexer (pure Scala),
 * FaissBuilder (FAISS-compatible), AnnoyBuilder (Annoy-compatible),
 * and MilvusBuilder (Milvus-compatible).
 */

/**
 * Builder factory for creating vector index builders.
 */
object BuilderFactory {
  enum RetrievalModel:
    case annoy, faiss, milvus, vector

  def apply(model: RetrievalModel, config: Map[String, Any] = Map.empty): BaseBuilder = {
    model match {
      case RetrievalModel.annoy =>
        val d = config.getOrElse("d", 128).asInstanceOf[Int]
        val metric = config.getOrElse("metric", IndexMetric.L2).asInstanceOf[IndexMetric]
        val nTrees = config.getOrElse("nTrees", 10).asInstanceOf[Int]
        val searchK = config.getOrElse("searchK", -1).asInstanceOf[Int]
        new AnnoyBuilder(d, metric, nTrees, searchK)

      case RetrievalModel.faiss =>
        val indexType = config.getOrElse("indexType", IndexType.FLAT).asInstanceOf[IndexType]
        val metric = config.getOrElse("metric", IndexMetric.L2).asInstanceOf[IndexMetric]
        val m = config.getOrElse("m", 32).asInstanceOf[Int]
        val nlist = config.getOrElse("nlist", 100).asInstanceOf[Int]
        val efSearch = config.get("efSearch").asInstanceOf[Option[Int]]
        val nprobe = config.get("nprobe").asInstanceOf[Option[Int]]
        new FaissBuilder(indexType, metric, m, nlist, efSearch, nprobe)

      case RetrievalModel.milvus =>
        val d = config.getOrElse("d", 128).asInstanceOf[Int]
        val indexType = config.getOrElse("indexType", IndexType.FLAT).asInstanceOf[IndexType]
        val metric = config.getOrElse("metric", IndexMetric.COSINE).asInstanceOf[IndexMetric]
        val m = config.getOrElse("m", 30).asInstanceOf[Int]
        val nlist = config.getOrElse("nlist", 128).asInstanceOf[Int]
        val ef = config.get("ef").asInstanceOf[Option[Int]]
        val nprobe = config.get("nprobe").asInstanceOf[Option[Int]]
        new MilvusBuilder(d, indexType, metric, m, nlist, ef, nprobe)

      case RetrievalModel.vector =>
        val metric = config.getOrElse("metric", IndexMetric.L2).asInstanceOf[IndexMetric]
        new VectorIndexerBuilder(metric)
    }
  }

  def apply(model: String, config: Map[String, Any]): BaseBuilder = {
    model.toLowerCase(java.util.Locale.ROOT) match {
      case "annoy" => BuilderFactory.apply(RetrievalModel.annoy, config)
      case "faiss" => BuilderFactory.apply(RetrievalModel.faiss, config)
      case "milvus" => BuilderFactory.apply(RetrievalModel.milvus, config)
      case "vector" | "default" => BuilderFactory.apply(RetrievalModel.vector, config)
      case _ => throw new NotImplementedError(s"Model '$model' is not implemented yet!")
    }
  }
}
