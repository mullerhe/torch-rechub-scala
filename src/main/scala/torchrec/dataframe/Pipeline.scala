package torchrec.dataframe

import scala.collection.mutable

/**
 * Pipeline for chaining multiple feature transformers.
 */
class Pipeline(stages: List[FeatureTransformer]) {

  private val fittedStages = mutable.ListBuffer[FeatureTransformer]()

  def addStage(transformer: FeatureTransformer): Pipeline = {
    new Pipeline(stages :+ transformer)
  }

  def fit(df: DataFrame): this.type = {
    var currentDF = df
    for (stage <- stages) {
      stage.fit(currentDF)
      fittedStages += stage
      currentDF = stage.transform(currentDF)
    }
    this
  }

  def transform(df: DataFrame): DataFrame = {
    var currentDF = df
    for (stage <- fittedStages) {
      currentDF = stage.transform(currentDF)
    }
    currentDF
  }

  def fitTransform(df: DataFrame): DataFrame = {
    fit(df)
    transform(df)
  }

  def getFittedStages: List[FeatureTransformer] = fittedStages.toList

  def getStage(name: String): Option[FeatureTransformer] = {
    fittedStages.find(_.name == name)
  }

  def numStages: Int = stages.length

  def save(path: String): Unit = {
    val stageInfo = stages.map(_.name).mkString("\n")
    java.nio.file.Files.write(
      java.nio.file.Paths.get(path),
      stageInfo.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )
  }

  def load(path: String): Pipeline = {
    val content = new String(
      java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(path)),
      java.nio.charset.StandardCharsets.UTF_8
    )
    val stageNames = content.split("\n").filter(_.nonEmpty)
    val loadedStages = stageNames.map { name =>
      FeatureTransformers.allTransformerNames.find(_ == name) match {
        case Some("StandardScaler") => FeatureTransformers.standardScaler()
        case Some("MinMaxScaler") => FeatureTransformers.minMaxScaler()
        case Some("MaxAbsScaler") => FeatureTransformers.maxAbsScaler()
        case Some("RobustScaler") => FeatureTransformers.robustScaler()
        case Some("LogTransformer") => FeatureTransformers.logTransformer()
        case Some("LabelEncoder") => FeatureTransformers.labelEncoder()
        case Some("OneHotEncoder") => FeatureTransformers.oneHotEncoder()
        case Some("TargetEncoder") => FeatureTransformers.targetEncoder()
        case Some("CountEncoder") => FeatureTransformers.countEncoder()
        case Some("HashEncoder") => FeatureTransformers.hashEncoder()
        case Some("OrdinalEncoder") => FeatureTransformers.ordinalEncoder()
        case Some("CatBoostEncoder") => FeatureTransformers.catBoostEncoder()
        case Some("WOEEncoder") => FeatureTransformers.woeEncoder()
        case Some("EmbeddingEncoder") => FeatureTransformers.embeddingEncoder()
        case Some("SequenceEncoder") => FeatureTransformers.sequenceEncoder()
        case Some("PolynomialFeatures") => FeatureTransformers.polynomialFeatures()
        case Some("InteractionFeatures") => FeatureTransformers.interactionFeatures()
        case Some("BinnedFeatures") => FeatureTransformers.binnedFeatures()
        case Some("DateTimeFeatures") => FeatureTransformers.dateTimeFeatures()
        case Some("TextFeatures") => FeatureTransformers.textFeatures()
        case Some("CrossFeatures") => FeatureTransformers.crossFeatures()
        case Some("QuantileFeatures") => FeatureTransformers.quantileFeatures()
        case Some("SVDFeatures") => FeatureTransformers.svdFeatures()
        case Some("VarianceThreshold") => FeatureTransformers.varianceThreshold()
        case Some("CorrelationSelector") => FeatureTransformers.correlationSelector()
        case Some("Chi2Selector") => FeatureTransformers.chi2Selector()
        case Some("MutualInfoSelector") => FeatureTransformers.mutualInfoSelector()
        case Some("FeatureImportanceSelector") => FeatureTransformers.featureImportanceSelector()
        case Some("SimpleImputer") => FeatureTransformers.simpleImputer()
        case Some("KNNImputer") => FeatureTransformers.knnImputer()
        case Some("IndicatorTransformer") => FeatureTransformers.indicatorTransformer()
        case Some("UserAgeBinTransformer") => FeatureTransformers.userAgeBinTransformer()
        case Some("ItemPopularityTransformer") => FeatureTransformers.itemPopularityTransformer()
        case Some("UserActivityTransformer") => FeatureTransformers.userActivityTransformer()
        case Some("CategoryEncoder") => FeatureTransformers.categoryEncoder()
        case Some("TagTransformer") => FeatureTransformers.tagTransformer()
        case Some("SequencePaddingTransformer") => FeatureTransformers.sequencePaddingTransformer()
        case Some("BehaviorHistoryTransformer") => FeatureTransformers.behaviorHistoryTransformer()
        case Some("TimeDecayTransformer") => FeatureTransformers.timeDecayTransformer()
        case Some("ItemAgeTransformer") => FeatureTransformers.itemAgeTransformer()
        case Some("UserClusterTransformer") => FeatureTransformers.userClusterTransformer()
        case Some("ItemClusterTransformer") => FeatureTransformers.itemClusterTransformer()
        case Some("PriceTransformer") => FeatureTransformers.priceTransformer()
        case Some("RatingTransformer") => FeatureTransformers.ratingTransformer()
        case Some("ContextualFeatureTransformer") => FeatureTransformers.contextualFeatureTransformer()
        case Some("ClickThroughRateTransformer") => FeatureTransformers.clickThroughRateTransformer()
        case Some("ConversionRateTransformer") => FeatureTransformers.conversionRateTransformer()
        case Some("UserEmbeddingTransformer") => FeatureTransformers.userEmbeddingTransformer()
        case Some("ItemEmbeddingTransformer") => FeatureTransformers.itemEmbeddingTransformer()
        case Some("GraphFeatureTransformer") => FeatureTransformers.graphFeatureTransformer()
        case Some("SequenceHistogramTransformer") => FeatureTransformers.sequenceHistogramTransformer()
        case Some("SequenceStatTransformer") => FeatureTransformers.sequenceStatTransformer()
        case _ => throw new IllegalArgumentException(s"Unknown transformer: $name")
      }
    }.toList

    new Pipeline(loadedStages)
  }
}

object PipelineBuilder {
  def create(): PipelineBuilder = new PipelineBuilder()
}

class PipelineBuilder() {
  private val stageBuilders = mutable.ListBuffer[FeatureTransformer]()

  def addScaling(name: String, scalerType: String, params: Map[String, Any]): PipelineBuilder = {
    scalerType match {
      case "standard" => stageBuilders += FeatureTransformers.standardScaler()
      case "minmax" => stageBuilders += FeatureTransformers.minMaxScaler()
      case "maxabs" => stageBuilders += FeatureTransformers.maxAbsScaler()
      case "robust" => stageBuilders += FeatureTransformers.robustScaler()
      case "log" => stageBuilders += FeatureTransformers.logTransformer(params.get("offset").map(_.asInstanceOf[Float]).getOrElse(1.0f))
      case _ => throw new IllegalArgumentException(s"Unknown scaler: $scalerType")
    }
    this
  }

  def addEncoding(name: String, encoderType: String, params: Map[String, Any]): PipelineBuilder = {
    encoderType match {
      case "label" =>
        val handleUnknown = params.get("handleUnknown").map(_.asInstanceOf[String]).getOrElse("encode")
        stageBuilders += FeatureTransformers.labelEncoder(handleUnknown)
      case "onehot" =>
        val sparseOutput = params.get("sparseOutput").map(_.asInstanceOf[Boolean]).getOrElse(true)
        val maxCategories = params.get("maxCategories").map(_.asInstanceOf[Int])
        stageBuilders += FeatureTransformers.oneHotEncoder(sparseOutput, maxCategories)
      case "target" =>
        val smoothing = params.get("smoothing").map(_.asInstanceOf[Float]).getOrElse(1.0f)
        stageBuilders += FeatureTransformers.targetEncoder(smoothing)
      case "count" => stageBuilders += FeatureTransformers.countEncoder()
      case "hash" =>
        val nFeatures = params.get("nFeatures").map(_.asInstanceOf[Int]).getOrElse(1024)
        stageBuilders += FeatureTransformers.hashEncoder(nFeatures)
      case "ordinal" => stageBuilders += FeatureTransformers.ordinalEncoder()
      case "catboost" =>
        val a = params.get("a").map(_.asInstanceOf[Float]).getOrElse(1.0f)
        val b = params.get("b").map(_.asInstanceOf[Float]).getOrElse(1.0f)
        stageBuilders += FeatureTransformers.catBoostEncoder(a, b)
      case "woe" =>
        val bins = params.get("bins").map(_.asInstanceOf[Int]).getOrElse(10)
        stageBuilders += FeatureTransformers.woeEncoder(bins)
      case "embedding" =>
        val embedDim = params.get("embedDim").map(_.asInstanceOf[Int]).getOrElse(8)
        stageBuilders += FeatureTransformers.embeddingEncoder(embedDim)
      case "sequence" =>
        val maxLen = params.get("maxLen").map(_.asInstanceOf[Int]).getOrElse(50)
        val padding = params.get("padding").map(_.asInstanceOf[String]).getOrElse("post")
        stageBuilders += FeatureTransformers.sequenceEncoder(maxLen, padding)
      case "category" => stageBuilders += FeatureTransformers.categoryEncoder()
      case _ => throw new IllegalArgumentException(s"Unknown encoder: $encoderType")
    }
    this
  }

  def addFeatureGen(name: String, featureType: String, params: Map[String, Any]): PipelineBuilder = {
    featureType match {
      case "polynomial" =>
        val degree = params.get("degree").map(_.asInstanceOf[Int]).getOrElse(2)
        val interactionOnly = params.get("interactionOnly").map(_.asInstanceOf[Boolean]).getOrElse(false)
        stageBuilders += FeatureTransformers.polynomialFeatures(degree, interactionOnly)
      case "interaction" => stageBuilders += FeatureTransformers.interactionFeatures()
      case "binned" =>
        val nBins = params.get("nBins").map(_.asInstanceOf[Int]).getOrElse(10)
        val strategy = params.get("strategy").map(_.asInstanceOf[String]).getOrElse("quantile")
        stageBuilders += FeatureTransformers.binnedFeatures(nBins, strategy)
      case "datetime" =>
        val features = params.get("features").map(_.asInstanceOf[List[String]]).getOrElse(List("year", "month", "day", "hour", "dayofweek"))
        stageBuilders += FeatureTransformers.dateTimeFeatures(features)
      case "text" => stageBuilders += FeatureTransformers.textFeatures()
      case "cross" =>
        val maxFeatures = params.get("maxCrossFeatures").map(_.asInstanceOf[Int]).getOrElse(100)
        stageBuilders += FeatureTransformers.crossFeatures(maxFeatures)
      case "quantile" =>
        val nQuantiles = params.get("nQuantiles").map(_.asInstanceOf[Int]).getOrElse(100)
        stageBuilders += FeatureTransformers.quantileFeatures(nQuantiles)
      case "svd" =>
        val nComponents = params.get("nComponents").map(_.asInstanceOf[Int]).getOrElse(32)
        stageBuilders += FeatureTransformers.svdFeatures(nComponents)
      case _ => throw new IllegalArgumentException(s"Unknown feature generator: $featureType")
    }
    this
  }

  def addRecommenderFeature(name: String, featureType: String, params: Map[String, Any]): PipelineBuilder = {
    featureType match {
      case "userAgeBin" => stageBuilders += FeatureTransformers.userAgeBinTransformer()
      case "itemPopularity" => stageBuilders += FeatureTransformers.itemPopularityTransformer()
      case "userActivity" => stageBuilders += FeatureTransformers.userActivityTransformer()
      case "tag" => stageBuilders += FeatureTransformers.tagTransformer()
      case "sequencePadding" => stageBuilders += FeatureTransformers.sequencePaddingTransformer()
//      case "sequenceTruncating" => stageBuilders += FeatureTransformers.sequenceTruncatingTransformer()
      case "behaviorHistory" => stageBuilders += FeatureTransformers.behaviorHistoryTransformer()
      case "timeDecay" => stageBuilders += FeatureTransformers.timeDecayTransformer()
      case "itemAge" => stageBuilders += FeatureTransformers.itemAgeTransformer()
      case "userCluster" =>
        val nClusters = params.get("nClusters").map(_.asInstanceOf[Int]).getOrElse(10)
        stageBuilders += new UserClusterTransformer(nClusters)
      case "itemCluster" =>
        val nClusters = params.get("nClusters").map(_.asInstanceOf[Int]).getOrElse(50)
        stageBuilders += new ItemClusterTransformer(nClusters)
      case "price" => stageBuilders += FeatureTransformers.priceTransformer()
      case "rating" => stageBuilders += FeatureTransformers.ratingTransformer()
      case "contextual" => stageBuilders += FeatureTransformers.contextualFeatureTransformer()
      case "ctr" | "clickThroughRate" => stageBuilders += FeatureTransformers.clickThroughRateTransformer()
      case "cvr" | "conversionRate" => stageBuilders += FeatureTransformers.conversionRateTransformer()
      case "userEmbedding" => stageBuilders += FeatureTransformers.userEmbeddingTransformer()
      case "itemEmbedding" => stageBuilders += FeatureTransformers.itemEmbeddingTransformer()
      case "graph" => stageBuilders += FeatureTransformers.graphFeatureTransformer()
      case "sequenceHistogram" => stageBuilders += FeatureTransformers.sequenceHistogramTransformer()
      case "sequenceStat" => stageBuilders += FeatureTransformers.sequenceStatTransformer()
      case _ => throw new IllegalArgumentException(s"Unknown recommender feature: $featureType")
    }
    this
  }

  def addImputer(name: String, imputerType: String, params: Map[String, Any]): PipelineBuilder = {
    imputerType match {
      case "simple" =>
        val strategy = params.get("strategy").map(_.asInstanceOf[String]).getOrElse("mean")
        val fillValue = params.get("fillValue").map(_.asInstanceOf[Float]).getOrElse(0.0f)
        stageBuilders += FeatureTransformers.simpleImputer(strategy, fillValue)
      case "knn" =>
        val k = params.get("k").map(_.asInstanceOf[Int]).getOrElse(5)
        stageBuilders += FeatureTransformers.knnImputer(k)
      case "indicator" => stageBuilders += FeatureTransformers.indicatorTransformer()
      case _ => throw new IllegalArgumentException(s"Unknown imputer: $imputerType")
    }
    this
  }

  def addFeatureSelection(name: String, selectorType: String, params: Map[String, Any]): PipelineBuilder = {
    selectorType match {
      case "variance" =>
        val threshold = params.get("threshold").map(_.asInstanceOf[Float]).getOrElse(0.0f)
        stageBuilders += FeatureTransformers.varianceThreshold(threshold)
      case "correlation" =>
        val threshold = params.get("threshold").map(_.asInstanceOf[Float]).getOrElse(0.95f)
        stageBuilders += FeatureTransformers.correlationSelector(threshold)
      case "chi2" =>
        val k = params.get("k").map(_.asInstanceOf[Int]).getOrElse(10)
        stageBuilders += FeatureTransformers.chi2Selector(k)
      case "mutualInfo" =>
        val k = params.get("k").map(_.asInstanceOf[Int]).getOrElse(10)
        stageBuilders += FeatureTransformers.mutualInfoSelector(k)
      case "importance" =>
        val threshold = params.get("threshold").map(_.asInstanceOf[Float]).getOrElse(0.01f)
        stageBuilders += FeatureTransformers.featureImportanceSelector(threshold)
      case _ => throw new IllegalArgumentException(s"Unknown selector: $selectorType")
    }
    this
  }

  def addCustom(transformer: FeatureTransformer): PipelineBuilder = {
    stageBuilders += transformer
    this
  }

  def build(): Pipeline = {
    new Pipeline(stageBuilders.toList)
  }
}