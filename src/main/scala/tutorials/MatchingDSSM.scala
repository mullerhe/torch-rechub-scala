package tutorials

import torchrec.Implicits._
import torchrec.data.DataGenerator
import torchrec.basic.features._
import torchrec.data._
import torchrec.models.matching._
import torchrec.trainers._

/**
 * Tutorial: Two-Tower Matching with DSSM
 *
 * DSSM (Deep Structured Semantic Model) uses a dual-tower architecture
 * for efficient user-item matching in large-scale retrieval.
 */
object MatchingDSSM {

  def main(args: Array[String]): Unit = {
    println("""
      |=============================================================
      | Tutorial: Two-Tower Matching with DSSM
      |=============================================================
      |
      | DSSM learns separate representations for users and items,
      | enabling efficient nearest-neighbor search for retrieval.
      |
      |""".stripMargin)

    // Define user and item features
    println("Defining features...")
    val userFeatures = List(
      SparseFeature("user_id", 10000, 16),
      SparseFeature("user_age", 100, 8),
      SparseFeature("user_gender", 2, 4)
    )

    val itemFeatures = List(
      SparseFeature("item_id", 50000, 16),
      SparseFeature("item_category", 1000, 8)
    )

    println("\nUser features:")
    userFeatures.foreach(f => println(f"  - ${f.name}: vocab=${f.vocabSize}%,d"))
    println("\nItem features:")
    itemFeatures.foreach(f => println(f"  - ${f.name}: vocab=${f.vocabSize}%,d"))

    // Create DSSM model
    println("\nCreating DSSM model...")
    val model = new DSSM(
      userFeatures = userFeatures,
      itemFeatures = itemFeatures,
      embedDim = 16,
      towerDims = List(256L, 128L, 64L),
      dropout = 0.2f
    )
    println("  Model: DSSM(embed_dim=16, tower_dims=[256, 128, 64])")

    // Generate matching data
    println("\nGenerating matching data...")
    val (trainData, _, testData) = DataGenerator.generateMatchingData(
      numUsers = 5000,
      numItems = 10000,
      avgSequenceLength = 10,
      numUserFeatures = userFeatures.size,
      numItemFeatures = itemFeatures.size,
      vocabSize = 100
    )
    println(f"  Train: ${trainData.size}%,d users")
    println(f"  Items: ${10000}%,d")

    // Create data loaders
    val trainLoader = new DataLoader(trainData, batchSize = 128, shuffle = true)
    val testLoader = new DataLoader(testData, batchSize = 128, shuffle = false)

    // Create trainer
    println("\nCreating MatchTrainer...")
    val trainer = new MatchTrainer(
      model = model,
      learningRate = 1e-3f,
      mode = 2,  // listwise
      numEpochs = 3,
      verbose = true
    )

    // Train
    println("\nTraining...")
    trainer.fit(trainLoader)

    // Extract embeddings
    println("\nExtracting user embeddings for inference...")
    val userEmbeds = trainer.inferenceEmbedding(testLoader, "user")
    println(f"  User embeddings shape: ${userEmbeds.head.shape.mkString(", ")}")

    println("""
      |
      |=============================================================
      | Tutorial Complete!
      |=============================================================
      |
      | Next: Build ANN index for fast retrieval
      |""".stripMargin)
  }
}