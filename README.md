[![Torch-RecHub Banner](doc/img/hub.png)]
# Torch-RecHub-Scala: Scala 3 Recommender System Framework

[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Scala](https://img.shields.io/badge/scala-3.8.3-DC322F?style=for-the-badge)](https://www.scala-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=for-the-badge)](https://pytorch.org/)
[![JavaCPP](https://img.shields.io/badge/JavaCPP-1.5.13-FF6600?style=for-the-badge)](https://github.com/bytedeco/javacpp)

[English](README.md) | [简体中文](README_zh.md)
[![Torch-RecHub Banner](doc/img/core.png)]
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**Torch-RecHub-Scala** is a lightweight, efficient, and easy-to-use recommender system framework built on Scala 3 and JavaCPP-PyTorch. It is a Scala implementation of the Python [Torch-RecHub](https://github.com/datawhalechina/torch-rechub), fully leveraging Scala's type safety, functional programming, and JavaCPP's high-performance PyTorch bindings.

## Features

- **Native Scala 3 Support**: Leverages Scala 3 features (Contextual Abstractions, Extension Methods, Multiversal Equality) to provide type-safe APIs
- **Deep JavaCPP Integration**: Directly calls PyTorch C++ APIs via JavaCPP, supporting CPU, CUDA GPU, and distributed training
- **Rich Model Library**: Covers **30+** mainstream recommendation algorithms (recall, ranking, multi-task, generative recommendation, etc.)
- **Modular Design**: Easy to add new models, datasets, and evaluation metrics
- **Standardized Pipeline**: Provides unified Scala Dataset/DataLoader, data loading, training, and evaluation
- **Deep JavaCPP DataLoader Support**:
    - `JavaDataset` / `JavaTensorDataset` - General datasets
    - `JavaStatefulDataset` / `JavaStatefulTensorDataset` - Stateful datasets
    - `JavaStreamDataset` / `JavaStreamTensorDataset` - Streaming datasets
    - `DistributedRandomSampler` / `DistributedSequentialSampler` - Distributed samplers
- **Distributed Training Support**:
    - `DDPTrainer` - DistributedDataParallel trainer
    - `FSDPTrainer` - FullyShardedDataParallel trainer
- **Pure JVM Ecosystem**: Seamless integration with Scala/Java/JVM ecosystem

## Installation

### Requirements

- Scala 3.8+
- sbt 1.9+
- Java 17+ (Java 21 recommended)
- PyTorch 2.10.0 (auto-loaded via JavaCPP)

### Installation Steps

```bash
# Clone the project
git clone https://github.com/your-repo/torch-rechub-scala.git
cd torch-rechub-scala

# Compile with sbt
sbt compile

# Run an example
sbt "runMain examples.ranking.DeepFMExample"
```

### Maven/Coursier Dependencies (Optional)

```scala
// build.sbt
libraryDependencies ++= Seq(
  "org.bytedeco" % "javacpp" % "1.5.13",
  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13",
  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13"
)
```

## Quick Start

### 1. CTR Ranking Model Training

```scala
import torchrec.data._
import torchrec.models.ranking._
import torchrec.trainers._
import torchrec.Implicits._

// Generate data
val (trainData, valData, testData) = DataGenerator.generateRankingData(
  numSamples = 10000,
  numSparseFeatures = 10,
  numDenseFeatures = 5,
  vocabSize = 100
)

// Create data loaders
val trainLoader = DataLoader.fromJavaRandom(trainData, batchSize = 256)
val valLoader = DataLoader.fromJavaSequential(valData, batchSize = 256)

// Define features
val features = (0 until 10).map { i =>
  SparseFeature(s"feat_$i", vocabSize = 100, embedDim = 8)
}

// Create model
val model = new DeepFM(features, embedDim = 8, mlpDims = List(64L, 32L))

// Train
val trainer = new CTRTrainer(model, learningRate = 0.001f)
trainer.fit(trainLoader, Some(valLoader))

// Evaluate
val metrics = trainer.evaluate(valLoader)
println(s"AUC: ${metrics("AUC")}")
```

### 2. Matching/Recall Model Training

```scala
import torchrec.models.matching._
import torchrec.trainers._

// Generate matching data
val (trainData, _, _) = DataGenerator.generateMatchingData(
  numUsers = 5000,
  numItems = 1000,
  vocabSize = 100
)

// Create data loader
val trainLoader = DataLoader.fromJavaRandom(trainData, batchSize = 128)

// Define user/item features
val userFeatures = (0 until 3).map { i =>
  SparseFeature(s"user_feat_$i", vocabSize = 100, embedDim = 16)
}
val itemFeatures = (0 until 2).map { i =>
  SparseFeature(s"item_feat_$i", vocabSize = 1000, embedDim = 16)
}

// Create DSSM model
val model = new DSSM(userFeatures, itemFeatures, embedDim = 16, towerDims = List(128L, 64L))

// Train
val trainer = new MatchTrainer(model, learningRate = 0.001f)
trainer.fit(trainLoader)
```

### 3. Multi-Task Learning

```scala
import torchrec.models.multi_task._

// Generate multi-task data
val taskNames = List("ctr", "cvr")
val (trainData, _, _) = DataGenerator.generateMultiTaskData(
  numSamples = 10000,
  numFeatures = 10,
  taskNames = taskNames
)

// Create MMOE model
val features = (0 until 10).map { i =>
  SparseFeature(s"feat_$i", vocabSize = 100, embedDim = 8)
}

val model = new MMOE(
  features,
  taskNames,
  taskTypes = List("classification", "classification"),
  embedDim = 8,
  numExperts = 4,
  expertDims = List(64L),
  towerDims = List(32L)
)

// Train
val trainer = new MTLTrainer(model, taskNames, learningRate = 0.001f)
trainer.fit(trainLoader)
```

## Project Structure

```
torch-rechub-scala/
├── README.md                    # Project documentation
├── build.sbt                   # sbt build configuration
├── src/main/scala/
│   ├── torchrec/               # Core library
│   │   ├── TorchRec.scala       # Main entry point
│   │   ├── Implicits.scala      # Implicit conversions
│   │   ├── TensorImplicits.scala # Tensor extensions
│   │   ├── basic/              # Basic components
│   │   │   ├── features/        # Feature definitions
│   │   │   │   └── Feature.scala
│   │   │   ├── layers/         # Neural network layers
│   │   │   │   ├── MLP.scala
│   │   │   │   ├── FM.scala
│   │   │   │   ├── CrossNetwork.scala
│   │   │   │   ├── CIN.scala
│   │   │   │   ├── SENETLayer.scala
│   │   │   │   └── EmbeddingLayer.scala
│   │   │   ├── losses/         # Loss functions
│   │   │   │   └── Loss.scala
│   │   │   └── metrics/        # Evaluation metrics
│   │   │       └── Metric.scala
│   │   ├── data/               # Data processing
│   │   │   ├── Dataset.scala   # Dataset base class
│   │   │   ├── DataLoader.scala # DataLoader
│   │   │   ├── DataGenerator.scala # Data generator
│   │   │   ├── JavaDatasetAdapters.scala    # JavaDataset adapters
│   │   │   ├── JavaTensorDatasetAdapters.scala # TensorDataset adapters
│   │   │   ├── JavaDistributedAdapters.scala # Distributed adapters
│   │   │   ├── JavaSamplerAdapters.scala    # Sampler factories
│   │   │   └── JavaDataLoaderAdapters.scala # DataLoader factories
│   │   ├── models/             # Recommendation models
│   │   │   ├── ranking/        # Ranking models
│   │   │   │   ├── DeepFM.scala
│   │   │   │   ├── WideDeep.scala
│   │   │   │   ├── DCN.scala
│   │   │   │   ├── DCNv2.scala
│   │   │   │   ├── DIN.scala
│   │   │   │   ├── DIEN.scala
│   │   │   │   ├── AFM.scala
│   │   │   │   ├── AutoInt.scala
│   │   │   │   ├── FiBiNet.scala
│   │   │   │   ├── DeepFFM.scala
│   │   │   │   └── EDCN.scala
│   │   │   ├── matching/        # Matching/recall models
│   │   │   │   ├── DSSM.scala
│   │   │   │   ├── YoutubeDNN.scala
│   │   │   │   ├── MIND.scala
│   │   │   │   ├── GRU4Rec.scala
│   │   │   │   ├── SASRec.scala
│   │   │   │   ├── NARM.scala
│   │   │   │   ├── STAMP.scala
│   │   │   │   ├── SINE.scala
│   │   │   │   ├── ComirecSA.scala
│   │   │   │   └── ComirecDR.scala
│   │   │   ├── multi_task/      # Multi-task models
│   │   │   │   ├── ESMM.scala
│   │   │   │   ├── MMOE.scala
│   │   │   │   ├── PLE.scala
│   │   │   │   ├── AITM.scala
│   │   │   │   └── SharedBottom.scala
│   │   │   └── generative/      # Generative recommendation
│   │   │       ├── HSTU.scala
│   │   │       ├── HLLM.scala
│   │   │       ├── TIGER.scala
│   │   │       └── RQVAE.scala
│   │   ├── trainers/           # Trainers
│   │   │   ├── CTRTrainer.scala    # CTR training
│   │   │   ├── MatchTrainer.scala  # Matching training
│   │   │   ├── MTLTrainer.scala   # Multi-task training
│   │   │   └── TrainLoop.scala      # Training loop
│   │   ├── distributed/        # Distributed training
│   │   │   ├── DDPConfig.scala
│   │   │   ├── DDPTrainer.scala
│   │   │   ├── FSDPConfig.scala
│   │   │   └── FSDPTrainer.scala
│   │   ├── utils/             # Utilities
│   │   │   ├── DataUtils.scala
│   │   │   ├── MatchUtils.scala
│   │   │   ├── ModelUtils.scala
│   │   │   └── Trie.scala
│   │   ├── quantization/       # Quantization
│   │   │   └── Quantizer.scala
│   │   └── serving/            # Online serving
│   │       ├── VectorIndexer.scala
│   │       └── package.scala
│   ├── examples/               # Examples
│   │   ├── ranking/
│   │   │   ├── DeepFMExample.scala
│   │   │   └── CriteoExample.scala
│   │   ├── matching/
│   │   │   ├── DSSMExample.scala
│   │   │   └── MovieLensExample.scala
│   │   ├── multi_task/
│   │   │   ├── MMOEExample.scala
│   │   │   └── CensusExample.scala
│   │   └── generative/
│   │       └── MovieLensSeqExample.scala
│   ├── tutorials/              # Tutorials
│   │   ├── QuickStartCTR.scala
│   │   ├── MatchingDSSM.scala
│   │   ├── MultiTaskMMOE.scala
│   │   └── RankingDIN.scala
│   └── benchmarks/            # Performance benchmarks
│       ├── BenchmarkRunner.scala
│       ├── DataGenerator.scala
│       └── ModelBenchmark.scala
```

## Supported Models

### Ranking Models - 12

| Model | Paper | Description |
|-------|-------|-------------|
| **DeepFM** | IJCAI 2017 | FM + Deep joint training |
| **Wide&Deep** | DLRS 2016 | Memorization + generalization |
| **DCN** | KDD 2017 | Explicit cross network |
| **DCN-v2** | WWW 2021 | Enhanced cross network |
| **DIN** | KDD 2018 | Attention for user interest |
| **DIEN** | AAAI 2019 | Interest evolution modeling |
| **AFM** | IJCAI 2017 | Attentive factorization machine |
| **AutoInt** | CIKM 2019 | Automatic feature interaction |
| **FiBiNET** | RecSys 2019 | Feature importance + bilinear interaction |
| **DeepFFM** | RecSys 2019 | Field-aware factorization machine |
| **EDCN** | KDD 2021 | Enhanced cross network |
| **BST** | DLP-KDD 2019 | Transformer for sequential modeling |

### Matching Models - 12

| Model | Paper | Description |
|-------|-------|-------------|
| **DSSM** | CIKM 2013 | Classic two-tower recall |
| **YoutubeDNN** | RecSys 2016 | YouTube deep recall |
| **MIND** | CIKM 2019 | Multi-interest dynamic routing |
| **GRU4Rec** | ICLR 2016 | GRU sequential recommendation |
| **SASRec** | ICDM 2018 | Self-attention sequential recommendation |
| **NARM** | CIKM 2017 | Neural attentive session recommendation |
| **STAMP** | KDD 2018 | Short-term attention memory priority |
| **SINE** | WSDM 2021 | Sparse interest network |
| **ComiRec-SA** | KDD 2020 | Controllable multi-interest recommendation |
| **ComiRec-DR** | KDD 2020 | Multi-interest retrieval |

### Multi-Task Models - 5

| Model | Paper | Description |
|-------|-------|-------------|
| **ESMM** | SIGIR 2018 | Full space multi-task modeling |
| **MMoE** | KDD 2018 | Multi-gate mixture of experts |
| **PLE** | RecSys 2020 | Progressive layered extraction |
| **AITM** | KDD 2021 | Adaptive information transfer |
| **SharedBottom** | - | Classic multi-task shared bottom |

### Generative Recommendation - 4

| Model | Paper | Description |
|-------|-------|-------------|
| **HSTU** | Meta 2024 | Hierarchical sequential transformer unit |
| **HLLM** | 2024 | Hierarchical large language model recommendation |
| **TIGER** | NeurIPS 2023 | T5-based generative retrieval |
| **RQVAE** | - | Residual quantized variational autoencoder |

## Supported Datasets

The framework has built-in support for the following common datasets:

- **MovieLens** - Movie rating recommendation
- **Criteo** - CTR prediction
- **Census-Income** - Income prediction
- **Amazon** - Product recommendation
- **Custom Datasets** - Generate synthetic data via DataGenerator

### Data Format

```scala
// Dataset supported data format
case class Batch(
  sparseFeatures: Map[String, Tensor],    // Sparse features
  denseFeatures: Map[String, Tensor],      // Dense features
  sequenceFeatures: Map[String, Tensor],    // Sequence features
  labels: Option[Tensor],                  // Labels
  tokens: Option[Tensor],                   // Tokens
  positions: Option[Tensor],                 // Positions
  timeDiffs: Option[Tensor],                // Time differences
  targets: Option[Tensor],                  // Targets
  itemFeatures: Map[String, Tensor]         // Item features
)
```

## Examples

All examples are located in `src/main/scala/examples/` and `src/main/scala/tutorials/`

### Running Examples

```bash
# Compile project
sbt compile

# Run QuickStart CTR tutorial
sbt "runMain tutorials.QuickStartCTR"

# Run DeepFM example
sbt "runMain examples.ranking.DeepFMExample"

# Run DSSM matching example
sbt "runMain tutorials.MatchingDSSM"

# Run MMOE multi-task example
sbt "runMain tutorials.MultiTaskMMOE"

# Run full Benchmark
sbt "runMain benchmarks.BenchmarkRunner"
```

### BenchmarkRunner Sample Output

```
============================================================
TorchRec Scala Benchmark Suite
============================================================

--- DeepFM Benchmark ---
--- WideDeep Benchmark ---
--- DCN Benchmark ---
--- DSSM Benchmark ---
--- MMOE Benchmark ---

================================================================================
Benchmark Results Summary
================================================================================
Task        Model       Dataset     Training Time  Throughput  AUC/Metric
--------------------------------------------------------------------------------
ranking     DeepFM      synthetic             65.79s      303.99/sAUC=0.5000
ranking     WideDeep    synthetic              2.11s     9501.19/sAUC=0.0000
ranking     DCN         synthetic              2.07s     9680.54/sAUC=0.0000
matching    DSSM        synthetic              0.05s   217391.30/sloss=0.5000
multitask   MMOE        synthetic              0.00s 10000000.00/scvr_auc=0.7500
================================================================================
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

*Last updated: 2026-06-06*


<div align="center">

# Torch-RecHub-Scala: A Production-Ready Lightweight Scala Recommender System Framework (JavaCPP / libtorch Native Interoperability)

**Scala Edition Note**: This repository is a Scala port/engineering version of Torch-RecHub, deeply integrated with JavaCPP's PyTorch bindings (org.bytedeco.pytorch), providing:

- Bidirectional interoperability with native JavaCPP `Dataset` / `DataLoader` (Scala Dataset <-> JavaDataset/JavaTensorDataset);
- High-quality adapters (Scala -> Java) and wrappers (Java -> Scala), supporting Random / Sequential / Stream / Stateful / Distributed scenarios;
- `JavaBackedDataLoader` (and enhanced version `JavaBackedDataLoaderEnhanced`), allowing direct use of JavaCPP's DataLoader/Dataset as backend, consumed as `Iterable[Batch]` on the Scala side;
- Preserving Scala native APIs (`Dataset` trait / `Batch` case class / `DataLoader`) while providing seamless switching to JavaCPP's native data pipeline.

</div>


## Key Features

- Native interoperability: Scala `Dataset` ⇄ JavaCPP `JavaDataset` / `JavaTensorDataset` mutual conversion;
- DataLoader factories: `JavaDataLoaderFactory` + `DataLoader` companion `fromJava*` factories, providing both JavaCPP DataLoader and Scala-iterable `DataLoader`;
- `JavaBackedDataLoader`: decodes JavaCPP Example/TensorExample vectors back to Scala `Batch`, ready for existing Scala training/evaluation pipelines;
- Encoding conventions: `Batch` companion provides `toExample` / `fromExample` methods, supporting sparse/dense/label packing/unpacking; enhanced `JavaBackedDataLoaderEnhanced` provides `EncodingConfig` with default aggregation strategies (first/last/mean/length) for sequence/tokens fields;
- Non-invasive: does not modify existing Scala APIs; achieves interoperability through adapters/wrappers, avoiding JVM method signature conflicts.


## Quick Reference

- Installation & Build (sbt)
- Running Examples & Smoke Tests
- Scala ↔ JavaCPP Interoperability Guide (API & Encoding Conventions)
- DataLoader & Adapter Quick Reference
- Encoding Configuration (sequence/token strategies)
- FAQ & Troubleshooting


---

## Requirements (Build / Runtime)

- JDK 11+ (Azul / OpenJDK recommended)
- sbt 1.5+ (project uses Scala 3)
- Native libtorch binaries from JavaCPP must be loadable at runtime (to run JavaCPP's native DataLoader/training pipeline)
  - In most cases, org.bytedeco.pytorch Maven dependencies auto-extract native libraries at runtime. For specific platforms (GPU, ROCm, Ascend), ensure the corresponding native toolchain and drivers are properly installed.
- For compilation/development only: JDK + sbt is sufficient.


## Build & Compile

From the repository root:

```bash
# Enter project
cd /home/muller/IdeaProjects/torch-rechub-scala

# Compile project
sbt compile
```

- If you see runtime errors about native libraries (e.g., libtorch not found), it is usually a runtime loading issue for JavaCPP native libs, but compilation itself should still pass.


## Quick Examples (Scala)

The examples below demonstrate how to:
- Construct a JavaCPP DataLoader using Scala native `TensorDataset` (via factories), and how to use `JavaBackedDataLoader` to consume JavaCPP Dataset as backend and iterate as `Batch` on the Scala side.

Example: create a small in-memory TensorDataset, then construct a Java-backed DataLoader:

```scala
import torchrec.data._
import org.bytedeco.pytorch._
import torchrec.Implicits._

// Construct simple tensors (Impls for creating Tensor assumed available)
val f1 = tensor(Array(1f,2f,3f), Array(3L)) // Example: 3 samples
val f2 = tensor(Array(10f,20f,30f), Array(3L))
val labels = tensor(Array(0f,1f,0f), Array(3L))

val td = new TensorDataset(Map("f1" -> f1, "f2" -> f2), Map.empty, Some(labels))

// Method A: get Scala-side DataLoader (internally calls JavaDataLoaderFactory, returns scala DataLoader)
val scalaDl: DataLoader = DataLoader.fromJavaRandomTensor(td, batchSize = 2, numWorkers = 0)
for (batch <- scalaDl) {
  println("Batch sparse keys = " + batch.sparseFeatures.keys)
  println("labels = " + batch.labels)
}

// Method B: get JavaTensorDataset directly and pass to JavaCPP factory (closer to native path)
val javaTd: org.bytedeco.pytorch.JavaTensorDataset = td.asJavaTensorDataset()
val jdl = org.bytedeco.pytorch.JavaRandomTensorDataLoader(javaTd, new org.bytedeco.pytorch.RandomSampler(javaTd.size()), new org.bytedeco.pytorch.DataLoaderOptions())
// If you want to iterate as Batch on Scala layer from Java Dataset (Example -> Batch directly)
val javaBacked = new JavaBackedDataLoader(javaTd, Seq("f1","f2"), batchSize = 2)
for (b <- javaBacked) {
  println(b)
}
```

Note: the example above uses `torchrec.Implicits.tensor` to construct Tensors; refer to `torchrec.Implicits` implementation in actual code.


## DataLoader.fromJava* and JavaBackedDataLoader

The project provides the following key APIs:

- `DataLoader.fromJavaRandom(backing: Dataset, batchSize, numWorkers, dropLast)`: Returns Scala `DataLoader` (Iterable[Batch]), internally calls `JavaDataLoaderFactory.random(...)` to ensure the JavaCPP path is triggered, but ultimately returns a Scala-layer `DataLoader` instance for seamless consumption by existing Scala training code.

- `DataLoader.fromJavaRandomTensor(...)` / `fromJavaSequentialTensor(...)` / `fromJavaStatefulTensor(...)`: TensorExample-based variants (same semantics).

- `Dataset.asJavaDataset()` / `asJavaTensorDataset()`: Convenience methods on Scala `Dataset`, returning JavaCPP native `JavaDataset` / `JavaTensorDataset` adapters, ready to be passed to JavaCPP DataLoader factories.

- `JavaBackedDataLoader(javaDs: JavaDataset, featureOrder, batchSize, shuffle)`: Directly uses JavaDataset as backend, pulls Example from Java side, decodes Example to Scala `Batch` and aggregates by batchSize, returns Scala Iterable[Batch].

- `JavaBackedDataLoaderEnhanced`: Supports `EncodingConfig` (sparseOrder, denseOrder, seqPolicies, tokensPolicy, includeLabel), provides simple aggregation (first/last/mean/length) for sequence/tokens fields, encoding aggregated results as single-element tensors into Example data vectors for later decoding back to Batch's sequenceFeatures / tokens fields.


## Batch ↔ Example/TensorExample Encoding Rules (Default Implementation)

- Default packing (pack) rules:
  - 1-D data vector consists of three parts in order: [sparse values (per sparseOrder) | dense values (per denseOrder) | optional label (scalar)];
  - Each feature's value takes the first element from the sample's corresponding tensor (single element or shape[1]); missing keys are filled with 0.
  - Finally constructs a float tensor, then casts to Long dtype (consistent with JavaTensorDatasetAdapters implementation).

- Unpacking (unpack) rules:
  - Splits data vector per sparseOrder/denseOrder, reconstructs single-element tensors into Batch.sparseFeatures / Batch.denseFeatures;
  - If `includeLabel=true`, the single scalar at the end of the data vector maps to Batch.labels;
  - sequence/tokens: for complex variable-length sequence fields, the default implementation encodes aggregation metrics (e.g., first/last/mean/length) as single-element tensors and places them in Batch.sequenceFeatures or Batch.tokens (see `EncodingConfig` strategies).


## Step-by-Step Guide: Migrating examples/benchmarks/tutorials to JavaCPP DataLoader

1. Keep existing Scala DataLoader calls unchanged, prefer using `DataLoader.fromJava*` which returns Scala `DataLoader`; most example code can then consume `for(batch <- dataloader)` without modification;
2. When you want to go fully native: convert your Scala Dataset to `JavaTensorDataset` via `asJavaTensorDataset()`, then call `JavaDataLoaderFactory.randomTensor(...)` (or manually create DataLoader with JavaCPP APIs), and use `JavaBackedDataLoader` to decode Java DataLoader results back to Scala `Batch` if needed;
3. Points to watch during migration:
   - Ensure `featureOrder` (feature order) is consistent between Scala packing and Java unpacking;
   - If using enhanced encoding for sequence/tokens, set the same `seqPolicies` / `tokensPolicy` in `EncodingConfig`;
   - If you encounter native library loading errors at runtime (common on GPU/ROCm/Ascend platforms), follow platform/driver documentation to fix, or run CPU-only mode in CI ( Scala compilation + unit tests only).


## FAQ & Troubleshooting

- `sbt compile` fails but logs show JavaCPP native library issues: usually means compilation passed, but some compile-time or runtime tasks attempted to load native libs causing exceptions. Compilation itself should succeed; if running sample programs fails, check `LD_LIBRARY_PATH`, drivers, CUDA / ROCm readiness, or try launching in CPU-only mode (set appropriate JVM environment variables).

- Fields missing after decoding Example/TensorExample back: check if `featureOrder` is consistent, confirm that sparseOrder/denseOrder/includeLabel configuration used during packing matches the `EncodingConfig` used during unpacking.

- Want to "make your existing Scala Dataset inherit from JavaCPP's JavaDataset"?
  - Not recommended: JavaCPP's Dataset method signatures (returning Example/TensorExample) conflict with Scala trait's `get(index): Batch` return type, causing method signature conflicts that prevent compilation.
  - Use the adapter/wrapper approach (already provided in the project), which is non-invasive and more robust.


## Development Workflow / Testing Suggestions (Harness Engineer Style)

1. Local compilation: `sbt compile` (ensure all compiler warnings are addressed).
2. Unit tests: if tests are added later, run `sbt test`.
3. Smoke-run: choose a small example (e.g., smallest dataset in `examples/...`), create loader with `DataLoader.fromJavaRandomTensor`, `println` the first batch's shapes in a Main, confirm that decoding mapping matches model input dimensions.
4. For full native path: on supported platforms, run `JavaRandomTensorDataLoader` and use `JavaBackedTensorDataLoaderEnhanced` for decoding, watch runtime logs for native library loading messages to confirm success.


## License & Contributing

- This project is licensed under the MIT License (see LICENSE file).
- Pull requests, issues, and discussions about design and implementation details are all welcome.


---

*Last updated: 2026-06-06*