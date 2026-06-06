[![Torch-RecHub Banner](doc/img/hub.png)]
# Torch-RecHub-Scala: Scala 3 推荐系统框架

[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Scala](https://img.shields.io/badge/scala-3.8.3-DC322F?style=for-the-badge)](https://www.scala-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=for-the-badge)](https://pytorch.org/)
[![JavaCPP](https://img.shields.io/badge/JavaCPP-1.5.13-FF6600?style=for-the-badge)](https://github.com/bytedeco/javacpp)

[English](README.md) | [简体中文](README_zh.md)
[![Torch-RecHub Banner](doc/img/core.png)]
## 目录

- [简介](#简介)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [支持的模型](#支持的模型)
- [支持的数据集](#支持的数据集)
- [示例](#示例)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 简介

**Torch-RecHub-Scala** 是一个基于 Scala 3 和 JavaCPP-PyTorch 的轻量级、高效、易用的推荐系统框架。它是 Python 版 [Torch-RecHub](https://github.com/datawhalechina/torch-rechub) 的 Scala 实现，充分利用了 Scala 的类型安全、函数式编程和 JavaCPP 的高性能 PyTorch 绑定。

## 特性

- **Scala 3 原生支持**: 利用 Scala 3 的新特性（Contextual Abstractions、Extension Methods、Multiversal Equality）提供类型安全的 API
- **JavaCPP 深度集成**: 通过 JavaCPP 直接调用 PyTorch C++ API，支持 CPU、CUDA GPU、分布式训练
- **丰富的模型库**: 涵盖 **30+** 主流推荐算法（召回、排序、多任务、生成式推荐等）
- **模块化设计**: 易于添加新模型、数据集和评估指标
- **标准化流程**: 提供统一的 Scala Dataset/DataLoader、数据加载、训练和评估流程
- **JavaCPP DataLoader 深度支持**:
  - `JavaDataset` / `JavaTensorDataset` - 通用数据集
  - `JavaStatefulDataset` / `JavaStatefulTensorDataset` - 有状态数据集
  - `JavaStreamDataset` / `JavaStreamTensorDataset` - 流式数据集
  - `DistributedRandomSampler` / `DistributedSequentialSampler` - 分布式采样器
- **分布式训练支持**:
  - `DDPTrainer` - DistributedDataParallel 训练器
  - `FSDPTrainer` - FullyShardedDataParallel 训练器
- **纯 JVM 生态**: 与 Scala/Java/JVM 生态系统无缝集成

## 安装

### 环境要求

- Scala 3.8+
- sbt 1.9+
- Java 17+ (推荐 Java 21)
- PyTorch 2.10.0 (通过 JavaCPP 自动加载)

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-repo/torch-rechub-scala.git
cd torch-rechub-scala

# 使用 sbt 编译
sbt compile

# 运行示例
sbt "runMain examples.ranking.DeepFMExample"
```

### Maven/Coursier 依赖 (可选)

```scala
// build.sbt
libraryDependencies ++= Seq(
  "org.bytedeco" % "javacpp" % "1.5.13",
  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13",
  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13"
)
```

## 快速开始

### 1. CTR 排序模型训练

```scala
import torchrec.data._
import torchrec.models.ranking._
import torchrec.trainers._
import torchrec.Implicits._

// 生成数据
val (trainData, valData, testData) = DataGenerator.generateRankingData(
  numSamples = 10000,
  numSparseFeatures = 10,
  numDenseFeatures = 5,
  vocabSize = 100
)

// 创建数据加载器
val trainLoader = DataLoader.fromJavaRandom(trainData, batchSize = 256)
val valLoader = DataLoader.fromJavaSequential(valData, batchSize = 256)

// 定义特征
val features = (0 until 10).map { i =>
  SparseFeature(s"feat_$i", vocabSize = 100, embedDim = 8)
}

// 创建模型
val model = new DeepFM(features, embedDim = 8, mlpDims = List(64L, 32L))

// 训练
val trainer = new CTRTrainer(model, learningRate = 0.001f)
trainer.fit(trainLoader, Some(valLoader))

// 评估
val metrics = trainer.evaluate(valLoader)
println(s"AUC: ${metrics("AUC")}")
```

### 2. 召回模型训练

```scala
import torchrec.models.matching._
import torchrec.trainers._

// 生成匹配数据
val (trainData, _, _) = DataGenerator.generateMatchingData(
  numUsers = 5000,
  numItems = 1000,
  vocabSize = 100
)

// 创建数据加载器
val trainLoader = DataLoader.fromJavaRandom(trainData, batchSize = 128)

// 定义用户/物品特征
val userFeatures = (0 until 3).map { i =>
  SparseFeature(s"user_feat_$i", vocabSize = 100, embedDim = 16)
}
val itemFeatures = (0 until 2).map { i =>
  SparseFeature(s"item_feat_$i", vocabSize = 1000, embedDim = 16)
}

// 创建 DSSM 模型
val model = new DSSM(userFeatures, itemFeatures, embedDim = 16, towerDims = List(128L, 64L))

// 训练
val trainer = new MatchTrainer(model, learningRate = 0.001f)
trainer.fit(trainLoader)
```

### 3. 多任务学习

```scala
import torchrec.models.multi_task._

// 生成多任务数据
val taskNames = List("ctr", "cvr")
val (trainData, _, _) = DataGenerator.generateMultiTaskData(
  numSamples = 10000,
  numFeatures = 10,
  taskNames = taskNames
)

// 创建 MMOE 模型
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

// 训练
val trainer = new MTLTrainer(model, taskNames, learningRate = 0.001f)
trainer.fit(trainLoader)
```

## 项目结构

```
torch-rechub-scala/
├── README.md                    # 项目文档
├── build.sbt                   # sbt 构建配置
├── src/main/scala/
│   ├── torchrec/               # 核心库
│   │   ├── TorchRec.scala       # 主入口
│   │   ├── Implicits.scala      # 隐式转换
│   │   ├── TensorImplicits.scala # Tensor 扩展
│   │   ├── basic/              # 基础组件
│   │   │   ├── features/        # 特征定义
│   │   │   │   └── Feature.scala
│   │   │   ├── layers/         # 神经网络层
│   │   │   │   ├── MLP.scala
│   │   │   │   ├── FM.scala
│   │   │   │   ├── CrossNetwork.scala
│   │   │   │   ├── CIN.scala
│   │   │   │   ├── SENETLayer.scala
│   │   │   │   └── EmbeddingLayer.scala
│   │   │   ├── losses/         # 损失函数
│   │   │   │   └── Loss.scala
│   │   │   └── metrics/        # 评估指标
│   │   │       └── Metric.scala
│   │   ├── data/               # 数据处理
│   │   │   ├── Dataset.scala   # Dataset 基类
│   │   │   ├── DataLoader.scala # DataLoader
│   │   │   ├── DataGenerator.scala # 数据生成器
│   │   │   ├── JavaDatasetAdapters.scala    # JavaDataset 适配器
│   │   │   ├── JavaTensorDatasetAdapters.scala # TensorDataset 适配器
│   │   │   ├── JavaDistributedAdapters.scala # 分布式适配器
│   │   │   ├── JavaSamplerAdapters.scala    # Sampler 工厂
│   │   │   └── JavaDataLoaderAdapters.scala # DataLoader 工厂
│   │   ├── models/             # 推荐模型
│   │   │   ├── ranking/        # 排序模型
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
│   │   │   ├── matching/        # 召回模型
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
│   │   │   ├── multi_task/      # 多任务模型
│   │   │   │   ├── ESMM.scala
│   │   │   │   ├── MMOE.scala
│   │   │   │   ├── PLE.scala
│   │   │   │   ├── AITM.scala
│   │   │   │   └── SharedBottom.scala
│   │   │   └── generative/      # 生成式推荐
│   │   │       ├── HSTU.scala
│   │   │       ├── HLLM.scala
│   │   │       ├── TIGER.scala
│   │   │       └── RQVAE.scala
│   │   ├── trainers/           # 训练器
│   │   │   ├── CTRTrainer.scala    # CTR 训练
│   │   │   ├── MatchTrainer.scala  # 召回训练
│   │   │   ├── MTLTrainer.scala   # 多任务训练
│   │   │   └── TrainLoop.scala      # 训练循环
│   │   ├── distributed/        # 分布式训练
│   │   │   ├── DDPConfig.scala
│   │   │   ├── DDPTrainer.scala
│   │   │   ├── FSDPConfig.scala
│   │   │   └── FSDPTrainer.scala
│   │   ├── utils/             # 工具函数
│   │   │   ├── DataUtils.scala
│   │   │   ├── MatchUtils.scala
│   │   │   ├── ModelUtils.scala
│   │   │   └── Trie.scala
│   │   ├── quantization/       # 量化
│   │   │   └── Quantizer.scala
│   │   └── serving/            # 在线服务
│   │       ├── VectorIndexer.scala
│   │       └── package.scala
│   ├── examples/               # 示例
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
│   ├── tutorials/              # 教程
│   │   ├── QuickStartCTR.scala
│   │   ├── MatchingDSSM.scala
│   │   ├── MultiTaskMMOE.scala
│   │   └── RankingDIN.scala
│   └── benchmarks/            # 性能测试
│       ├── BenchmarkRunner.scala
│       ├── DataGenerator.scala
│       └── ModelBenchmark.scala
```

## 支持的模型

### 排序模型 (Ranking) - 12个

| 模型 | 论文 | 简介 |
|------|------|------|
| **DeepFM** | IJCAI 2017 | FM + Deep 联合训练 |
| **Wide&Deep** | DLRS 2016 | 记忆 + 泛化能力结合 |
| **DCN** | KDD 2017 | 显式特征交叉网络 |
| **DCN-v2** | WWW 2021 | 增强版交叉网络 |
| **DIN** | KDD 2018 | 注意力机制捕捉用户兴趣 |
| **DIEN** | AAAI 2019 | 兴趣演化建模 |
| **AFM** | IJCAI 2017 | 注意力因子分解机 |
| **AutoInt** | CIKM 2019 | 自动特征交互学习 |
| **FiBiNET** | RecSys 2019 | 特征重要性 + 双线性交互 |
| **DeepFFM** | RecSys 2019 | 场感知因子分解机 |
| **EDCN** | KDD 2021 | 增强型交叉网络 |
| **BST** | DLP-KDD 2019 | Transformer 序列建模 |

### 召回模型 (Matching) - 12个

| 模型 | 论文 | 简介 |
|------|------|------|
| **DSSM** | CIKM 2013 | 经典双塔召回模型 |
| **YoutubeDNN** | RecSys 2016 | YouTube 深度召回 |
| **MIND** | CIKM 2019 | 多兴趣动态路由 |
| **GRU4Rec** | ICLR 2016 | GRU 序列推荐 |
| **SASRec** | ICDM 2018 | 自注意力序列推荐 |
| **NARM** | CIKM 2017 | 神经注意力会话推荐 |
| **STAMP** | KDD 2018 | 短期注意力记忆优先 |
| **SINE** | WSDM 2021 | 稀疏兴趣网络 |
| **ComiRec-SA** | KDD 2020 | 可控多兴趣推荐 |
| **ComiRec-DR** | KDD 2020 | 多兴趣检索 |

### 多任务模型 (Multi-Task) - 5个

| 模型 | 论文 | 简介 |
|------|------|------|
| **ESMM** | SIGIR 2018 | 全空间多任务建模 |
| **MMoE** | KDD 2018 | 多门控专家混合 |
| **PLE** | RecSys 2020 | 渐进式分层提取 |
| **AITM** | KDD 2021 | 自适应信息迁移 |
| **SharedBottom** | - | 经典多任务共享底层 |

### 生成式推荐 (Generative) - 4个

| 模型 | 论文 | 简介 |
|------|------|------|
| **HSTU** | Meta 2024 | 层级序列转换单元 |
| **HLLM** | 2024 | 层级大语言模型推荐 |
| **TIGER** | NeurIPS 2023 | 基于 T5 的生成式检索 |
| **RQVAE** | - | 残差量化变分自编码器 |

## 支持的数据集

框架内置了对以下常见数据集的支持：

- **MovieLens** - 电影评分推荐
- **Criteo** - CTR 预测
- **Census-Income** - 收入预测
- **Amazon** - 商品推荐
- **自定义数据集** - 通过 DataGenerator 生成合成数据

### 数据格式

```scala
// Dataset 支持的数据格式
case class Batch(
  sparseFeatures: Map[String, Tensor],    // 稀疏特征
  denseFeatures: Map[String, Tensor],      // 密集特征
  sequenceFeatures: Map[String, Tensor],    // 序列特征
  labels: Option[Tensor],                  // 标签
  tokens: Option[Tensor],                   // Token
  positions: Option[Tensor],                 // 位置
  timeDiffs: Option[Tensor],                // 时间差
  targets: Option[Tensor],                  // 目标
  itemFeatures: Map[String, Tensor]         // 物品特征
)
```

## 示例

所有示例位于 `src/main/scala/examples/` 和 `src/main/scala/tutorials/`

### 运行示例

```bash
# 编译项目
sbt compile

# 运行 QuickStart CTR 教程
sbt "runMain tutorials.QuickStartCTR"

# 运行 DeepFM 示例
sbt "runMain examples.ranking.DeepFMExample"

# 运行 DSSM 召回示例
sbt "runMain tutorials.MatchingDSSM"

# 运行 MMOE 多任务示例
sbt "runMain tutorials.MultiTaskMMOE"

# 运行完整 Benchmark
sbt "runMain benchmarks.BenchmarkRunner"
```

### BenchmarkRunner 输出示例

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

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

*最后更新: 2026-06-06*
