// torch-rechub-scala build.sbt
// Scala 3 Recommendation System Framework using JavaCPP-PyTorch

ThisBuild / scalaVersion := "3.8.3"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / organization := "torchrec"

//val pytorchVersion = "2.12.0-1.5.14-SNAPSHOT"

resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"
resolvers += "aliyun" at "https://maven.aliyun.com/repository/public"
resolvers ++= Seq(
  Resolver.sonatypeCentralSnapshots,
  "aliyun-snapshot" at "https://maven.aliyun.com/repository/public"
)
updateOptions := updateOptions.value.withLatestSnapshots(true)
resolvers += "aliyunmaven" at "https://maven.aliyun.com/repository/public"
// Source: https://mvnrepository.com/artifact/com.lihaoyi/requests
libraryDependencies += "com.lihaoyi" %% "requests" % "0.9.3"
lazy val root = (project in file("."))
  .settings(
    name := "torch-rechub-scala",
    description := "Scala 3 Recommendation System Framework using JavaCPP-PyTorch",

    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Yexplicit-nulls",
      "-language:postfixOps",
      "-language:implicitConversions"
    ),

    libraryDependencies ++= Seq(
      // JavaCPP PyTorch
//      "org.bytedeco" % "javacpp-pytorch" % pytorchVersion,
//      "org.bytedeco" % "javacpp" % "1.5.14-SNAPSHOT",
//
//      // JavaCPP CUDA support
//      "org.bytedeco" % "javacpp-cuda" % "13.1-9.19-1.5.14-SNAPSHOT",

      // Logging
      "org.slf4j" % "slf4j-api" % "2.0.16",
      "org.slf4j" % "slf4j-simple" % "2.0.16",

      // JSON parsing
      "com.google.code.gson" % "gson" % "2.10.1",

      // ScalaTest for testing
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),

    // Disable publishing for now
    publish := {},
    publishLocal := {}
  )

// Enable parallel execution
ThisBuild / concurrentRestrictions := Seq.empty

// Compiler settings
scalacOptions ++= Seq(
  "-WnonLocalReturn"
)

resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"
resolvers += "aliyun" at "https://maven.aliyun.com/repository/public"
resolvers ++= Seq(
  Resolver.sonatypeCentralSnapshots,
  "aliyun-snapshot" at "https://maven.aliyun.com/repository/public"
)
updateOptions := updateOptions.value.withLatestSnapshots(true)
resolvers += "aliyunmaven" at "https://maven.aliyun.com/repository/public"

libraryDependencies ++= Seq(
  // JavaCPP
  "org.bytedeco" % "javacpp" % "1.5.13",
  "org.bytedeco" % "javacpp" % "1.5.13" classifier "linux-x86_64",

  // OpenBLAS
  "org.bytedeco" % "openblas" % "0.3.31-1.5.13",
  "org.bytedeco" % "openblas" % "0.3.31-1.5.13" classifier "linux-x86_64",

  // PyTorch
  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13",
  "org.bytedeco" % "pytorch" % "2.10.0-1.5.13" classifier "linux-x86_64",

  "org.bytedeco" % "pytorch-platform-gpu" % "2.10.0-1.5.13" excludeAll(
    ExclusionRule("org.bytedeco", "cuda-platform"),
    ExclusionRule("org.bytedeco", "javacpp-platform"),
    ExclusionRule("org.bytedeco", "openblas-platform")
  ),

  // CUDA
  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda" % "13.1-9.19-1.5.13" classifier "linux-x86_64",

  "org.bytedeco" % "cuda-redist" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-cublas" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-cudnn" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-cusolver" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-cusparse" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-npp" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-nccl" % "13.1-9.19-1.5.13",
  "org.bytedeco" % "cuda-redist-nvcomp" % "13.1-9.19-1.5.13",

  "org.bytedeco" % "cuda-redist" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cublas" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cudnn" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cusolver" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-cusparse" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-npp" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-nccl" % "13.1-9.19-1.5.13" classifier "linux-x86_64",
  "org.bytedeco" % "cuda-redist-nvcomp" % "13.1-9.19-1.5.13" classifier "linux-x86_64",

  // OpenCV
  "org.bytedeco" % "opencv" % "4.13.0-1.5.13",
  "org.bytedeco" % "opencv" % "4.13.0-1.5.13" classifier "linux-x86_64",

  // FFmpeg
  "org.bytedeco" % "ffmpeg" % "8.0.1-1.5.13",
  "org.bytedeco" % "ffmpeg" % "8.0.1-1.5.13" classifier "linux-x86_64",

  // Gson
  "com.google.code.gson" % "gson" % "2.14.0",

  // HTTP client
  "com.lihaoyi" %% "requests" % "0.9.3",

  // ScalaTest
  "org.scalatest" %% "scalatest" % "3.2.18"
)