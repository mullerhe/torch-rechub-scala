package torchrec.dataframe

import java.io.File
import scala.collection.mutable

object DataFrameBenchmarkApp {

  private case class BenchmarkCase(name: String, run: () => Unit)

  def main(args: Array[String]): Unit = {
    val csvPath = "/home/muller/IdeaProjects/torch-rechub-scala/data/fraud_data.csv"
    if (!new File(csvPath).exists()) {
      System.err.println(s"Error: $csvPath does not exist.")
      sys.exit(1)
    }

    println(s"Loading dataset from $csvPath...")

    val rawDf = DataFrame.readCSV(csvPath).drop("V1","V2")
    rawDf.show(30)
//    rawDf.describe().show(30)
    rawDf.info()
//    rawDf.apply(row => row.)

    val df = if (rawDf.numRows > 2000) rawDf.head(2000) else rawDf
    println(s"Loaded DataFrame with ${df.numRows} rows and ${df.numCols} columns.")
    println(s"Columns: ${df.columns.mkString(", ")}")

    val numericCols = df.columns.filter(c => df.columnData(c).dtype.isPrimitive)
    val stringCols = df.columns.filter(c => df.columnData(c).dtype == DataType.String)
    val primaryKey = df.columns.headOption.getOrElse("id")
    val numericCol = numericCols.headOption.getOrElse(primaryKey)
    val stringCol = stringCols.headOption.getOrElse(primaryKey)
    val dateCol = primaryKey

    val cases = mutable.ArrayBuffer[BenchmarkCase]()
    def add(name: String)(body: => Unit): Unit = cases += BenchmarkCase(name, () => body)

    def repeat(prefix: String, times: Int)(body: Int => Unit): Unit =
      (0 until times).foreach(i => add(f"$prefix-${i + 1}%03d") { body(i) })

    repeat("attrs", 40) { i =>
      val rows = df.numRows
      val cols = df.numCols
      val shape = df.shape
      val dtypes = df.dtypes
      val values = df.values
      require(rows == shape._1 && cols == shape._2)
      require(dtypes.nonEmpty || df.isEmpty)
      require(values.length == rows)
      require(df.index.nonEmpty || rows == 0)
      if (i % 5 == 0) {
        require(df.to_numpy().length == rows)
        require(df.get_values().length == rows)
      }
    }

    repeat("selection", 40) { i =>
      val takeN = math.max(1, math.min(df.numCols, (i % math.max(df.numCols, 1)) + 1))
      val cols = df.columns.take(takeN)
      val sel = df.select(cols*)
      val dropped = if (cols.nonEmpty) df.drop(cols.head) else df
      require(sel.numCols == cols.length)
      require(dropped.numCols <= df.numCols)
      require(df.selectCols(0 until takeN).numCols == takeN)
      require(df.loc(cols*).numCols == cols.length)
      require(df.get(primaryKey, null) != null || df.columns.nonEmpty)
    }

    repeat("row-access", 40) { i =>
      val rowIdx = if (df.numRows == 0) 0 else i % df.numRows
      if (df.numRows > 0) {
        val row = df.iloc(rowIdx)
        require(row.toMap.size == df.numCols)
        require(df.iloc(Seq(rowIdx)).numRows == 1)
        require(df.iloc(0 until math.min(3, df.numRows)).numRows == math.min(3, df.numRows))
        if (df.numCols > 0) {
          df.iat(rowIdx, 0)
          df.at(rowIdx, primaryKey)
        }
      }
      require(df.select().numCols == 0)
    }

    repeat("missing", 40) { i =>
      val na = df.isna()
      val nna = df.notna()
      val filled = df.fillna(value = 0)
      val dropped = df.dropna()
      require(na.numRows == df.numRows && na.numCols == df.numCols)
      require(nna.numRows == df.numRows && nna.numCols == df.numCols)
      require(filled.numRows == df.numRows)
      require(dropped.numRows <= df.numRows)
      if (i % 2 == 0) require(df.isnull().numCols == df.numCols)
    }

    repeat("mutate", 40) { i =>
      val baseCol = df.columnData(primaryKey)
      val cond = df.isna()
      val masked = df.mask(cond, other = -1)
      val where = df.where(cond, other = -2)
      val replaced = df.replace(to_replace = null, value = 0)
      val shifted = df.shift(fill_value = 0)
      require(masked.numRows == df.numRows)
      require(where.numCols == df.numCols)
      require(replaced.numCols == df.numCols)
      require(shifted.numRows == df.numRows)
      if (baseCol.length > 0 && i % 3 == 0) {
        require(df.duplicated().length == df.numRows)
      }
    }

    repeat("math", 40) { i =>
      val added = df.add(1)
      val subbed = df.sub(1)
      val mul = df.mul(2)
      val div = df.div(2)
      val pow = df.pow(2)
      val mod = df.mod(2)
      val rdiv = df.rdiv(2)
      val radd = df.radd(2)
      require(added.numCols == df.numCols)
      require(subbed.numCols == df.numCols)
      require(mul.numCols == df.numCols)
      require(div.numCols == df.numCols)
      require(pow.numCols == df.numCols)
      require(mod.numCols == df.numCols)
      require(rdiv.numCols == df.numCols)
      require(radd.numCols == df.numCols)
      if (numericCols.nonEmpty && i % 4 == 0) {
        require(df.abs().numCols == df.numCols)
        require(df.sign().numCols == df.numCols)
        require(df.round().numCols == df.numCols)
      }
    }

    repeat("comparison", 40) { i =>
      val lt = df.lt(1)
      val le = df.le(1)
      val gt = df.gt(1)
      val ge = df.ge(1)
      val eq = df.eq(1)
      val ne = df.ne(1)
      val between = df.between(0, 1)
      require(lt.numCols == df.numCols)
      require(le.numCols == df.numCols)
      require(gt.numCols == df.numCols)
      require(ge.numCols == df.numCols)
      require(eq.numCols == df.numCols)
      require(ne.numCols == df.numCols)
      require(between.numCols == df.numCols)
      if (i % 6 == 0) require(df.any().nonEmpty && df.all().nonEmpty)
    }

    repeat("sort-rank", 40) { i =>
      val sorted = df.sort_values(by = primaryKey)
      val sortedCols = df.sort_columns()
      val ranked = df.rank()
      val nlargest = if (numericCols.nonEmpty) df.nlargest(1, numericCol) else df.head(1)
      val nsmallest = if (numericCols.nonEmpty) df.nsmallest(1, numericCol) else df.head(1)
      require(sorted.numRows == df.numRows)
      require(sortedCols.numCols == df.numCols)
      require(ranked.numCols == df.numCols)
      require(nlargest.numRows <= df.numRows)
      require(nsmallest.numRows <= df.numRows)
      if (i % 5 == 0) df.T
    }

    repeat("groupby-window", 40) { i =>
      val grouped = df.groupby(primaryKey)
      val count = grouped.count()
      val firstGroup = grouped.groups.keys.headOption.map(k => grouped.get_group(k)).getOrElse(df.head(0))
      val rolling = df.rolling(math.min(5, math.max(df.numRows, 1)))
      val expanding = df.expanding()
      val ewm = df.ewm()
      val window = df.window(math.min(5, math.max(df.numRows, 1)))
      require(count.numCols >= 1)
      require(firstGroup.numCols <= df.numCols)
      require(rolling.mean().numCols == df.numCols)
      require(expanding.sum().numCols == df.numCols)
      require(ewm.mean().numCols == df.numCols)
      require(window.get_window_size() >= 1)
      if (i % 2 == 0) {
        require(grouped.window(3).get_window_size() == 3)
        require(grouped.expanding().count().numCols == df.numCols)
      }
    }

    repeat("reshape", 40) { i =>
      val cols = if (df.numCols >= 2) df.columns.take(2) else df.columns
      val meltDf = if (cols.nonEmpty) df.melt(value_vars = cols, id_vars = df.columns.take(math.max(df.numCols - cols.length, 0))) else df
      val pivot = if (df.numCols >= 3) df.pivot(df.columns.head, df.columns(1), df.columns(2)) else df
      val explodeDf = if (df.numRows > 0) df.withColumn("tmp_list", Column("tmp_list", Seq.fill(df.numRows)(Seq("a", "b")), DataType.List)).explode("tmp_list") else df
      val csv = df.to_csv()
      val json = df.to_json()
      val records = df.to_records()
      require(meltDf.numCols >= 1)
      require(pivot.numCols >= 1)
      require(explodeDf.numRows >= df.numRows)
      require(csv.nonEmpty)
      require(json.nonEmpty)
      require(records.length == df.numRows)
      if (i % 4 == 0) {
        require(df.to_string().nonEmpty)
        require(df.to_list().length == df.numRows)
      }
    }

    repeat("compatibility", 100) { i =>
      val op = i % 20
      op match {
        case 0 => require(df.to_numpy().length == df.numRows)
        case 1 => require(df.get_values().length == df.numRows)
        case 2 => require(df._get_values().length == df.numRows)
        case 3 => require(!df._is_view)
        case 4 => require(!df.is_copy)
        case 5 => require(df._mgr.nonEmpty)
        case 6 => require(df.get_loc(primaryKey) >= 0)
        case 7 => require(df.get_level_values(0).nonEmpty)
        case 8 => require(df.factorize()._1.length == df.numRows || df.numRows == 0)
        case 9 => require(df.unique().nonEmpty)
        case 10 => require(df.mode().nonEmpty)
        case 11 => require(df.convert_dtypes().numCols == df.numCols)
        case 12 => require(df.infer_objects().numCols == df.numCols)
        case 13 =>
          val first = df.columns.headOption.getOrElse(primaryKey)
          require(df.astype(Map(first -> df.columnData(first).dtype.toPandas)).numCols == df.numCols)
        case 14 => require(df.is_dtype_equal(df.copy()))
        case 15 => require(df.get_dummies(Seq(stringCol)).numCols >= df.numCols - 1)
        case 16 => require(df.crosstab(primaryKey, primaryKey).numCols >= 1)
        case 17 => require(df.cut(numericCol, Seq(0.0, 1.0, 2.0)).dtype == DataType.String)
        case 18 => require(df.timedelta(1, 1) == 86401L)
        case 19 => require(df.copy().numRows == df.numRows)
      }
    }

//    require(cases.length == 500, s"Expected 500 benchmark cases, found ${cases.length}")
    println(s"Running ${cases.length} API benchmark cases...")

    var passed = 0
    var failed = 0
    val timings = mutable.ArrayBuffer[(String, Long)]()

    cases.foreach { benchmarkCase =>
      val start = System.nanoTime()
      try {
        benchmarkCase.run()
        passed += 1
      } catch {
        case t: Throwable =>
          failed += 1
          System.err.println(s"[FAIL] ${benchmarkCase.name}: $t")
          t.printStackTrace()
      }
      val elapsed = System.nanoTime() - start
      timings += benchmarkCase.name -> elapsed
    }

    val avgMs = if (timings.nonEmpty) timings.map(_._2).sum.toDouble / timings.length / 1e6 else 0.0
    val maxMs = if (timings.nonEmpty) timings.map(_._2).max.toDouble / 1e6 else 0.0
    val minMs = if (timings.nonEmpty) timings.map(_._2).min.toDouble / 1e6 else 0.0

    println("=" * 72)
    println(s"Benchmark summary: passed=$passed failed=$failed total=${cases.length}")
    println(f"Latency: avg=$avgMs%.3f ms min=$minMs%.3f ms max=$maxMs%.3f ms")
    println("=" * 72)

    if (failed > 0) sys.exit(1)
  }
}

