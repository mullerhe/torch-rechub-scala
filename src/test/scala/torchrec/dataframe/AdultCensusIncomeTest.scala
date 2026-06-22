package torchrec.dataframe

import scala.collection.mutable

/**
 * Scala 3 DataFrame translation of Adult Census Income Jupyter notebook
 * Original notebook: /home/muller/IdeaProjects/torch-rechub-scala/data/Adult Census Income.ipynb
 * Dataset: /home/muller/IdeaProjects/torch-rechub-scala/data/adult.csv
 */
object AdultCensusIncomeTest {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("Adult Census Income Analysis - Scala DataFrame Translation")
    println("=" * 80)

    // 1. Load data from CSV
    val df = DataFrame.readCSV("/home/muller/IdeaProjects/torch-rechub-scala/data/adult.csv")
    println(s"\n【1. Data Loading】")
    println(s"Number of Observations in adult dataset: ${df.shape}")

    // 2. Show first 5 rows
    println(s"\n【2. First 5 rows】")
    df.head(5).show()

    // 3. DataFrame info
    println(s"\n【3. DataFrame Info】")
    df.info()

    // 4. Describe numerical columns
    println(s"\n【4. Describe Numerical Columns】")
    val numCols = List("age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week")
    df.describe(numCols*).show()

    // 5. Separate categorical and numerical columns
    println(s"\n【5. Column Types Separation】")
    val allCols = df.columns
    val catCols = allCols.filter(c => df.columnData(c).dtype == DataType.String)
    val numericColsList = allCols.filter(c => df.columnData(c).dtype != DataType.String)
    println(s"Categorical columns: ${catCols.mkString(", ")}")
    println(s"Numerical columns: ${numericColsList.mkString(", ")}")

    // 6. Value counts for categorical columns
    println(s"\n【6. Value Counts for Categorical Columns】")
    for (col <- catCols) {
      println(s"\n--- $col ---")
      val valueCounts = df.groupby(col).count().sort_values("count", ascending = false)
      valueCounts.show(10)
    }

    // 7. Replace '?' with 'unknown' in specific columns
    println(s"\n【7. Replace '?' with 'unknown'】")
    val editCols = List("native.country", "occupation", "workclass")
    var adultDf = df
    for (col <- editCols) {
      adultDf = adultDf.withColumn(col,
        adultDf.columnData(col).map(v => if (v == "?" || v == null) "unknown" else v))
    }
    println("Replaced '?' with 'unknown' in native.country, occupation, workclass")

    // Verify no '?' remaining
    for (col <- editCols) {
      val hasQuestionMark = adultDf.columnData(col).toSeq.exists(v => v == "?")
      println(s"? in $col: $hasQuestionMark")
    }

    // 8. Education grouping
    println(s"\n【8. Education Grouping】")
    val hsGrad = List("HS-grad", "11th", "10th", "9th", "12th")
    val elementary = List("1st-4th", "5th-6th", "7th-8th")

    adultDf = adultDf.withColumn("education",
      adultDf.columnData("education").map(v =>
        if (hsGrad.contains(v)) "HS-grad"
        else if (elementary.contains(v)) "elementary_school"
        else v))

    println("Grouped education: HS-grad (9th-12th), elementary_school (1st-8th)")
    adultDf.groupby("education").count().sort_values("count", ascending = false).show()

    // 9. Marital Status grouping
    println(s"\n【9. Marital Status Grouping】")
    val married = List("Married-spouse-absent", "Married-civ-spouse", "Married-AF-spouse")
    val separated = List("Separated", "Divorced")

    adultDf = adultDf.withColumn("marital.status",
      adultDf.columnData("marital.status").map(v =>
        if (married.contains(v)) "Married"
        else if (separated.contains(v)) "Separated"
        else v))

    println("Grouped marital.status: Married, Separated")
    adultDf.groupby("marital.status").count().sort_values("count", ascending = false).show()

    // 10. Workclass grouping
    println(s"\n【10. Workclass Grouping】")
    val selfEmployed = List("Self-emp-not-inc", "Self-emp-inc")
    val govtEmployees = List("Local-gov", "State-gov", "Federal-gov")

    adultDf = adultDf.withColumn("workclass",
      adultDf.columnData("workclass").map(v =>
        if (selfEmployed.contains(v)) "Self_employed"
        else if (govtEmployees.contains(v)) "Govt_employees"
        else v))

    println("Grouped workclass: Self_employed, Govt_employees")
    adultDf.groupby("workclass").count().sort_values("count", ascending = false).show()

    // 11. Income distribution
    println(s"\n【11. Income Distribution】")
    adultDf.groupby("income").count().show()
    val incomeCounts = adultDf.groupby("income").count().collect()
    val total = incomeCounts.map(_.getDouble("count")).sum
    incomeCounts.foreach(r => {
      val income = r("income")
      val count = r.getDouble("count")
      val percent = count / total * 100
      println(s"$income: $count (${f"$percent%.2f%%"})")
    })

    // 12. Capital gain and loss analysis
    println(s"\n【12. Capital Gain/Loss Analysis】")
    val capitalGainMask = new Column("temp",
      adultDf.columnData("capital.gain").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() > 0.0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    val capitalGainDf = adultDf.filter(capitalGainMask)

    val capitalLossMask = new Column("temp",
      adultDf.columnData("capital.loss").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() > 0.0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    val capitalLossDf = adultDf.filter(capitalLossMask)

    println(s"Number of observations with capital.gain > 0: ${capitalGainDf.numRows}")
    println(s"Number of observations with capital.loss > 0: ${capitalLossDf.numRows}")

    val capitalGainPercent = capitalGainDf.numRows.toDouble / adultDf.numRows * 100
    val capitalLossPercent = capitalLossDf.numRows.toDouble / adultDf.numRows * 100
    println(f"Percentage with capital.gain > 0: $capitalGainPercent%.4f%%")
    println(f"Percentage with capital.loss > 0: $capitalLossPercent%.4f%%")

    // 13. Both capital gain and loss are zero
    println(s"\n【13. Capital Gain/Loss Both Zero Analysis】")
    val bothZeroMask = new Column("temp",
      adultDf.columnData("capital.gain").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() == 0.0).zip(
        adultDf.columnData("capital.loss").toSeq.map(v =>
          v.asInstanceOf[Number].doubleValue() == 0.0)).map(p => p._1 && p._2).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)

    val bothZeroDf = adultDf.filter(bothZeroMask)
    println(s"Number of observations with both capital.gain and capital.loss = 0: ${bothZeroDf.numRows}")
    println(f"Percentage: ${bothZeroDf.numRows.toDouble / adultDf.numRows * 100}%.4f%%")

    // Show income distribution for both zero group
    println("\nIncome distribution for both zero group:")
    bothZeroDf.groupby("income").count().show()

    // 14. Capital gain = 99999 analysis (outlier)
    println(s"\n【14. Capital Gain = 99999 Outlier Analysis】")
    val maxGainMask = new Column("temp",
      adultDf.columnData("capital.gain").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() == 99999.0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    val maxGainDf = adultDf.filter(maxGainMask)
    println(s"Number of observations with capital.gain = 99999: ${maxGainDf.numRows}")

    if (maxGainDf.numRows > 0) {
      println("Income distribution for capital.gain = 99999:")
      maxGainDf.groupby("income").count().show()
      println("Observation: All have income > 50K")
    }

    // 15. Describe capital.gain > 0 subset
    println(s"\n【15. Describe Capital.Gain > 0 Subset】")
    capitalGainDf.describe("age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week").show()

    // 16. Cross-tabulation analysis
    println(s"\n【16. Cross-tabulation Analysis】")

    // Occupation vs Income
    println("\n--- Occupation vs Income ---")
    val occIncome = adultDf.groupBy("occupation", "income").count()
    occIncome.show(20)

    // Workclass vs Income
    println("\n--- Workclass vs Income ---")
    val wcIncome = adultDf.groupBy("workclass", "income").count()
    wcIncome.show(10)

    // Education vs Income
    println("\n--- Education vs Income ---")
    val eduIncome = adultDf.groupBy("education", "income").count()
    eduIncome.show(15)

    // Marital Status vs Income
    println("\n--- Marital Status vs Income ---")
    val msIncome = adultDf.groupBy("marital.status", "income").count()
    msIncome.show(10)

    // Race vs Income
    println("\n--- Race vs Income ---")
    val raceIncome = adultDf.groupBy("race", "income").count()
    raceIncome.show(10)

    // Sex vs Income
    println("\n--- Sex vs Income ---")
    val sexIncome = adultDf.groupBy("sex", "income").count()
    sexIncome.show(10)

    // 17. Correlation analysis
    println(s"\n【17. Correlation Analysis】")
    val corrDf = adultDf.corr()
    corrDf.show()

    // 18. Filter data for age = 90 analysis
    println(s"\n【18. Age = 90 Analysis】")
    val age90Mask = new Column("temp",
      adultDf.columnData("age").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() == 90.0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    val age90Df = adultDf.filter(age90Mask)
    println(s"Number of observations with age = 90: ${age90Df.numRows}")

    if (age90Df.numRows > 0) {
      println("\nWorkclass distribution for age 90:")
      age90Df.groupby("workclass").count().show()
      println("\nOccupation distribution for age 90:")
      age90Df.groupby("occupation").count().show()
      println("\nIncome distribution for age 90:")
      age90Df.groupby("income").count().show()
    }

    // 19. Data Cleaning - Remove age = 90 outliers
    println(s"\n【19. Data Cleaning - Remove Age 90】")
    val beforeRows = adultDf.numRows
    val notAge90Mask = new Column("temp",
      adultDf.columnData("age").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() != 90.0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    adultDf = adultDf.filter(notAge90Mask)
    val afterRows = adultDf.numRows
    println(s"Rows before removing age 90: $beforeRows")
    println(s"Rows after removing age 90: $afterRows")
    println(s"Removed: ${beforeRows - afterRows} rows")

    // 20. Data Cleaning - Remove capital.gain = 99999 outliers
    println(s"\n【20. Data Cleaning - Remove Capital.Gain 99999】")
    val beforeRows2 = adultDf.numRows
    val notMaxGainMask = new Column("temp",
      adultDf.columnData("capital.gain").toSeq.map(v =>
        v.asInstanceOf[Number].doubleValue() != 99999.0).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    adultDf = adultDf.filter(notMaxGainMask)
    val afterRows2 = adultDf.numRows
    println(s"Rows before removing capital.gain = 99999: $beforeRows2")
    println(s"Rows after removing capital.gain = 99999: $afterRows2")
    println(s"Removed: ${beforeRows2 - afterRows2} rows")

    // 21. Final dataset shape
    println(s"\n【21. Final Dataset Shape】")
    println(s"Final number of observations: ${adultDf.numRows}")
    println(s"Final number of columns: ${adultDf.numCols}")

    // 22. Final column lists
    println(s"\n【22. Final Column Types】")
    val finalNumCols = adultDf.columns.filter(c => adultDf.columnData(c).dtype != DataType.String)
    val finalCatCols = adultDf.columns.filter(c => adultDf.columnData(c).dtype == DataType.String)
    println(s"Numerical columns (${finalNumCols.size}): ${finalNumCols.mkString(", ")}")
    println(s"Categorical columns (${finalCatCols.size}): ${finalCatCols.mkString(", ")}")

    // 23. Summary statistics
    println(s"\n【23. Summary Statistics】")
    println("\nSum:")
    adultDf.sum().foreach { case (col, v) => println(s"  $col: $v") }
    println("\nMean:")
    adultDf.mean().foreach { case (col, v) => println(f"  $col: $v%.4f") }
    println("\nStd:")
    adultDf.std().foreach { case (col, v) => println(f"  $col: $v%.4f") }

    // 24. Group statistics by income
    println(s"\n【24. Group Statistics by Income】")
    val incomeGroups = adultDf.groupBy("income")
    println("\nMean by income group:")
    incomeGroups.mean().show()
    println("\nCount by income group:")
    incomeGroups.count().show()

    // 25. Sex-based income analysis
    println(s"\n【25. Sex-based Income Analysis】")
    val sexGroups = adultDf.groupBy("sex", "income").count()
    sexGroups.show()

    // Calculate percentages
    val sexTotals = adultDf.groupBy("sex").count().collect().map(r =>
      r("sex") -> r.getDouble("count")).toMap
    sexGroups.collect().foreach(r => {
      val sex = r("sex")
      val income = r("income")
      val count = r.getDouble("count")
      val total = sexTotals.getOrElse(sex, 1.0)
      val pct = count / total * 100
      println(f"$sex, $income: $count ($pct%.2f%%)")
    })

    println("\n" + "=" * 80)
    println("Analysis Complete!")
    println("=" * 80)
  }
}