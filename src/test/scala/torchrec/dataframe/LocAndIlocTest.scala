package torchrec.dataframe

import scala.collection.mutable

/**
 * Scala 3 DataFrame translation of loc_and_iloc Jupyter notebook
 * Original notebook: /home/muller/IdeaProjects/torch-rechub-scala/data/Datasets/loc_and_iloc.ipynb
 * Dataset: cars.csv (built-in test data)
 */
object LocAndIlocTest {

  def main(args: Array[String]): Unit = {
    println("=" * 80)
    println("loc and iloc Indexing - Scala DataFrame Translation")
    println("=" * 80)

    // Create sample cars dataset
    val df = createCarsDataFrame()

    // 1. Basic iloc[row_index]
    println(s"\n【1. iloc(3) - 4th row】")
    val row3 = df.iloc(3)
    println(s"YEAR: ${row3("YEAR")}")
    println(s"Make: ${row3("Make")}")
    println(s"Model: ${row3("Model")}")
    println(s"Size: ${row3("Size")}")

    // 2. Select single column by name
    println(s"\n【2. select column 'Model' - All rows, 3rd column (Model)】")
    val col2 = df.select("Model")
    col2.show(10)

    // 3. iloc with row range
    println(s"\n【3. iloc(10 until 15) - Rows 10 to 14】")
    df.iloc(10 until 15).show()

    // 4. iloc with row and column indices
    println(s"\n【4. iloc(Seq(10,11,12,13,14), Seq(3,4,5,6,7,8)) - Rows 10-14, Columns 3-8】")
    df.iloc(Seq(10,11,12,13,14), Seq(3,4,5,6,7,8)).show()

    // 5. iloc with specific row indices
    println(s"\n【5. iloc(Seq(2, 6, 10, 25, 28, 33)) - Specific rows】")
    df.iloc(Seq(2, 6, 10, 25, 28, 33)).show()

    // 6. iloc with specific row and column indices
    println(s"\n【6. iloc(Seq(2, 6, 10, 25, 28, 33), Seq(0, 1, 6, 10))】")
    df.iloc(Seq(2, 6, 10, 25, 28, 33), Seq(0, 1, 6, 10)).show()

    // 7. iloc with row list and column range
    println(s"\n【7. iloc(Seq(2, 6, 10, 25, 28, 33), Seq(1,2,3,4,5))】")
    df.iloc(Seq(2, 6, 10, 25, 28, 33), Seq(1,2,3,4,5)).show()

    // 8. First n rows using head
    println(s"\n【8. head(6) - First 6 rows】")
    df.head(6).show()

    // 9. head() - First n rows
    println(s"\n【9. head() - First 5 rows】")
    df.head(5).show()

    // 10. Set index to YEAR
    println(s"\n【10. set_index('YEAR')】")
    val dfIndexed = df.set_index("YEAR")
    println("Set YEAR as index")
    dfIndexed.head(5).show()

    // 11. loc with single index value (string-based indexing after set_index)
    println(s"\n【11. loc(\"2013\") - Rows where YEAR = 2013】")
    dfIndexed.loc("2013").show()

    // 12. loc with multiple index values
    println(s"\n【12. loc(Seq(\"2012\", \"2013\", \"2016\")) - Multiple index values】")
    dfIndexed.loc(Seq("2012", "2013", "2016")).show()

    // 13. loc with index range
    println(s"\n【14. loc(\"2012\", \"2016\") - Index range (rowStart, rowEnd)】")
    dfIndexed.loc("2012", "2016", "", "").show()

    // 14. loc with column range
    println(s"\n【15. loc(\"2012\", \"2014\", \"Make\", \"(kW)\") - Row range and column range】")
    dfIndexed.loc("2012", "2014", "Make", "(kW)").show()

    // 15. loc with specific column list
    println(s"\n【16. loc(\"2012\", \"2014\", Seq(\"Make\", \"Model\", \"TYPE\", \"(kW)\", \"(km)\"))】")
    dfIndexed.loc("2012", "Make", "(kW)").show()

    // 16. loc with specific indices and columns
    println(s"\n【17. loc(Seq(\"2012\", \"2014\", \"2016\"), Seq(\"Make\", \"Model\", \"TYPE\", \"(kW)\", \"(km)\"))】")
    dfIndexed.loc(Seq("2012", "2014", "2016")).show()

    // 17. Boolean filtering with filter
    println(s"\n【18. filter(Make == 'FORD') - Boolean mask filtering】")
    val fordMask = new Column("temp",
      df.columnData("Make").toSeq.map(_.toString == "FORD").to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    df.filter(fordMask).show()

    // 18. Direct boolean filtering using filter with predicate
    println(s"\n【19. filter(row => ...) - Direct boolean filtering】")
    val fordDf = df.filter(row => row.getString("Make") == "FORD")
    fordDf.show()

    // 19. Filter and select specific columns
    println(s"\n【20. Filter Make == 'FORD', select Make:'TYPE'】")
    fordDf.select("Make", "Model", "Size", "(kW)", "Unnamed: 5", "TYPE").show()

    // 20. Multiple conditions filtering
    println(s"\n【21. Make == 'TESLA' & Model == 'MODEL S 70D'】")
    val tesla70D = df.filter(row =>
      row.getString("Make") == "TESLA" && row.getString("Model") == "MODEL S 70D")
    tesla70D.show()

    // 21. Filter and select specific columns
    println(s"\n【22. Filter TESLA & MODEL S 70D, select ['Make','Model', 'Size', '(kW)']】")
    tesla70D.select("Make", "Model", "Size", "(kW)").show()

    // 22. Filter with apply-like function - Make length == 5
    println(s"\n【23. Filter - Make length == 5】")
    val len5Mask = new Column("temp",
      df.columnData("Make").toSeq.map(_.toString.length == 5).to(mutable.ArrayBuffer).asInstanceOf[mutable.ArrayBuffer[Any]],
      DataType.Boolean)
    df.filter(len5Mask).show()

    // 23. Filter Make length == 5, select columns
    println(s"\n【24. Filter Make length == 5, select Make:'TYPE'】")
    val len5Df = df.filter(row => row.getString("Make").length == 5)
    len5Df.select("Make", "Model", "Size", "(kW)", "Unnamed: 5", "TYPE").show()

    // 24. at - single cell access
    println(s"\n【25. at(row, col) - Single cell access】")
    println(s"df.at(0, 'Make') = ${df.at(0, "Make")}")
    println(s"df.at(0, 'Model') = ${df.at(0, "Model")}")

    // 25. iat - single cell by integer position
    println(s"\n【26. iat(row_index, col_index) - Single cell by position】")
    println(s"df.iat(0, 1) = ${df.iat(0, 1)}")  // Make column
    println(s"df.iat(0, 2) = ${df.iat(0, 2)}")  // Model column

    // 26. select specific columns
    println(s"\n【27. select() - Specific columns】")
    df.select("YEAR", "Make", "Model", "TYPE", "(km)").show(5)

    // 27. drop columns
    println(s"\n【28. drop() - Remove columns】")
    df.drop("Unnamed: 5", "RATING").show(5)

    // 28. limit - first n rows
    println(s"\n【29. limit(10) - First 10 rows】")
    df.limit(10).show()

    // 29. tail - last n rows
    println(s"\n【30. tail(10) - Last 10 rows】")
    df.tail(10).show()

    // 30. sample - random rows
    println(s"\n【31. sample(5) - Random 5 rows】")
    df.sample(n = Some(5)).show()

    println("\n" + "=" * 80)
    println("loc and iloc Indexing Examples Complete!")
    println("=" * 80)
  }

  /**
   * Create sample cars dataset similar to cars.csv
   */
  private def createCarsDataFrame(): DataFrame = {
    val data = mutable.Map[String, Column]()

    // Sample car data
    val carData = Seq(
      (2012, "MITSUBISHI", "i-MiEV", "SUBCOMPACT", 49, "A1", "B", 16.9, 21.4, 18.7, 1.9, 2.4, 2.1, 0, null, 100, 7),
      (2012, "NISSAN", "LEAF", "MID-SIZE", 80, "A1", "B", 19.3, 23.0, 21.1, 2.2, 2.6, 2.4, 0, null, 117, 7),
      (2013, "FORD", "FOCUS ELECTRIC", "COMPACT", 107, "A1", "B", 19.0, 21.1, 20.0, 2.1, 2.4, 2.2, 0, null, 122, 4),
      (2013, "MITSUBISHI", "i-MiEV", "SUBCOMPACT", 49, "A1", "B", 16.9, 21.4, 18.7, 1.9, 2.4, 2.1, 0, null, 100, 7),
      (2013, "NISSAN", "LEAF", "MID-SIZE", 80, "A1", "B", 19.3, 23.0, 21.1, 2.2, 2.6, 2.4, 0, null, 117, 7),
      (2013, "SMART", "FORTWO ELECTRIC DRIVE CABRIOLET", "TWO-SEATER", 35, "A1", "B", 17.2, 22.5, 19.6, 1.9, 2.5, 2.2, 0, null, 109, 8),
      (2013, "SMART", "FORTWO ELECTRIC DRIVE COUPE", "TWO-SEATER", 35, "A1", "B", 17.2, 22.5, 19.6, 1.9, 2.5, 2.2, 0, null, 109, 8),
      (2013, "TESLA", "MODEL S (40 kWh battery)", "FULL-SIZE", 270, "A1", "B", 22.4, 21.9, 22.2, 2.5, 2.5, 2.5, 0, null, 224, 6),
      (2013, "TESLA", "MODEL S (60 kWh battery)", "FULL-SIZE", 270, "A1", "B", 22.2, 21.7, 21.9, 2.5, 2.4, 2.5, 0, null, 335, 10),
      (2013, "TESLA", "MODEL S (85 kWh battery)", "FULL-SIZE", 270, "A1", "B", 23.8, 23.2, 23.6, 2.7, 2.6, 2.6, 0, null, 426, 12),
      (2013, "TESLA", "MODEL S PERFORMANCE", "FULL-SIZE", 310, "A1", "B", 23.9, 23.2, 23.6, 2.7, 2.6, 2.6, 0, null, 426, 12),
      (2014, "CHEVROLET", "SPARK EV", "SUBCOMPACT", 104, "A1", "B", 16.0, 19.6, 17.8, 1.8, 2.2, 2.0, 0, null, 131, 7),
      (2014, "FORD", "FOCUS ELECTRIC", "COMPACT", 107, "A1", "B", 19.0, 21.1, 20.0, 2.1, 2.4, 2.2, 0, null, 122, 4),
      (2014, "MITSUBISHI", "i-MiEV", "SUBCOMPACT", 49, "A1", "B", 16.9, 21.4, 18.7, 1.9, 2.4, 2.1, 0, null, 100, 7),
      (2014, "NISSAN", "LEAF", "MID-SIZE", 80, "A1", "B", 16.5, 20.8, 18.4, 1.9, 2.3, 2.1, 0, null, 135, 5),
      (2014, "SMART", "FORTWO ELECTRIC DRIVE CABRIOLET", "TWO-SEATER", 35, "A1", "B", 17.2, 22.5, 19.6, 1.9, 2.5, 2.2, 0, null, 109, 8),
      (2014, "SMART", "FORTWO ELECTRIC DRIVE COUPE", "TWO-SEATER", 35, "A1", "B", 17.2, 22.5, 19.6, 1.9, 2.5, 2.2, 0, null, 109, 8),
      (2014, "TESLA", "MODEL S (60 kWh battery)", "FULL-SIZE", 225, "A1", "B", 22.2, 21.7, 21.9, 2.5, 2.4, 2.5, 0, null, 335, 10),
      (2014, "TESLA", "MODEL S (85 kWh battery)", "FULL-SIZE", 270, "A1", "B", 23.8, 23.2, 23.6, 2.7, 2.6, 2.6, 0, null, 426, 12),
      (2014, "TESLA", "MODEL S PERFORMANCE", "FULL-SIZE", 310, "A1", "B", 23.9, 23.2, 23.6, 2.7, 2.6, 2.6, 0, null, 426, 12),
      (2015, "NISSAN", "LEAF", "MID-SIZE", 80, "A1", "B", 16.5, 20.8, 18.4, 1.9, 2.3, 2.1, 0, null, 135, 5),
      (2015, "TESLA", "MODEL S (60 kWh battery)", "FULL-SIZE", 283, "A1", "B", 22.2, 21.7, 21.9, 2.5, 2.4, 2.5, 0, null, 335, 10),
      (2015, "TESLA", "MODEL S (70 kWh battery)", "FULL-SIZE", 283, "A1", "B", 20.8, 20.6, 20.7, 2.3, 2.3, 2.3, 0, null, 386, 12),
      (2015, "TESLA", "MODEL S 70D", "FULL-SIZE", 280, "A1", "B", 20.8, 20.6, 20.7, 2.3, 2.3, 2.3, 0, null, 386, 12),
      (2015, "TESLA", "MODEL S 85D/90D", "FULL-SIZE", 280, "A1", "B", 22.0, 19.8, 21.0, 2.5, 2.2, 2.4, 0, null, 435, 12),
      (2015, "TESLA", "MODEL S P85D/P90D", "FULL-SIZE", 515, "A1", "B", 23.4, 21.5, 22.5, 2.6, 2.4, 2.5, 0, null, 407, 12),
      (2015, "TESLA", "MODEL X 90D", "SUV - STANDARD", 386, "A1", "B", 20.8, 19.7, 20.3, 2.3, 2.2, 2.3, 0, null, 414, 12),
      (2015, "TESLA", "MODEL X P90D", "SUV - STANDARD", 568, "A1", "B", 23.2, 22.2, 22.7, 2.6, 2.5, 2.6, 0, null, 402, 12),
      (2016, "BMW", "i3", "SUBCOMPACT", 125, "A1", "B", 15.2, 18.8, 16.8, 1.7, 2.1, 1.9, 0, 10.0, 130, 4),
      (2016, "CHEVROLET", "SPARK EV", "SUBCOMPACT", 104, "A1", "B", 16.0, 19.6, 17.8, 1.8, 2.2, 2.0, 0, 10.0, 131, 7),
      (2016, "FORD", "FOCUS ELECTRIC", "COMPACT", 107, "A1", "B", 19.0, 21.1, 20.0, 2.1, 2.4, 2.2, 0, 10.0, 122, 4),
      (2016, "KIA", "SOUL EV", "STATION WAGON - SMALL", 81, "A1", "B", 17.5, 22.7, 19.9, 2.0, 2.6, 2.2, 0, 10.0, 149, 4),
      (2016, "MITSUBISHI", "i-MiEV", "SUBCOMPACT", 49, "A1", "B", 16.9, 21.4, 18.7, 1.9, 2.4, 2.1, 0, 10.0, 100, 7),
      (2016, "NISSAN", "LEAF (24 kWh battery)", "MID-SIZE", 80, "A1", "B", 16.5, 20.8, 18.4, 1.9, 2.3, 2.1, 0, 10.0, 135, 5),
      (2016, "NISSAN", "LEAF (30 kWh battery)", "MID-SIZE", 80, "A1", "B", 17.0, 20.7, 18.6, 1.9, 2.3, 2.1, 0, 10.0, 172, 6),
      (2016, "SMART", "FORTWO ELECTRIC DRIVE CABRIOLET", "TWO-SEATER", 35, "A1", "B", 17.2, 22.5, 19.6, 1.9, 2.5, 2.2, 0, 10.0, 109, 8),
      (2016, "SMART", "FORTWO ELECTRIC DRIVE COUPE", "TWO-SEATER", 35, "A1", "B", 17.2, 22.5, 19.6, 1.9, 2.5, 2.2, 0, 10.0, 109, 8),
      (2016, "TESLA", "MODEL S (60 kWh battery)", "FULL-SIZE", 283, "A1", "B", 22.2, 21.7, 21.9, 2.5, 2.4, 2.5, 0, 10.0, 335, 10),
      (2016, "TESLA", "MODEL S (70 kWh battery)", "FULL-SIZE", 283, "A1", "B", 20.8, 20.6, 20.7, 2.3, 2.3, 2.3, 0, 10.0, 377, 12),
      (2016, "TESLA", "MODEL S (85/90 kWh battery)", "FULL-SIZE", 283, "A1", "B", 23.8, 23.2, 23.6, 2.7, 2.6, 2.6, 0, 10.0, 426, 12),
      (2016, "TESLA", "MODEL S 70D", "FULL-SIZE", 386, "A1", "B", 20.8, 20.6, 20.7, 2.3, 2.3, 2.3, 0, 10.0, 386, 12),
      (2016, "TESLA", "MODEL S 85D/90D", "FULL-SIZE", 386, "A1", "B", 22.0, 19.8, 21.0, 2.5, 2.2, 2.4, 0, 10.0, 435, 12),
      (2016, "TESLA", "MODEL S 90D (Refresh)", "FULL-SIZE", 386, "A1", "B", 20.8, 19.7, 20.3, 2.3, 2.2, 2.3, 0, 10.0, 473, 12),
      (2016, "TESLA", "MODEL S P85D/P90D", "FULL-SIZE", 568, "A1", "B", 23.4, 21.5, 22.5, 2.6, 2.4, 2.5, 0, 10.0, 407, 12),
      (2016, "TESLA", "MODEL S P90D (Refresh)", "FULL-SIZE", 568, "A1", "B", 22.9, 21.0, 22.1, 2.6, 2.4, 2.5, 0, 10.0, 435, 12),
      (2016, "TESLA", "MODEL X 90D", "SUV - STANDARD", 386, "A1", "B", 23.2, 22.2, 22.7, 2.6, 2.5, 2.6, 0, 10.0, 414, 12),
      (2016, "TESLA", "MODEL X P90D", "SUV - STANDARD", 568, "A1", "B", 23.6, 23.3, 23.5, 2.7, 2.6, 2.6, 0, 10.0, 402, 12)
    )

    val years = mutable.ArrayBuffer[Any]()
    val makes = mutable.ArrayBuffer[Any]()
    val models = mutable.ArrayBuffer[Any]()
    val sizes = mutable.ArrayBuffer[Any]()
    val kw = mutable.ArrayBuffer[Any]()
    val unnamed = mutable.ArrayBuffer[Any]()
    val types = mutable.ArrayBuffer[Any]()
    val cityKwh = mutable.ArrayBuffer[Any]()
    val hwyKwh = mutable.ArrayBuffer[Any]()
    val combKwh = mutable.ArrayBuffer[Any]()
    val cityLe = mutable.ArrayBuffer[Any]()
    val hwyLe = mutable.ArrayBuffer[Any]()
    val combLe = mutable.ArrayBuffer[Any]()
    val gkm = mutable.ArrayBuffer[Any]()
    val rating = mutable.ArrayBuffer[Any]()
    val km = mutable.ArrayBuffer[Any]()
    val time = mutable.ArrayBuffer[Any]()

    for (car <- carData) {
      years += car._1
      makes += car._2
      models += car._3
      sizes += car._4
      kw += car._5
      unnamed += car._6
      types += car._7
      cityKwh += car._8
      hwyKwh += car._9
      combKwh += car._10
      cityLe += car._11
      hwyLe += car._12
      combLe += car._13
      gkm += car._14
      rating += car._15
      km += car._16
      time += car._17
    }

    data("YEAR") = new Column("YEAR", years.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
    data("Make") = new Column("Make", makes.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
    data("Model") = new Column("Model", models.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
    data("Size") = new Column("Size", sizes.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
    data("(kW)") = new Column("(kW)", kw.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
    data("Unnamed: 5") = new Column("Unnamed: 5", unnamed.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
    data("TYPE") = new Column("TYPE", types.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.String)
    data("CITY (kWh/100 km)") = new Column("CITY (kWh/100 km)", cityKwh.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("HWY (kWh/100 km)") = new Column("HWY (kWh/100 km)", hwyKwh.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("COMB (kWh/100 km)") = new Column("COMB (kWh/100 km)", combKwh.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("CITY (Le/100 km)") = new Column("CITY (Le/100 km)", cityLe.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("HWY (Le/100 km)") = new Column("HWY (Le/100 km)", hwyLe.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("COMB (Le/100 km)") = new Column("COMB (Le/100 km)", combLe.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("(g/km)") = new Column("(g/km)", gkm.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
    data("RATING") = new Column("RATING", rating.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Float64)
    data("(km)") = new Column("(km)", km.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)
    data("TIME (h)") = new Column("TIME (h)", time.asInstanceOf[mutable.ArrayBuffer[Any]], DataType.Int32)

    new DataFrame(data.toMap)
  }
}