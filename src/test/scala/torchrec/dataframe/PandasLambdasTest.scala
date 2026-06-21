package torchrec.dataframe

import org.scalatest.funsuite.AnyFunSuite
import java.time.LocalDate
import java.io.File

class PandasCompliantTest extends AnyFunSuite {

  // Initialize df as per first 100 examples
  val df = {
    val colMap = collection.immutable.ListMap(
      "num1" -> Column("num1", Seq(1.0, 2.0, 3.0, 4.0, 5.0, Double.NaN, 6.0, 7.0, 8.0, 9.0, 10.0), DataType.Float64),
      "num2" -> Column("num2", Seq(10.0, 9.0, 8.0, 7.0, 6.0, 5.0, Double.NaN, 3.0, 2.0, 1.0, 0.0), DataType.Float64),
      "text" -> Column("text", Seq("apple", "banana", "", "orange", null, "grape", "pear", "melon", "peach", "plum", "date"), DataType.String),
      "score" -> Column("score", Seq(85.0, 92.0, 59.0, 77.0, 44.0, 96.0, 63.0, 71.0, 55.0, 88.0, 100.0), DataType.Float64),
      "gender" -> Column("gender", Seq("M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M"), DataType.String),
      "birth" -> Column("birth", Seq("2000-01-05", "1999-08-12", "2001-03-22", "1998-11-09", "2002-07-17", "1997-04-30", "2003-02-14", "1996-09-25", "2004-05-06", "1995-12-03", "2000-10-19").map(LocalDate.parse), DataType.Date)
    )
    new DataFrame(colMap)
  }

  // Initialize df2 as per second 100 examples
  val df2 = {
    val colMap = collection.immutable.ListMap(
      "v1" -> Column("v1", Seq(12.0, -3.0, Double.NaN, 47.0, 29.0, -18.0, 0.0, 93.0, 55.0, 81.0, 66.0, -9.0, 73.0), DataType.Float64),
      "v2" -> Column("v2", Seq(77.0, 22.0, 91.0, Double.NaN, -5.0, 38.0, 62.0, 4.0, 88.0, 19.0, Double.NaN, 33.0, 51.0), DataType.Float64),
      "val" -> Column("val", Seq(82.3, 55.6, 91.2, 77.9, 33.1, 66.8, 99.0, 21.7, 44.4, 88.8, 11.2, 79.5, 63.6), DataType.Float64),
      "txt" -> Column("txt", Seq("A1b_9", "", null, "X7#kL2", "Z99-pp", "   s6G8  ", "007abc", "M@2026", "", "qwe_66", "T3-P9", null, "55JJ"), DataType.String),
      "flag" -> Column("flag", Seq(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1), DataType.Int32),
      "dt" -> Column("dt", (0 until 13).map(i => java.time.LocalDateTime.of(2023, 1, 1, 0, 0).plusHours(i * 23)), DataType.Timestamp)
    )
    new DataFrame(colMap)
  }

  test("First 100 Examples: 1-20 (Column Axis=0)") {
    // 1. res = df["num1"].apply(lambda x: x + 1)
    val res1 = df("num1").map(x => if (pd.isna(x)) null else x.asInstanceOf[Double] + 1)

    // 2. res = df["num1"].apply(lambda x: x ** 2)
    val res2 = df("num1").map(x => if (pd.isna(x)) null else math.pow(x.asInstanceOf[Double], 2))

    // 3. res = df["num1"].apply(lambda x: np.sqrt(x) if not pd.isna(x) else np.nan)
    val res3 = df("num1").map(x => if (pd.notna(x)) np.sqrt(x) else np.nan)

    // 4. res = df["num1"].apply(lambda x: -x if pd.notna(x) else 0)
    val res4 = df("num1").map(x => if (pd.notna(x)) -x.asInstanceOf[Double] else 0.0)

    // 5. res = df["num1"].apply(lambda x: x * 1.5)
    val res5 = df("num1").map(x => if (pd.isna(x)) null else x.asInstanceOf[Double] * 1.5)

    // 6. res = df["num1"].apply(lambda x: x // 2 if not pd.isna(x) else None)
    val res6 = df("num1").map(x => if (pd.notna(x)) math.floor(x.asInstanceOf[Double] / 2) else null)

    // 7. res = df["num1"].apply(lambda x: x % 3 if pd.notna(x) else np.nan)
    val res7 = df("num1").map(x => if (pd.notna(x)) x.asInstanceOf[Double] % 3 else np.nan)

    // 8. res = df["num1"].apply(lambda x: x % 2 == 0 if pd.notna(x) else False)
    val res8 = df("num1").map(x => if (pd.notna(x)) x.asInstanceOf[Double] % 2 == 0 else false)

    // 9. res = df["num1"].apply(lambda x: x > 5 if pd.notna(x) else False)
    val res9 = df("num1").map(x => if (pd.notna(x)) x.asInstanceOf[Double] > 5 else false)

    // 10. res = df["num1"].apply(lambda x: "大" if x > 5 else "小" if pd.notna(x) else "空值")
    val res10 = df("num1").map(x => if (pd.isna(x)) "空值" else if (x.asInstanceOf[Double] > 5) "大" else "小")

    // 11. res = df["num1"].apply(lambda x: f"数值:{x}" if pd.notna(x) else "无数据")
    val res11 = df("num1").map(x => if (pd.notna(x)) s"数值:$x" else "无数据")

    // 12. res = df["num1"].apply(lambda x: 0 if pd.isna(x) else x)
    val res12 = df("num1").map(x => if (pd.isna(x)) 0.0 else x)

    // 13. mean_val = df["num1"].mean()
    val mean_val = df("num1").stats().mean
    val res13 = df("num1").map(x => if (pd.isna(x)) mean_val else x)

    // 14. res = df["num1"].apply(lambda x: min(x, 8) if pd.notna(x) else np.nan)
    val res14 = df("num1").map(x => if (pd.notna(x)) math.min(x.asInstanceOf[Double], 8.0) else np.nan)

    // 15. res = df["num1"].apply(lambda x: max(x, 2) if pd.notna(x) else np.nan)
    val res15 = df("num1").map(x => if (pd.notna(x)) math.max(x.asInstanceOf[Double], 2.0) else np.nan)

    // 16. res = df["num1"].apply(lambda x: round(x, 2) if pd.notna(x) else np.nan)
    val res16 = df("num1").map(x => if (pd.notna(x)) BigDecimal(x.asInstanceOf[Double]).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble else np.nan)

    // 17. res = df["num2"].apply(lambda x: abs(x) if pd.notna(x) else np.nan)
    val res17 = df("num2").map(x => if (pd.notna(x)) math.abs(x.asInstanceOf[Double]) else np.nan)

    // 18. res = df["num1"].apply(lambda x: "低" if x<=3 else "中" if x<=7 else "高" if pd.notna(x) else "未知")
    val res18 = df("num1").map(x => if (pd.isna(x)) "未知" else if (x.asInstanceOf[Double] <= 3) "低" else if (x.asInstanceOf[Double] <= 7) "中" else "高")

    // 19. res = df["num1"].apply(lambda x: np.log(x) if x>0 and pd.notna(x) else np.nan)
    val res19 = df("num1").map(x => if (pd.notna(x) && x.asInstanceOf[Double] > 0) math.log(x.asInstanceOf[Double]) else np.nan)

    // 20. res = df["num1"].apply(lambda x: bin(int(x))[2:] if pd.notna(x) else "NaN")
    val res20 = df("num1").map(x => if (pd.notna(x)) x.asInstanceOf[Double].toInt.toBinaryString else "NaN")
  }

  test("First 100 Examples: 21-40 (Text Operations)") {
    // 21. res = df["text"].apply(lambda s: s.upper() if pd.notna(s) else "空")
    val res21 = df("text").map(s => if (pd.notna(s) && s != null) s.toString.toUpperCase else "空")

    // 22. res = df["text"].apply(lambda s: s.lower() if pd.notna(s) else "空")
    val res22 = df("text").map(s => if (pd.notna(s) && s != null) s.toString.toLowerCase else "空")

    // 23. res = df["text"].apply(lambda s: s.strip() if pd.notna(s) else "")
    val res23 = df("text").map(s => if (pd.notna(s) && s != null) s.toString.trim else "")

    // 24. res = df["text"].apply(lambda s: len(s) if pd.notna(s) else 0)
    val res24 = df("text").map(s => if (pd.notna(s) && s != null) s.toString.length else 0)

    // 25. res = df("text").apply(lambda s: "水果:" + s if pd.notna(s) else "未知水果")
    val res25 = df("text").map(s => if (pd.notna(s) && s != null) "水果:" + s.toString else "未知水果")

    // 26. res = df["text"].apply(lambda s: "缺失" if pd.isna(s) or s == "" else "正常")
    val res26 = df("text").map(s => if (pd.isna(s) || s.toString == "") "缺失" else "正常")

    // 27. res = df["text"].apply(lambda s: "有a" if pd.notna(s) and "a" in s else "无a")
    val res27 = df("text").map(s => if (pd.notna(s) && s.toString.contains("a")) "有a" else "无a")

    // 28. res = df["text"].apply(lambda s: s[:2] if pd.notna(s) else "")
    val res28 = df("text").map(s => if (pd.notna(s)) s.toString.take(2) else "")

    // 29. res = df["text"].apply(lambda s: s[::-1] if pd.notna(s) else "")
    val res29 = df("text").map(s => if (pd.notna(s)) s.toString.reverse else "")

    // 30. res = df["text"].apply(lambda s: "未知" if s == "" or pd.isna(s) else s)
    val res30 = df("text").map(s => if (s == null || s.toString == "") "未知" else s)

    // 31. res = df["text"].apply(lambda s: "-".join(list(s)) if pd.notna(s) else "")
    val res31 = df("text").map(s => if (pd.notna(s)) s.toString.mkString("-") else "")

    // 32. res = df["text"].apply(lambda s: True if pd.notna(s) and s.startswith("a") else False)
    val res32 = df("text").map(s => if (pd.notna(s) && s.toString.startsWith("a")) true else false)

    // 33. res = df["text"].apply(lambda s: True if pd.notna(s) and s.endswith("e") else False)
    val res33 = df("text").map(s => if (pd.notna(s) && s.toString.endsWith("e")) true else false)

    // 34. res = df["text"].apply(lambda s: s*2 if pd.notna(s) else "")
    val res34 = df("text").map(s => if (pd.notna(s)) s.toString * 2 else "")

    // 35. res = df["text"].apply(lambda s: s.replace("b","#") if pd.notna(s) else "")
    val res35 = df("text").map(s => if (pd.notna(s)) s.toString.replace("b", "#") else "")

    // 36. res = df["text"].apply(lambda s: s.capitalize() if pd.notna(s) else "")
    val res36 = df("text").map(s => if (pd.notna(s)) s.toString.capitalize else "")

    // 37. res = df["text"].apply(lambda s: s.title() if pd.notna(s) else "")
    val res37 = df("text").map(s => if (pd.notna(s)) s.toString.split(" ").map(_.capitalize).mkString(" ") else "")

    // 38. res = df["text"].apply(lambda s: "长文本" if pd.notna(s) and len(s)>4 else "短文本")
    val res38 = df("text").map(s => if (pd.notna(s) && s.toString.length > 4) "长文本" else "短文本")

    // 39. res = df["text"].apply(lambda s: "" if pd.isna(s) else s)
    val res39 = df("text").map(s => if (pd.isna(s)) "" else s)

    // 40. import hashlib; res = df["text"].apply(lambda s: hashlib.md5(s.encode()).hexdigest() if pd.notna(s) else None)
//    val res40 = df("text").str("text").md5 // Using my helper

  }

  test("First 100 Examples: 41-60 (Row Axis=1 Operations)") {
    // 41. df["sum_col"] = df.apply(lambda row: row["num1"] + row["num2"], axis=1)
    val df41 = df.withColumn("sum_col", Column("sum_col", df.applyPerRow(r => r.getDouble("num1") + r.getDouble("num2")), DataType.Float64))

    // 42. df["diff_col"] = df.apply(lambda row: row["num1"] - row["num2"], axis=1)
    val df42 = df.withColumn("diff_col", Column("diff_col", df.applyPerRow(r => r.getDouble("num1") - r.getDouble("num2")), DataType.Float64))

    // 43. 两数乘积
    val df43 = df.withColumn("mul_col", Column("mul_col", df.applyPerRow(r => r.getDouble("num1") * r.getDouble("num2")), DataType.Float64))

    // 44. 两数平均值
    val df44 = df.withColumn("mean_col", Column("mean_col", df.applyPerRow(r => (r.getDouble("num1") + r.getDouble("num2")) / 2.0), DataType.Float64))

    // 45. 取两列最大值
    val df45 = df.withColumn("max_col", Column("max_col", df.applyPerRow(r => math.max(r.getDouble("num1"), r.getDouble("num2"))), DataType.Float64))

    // 46. 取两列最小值
    val df46 = df.withColumn("min_col", Column("min_col", df.applyPerRow(r => math.min(r.getDouble("num1"), r.getDouble("num2"))), DataType.Float64))

    // 47. 判断 num1 是否大于 num2
    val df47 = df.withColumn("compare", Column("compare", df.applyPerRow(r => if (r.getDouble("num1") > r.getDouble("num2")) "num1大" else "num2大"), DataType.String))

    // 48. 分数分级
    val df48 = df.withColumn("level", Column("level", df.applyPerRow(r => {
      val s = r.getDouble("score")
      if (s >= 90) "优秀" else if (s >= 70) "良好" else "不及格"
    }), DataType.String))

    // 49. 性别 + 分数组合标签
    val df49 = df.withColumn("tag", Column("tag", df.applyPerRow(r => s"${r("gender")}_${r("score")}分"), DataType.String))

    // 50. 文本 + 数字拼接
    val df50 = df.withColumn("mix_text", Column("mix_text", df.applyPerRow(r => {
      val t = r("text"); val n = r("num1")
      if (pd.notna(t)) s"${t}_$n" else s"空_$n"
    }), DataType.String))

    // 51. 总分加权：num1*0.3 + score*0.7
    val df51 = df.withColumn("weight_score", Column("weight_score", df.applyPerRow(r => r.getDouble("num1") * 0.3 + r.getDouble("score") * 0.7), DataType.Float64))

    // 52. 空值判断：任意一列缺失标记异常
    val df52 = df.withColumn("is_err", Column("is_err", df.applyPerRow(r => pd.isna(r("num1")) || pd.isna(r("num2"))), DataType.Boolean))

    // 53. 三列最大值 num1,num2,score
    val df53 = df.withColumn("three_max", Column("three_max", df.applyPerRow(r => List(r.getDouble("num1"), r.getDouble("num2"), r.getDouble("score")).max), DataType.Float64))

    // 54. 分数修正：男性 + 5 分
    val df54 = df.withColumn("score_fix", Column("score_fix", df.applyPerRow(r => {
      val s = r.getDouble("score")
      if (r("gender") == "M") s + 5 else s
    }), DataType.Float64))

    // 55. 分数超过 90 且男性标记精英
    val df55 = df.withColumn("elite", Column("elite", df.applyPerRow(r => {
      if (r("gender") == "M" && r.getDouble("score") >= 90) "精英" else "普通"
    }), DataType.String))

    // 56. num1 偶数且分数 > 80 标记达标
    val df56 = df.withColumn("pass_flag", Column("pass_flag", df.applyPerRow(r => {
      if (r.getInt("num1") % 2 == 0 && r.getDouble("score") > 80) "达标" else "不达标"
    }), DataType.String))

    // 57. 两数差值绝对值
    val df57 = df.withColumn("abs_diff", Column("abs_diff", df.applyPerRow(r => {
      val n1 = r.getDouble("num1"); val n2 = r.getDouble("num2")
      if (pd.notna(n1) && pd.notna(n2)) math.abs(n1 - n2) else Double.NaN
    }), DataType.Float64))

    // 58. 文本为空且分数 < 60 标记作废
    val df58 = df.withColumn("invalid", Column("invalid", df.applyPerRow(r => {
      if ((pd.isna(r("text")) || r("text") == "") && r.getDouble("score") < 60) "作废" else "有效"
    }), DataType.String))

    // 59. 数字乘积取平方根
    val df59 = df.withColumn("sqrt_mul", Column("sqrt_mul", df.applyPerRow(r => {
      val n1 = r.getDouble("num1"); val n2 = r.getDouble("num2")
      if (pd.notna(n1) && pd.notna(n2)) math.sqrt(n1 * n2) else Double.NaN
    }), DataType.Float64))

    // 60. 输出多字符状态：高低分 + 男女
    val df60 = df.withColumn("status", Column("status", df.applyPerRow(r => {
      val g = r("gender"); val s = r.getDouble("score")
      if (g == "M" && s >= 80) "男高分" else if (g == "F" && s < 80) "女低分" else "其他"
    }), DataType.String))
  }

  test("First 100 Examples: 61-80 (Date and Nested Logic)") {
    // 61. 计算年龄 (2026 - 出生年份)
    val df61 = df.withColumn("age", Column("age", df.applyPerRow(r => 2026 - r("birth").asInstanceOf[LocalDate].getYear), DataType.Int32))

    // 62. 提取出生月份
    val df62 = df.withColumn("birth_month", Column("birth_month", df.applyPerRow(r => r("birth").asInstanceOf[LocalDate].getMonthValue), DataType.Int32))

    // 63. 判断是否上半年出生
    val df63 = df.withColumn("half_year", Column("half_year", df.applyPerRow(r => if (r("birth").asInstanceOf[LocalDate].getMonthValue <= 6) "上半年" else "下半年"), DataType.String))

    // 64. 生日完整字符串格式化
    val df64 = df.withColumn("birth_str", Column("birth_str", df.applyPerRow(r => r("birth").asInstanceOf[LocalDate].format(java.time.format.DateTimeFormatter.ofPattern("yyyy年MM月dd日"))), DataType.String))

    // 65. 判断是否当月 15 号之后出生
    val df65 = df.withColumn("late_birth", Column("late_birth", df.applyPerRow(r => r("birth").asInstanceOf[LocalDate].getDayOfMonth > 15), DataType.Boolean))

    // 66. 年份大于 2000 标记新生代
    val df66 = df.withColumn("gen", Column("gen", df.applyPerRow(r => if (r("birth").asInstanceOf[LocalDate].getYear > 2000) "新生代" else "90后"), DataType.String))

    // 67. 出生星期名称
    val df67 = df.withColumn("week_name", Column("week_name", df.applyPerRow(r => r("birth").asInstanceOf[LocalDate].getDayOfWeek.toString), DataType.String))

    // 68. 距离 2026-06-01 天数
    val base_dt = LocalDate.parse("2026-06-01")
    val df68 = df.withColumn("day_diff", Column("day_diff", df.applyPerRow(r => java.time.temporal.ChronoUnit.DAYS.between(r("birth").asInstanceOf[LocalDate], base_dt)), DataType.Int64))

    // 69. 季度标记
    val df69 = df.withColumn("quarter", Column("quarter", df.applyPerRow(r => s"Q${(r("birth").asInstanceOf[LocalDate].getMonthValue - 1) / 3 + 1}"), DataType.String))

    // 70. 年份 + 性别组合标签
    val df70 = df.withColumn("year_gender", Column("year_gender", df.applyPerRow(r => s"${r("birth").asInstanceOf[LocalDate].getYear}_${r("gender")}"), DataType.String))

    // 71. 分数多层评级
    val df71 = df.withColumn("rank_all", Column("rank_all", df.applyPerRow(r => {
      val s = r.getDouble("score")
      if (s >= 95) "S" else if (s >= 85) "A" else if (s >= 70) "B" else if (s >= 60) "C" else "D"
    }), DataType.String))

    // 72. 数值空值 + 性别双重判断
    val df72 = df.withColumn("complex_flag", Column("complex_flag", df.applyPerRow(r => {
      if (pd.isna(r("num1")) && r("gender") == "M") "男缺失" else if (pd.isna(r("num1")) && r("gender") == "F") "女缺失" else "正常"
    }), DataType.String))

    // 73. 文本长度 + 分数联合分级
    val df73 = df.withColumn("mix_level", Column("mix_level", df.applyPerRow(r => {
      if (r.getString("text").length > 3 && r.getDouble("score") > 85) "优长" else if (r.getString("text").length <= 3 && r.getDouble("score") > 70) "良短" else "普通"
    }), DataType.String))

    // 74. 三条件同时满足标记完美
    val df74 = df.withColumn("perfect", Column("perfect", df.applyPerRow(r => {
      if (r.getDouble("num1") > 5 && r.getDouble("score") > 90 && r("gender") == "F") "完美" else "一般"
    }), DataType.String))

    // 75. 差值区间分段
    val df75 = df.withColumn("diff_level", Column("diff_level", df.applyPerRow(r => {
      val d = math.abs(r.getDouble("num1") - r.getDouble("num2"))
      if (d <= 2) "近" else if (d <= 5) "中等差距" else "差距大"
    }), DataType.String))

    // 76. 年份 + 分数双层筛选
    val df76 = df.withColumn("special", Column("special", df.applyPerRow(r => {
      if (r("birth").asInstanceOf[LocalDate].getYear >= 2000 && r.getDouble("score") >= 90) "00后高分" else if (r("birth").asInstanceOf[LocalDate].getYear < 2000 && r.getDouble("score") < 70) "90后低分" else "普通学生"
    }), DataType.String))

    // 77. 空文本且偶数 num1 标记异常数据
    val df77 = df.withColumn("data_err", Column("data_err", df.applyPerRow(r => {
      if ((pd.isna(r("text")) || r("text") == "") && r.getInt("num1") % 2 == 0) "异常" else "正常数据"
    }), DataType.String))

    // 78. 加权分多层评级
    val df78 = df.withColumn("weight_rank", Column("weight_rank", df.applyPerRow(r => {
      val ws = r.getDouble("num1") * 0.3 + r.getDouble("score") * 0.7
      if (ws > 90) "五星" else if (ws > 75) "四星" else "三星"
    }), DataType.String))

    // 79. 多列同时非空校验
    val df79 = df.withColumn("full_data", Column("full_data", df.applyPerRow(r => {
      if (pd.notna(r("num1")) && pd.notna(r("num2")) && pd.notna(r("text"))) "完整" else "缺字段"
    }), DataType.String))

    // 80. 偶数 num1 + 下半年出生组合标签
    val df80 = df.withColumn("mix_tag2", Column("mix_tag2", df.applyPerRow(r => {
      if (r.getInt("num1") % 2 == 0 && r("birth").asInstanceOf[LocalDate].getMonthValue > 6) "偶下半年" else "其他"
    }), DataType.String))
  }

  test("First 100 Examples: 81-100 (Advanced)") {
    // 81. 分数取模 3 分类
    val df81 = df.withColumn("mod3", Column("mod3", df.applyPerRow(r => s"余${r.getDouble("score").toInt % 3}"), DataType.String))

    // 82. 文本包含 p 且分数高于 80
    val df82 = df.withColumn("text_score_match", Column("text_score_match", df.applyPerRow(r => if (r.getString("text").contains("p") && r.getDouble("score") > 80) "匹配" else "不匹配"), DataType.String))

    // 83. 两数之和大于 12 且女性
    val df83 = df.withColumn("sum_female", Column("sum_female", df.applyPerRow(r => if ((r.getDouble("num1") + r.getDouble("num2")) > 12 && r("gender") == "F") "达标女生" else "未达标"), DataType.String))

    // 84. 出生季度 Q1 且 num1 大于 4
    val df84 = df.withColumn("q1_high", Column("q1_high", df.applyPerRow(r => {
        val q = (r("birth").asInstanceOf[LocalDate].getMonthValue - 1) / 3 + 1
        if (q == 1 && r.getDouble("num1") > 4) "Q1高分num" else "其他季度"
    }), DataType.String))

    // 85. 综合多维度输出长描述字符串
    val df85 = df.withColumn("desc", Column("desc", df.applyPerRow(r => s"${r("gender")},${2026 - r("birth").asInstanceOf[LocalDate].getYear}岁,分数${r("score")}"), DataType.String))

    // 86. apply 一行返回多个值 (result_type="expand") -> Simulated by adding multiple columns
    val sums = df.applyPerRow(r => r.getDouble("num1") + r.getDouble("num2"))
    val diffs = df.applyPerRow(r => r.getDouble("num1") - r.getDouble("num2"))
    val df86 = df.withColumn("sum_v", Column("sum_v", sums, DataType.Float64)).withColumn("diff_v", Column("diff_v", diffs, DataType.Float64))

    // 87. lambda 内调用自定义函数
    val calc = (a: Double, b: Double) => math.pow(a, 2) + b * 3
    val df87 = df.withColumn("custom_calc", Column("custom_calc", df.applyPerRow(r => calc(r.getDouble("num1"), r.getDouble("num2"))), DataType.Float64))

    // 88. 行数据全部转为列表输出
    val df88 = df.withColumn("row_list", Column("row_list", df.applyPerRow(r => List(r("num1"), r("num2"), r("score"))), DataType.List))

    // 89. lambda 内三元表达式嵌套多层数学运算
    val df89 = df.withColumn("calc_nest", Column("calc_nest", df.applyPerRow(r => {
      if (r("gender") == "M") math.sqrt(r.getDouble("num1")) + r.getDouble("score") / 10 else r.getDouble("num2") * 0.8 - r.getDouble("num1")
    }), DataType.Float64))

    // 90. 过滤空行
    val mask = df.applyPerRow(r => pd.notna(r("text")) && r.getDouble("score") > 60)
    val df90 = df.filter(Column("mask", mask.map(_.asInstanceOf[Any]), DataType.Boolean))

    // 91. 数值标准化 lambda 实现 min-max 缩放
    val min_s = df("score").stats().min
    val max_s = df("score").stats().max
    val df91 = df.withColumn("score_norm", Column("score_norm", df.applyPerRow(r => (r.getDouble("score") - min_s) / (max_s - min_s)), DataType.Float64))

    // 92. 文本长度乘以分数复合指标
    val df92 = df.withColumn("text_score_idx", Column("text_score_idx", df.applyPerRow(r => r.getString("text").length * r.getDouble("score")), DataType.Float64))

    // 93. groupby 分组内使用 apply+lambda
    val group_res = df.groupby("gender").apply(g => g("score").map(x => x.asInstanceOf[Double] + 10).stats().mean)

    // 94. df.pipe 结合 apply 链式调用
    val res_pipe = df.pipe(d => d.withColumn("new_col", Column("new_col", d.applyPerRow(r => r.getDouble("num1") * 2), DataType.Float64)))

    // 95. 判断是否为质数简易 lambda
    val is_prime = (n: Int) => if (n > 1) (2 to math.sqrt(n).toInt).forall(n % _ != 0) else false
    
//    val df95 = df.withColumn("num1_prime", Column("num1_prime", df("num1").map(x => if (pd.notna(x)) is_prime(x.asInstanceOf[Double].toInt) else false), DataType.Boolean))
  }

  test("Second 100 Examples: Complex Mathematical Nested Logic") {
    // 101 (c1 in MD). df["c1"] = df.apply(lambda r: (np.tanh(np.log(abs(r.v1)+1)) * np.clip(r.val/10, 0.1, 9)) if pd.notna(r.v1) and pd.notna(r.val) else (np.sin(r.val/100)*5 if pd.notna(r.val) else -999), axis=1)
    val c1 = df2.applyPerRow(r => {
      val v1 = r.getDouble("v1"); val val_v = r.getDouble("val")
      if (pd.notna(v1) && pd.notna(val_v)) {
        np.tanh(np.log(math.abs(v1) + 1)) * np.clip(val_v / 10.0, 0.1, 9.0)
      } else if (pd.notna(val_v)) {
        math.sin(val_v / 100.0) * 5.0
      } else -999.0
    })
    val df_c1 = df2.withColumn("c1", Column("c1", c1, DataType.Float64))

    // 102 (c2 in MD). df["c2"] = df.apply(lambda r: (np.exp(abs(r.v2)/20) - np.power(r.v1/10,2)) if (r.v1>0 and pd.notna(r.v1) and pd.notna(r.v2)) else (np.sqrt(abs(r.v1)+abs(r.v2))*0.8 if pd.notna(r.v1) or pd.notna(r.v2) else 0), axis=1)
    val c2 = df2.applyPerRow(r => {
      val v1 = r.getDouble("v1"); val v2 = r.getDouble("v2")
      if (v1 > 0 && pd.notna(v1) && pd.notna(v2)) {
        math.exp(math.abs(v2) / 20.0) - math.pow(v1 / 10.0, 2)
      } else if (pd.notna(v1) || pd.notna(v2)) {
        math.sqrt(math.abs(v1) + math.abs(v2)) * 0.8
      } else 0.0
    })
    val df_c2 = df2.withColumn("c2", Column("c2", c2, DataType.Float64))

    // 103 (c3 in MD). 分段加权归一化
    val c3 = df2.applyPerRow(r => {
        val v1 = r.getDouble("v1"); val v2 = r.getDouble("v2"); val val_v = r.getDouble("val")
        if (v1 > 10 && v2 < 80 && pd.notna(v1) && pd.notna(v2)) {
            (v1 * 0.7 + v2 * 0.3) / math.max(math.abs(v1), math.abs(v2))
        } else if (v1 < -5) {
            (math.pow(v1, 1.2) - math.pow(v2, 0.9)) / 2.0
        } else val_v / 10.0
    })
    val df_c3 = df2.withColumn("c3", Column("c3", c3, DataType.Float64))

    // 104 (c4 in MD). 三角函数嵌套
    val c4 = df2.applyPerRow(r => {
        val v1 = r.getDouble("v1"); val val_v = r.getDouble("val"); val v2 = r.getDouble("v2")
        if (pd.notna(v1) && val_v > 50) math.cos(val_v/15.0) * math.sinh(math.abs(v1)/30.0)
        else if (pd.notna(v2)) math.tan(math.abs(v2+1)/40.0)
        else 0.001
    })
    val df_c4 = df2.withColumn("c4", Column("c4", c4, DataType.Float64))

    // 105 (c8 in MD). 动态阈值区间多级分流 (5层三元)
    val c8 = df2.applyPerRow(r => {
        val val_v = r.getDouble("val")
        if (val_v > 90) 9.9 else if (val_v > 70) 7.7 else if (val_v > 40) 5.5 else if (val_v > 20) 2.2 else 0.1
    })
    val df_c8 = df2.withColumn("c8", Column("c8", c8, DataType.Float64))

    // 106 (c12 in MD). 指数衰减模拟
    val c12 = df2.applyPerRow(r => {
        val v1 = r.getDouble("v1"); val v2 = r.getDouble("v2"); val val_v = r.getDouble("val")
        if (math.abs(v1) < 60) math.exp(-math.abs(v1)/50.0) * val_v else math.exp(-math.abs(v2)/30.0) * val_v * 0.8
    })
    val df_c12 = df2.withColumn("c12", Column("c12", c12, DataType.Float64))
  }

  test("Second 100 Examples: Text and Regex") {
    import java.util.regex.Pattern
    // 121 (c21 in MD). 正则提取数字 + 字母校验
    val c21 = df2.applyPerRow(r => {
        val txt = r.getString("txt")
        if (pd.notna(txt) && txt.trim.nonEmpty) {
            val digits = txt.count(_.isDigit)
            val letters = txt.count(_.isLetter)
            digits * 10 + letters * 3
        } else if (txt == "") -1 else -2
    })
    val df_c21 = df2.withColumn("c21", Column("c21", c21, DataType.Int32))

    // 122 (c22 in MD). 文本合规性判定
    val c22 = df2.applyPerRow(r => {
        val txt = r.getString("txt").trim
        if (txt.length >= 4 && txt.exists(_.isUpper) && txt.exists(_.isDigit)) "强合法"
        else if (txt.nonEmpty) "弱合法"
        else if (r("txt") == "") "空文本"
        else "无效缺失"
    })
    val df_c22 = df2.withColumn("c22", Column("c22", c22, DataType.String))

    // 129 (c29 in MD). 纯文本 / 纯数字 / 混合
    val c29 = df2.applyPerRow(r => {
        val txt = r.getString("txt").trim
        if (txt.forall(_.isDigit) && txt.nonEmpty) "纯数字"
        else if (txt.forall(_.isLetter) && txt.nonEmpty) "纯字母"
        else if (txt.nonEmpty) "混合文本"
        else "空值"
    })
    val df_c29 = df2.withColumn("c29", Column("c29", c29, DataType.String))
  }

  test("Second 100 Examples: Timestamp Operations") {
    // 141 (c41 in MD). 年月日时分秒复合
    val c41 = df2.applyPerRow(r => {
        val dt = r("dt").asInstanceOf[java.time.LocalDateTime]
        if (pd.notna(dt)) dt.getYear * 1000 + dt.getMonthValue * 100 + dt.getDayOfMonth * 10 + dt.getHour else 0
    })
    val df_c41 = df2.withColumn("c41", Column("c41", c41, DataType.Int32))

    // 142 (c42 in MD). 昼夜分段
    val c42 = df2.applyPerRow(r => {
        val h = r("dt").asInstanceOf[java.time.LocalDateTime].getHour
        if (h < 6) "凌晨" else if (h < 12) "上午" else if (h < 18) "下午" else "夜间"
    })
    val df_c42 = df2.withColumn("c42", Column("c42", c42, DataType.String))

    // 149 (c49 in MD). 周末 / 工作日
    val c49 = df2.applyPerRow(r => {
        val dow = r("dt").asInstanceOf[java.time.LocalDateTime].getDayOfWeek.getValue // 1-7
        if (dow >= 6) "周末" else "工作日"
    })
    val df_c49 = df2.withColumn("c49", Column("c49", c49, DataType.String))
  }

  test("Final Complex Examples: 191-200") {
    // 191 (c91 in MD). 六维特征融合打分
    val c91 = df2.applyPerRow(r => {
        val v1 = r.getDouble("v1"); val v2 = r.getDouble("v2"); val val_v = r.getDouble("val")
        val txt = r.getString("txt"); val flag = r.getInt("flag"); val dt = r("dt").asInstanceOf[java.time.LocalDateTime]
        if (pd.notna(v1) && pd.notna(v2)) {
            (math.tanh(math.abs(v1)/40.0)*0.25) +
            (math.sinh(math.abs(v2)/60.0)*0.2) +
            (val_v/100.0*0.3) +
            (txt.trim.length/10.0*0.1) +
            (flag*0.05) +
            (java.time.temporal.ChronoUnit.DAYS.between(dt, java.time.LocalDateTime.now())/365.0*0.1)
        } else val_v / 100.0
    })
    val df_c91 = df2.withColumn("c91", Column("c91", c91, DataType.Float64))

    // 200 (c100 in MD). 终极综合超级复杂全维度融合模型
    val c100 = df2.applyPerRow(r => {
        val v1 = r.getDouble("v1"); val v2 = r.getDouble("v2"); val val_v = r.getDouble("val")
        val txt = r.getString("txt"); val flag = r.getInt("flag"); val dt = r("dt").asInstanceOf[java.time.LocalDateTime]
        if (pd.notna(v1) && pd.notna(v2)) {
            val score = (math.exp(math.abs(v1)/100.0)*0.15) +
                        (math.log1p(math.abs(v2))*0.2) +
                        (math.pow(val_v/100.0, 1.2)*0.35) +
                        (txt.length/20.0*0.1) +
                        (1.0 - java.time.temporal.ChronoUnit.DAYS.between(dt, java.time.LocalDateTime.now())/365.0)*0.15 +
                        (flag*0.05)
            math.max(0.0, math.min(9.99, score))
        } else 1.11
    })
    val df_c100 = df2.withColumn("c100", Column("c100", c100, DataType.Float64))
  }
}



