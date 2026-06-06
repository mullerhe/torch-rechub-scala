package torchrec.data

import java.io._
import java.util.zip.{ZipInputStream, GZIPInputStream}

/**
 * HTTP downloader with disk caching and progress reporting.
 * Handles zip/gzip extraction automatically.
 */
object DatasetDownloader {

  private val cacheDir: File = {
    val home = System.getProperty("user.home")
    val dir = new File(home, ".torchrec-datasets")
    if (!dir.exists()) dir.mkdirs()
    dir
  }

  /**
   * Download a file from URL, cache locally, optionally extract.
   * @param urlString Source URL
   * @param name Human-readable dataset name (used for cache filename)
   * @param forceRedownload Override cached file
   * @return Path to downloaded/extracted file or directory
   */
  def download(
    urlString: String,
    name: String,
    forceRedownload: Boolean = false
  ): File = {
    val isZip = urlString.endsWith(".zip")
    val isGz = urlString.endsWith(".gz") || urlString.endsWith(".gzip")

    val cachedFile = new File(cacheDir, name)
    val targetFile = if (isZip) new File(cacheDir, s"$name.zip")
                     else if (isGz) new File(cacheDir, s"$name.txt")
                     else new File(cacheDir, name)

    // Check if already cached: either as a file (length > 1000) or as a directory
    val alreadyCached = if (!forceRedownload) {
      if (targetFile.isDirectory) {
        true  // Extracted directory exists
      } else {
        targetFile.exists() && targetFile.length() > 1000
      }
    } else {
      false
    }

    if (alreadyCached) {
      println(s"  [Cache] Using cached: ${targetFile.getAbsolutePath}")
      return targetFile
    }

    println(s"  [Download] $urlString")
    println(s"  [Save to] ${targetFile.getAbsolutePath}")

    val tempFile = new File(cacheDir, s"$name.tmp")
    if (tempFile.exists()) tempFile.delete()

    // Launch curl with --progress-bar, stream to /dev/tty for real-time progress
    val ttyOpt = try { Some(new java.io.FileOutputStream("/dev/tty")) } catch { case _ => None }
    val outSink = ttyOpt.getOrElse(System.out)
    val errSink = ttyOpt.getOrElse(System.err)
    val pio = new scala.sys.process.ProcessIO(
      in => (),
      out => scala.io.Source.fromInputStream(out).foreach { c =>  outSink.flush() },
      err => scala.io.Source.fromInputStream(err).foreach { c => errSink.flush() }
    )
    val proc = scala.sys.process.Process(
      Seq("curl", "-L",
          "--progress-bar",
          "--connect-timeout", "60",
          "--max-time", "1800",
          "--retry", "2",
          "-o", tempFile.getAbsolutePath,
          urlString)
    ).run(pio)

    val exitCode = proc.exitValue()
    if (exitCode != 0 || !tempFile.exists() || tempFile.length() < 1000) {
      throw new RuntimeException(s"curl failed with exit code $exitCode for $urlString")
    }

    val downloadedSize = tempFile.length()
    println(s"  [Done] ${formatSize(downloadedSize)} downloaded")

    // Atomically rename temp file
    if (targetFile.exists()) targetFile.delete()
    tempFile.renameTo(targetFile)

    // Extract if needed
    if (isZip) {
      println(s"  [Extract] $name.zip")
      extractZip(targetFile, cacheDir)
    } else if (isGz) {
      println(s"  [Extract] gzip to .txt")
      extractGzip(targetFile, cacheDir)
    } else {
      targetFile
    }
  }

  /** Try multiple URLs in sequence, return first successful download */
  def tryMirrors(urls: List[String], name: String): File = {
    for (url <- urls) {
      try {
        println(s"  [Try] $url")
        val file = download(url, name, forceRedownload = false)
        if (file != null && file.exists() && file.length() > 1000) {
          return file
        }
      } catch {
        case e: Throwable =>
          println(s"  [Fail] ${e.getMessage}")
      }
    }
    throw new RuntimeException(s"All mirrors failed for $name")
  }

  private def extractZip(zipFile: File, destDir: File): File = {
    val extractor = new ZipExtractor
    extractor.extract(zipFile, destDir)
  }

  private def extractGzip(gzFile: File, destDir: File): File = {
    var fis: FileInputStream = null.asInstanceOf[FileInputStream]
    var gis: GZIPInputStream = null.asInstanceOf[GZIPInputStream]
    var fos: FileOutputStream = null.asInstanceOf[FileOutputStream]
    var bos: BufferedOutputStream = null.asInstanceOf[BufferedOutputStream]

    try {
      val outName = gzFile.getName.replaceAll("\\.gz$", "").replaceAll("\\.gzip$", "")
      val outFile = new File(destDir, outName)
      if (outFile.exists()) return outFile

      fis = new FileInputStream(gzFile)
      gis = new GZIPInputStream(fis)
      fos = new FileOutputStream(outFile)
      bos = new BufferedOutputStream(fos)

      val buffer = new Array[Byte](8192)
      var read: Int = 0
      while ({ read = gis.read(buffer); read != -1 }) {
        bos.write(buffer, 0, read)
      }
      bos.flush()
      outFile
    } finally {
      try { if (bos != null) bos.close() } catch { case _: Throwable => }
      try { if (fos != null) fos.close() } catch { case _: Throwable => }
      try { if (gis != null) gis.close() } catch { case _: Throwable => }
      try { if (fis != null) fis.close() } catch { case _: Throwable => }
    }
  }

  private def formatSize(bytes: Long): String = {
    if (bytes < 1024) s"${bytes}B"
    else if (bytes < 1024 * 1024) s"${bytes / 1024}KB"
    else if (bytes < 1024 * 1024 * 1024) s"${bytes / (1024 * 1024)}MB"
    else s"${bytes / (1024 * 1024 * 1024)}GB"
  }

  /**
   * Get the cache directory path
   */
  def cachePath: String = cacheDir.getAbsolutePath

  private class ZipExtractor {
    def extract(zipFile: File, destDir: File): File = {
      var zis: ZipInputStream = null.asInstanceOf[ZipInputStream]
      var extractedFile: File = null.asInstanceOf[File]

      try {
        zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile)))
        var entry = zis.getNextEntry

        while (entry != null) {
          if (!entry.isDirectory && !entry.getName.contains("__MACOSX")) {
            val outFile = new File(destDir, entry.getName)
            // Create parent directories
            outFile.getParentFile.mkdirs()

            val fos = new FileOutputStream(outFile)
            val bos = new BufferedOutputStream(fos)
            val buffer = new Array[Byte](8192)
            var read = 0
            while ({ read = zis.read(buffer); read != -1 }) {
              bos.write(buffer, 0, read)
            }
            bos.close()

            if (extractedFile == null) extractedFile = outFile
          }
          zis.closeEntry()
          entry = zis.getNextEntry
        }

        // If zip contained a single directory, return that directory
        if (extractedFile == null && zipFile.exists) {
          // Look at zip contents
          val zip = new java.util.zip.ZipFile(zipFile)
          val entries = zip.entries()
          while (entries.hasMoreElements) {
            val e = entries.nextElement()
            if (!e.isDirectory && !e.getName.contains("__MACOSX")) {
              val f = new File(destDir, e.getName)
              if (f.exists) { extractedFile = f; if (!f.isDirectory) {} }
            }
          }
          zip.close()
        }

        // Return the directory containing the extracted files, or the first file
        if (extractedFile != null) extractedFile
        else destDir
      } finally {
        try { if (zis != null) zis.close() } catch { case _: Throwable => }
      }
    }
  }

  /**
   * Parse a CSV/TSV file lazily (one line at a time) to avoid OOM on large files.
   */
  def readLines(
    file: File,
    delimiter: String = "\t",
    skipHeader: Boolean = false,
    maxLines: Long = Long.MaxValue
  ): Iterator[Array[String]] = new Iterator[Array[String]] {
    private val reader = new BufferedReader(new FileReader(file))
    private var nextLine: Array[String] = null.asInstanceOf[Array[String]]
    private var linesRead = 0L

    if (skipHeader) reader.readLine()

    private def fetchNext(): Unit = {
      if (linesRead >= maxLines) { nextLine = null.asInstanceOf[Array[String]]; return }
      val line = reader.readLine()
      if (line == null) { nextLine = null.asInstanceOf[Array[String]]; return }
      linesRead += 1
      nextLine = line.split(delimiter, -1)
    }
    fetchNext()

    def hasNext: Boolean = nextLine != null
    def next(): Array[String] = {
      if (nextLine == null) throw new NoSuchElementException()
      val cur = nextLine
      fetchNext()
      cur
    }

    override def finalize(): Unit = {
      try { reader.close() } catch { case _: Throwable => }
    }
  }
}
