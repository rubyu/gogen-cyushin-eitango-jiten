
import java.awt.image.BufferedImage
import java.nio.{FloatBuffer, ByteBuffer}
import java.io._
import javax.imageio.ImageIO
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.indexer.UByteIndexer
import org.bytedeco.javacv.{OpenCVFrameConverter, Java2DFrameConverter, CanvasFrame}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_objdetect._

import scala.collection.{mutable, immutable}

object Main {

  trait PageType
  object PageTypeA extends PageType
  object PageTypeB extends PageType

  val debugOn = false
  val guiOn = false
  val outputOn = false

  val BLACK = cvScalar(0, 0, 0, 0)
  val BLUE = cvScalar(255, 0, 0, 0)
  val GREEN = cvScalar(0, 255, 0, 0)
  val RED = cvScalar(0, 0, 255, 0)
  val YELLOW = cvScalar(0, 255, 255, 0)

  val FONT = cvFont(2, 2)
  cvInitFont(FONT, CV_FONT_HERSHEY_TRIPLEX, 2, 2, 1, 2, CV_AA)

  val converter1 = new OpenCVFrameConverter.ToMat()
  val converter2 = new OpenCVFrameConverter.ToIplImage()
  val converter3 = new Java2DFrameConverter()

  def bufferedImageToMat(img: BufferedImage): Mat = converter1.convert(converter3.convert(img))
  def bufferedImageToIplImage(img: BufferedImage): IplImage = converter2.convert(converter3.convert(img))

  // PageTypeがAであるかBであるかを決定する境界値
  val header_border = 450
  // ページを切断する際のマージン値
  val page_cut_margin = 15

  def scan_page(i0: IplImage, debug: IplImage): (PageType, Int, Int) = {
    val page_size = cvGetSize(i0)
    val i1 = cvCreateImage(page_size, IPL_DEPTH_8U, 1)
    val i2 = cvCreateImage(cvSize(1, page_size.height), IPL_DEPTH_32F, 1)
    val i3 = cvCreateImage(cvSize(1, page_size.height), IPL_DEPTH_32F, 1)

    // binarize
    cvThreshold(i0, i1, 0.9, 1.0, CV_THRESH_BINARY)
    // calculate average value of rows
    cvReduce(i1, i2, 1, CV_REDUCE_AVG)
    // binarize
    // ちょいセンシティブ。例えば0.995だとp108の先頭の…(縦)が無視される。
    // ダメそうならパラメータ化して、ページごとに設定するなどする。
    cvThreshold(i2, i3, 0.997, 1.0, CV_THRESH_BINARY)

    class PixelFinder(buf: FloatBuffer) {
      def find(r : collection.immutable.Range, f: Float => Boolean): Option[Int] = {
        for (i <- r) {
          if (f(buf get i)) {
            return Some(i)
          }
        }
        None
      }
    }
    val finder = new PixelFinder(i3.createBuffer[FloatBuffer]())

    cvReleaseImage(i1)
    cvReleaseImage(i2)
    cvReleaseImage(i3)

    val fpx = finder.find(0 until page_size.height, _ < 1.0).get
    println("fpx:", fpx)
    val lpx = finder.find(page_size.height-1 to 0 by -1, _ < 1.0).get
    println("lpx:", lpx)

    if (debugOn) {
      cvPutText(debug, "first_px", cvPoint(0, fpx+10), FONT, BLUE)
      cvLine(debug, cvPoint(0, fpx), cvPoint(page_size.width, fpx), BLUE, 2, CV_AA, 0)

      cvPutText(debug, "header_border", cvPoint(0, header_border+10), FONT, BLACK)
      cvLine(debug, cvPoint(0, header_border), cvPoint(page_size.width, header_border), BLACK, 2, CV_AA, 0)
    }

    if (fpx > header_border) {
      // Type A; The pages start with "第～篇 (The n-th Chapter)".
      if (debugOn) {
        cvPutText(debug, "Type: A", cvPoint(0, 50), FONT, BLACK)
      }

      val fpx2 = fpx - page_cut_margin
      //再びスキャンを行い、末尾を見つける
      val lpx2 = finder.find(lpx-75 to 0 by -1, _ < 1.0).get
      val lpx2_adjusted = lpx2 + page_cut_margin

      if (debugOn) {
        cvPutText(debug, "first_px+margin", cvPoint(0, fpx2 + 10), FONT, RED)
        cvLine(debug, cvPoint(0, fpx2), cvPoint(page_size.width, fpx2), RED, 2, CV_AA, 0)

        cvPutText(debug, "last_px", cvPoint(0, lpx + 10), FONT, BLUE)
        cvLine(debug, cvPoint(0, lpx), cvPoint(page_size.width, lpx), BLUE, 2, CV_AA, 0)

        cvPutText(debug, "last_px(adjusted)", cvPoint(0, lpx2_adjusted + 10), FONT, GREEN)
        cvLine(debug, cvPoint(0, lpx2_adjusted), cvPoint(page_size.width, lpx2_adjusted), GREEN, 2, CV_AA, 0)
      }

      (PageTypeA, fpx2, lpx2_adjusted)
    } else {
      // Type B; Normal pages.
      if (debugOn) {
        cvPutText(debug, "Type: B", cvPoint(0, 50), FONT, BLACK)
      }

      //再びスキャンを行い、先端を見つける
      val fpx2 = finder.find(fpx+80 until page_size.height, _ < 1.0).get
      val fpx2_adjusted = fpx2 - page_cut_margin
      val lpx2 = lpx + page_cut_margin

      if (debugOn) {
        cvPutText(debug, "first_px(adjusted)", cvPoint(0, fpx2_adjusted + 10), FONT, GREEN)
        cvLine(debug, cvPoint(0, fpx2_adjusted), cvPoint(page_size.width, fpx2_adjusted), GREEN, 2, CV_AA, 0)

        cvPutText(debug, "last_px", cvPoint(0, lpx + 10), FONT, BLUE)
        cvLine(debug, cvPoint(0, lpx), cvPoint(page_size.width, lpx), BLUE, 2, CV_AA, 0)

        cvPutText(debug, "last_px+margin", cvPoint(0, lpx2 + 10), FONT, RED)
        cvLine(debug, cvPoint(0, lpx2), cvPoint(page_size.width, lpx2), RED, 2, CV_AA, 0)
      }

      (PageTypeB, fpx2_adjusted, lpx2)
    }
  }

  // 想定されるアイテムの最短の高さ
  val min_item_height = 300
  // アイテムを切り出す際のマージン
  val item_split_margin = 15
  // 検出した線のx方向に取るマージン
  val item_x_margin = 5
  // 線の上部のスキャン領域の高さ
  val item_top_height = 50
  // 線の上部のスキャン領域のマージン
  val item_top_y_margin = 5
  // 線の下部のスキャン領域の高さ
  val item_bottom_height = 15
  // 線の下部のスキャン領域のマージン
  val item_bottom_y_margin = 5
  
  def scan_items(i0: IplImage, debug: IplImage): List[Int] = {
    val page_size = cvGetSize(i0)
    val i1 = cvCreateImage(page_size, IPL_DEPTH_32F, 1)
    val i2 = cvCreateImage(page_size, IPL_DEPTH_8U, 1)
    val storage = cvCreateMemStorage(0)
    cvSmooth(i0, i1, CV_GAUSSIAN, 9, 9, 0, 0)
    cvThreshold(i1, i2, 0.95, 255, CV_THRESH_BINARY_INV)

    val lines = cvHoughLines2(i2, storage, CV_HOUGH_PROBABILISTIC, 0.025, CV_PI / 180, 1, 200, 4)

    val segments = (for (i <- 0 until lines.total) yield {
      val segment = cvGetSeqElem(lines, i)
      (new CvPoint(segment).position(0), new CvPoint(segment).position(1))
    }).sortWith((p0, p1) => Math.min(p0._1.y, p0._2.y) < Math.min(p1._1.y, p1._2.y))

    val items = new mutable.MutableList[Int]

    for ((p0, p1) <- segments) {
      if (debugOn) {
        cvLine(debug, cvPoint(p0.x, p0.y), cvPoint(p1.x, p1.y), RED, 2, CV_AA, 0)
      }
      val x = Math.min(p0.x, p1.x) + item_x_margin
      val width = Math.abs(p0.x - p1.x) - item_x_margin * 2
      val y = Math.min(p0.y, p1.y) - item_split_margin
      val top_y = Math.min(p0.y, p1.y) - item_top_height - item_top_y_margin
      val bottom_y = Math.max(p0.y, p1.y) + item_bottom_y_margin

      if (x >= 0 && top_y >= 0 && width > 0 && bottom_y + item_bottom_height < page_size.height) {
        val v0 = cvCreateImage(cvSize(width, 1), IPL_DEPTH_32F, 1)
        val v1 = cvCreateImage(cvSize(1, 1), IPL_DEPTH_32F, 1)

        cvSetImageROI(i2, cvRect(x, top_y, width, item_top_height))
        cvReduce(i2, v0, 0, CV_REDUCE_AVG)
        cvReduce(v0, v1, 1, CV_REDUCE_AVG)
        val avg1 = v1.createBuffer[FloatBuffer]().get(0)
        cvResetImageROI(i2)
        cvSetImageROI(i2, cvRect(x, bottom_y, width, item_bottom_height))
        cvReduce(i2, v0, 0, CV_REDUCE_AVG)
        cvReduce(v0, v1, 1, CV_REDUCE_AVG)
        val avg = (avg1 + v1.createBuffer[FloatBuffer]().get(0)) / 512
        println("p0:", p0, "p1:", p1, "avg:", avg)
        if (avg < 0.15) {
          if (items.isEmpty || y - items.last > min_item_height) {
            items += y
            if (debugOn) {
              cvLine(debug, cvPoint(0, y - item_split_margin), cvPoint(page_size.width, y - item_split_margin), BLUE, 2, CV_AA, 0)
            }
          }
          if (debugOn) {
            cvRectangle(debug, cvPoint(x, top_y), cvPoint(x + width, top_y + item_top_height), BLUE, 1, CV_AA, 0)
            cvRectangle(debug, cvPoint(x, bottom_y), cvPoint(x + width, bottom_y + item_bottom_height), GREEN, 1, CV_AA, 0)
          }
        }
        cvReleaseImage(v0)
        cvReleaseImage(v1)
      }
      cvResetImageROI(i2)
    }
    cvReleaseImage(i1)
    cvReleaseImage(i2)
    cvReleaseMemStorage(storage)

    items.toList
  }

  def cvGamma(src: IplImage, dst: IplImage, gamma: Double) {
    val lut = cvCreateMat(1, 256, CV_8U)
    val lut_indexer = lut.createIndexer[UByteIndexer]()
    for (i <- 0 until 256) {
      lut_indexer.put(i, (Math.pow(i.toDouble / 255, 1.0 / gamma) * 255).toInt)
    }
    cvLUT(src, dst, lut)
    cvReleaseMat(lut)
  }

  val gamma_adjust = 1.6

  val page_clip_A = 150
  val page_clip_B = 200

  def main(args: Array[String]) {

    println("enumerating supported file types")
    println("R:", ImageIO.getReaderFormatNames().mkString(", "))
    println("W:", ImageIO.getWriterFormatNames().mkString(", "))
    println("")

    val source_dir = "C:\\!変換データ\\01"
//    val source_dir = "data"
    var item_index = -1
    var img_index = -1
    val csv = new PrintWriter("output/chapter01.csv")

    for ((file, i) <- new File(source_dir).listFiles.zipWithIndex if file.isFile) {
      println("-" * 20)
      println(file.getName)

      val image = ImageIO.read(file)
      val i0 = bufferedImageToIplImage(image)
      val pageSize = cvGetSize(i0)
      val result = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
      val debug = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
      val i1 = cvCreateImage(pageSize, IPL_DEPTH_32F, 3)
      val i2 = cvCreateImage(pageSize, IPL_DEPTH_32F, 1)
      val i3 = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)

      // up-sampling 8bit integer to 32bit floating-point
      cvConvertScale(i0, i1, 1.0 / 255, 0)
      // convert color to gray-scale
      cvCvtColor(i1, i2, CV_BGR2GRAY)
      // adjust gamma
      cvGamma(i0, i3, 1.0 / gamma_adjust)

      if (debugOn) {
        cvCopy(i0, debug)
      }

      val (pageLeft, pageRight) = if (i % 2 == 0) (page_clip_A, page_clip_B) else (page_clip_B, page_clip_A)
      val pageWidth = pageSize.width - pageLeft -pageRight
      println("page left", pageLeft)
      println("page right", pageRight)
      println("page width", pageWidth)

      val (pageType, pageTop, pageBottom) = scan_page(i2, debug)
      val detectedItems = scan_items(i2, debug)
      var items = immutable.TreeSet.empty(Ordering.fromLessThan[Int](_ < _)) ++ detectedItems

      println("page type", pageType)
      println("page top", pageTop)
      println("page bottom", pageBottom)
      println("item list", items)

      if (guiOn) {
        val title = "Preview"
        val preview = cvCreateImage(cvSize(pageSize.width / 4, pageSize.height / 4), IPL_DEPTH_8U, 3)

        def refresh() {
          cvCopy(if (debugOn) debug else i3, result)
          cvPutText(result, "top", cvPoint(200, pageTop-10), FONT, GREEN)
          cvLine(result, cvPoint(0, pageTop), cvPoint(pageSize.width, pageTop), GREEN, 2, CV_AA, 0)
          cvPutText(result, "bottom", cvPoint(200, pageBottom-10), FONT, GREEN)
          cvLine(result, cvPoint(0, pageBottom), cvPoint(pageSize.width, pageBottom), GREEN, 2, CV_AA, 0)

          cvPutText(result, "left", cvPoint(pageLeft, 50), FONT, GREEN)
          cvLine(result, cvPoint(pageLeft, 0), cvPoint(pageLeft, pageSize.height), GREEN, 2, CV_AA, 0)
          cvPutText(result, "right", cvPoint(pageLeft + pageWidth - 200, 50), FONT, GREEN)
          cvLine(result, cvPoint(pageLeft + pageWidth, 0), cvPoint(pageLeft + pageWidth, pageSize.height), GREEN, 2, CV_AA, 0)

          for ((item, i) <- items.zipWithIndex) {
            cvPutText(result, "[" + i + "]", cvPoint(10*i, item-10), FONT, BLUE)
            cvLine(result, cvPoint(0, item), cvPoint(pageSize.width, item), BLUE, 2, CV_AA, 0)
          }
          cvResize(result, preview, INTER_LINEAR)
          cvShowImage(title, preview)
          cvResizeWindow(title, preview.arrayWidth(), preview.arrayHeight())
        }

        val callback = new CvMouseCallback() {
          var lastPos: Option[Int] = None
          override def call(evt: Int, x: Int, y: Int, flags: Int, param: Pointer) {
            val current_y = y * 4
            evt match {
              case CV_EVENT_LBUTTONUP =>
                lastPos match {
                  case Some(last_y) if last_y == current_y =>
                    if (items contains current_y) {
                      println(y, "already be contained to item list")
                    } else {
                      items = items + current_y
                      println("add", y)
                      println("item list", items)
                      refresh()
                    }
                  case Some(last_y) =>
                    val a = Math.min(last_y, current_y)
                    val b = Math.max(last_y, current_y)
                    items = items.filter(i => i < a || b < i)
                    println("filter item list", a , b)
                    println("item list", items)
                    refresh()
                  case None =>
                }
                println("CV_EVENT_LBUTTONUP", x, y)
              case CV_EVENT_LBUTTONDOWN =>
                println("CV_EVENT_LBUTTONDOWN", x, y)
                lastPos = Some(y * 4)
              case CV_EVENT_RBUTTONDOWN =>
                //reset
                println("CV_EVENT_RBUTTONDOWN", x, y)
                items = items.empty ++ detectedItems
                refresh()
              case _ => {}
            }
          }
        }
        cvNamedWindow(title)
        cvSetMouseCallback(title, callback)
        refresh()
        cvWaitKey()

        cvReleaseImage(preview)
      }

      if (outputOn) {
        // output item images
        println("items", items)
        def fileId = f"00-00-$item_index%04d-$img_index%02d"
        def filename = s"output/$fileId.bmp"

        // 見出しがある
        if (pageType == PageTypeA) {
          item_index += 1
        }

        if (items.isEmpty || pageTop < items.head) {
          img_index += 1
          val rect = cvRect(pageLeft, pageTop, pageWidth, items.headOption.getOrElse(pageBottom) - pageTop)
          println("filename", filename)
          println("rect", rect)
          if (img_index > 0) {
            csv.print(", ")
          }
          csv.print(fileId)
          cvSetImageROI(i3, rect)
          cvSaveImage(filename, i3)
          cvResetImageROI(i3)
        }

        for ((a, b) <- items zip items.drop(1) + pageBottom) {
          img_index = 0
          item_index += 1
          val rect = cvRect(pageLeft, a, pageWidth, b - a)
          println("filename", filename)
          println("rect", rect)
          csv.println()
          csv.print(fileId)
          cvSetImageROI(i3, rect)
          cvSaveImage(filename, i3)
          cvResetImageROI(i3)
        }
      }

      //cvReleaseImage(i0)
      cvReleaseImage(i1)
      cvReleaseImage(i2)
      cvReleaseImage(i3)
      cvReleaseImage(result)
      cvReleaseImage(debug)
    }
    csv.flush()
    csv.close()
  }
}