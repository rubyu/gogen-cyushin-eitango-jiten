
import java.awt.image.BufferedImage
import java.nio.{FloatBuffer, ByteBuffer}
import java.io._
import javax.imageio.ImageIO
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacv.{OpenCVFrameConverter, Java2DFrameConverter, CanvasFrame}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_objdetect._

import scala.collection.{mutable, immutable}

object HeaderClipper {

  trait PageType
  object PageTypeA extends PageType
  object PageTypeB extends PageType

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
    }).sortWith((p0, p1) => Math.min(p0._1.y, p0._2.y) > Math.min(p1._1.y, p1._2.y))

    val items = new mutable.MutableList[Int]

    for ((p0, p1) <- segments) {
//      cvLine(debug, cvPoint(p0.x, p0.y), cvPoint(p1.x, p1.y), RED, 2, CV_AA, 0)

      val x = Math.min(p0.x, p1.x) + item_x_margin
      val width = Math.abs(p0.x - p1.x) - item_x_margin * 2
      val y = Math.min(p0.y, p1.y) + item_split_margin
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
          if (items.isEmpty || items.last - y > min_item_height) {
            items += y
//            cvLine(debug, cvPoint(0, y-item_split_margin), cvPoint(page_size.width, y-item_split_margin), BLUE, 2, CV_AA, 0)
          }
//          cvRectangle(debug, cvPoint(x, top_y), cvPoint(x+width, top_y+item_top_height), BLUE, 1, CV_AA, 0)
//          cvRectangle(debug, cvPoint(x, bottom_y), cvPoint(x+width, bottom_y+item_bottom_height), GREEN, 1, CV_AA, 0)
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

  def main(args: Array[String]) {

    println("enumerating supported file types")
    println("R:", ImageIO.getReaderFormatNames().mkString(", "))
    println("W:", ImageIO.getWriterFormatNames().mkString(", "))
    println("")

    val source_dir = "C:\\ee_bmp\\00"

    new File(source_dir).listFiles.map {
      case f if f.isDirectory => {}
      case file => {
        println("-" * 20)
        println(file.getName)
        val image = ImageIO.read(file)
        val i0 = bufferedImageToIplImage(image)
        val page_size = cvGetSize(i0)
        val result = cvCreateImage(page_size, IPL_DEPTH_8U, 3)
//        val debug = cvCreateImage(page_size, IPL_DEPTH_8U, 3)
//        cvCopy(i0, debug)
        val i1 = cvCreateImage(page_size, IPL_DEPTH_32F, 3)
        val i2 = cvCreateImage(page_size, IPL_DEPTH_32F, 1)
        // up-sampling 8bit integer to 32bit floating-point
        cvConvertScale(i0, i1, 1.0 / 255, 0)
        // convert color to gray-scale
        cvCvtColor(i1, i2, CV_BGR2GRAY)

        val detected_items = scan_items(i2, result)
        var items = immutable.TreeSet.empty(Ordering.fromLessThan[Int](_ < _)) ++ detected_items

        println("item list", items)

        val title = "Preview"
        val preview = cvCreateImage(cvSize(page_size.width / 4, page_size.height / 4), IPL_DEPTH_8U, 3)


        def refresh() {
          cvCopy(i0, result)

          for ((item, i) <- items.zipWithIndex) {
            cvPutText(result, "[" + i + "]", cvPoint(10*i, item-10), FONT, BLUE)
            cvLine(result, cvPoint(0, item), cvPoint(page_size.width, item), BLUE, 2, CV_AA, 0)
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
                items = items.empty ++ detected_items
                refresh()
              case _ => {}
            }
          }
        }
        cvNamedWindow(title)
        cvSetMouseCallback(title, callback)
        refresh()
        cvWaitKey()

        // output item images
        println("items", items)
        def filename = s"output/${file.getName.split('.')(0)}-header.bmp"

        val rect = cvRect(0, 0, page_size.width, items.headOption.getOrElse(page_size.height))
        println("filename", filename)
        println("rect", rect)
        cvSetImageROI(i0, rect)
        cvSaveImage(filename, i0)
        cvResetImageROI(i0)

        //cvReleaseImage(i0)
        cvReleaseImage(i1)
        cvReleaseImage(i2)
        cvReleaseImage(result)
        cvReleaseImage(preview)
      }
    }
  }
}