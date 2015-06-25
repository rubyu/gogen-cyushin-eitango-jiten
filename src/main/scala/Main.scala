
import java.awt.image.BufferedImage
import java.nio.{FloatBuffer, ByteBuffer}
import java.io._
import javax.imageio.ImageIO
import org.bytedeco.javacv.{OpenCVFrameConverter, Java2DFrameConverter, CanvasFrame}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_objdetect._

import scala.collection.mutable

object Main {

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

  // PageTypeがAであるかBであるかを決定する境界値
  val header_border = 450
  // ページを切断する際のマージン値
  val page_cut_margin = 15

  def scan_page(i0: IplImage, result: IplImage): (PageType, Int, Int) = {

    println("i0:", i0)

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

//    cvNamedWindow("i3")
//    cvShowImage("i3", i3)
//    cvWaitKey()

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

    cvPutText(result, "first_px", cvPoint(0, fpx+10), FONT, BLUE)
    cvLine(result, cvPoint(0, fpx), cvPoint(page_size.width, fpx), BLUE, 2, CV_AA, 0)

    cvPutText(result, "header_border", cvPoint(0, header_border+10), FONT, BLACK)
    cvLine(result, cvPoint(0, header_border), cvPoint(page_size.width, header_border), BLACK, 2, CV_AA, 0)

    if (fpx > header_border) {
      // Type A; The pages start with "第～篇 (The n-th Chapter)".
      cvPutText(result, "Type: A", cvPoint(0, 50), FONT, BLACK)

      val fpx2 = fpx - page_cut_margin
      //再びスキャンを行い、末尾を見つける
      val lpx2 = finder.find(lpx-75 to 0 by -1, _ < 1.0).get
      val lpx2_adjusted = lpx2 + page_cut_margin

      cvPutText(result, "first_px+margin", cvPoint(0, fpx2+10), FONT, RED)
      cvLine(result, cvPoint(0, fpx2), cvPoint(page_size.width, fpx2), RED, 2, CV_AA, 0)

      cvPutText(result, "last_px", cvPoint(0, lpx+10), FONT, BLUE)
      cvLine(result, cvPoint(0, lpx), cvPoint(page_size.width, lpx), BLUE, 2, CV_AA, 0)

      cvPutText(result, "last_px(adjusted)", cvPoint(0, lpx2_adjusted+10), FONT, GREEN)
      cvLine(result, cvPoint(0, lpx2_adjusted), cvPoint(page_size.width, lpx2_adjusted), GREEN, 2, CV_AA, 0)

      (PageTypeA, fpx2, lpx2_adjusted)
    } else {
      // Type B; Normal pages.
      cvPutText(result, "Type: B", cvPoint(0, 50), FONT, BLACK)

      //再びスキャンを行い、先端を見つける
      val fpx2 = finder.find(fpx+80 until page_size.height, _ < 1.0).get
      val fpx2_adjusted = fpx2 - page_cut_margin
      val lpx2 = lpx + page_cut_margin

      cvPutText(result, "first_px(adjusted)", cvPoint(0, fpx2_adjusted+10), FONT, GREEN)
      cvLine(result, cvPoint(0, fpx2_adjusted), cvPoint(page_size.width, fpx2_adjusted), GREEN, 2, CV_AA, 0)

      cvPutText(result, "last_px", cvPoint(0, lpx+10), FONT, BLUE)
      cvLine(result, cvPoint(0, lpx), cvPoint(page_size.width, lpx), BLUE, 2, CV_AA, 0)

      cvPutText(result, "last_px+margin", cvPoint(0, lpx2+10), FONT, RED)
      cvLine(result, cvPoint(0, lpx2), cvPoint(page_size.width, lpx2), RED, 2, CV_AA, 0)

      (PageTypeB, fpx2_adjusted, lpx2)
    }
  }

  def main(args: Array[String]) {

    println("enumerating supported file types")
    println("R:", ImageIO.getReaderFormatNames().mkString(", "))
    println("W:", ImageIO.getWriterFormatNames().mkString(", "))
    println("")

//    val source_dir = "Z:\\BTSync\\rubyu\\code\\english etymology\\!変換データ"
    val source_dir = "data"
    new File(source_dir).listFiles.map {
      case f if f.isDirectory => {}
      case file => {
        println("-" * 20)
        println(file.getName)
        val image = ImageIO.read(file)
        println("Width:", image.getWidth)
        println("Height:", image.getHeight)
        println("ColorModel:", image.getColorModel)

        val i0 = bufferedImageToIplImage(image)
        println("i0:", i0)
        val page_size = cvGetSize(i0)
        val result = cvCreateImage(page_size, IPL_DEPTH_8U, 3)
        cvCopy(i0, result)
        val i1 = cvCreateImage(page_size, IPL_DEPTH_32F, 3)
        val i2 = cvCreateImage(page_size, IPL_DEPTH_32F, 1)
        // up-sampling 8bit integer to 32bit floating-point
        cvConvertScale(i0, i1, 1.0 / 255, 0)
        // convert color to gray-scale
        cvCvtColor(i1, i2, CV_BGR2GRAY)

        scan_page(i2, result)

        val i3 = cvCreateImage(page_size, IPL_DEPTH_32F, 1)
        val i4 = cvCreateImage(cvGetSize(i0), IPL_DEPTH_8U, 1)
        val storage = cvCreateMemStorage(0)
        cvSmooth(i2, i3, CV_GAUSSIAN, 9, 9, 0, 0)
        cvThreshold(i3, i4, 0.95, 255, CV_THRESH_BINARY_INV)

        val lines = cvHoughLines2(i4, storage, CV_HOUGH_PROBABILISTIC, 0.025, CV_PI / 180, 1, 200, 4)

        val segments = (for (i <- 0 until lines.total) yield {
          val segment = cvGetSeqElem(lines, i)
          (new CvPoint(segment).position(0), new CvPoint(segment).position(1))
        }).sortWith((p0, p1) => Math.min(p0._1.y, p0._2.y) < Math.min(p1._1.y, p1._2.y))

        val min_item_height = 300
        val item_split_margin = 20
        val items = new mutable.MutableList[Int]

        for ((p0, p1) <- segments) {
          cvLine(result, cvPoint(p0.x, p0.y), cvPoint(p1.x, p1.y), RED, 2, CV_AA, 0)

          val x_margin = 5
          val top_height = 50
          val top_margin = 5
          val bottom_height = 15
          val bottom_margin = 5
          val x = Math.min(p0.x, p1.x) + x_margin
          val width = Math.abs(p0.x - p1.x) - x_margin * 2
          val y = Math.min(p0.y, p1.y)
          val top_y = Math.min(p0.y, p1.y) - top_height - top_margin
          val bottom_y = Math.max(p0.y, p1.y) + bottom_margin


          if (x >= 0 && top_y >= 0 && width > 0 && bottom_y + bottom_height < page_size.height) {
            val v0 = cvCreateImage(cvSize(width, 1), IPL_DEPTH_32F, 1)
            val v1 = cvCreateImage(cvSize(1, 1), IPL_DEPTH_32F, 1)

            cvSetImageROI(i4, cvRect(x, top_y, width, top_height))
            cvReduce(i4, v0, 0, CV_REDUCE_AVG)
            cvReduce(v0, v1, 1, CV_REDUCE_AVG)
            val avg1 = v1.createBuffer[FloatBuffer]().get(0)
            cvResetImageROI(i4)
            cvSetImageROI(i4, cvRect(x, bottom_y, width, bottom_height))
            cvReduce(i4, v0, 0, CV_REDUCE_AVG)
            cvReduce(v0, v1, 1, CV_REDUCE_AVG)
            val avg = (avg1 + v1.createBuffer[FloatBuffer]().get(0)) / 512
            println("p0:", p0, "p1:", p1, "avg:", avg)
            if (avg < 0.15) {
              if (items.isEmpty || y - items.last > min_item_height) {
                items += y
                cvLine(result, cvPoint(0, y-item_split_margin), cvPoint(page_size.width, y-item_split_margin), BLUE, 2, CV_AA, 0)
              }
              cvRectangle(result, cvPoint(x, top_y), cvPoint(x+width, top_y+top_height), BLUE, 1, CV_AA, 0)
              cvRectangle(result, cvPoint(x, bottom_y), cvPoint(x+width, bottom_y+bottom_height), GREEN, 1, CV_AA, 0)
            }
            cvReleaseImage(v0)
            cvReleaseImage(v1)
          }
          cvResetImageROI(i4)
        }

        // patch for o

        // split

        val preview = cvCreateImage(cvSize(page_size.width / 4, page_size.height / 4), IPL_DEPTH_8U, 3)
        cvResize(result, preview, INTER_LINEAR)
        println("preview: " + preview)

        cvNamedWindow("Preview")
        cvShowImage("Preview", preview)
        cvResizeWindow("Preview", preview.arrayWidth(), preview.arrayHeight())
        cvWaitKey()

        //cvReleaseImage(i0)
        cvReleaseImage(i1)
        cvReleaseImage(i2)
        cvReleaseImage(i3)
        cvReleaseImage(i4)
        cvReleaseImage(result)
        cvReleaseMemStorage(storage)
        cvReleaseImage(preview)
      }
    }
  }
}