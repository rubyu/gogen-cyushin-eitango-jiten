
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

  val converter1 = new OpenCVFrameConverter.ToMat()
  val converter2 = new OpenCVFrameConverter.ToIplImage()
  val converter3 = new Java2DFrameConverter()

  def bufferedImageToMat(img: BufferedImage): Mat = {
    converter1.convert(converter3.convert(img))
  }

  def bufferedImageToIplImage(img: BufferedImage): IplImage = {
    converter2.convert(converter3.convert(img))
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
        println(file.getName)
        val image = ImageIO.read(file)
        if (image != null) {
          println("Width:", image.getWidth)
          println("Height:", image.getHeight)
          println("ColorModel:", image.getColorModel)
        }
        val m0 = bufferedImageToMat(image)
        println("original:", m0)

        val ow = m0.arrayWidth()
        val oh = m0.arrayHeight()
        val m1, m2, m3, m4, m5, m6, m7, result = new Mat()

        m0.copyTo(result)
        // up-sampling 8bit integer to 32bit floating-point
        m0.convertTo(m1, CV_32F, 1.0 / 255, 0)
        // convert color to gray-scale
        cvtColor(m1, m2, CV_BGR2GRAY)
        // binarize
        threshold(m2, m3, 0.9, 1.0, CV_THRESH_BINARY)
        // calculate average value of rows
        reduce(m3, m4, 1, CV_REDUCE_AVG)
        // binarize
        // ちょいセンシティブ。例えば0.995だとp108の先頭の…(縦)が無視される。
        // ダメそうならパラメータ化して、ページごとに設定するなどする。
        threshold(m4, m5, 0.997, 1.0, CV_THRESH_BINARY)

        //    namedWindow("m5")
        //    imshow("m5", m5)
        //    waitKey()

        val m5_buf = m5.createBuffer[FloatBuffer]()

        def find_px(buf: FloatBuffer, r : collection.immutable.Range, f: Float => Boolean): Option[Int] = {
          for (i <- r) {
            if (f(buf.get(i))) {
              return Some(i)
            }
          }
          None
        }

        val first_px = find_px(m5_buf, 0 until oh, _ < 1.0)
        println("first_px: " + first_px)

        val last_px = find_px(m5_buf, oh-1 to 0 by -1, _ < 1.0)
        println("last_px: " + last_px)

        val fpx = first_px.get
        val lpx = last_px.get
        val margin = 15
        val h_border = 450
        val BLACK = new Scalar(0, 0, 0, 0)
        val BLUE = new Scalar(255, 0, 0, 0)
        val GREEN = new Scalar(0, 255, 0, 0)
        val RED = new Scalar(0, 0, 255, 0)
        val YELLOW = new Scalar(0, 255, 255, 0)

        putText(result, "first_px",new Point(0, fpx+10), FONT_HERSHEY_TRIPLEX, 2.0, BLUE, 2, CV_AA, false)
        line(result, new Point(0, fpx), new Point(ow, fpx), BLUE, 2, CV_AA, 0)

        putText(result, "h_border", new Point(0, h_border+10), FONT_HERSHEY_TRIPLEX, 2.0, BLACK, 2, CV_AA, false)
        line(result, new Point(0, h_border), new Point(ow, h_border), BLACK, 2, CV_AA, 0)

        if (fpx > h_border) {
          // Type A; The pages start with "第～篇 (The n-th Chapter)".
          putText(result, "Type: A", new Point(0, 50), FONT_HERSHEY_TRIPLEX, 2.0, BLACK, 2, CV_AA, false)

          putText(result, "first_px+margin", new Point(0, fpx-margin+10), FONT_HERSHEY_TRIPLEX, 2.0, RED, 2, CV_AA, false)
          line(result, new Point(0, fpx-margin), new Point(ow, fpx-margin), RED, 2, CV_AA, 0)

          putText(result, "last_px", new Point(0, lpx+10), FONT_HERSHEY_TRIPLEX, 2.0, BLUE, 2, CV_AA, false)
          line(result, new Point(0, lpx), new Point(ow, lpx), BLUE, 2, CV_AA, 0)

          //再びスキャンを行い、末尾を見つける
          val last_px2 = find_px(m5_buf, lpx-75 to 0 by -1, _ < 1.0)
          val lpx2 = last_px2.get

          putText(result, "last_px(adjusted)", new Point(0, lpx2+margin+10), FONT_HERSHEY_TRIPLEX, 2.0, GREEN, 2, CV_AA, false)
          line(result, new Point(0, lpx2+margin), new Point(ow, lpx2+margin), GREEN, 2, CV_AA, 0)
        } else {
          // Type B; Normal pages.
          putText(result, "Type: B", new Point(0, 50), FONT_HERSHEY_TRIPLEX, 2.0, BLACK, 2, CV_AA, false)

          //再びスキャンを行い、先端を見つける
          val first_px2 = find_px(m5_buf, fpx+80 until oh, _ < 1.0)
          val fpx2 = first_px2.get

          putText(result, "first_px(adjusted)", new Point(0, fpx2-margin+10), FONT_HERSHEY_TRIPLEX, 2.0, GREEN, 2, CV_AA, false)
          line(result, new Point(0, fpx2-margin), new Point(ow, fpx2-margin), GREEN, 2, CV_AA, 0)

          putText(result, "last_px", new Point(0, lpx+10), FONT_HERSHEY_TRIPLEX, 2.0, BLUE, 2, CV_AA, false)
          line(result, new Point(0, lpx), new Point(ow, lpx), BLUE, 2, CV_AA, 0)

          putText(result, "last_px+margin", new Point(0, lpx+margin+10), FONT_HERSHEY_TRIPLEX, 2.0, RED, 2, CV_AA, false)
          line(result, new Point(0, lpx+margin), new Point(ow, lpx+margin), RED, 2, CV_AA, 0)
        }

        val i0 = m0.asIplImage()
        val i1, i2 = cvCreateImage(cvGetSize(i0), IPL_DEPTH_8U, 1)
        val storage = cvCreateMemStorage(0)
        cvCvtColor(i0, i1, CV_BGR2GRAY)
        cvSmooth(i1, i2, CV_GAUSSIAN, 9, 9, 0, 0)
        cvCopy(i2, i1)
        cvThreshold(i1, i2, 250, 255, CV_THRESH_BINARY_INV)
        val lines = cvHoughLines2(i2, storage, CV_HOUGH_PROBABILISTIC, 0.025, CV_PI / 180, 1, 200, 4)

        val segments = (for (i <- 0 until lines.total) yield {
          val segment = cvGetSeqElem(lines, i)
          (new CvPoint(segment).position(0), new CvPoint(segment).position(1))
        }).sortWith((p0, p1) => Math.min(p0._1.y, p0._2.y) < Math.min(p1._1.y, p1._2.y))

        val min_item_height = 300
        val item_split_margin = 20
        val items = new mutable.MutableList[Int]

        for ((p0, p1) <- segments) {
          line(result, new Point(p0.x, p0.y), new Point(p1.x, p1.y), RED, 2, CV_AA, 0)

          val x_margin = 5
          val top_height = 50
          val top_margin = 5
          val bottom_height = 15
          val bottom_margin = 5
          val x = Math.min(p0.x, p1.x) + x_margin
          val width = Math.abs(p0.x - p1.x) - x_margin * 2
          val y = Math.min(p0.y, p1.y)
          val top_y = y - top_height - top_margin
          val bottom_y = y + bottom_margin

          if (x >= 0 && top_y >= 0 && width > 0 && bottom_y + bottom_height < oh) {
            val v0 = cvCreateImage(cvSize(width, 1), IPL_DEPTH_32F, 1)
            val v1 = cvCreateImage(cvSize(1, 1), IPL_DEPTH_32F, 1)

            cvSetImageROI(i2, cvRect(x, top_y, width, top_height))
            cvReduce(i2, v0, 0, CV_REDUCE_AVG)
            cvReduce(v0, v1, 1, CV_REDUCE_AVG)
            val avg1 = v1.createBuffer[FloatBuffer]().get(0)
            cvResetImageROI(i2)
            cvSetImageROI(i2, cvRect(x, bottom_y, width, bottom_height))
            cvReduce(i2, v0, 0, CV_REDUCE_AVG)
            cvReduce(v0, v1, 1, CV_REDUCE_AVG)
            val avg = (avg1 + v1.createBuffer[FloatBuffer]().get(0)) / 512
            println("p0:", p0, "p1:", p1, "avg:", avg)
            if (avg < 0.15) {
              if (items.isEmpty || y - items.last > min_item_height) {
                items += y
                line(result, new Point(0, y-item_split_margin), new Point(ow, y-item_split_margin), BLUE, 2, CV_AA, 0)
              }
              rectangle(result, new Rect(x, top_y, width, top_height), BLUE, 1, CV_AA, 0)
              rectangle(result, new Rect(x, bottom_y, width, bottom_height), GREEN, 1, CV_AA, 0)
            }
          }
          cvResetImageROI(i2)
        }

        val preview = new Mat()
        resize(result, preview, new Size(), 0.25, 0.25, INTER_LINEAR)
        println("preview: " + preview)

        namedWindow("Preview")
        imshow("Preview", preview)
        resizeWindow("Preview", preview.arrayWidth(), preview.arrayHeight())
        waitKey()
      }
    }
  }
}