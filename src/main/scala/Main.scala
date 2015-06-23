
import java.awt.image.BufferedImage
import java.nio.{FloatBuffer, ByteBuffer}
import java.io._
import javax.imageio.ImageIO
import org.bytedeco.javacv.{OpenCVFrameConverter, Java2DFrameConverter, CanvasFrame}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_objdetect._

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

    val source_dir = "Z:\\BTSync\\rubyu\\code\\english etymology\\!変換データ"
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
        val m1, m2, m3, m4, m5 = new Mat()
        //up-sampling to 32bit
        m0.convertTo(m1, CV_32F, 1.0 / 255, 0)
        // convert color to gray-scale
        cvtColor(m1, m2, CV_BGR2GRAY)
        // binarize
        threshold(m2, m3, 0.9, 1.0, CV_THRESH_BINARY)
        // calculate average value of rows
        reduce(m3, m4, 1, CV_REDUCE_AVG)
        // binarize
        threshold(m4, m5, 0.995, 1.0, CV_THRESH_BINARY)
        println("m5: " + m5)

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

        putText(m0, "first_px",new Point(0, fpx+10), FONT_HERSHEY_TRIPLEX, 2.0, BLUE, 2, CV_AA, false)
        line(m0, new Point(0, fpx), new Point(ow, fpx), BLUE, 2, CV_AA, 0)

        putText(m0, "h_border", new Point(0, h_border+10), FONT_HERSHEY_TRIPLEX, 2.0, BLACK, 2, CV_AA, false)
        line(m0, new Point(0, h_border), new Point(ow, h_border), BLACK, 2, CV_AA, 0)

        if (fpx > h_border) {
          putText(m0, "Type: A", new Point(0, 50), FONT_HERSHEY_TRIPLEX, 2.0, BLACK, 2, CV_AA, false)

          putText(m0, "first_px+margin", new Point(0, fpx-margin+10), FONT_HERSHEY_TRIPLEX, 2.0, RED, 2, CV_AA, false)
          line(m0, new Point(0, fpx-margin), new Point(ow, fpx-margin), RED, 2, CV_AA, 0)

          putText(m0, "last_px", new Point(0, lpx+10), FONT_HERSHEY_TRIPLEX, 2.0, BLUE, 2, CV_AA, false)
          line(m0, new Point(0, lpx), new Point(ow, lpx), BLUE, 2, CV_AA, 0)

          //再びスキャンを行い、末尾を見つける
          val last_px2 = find_px(m5_buf, lpx-75 to 0 by -1, _ < 1.0)
          val lpx2 = last_px2.get

          putText(m0, "last_px(adjusted)", new Point(0, lpx2+margin+10), FONT_HERSHEY_TRIPLEX, 2.0, GREEN, 2, CV_AA, false)
          line(m0, new Point(0, lpx2+margin), new Point(ow, lpx2+margin), GREEN, 2, CV_AA, 0)
        } else {
          putText(m0, "Type: B", new Point(0, 50), FONT_HERSHEY_TRIPLEX, 2.0, BLACK, 2, CV_AA, false)

          //再びスキャンを行い、先端を見つける
          val first_px2 = find_px(m5_buf, fpx+80 until oh, _ < 1.0)
          val fpx2 = first_px2.get

          putText(m0, "first_px(adjusted)", new Point(0, fpx2-margin+10), FONT_HERSHEY_TRIPLEX, 2.0, GREEN, 2, CV_AA, false)
          line(m0, new Point(0, fpx2-margin), new Point(ow, fpx2-margin), GREEN, 2, CV_AA, 0)

          putText(m0, "last_px", new Point(0, lpx+10), FONT_HERSHEY_TRIPLEX, 2.0, BLUE, 2, CV_AA, false)
          line(m0, new Point(0, lpx), new Point(ow, lpx), BLUE, 2, CV_AA, 0)

          putText(m0, "last_px+margin", new Point(0, lpx+margin+10), FONT_HERSHEY_TRIPLEX, 2.0, RED, 2, CV_AA, false)
          line(m0, new Point(0, lpx+margin), new Point(ow, lpx+margin), RED, 2, CV_AA, 0)
        }

        val preview = new Mat()
        resize(m0, preview, new Size(), 0.25, 0.25, INTER_LINEAR)
        println("preview: " + preview)

        namedWindow("Preview")
        imshow("Preview", preview)
        resizeWindow("Preview", preview.arrayWidth(), preview.arrayHeight())
        waitKey()
      }
    }
  }
}