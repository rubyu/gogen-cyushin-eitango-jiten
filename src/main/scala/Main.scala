
import java.awt.image.BufferedImage
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

    // サポートする画像形式の読み出し
    println("enumerating supported file types")
    println("R: " + ImageIO.getReaderFormatNames().mkString(", "))
    println("W: " + ImageIO.getWriterFormatNames().mkString(", "))
    println("")

    val file = new File("C:\\english etymology\\0.tif")
    println(file.getName)
    val image = ImageIO.read(file)
    if (image != null) {
      println("source")
      println("Width: " + image.getWidth)
      println("Height: " + image.getHeight)
      println("ColorModel: " + image.getColorModel)
    }
    println("")

    val mat_0 = bufferedImageToMat(image)

    println("original")
    println("Width: " + mat_0.arrayWidth())
    println("Height: " + mat_0.arrayHeight())
    println("Channels: " + mat_0.channels())
    println("")

    val mat_1 = new Mat()
    cvtColor(mat_0, mat_1, CV_BGR2GRAY)

    val preview = new Mat()
    resize(mat_1, preview, new Size(), 0.25, 0.25, INTER_LINEAR)
    println("preview")
    println("Width: " + preview.arrayWidth())
    println("Height: " + preview.arrayHeight())
    println("Channels: " + preview.channels())
    println("")

    namedWindow(file.getName)
    imshow(file.getName, preview)
    resizeWindow(file.getName, preview.arrayWidth(), preview.arrayHeight())
    waitKey()

    /*
    val canvas = new CanvasFrame("Preview")
    canvas.setSize(image.getWidth / 2, image.getHeight / 2)
    canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE)
    canvas.showImage(image)
    */

    /*
    val source_dir = "Z:\\BTSync\\rubyu\\code\\english etymology\\!変換データ"
    new File(source_dir).listFiles.map {
      case f if f.isDirectory => {}
      case x => {
        println(x.getAbsolutePath)
        val image = ImageIO.read(new File("C:\\english etymology\\0.tif"))

        if (image == null) {
          println("  -> null")
        }


        /*
        val converter1 = new OpenCVFrameConverter.ToIplImage()
        val converter2 = new Java2DFrameConverter()

        println(x.getAbsolutePath)

        val iplimage = cvLoadImage(x.getAbsolutePath, CV_LOAD_IMAGE_GRAYSCALE)

        if (iplimage == null) {
          println("  -> null")
        }

        val image = converter2.convert(converter1.convert(iplimage))
        */

        val canvas = new CanvasFrame("Preview")
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE)
        canvas.showImage(image)
      }
    }
    */
  }
}