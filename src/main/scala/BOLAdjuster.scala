
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import java.io._
import javax.imageio.ImageIO
import org.bytedeco.javacv.{OpenCVFrameConverter, Java2DFrameConverter}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_highgui._

import scala.collection.{mutable, immutable}

object BOLAdjuster {

  val debugOn = false
  val guiOn = false
  val outputOn = true

  val WHITE = cvScalar(255, 255, 255, 0)
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

  def scanPage(i0: IplImage, debug: IplImage): (Int, Int) = {
    val pageSize = cvGetSize(i0)
    val i1 = cvCreateImage(pageSize, IPL_DEPTH_8U, 1)
    val i2 = cvCreateImage(cvSize(pageSize.width, 1), IPL_DEPTH_32F, 1)
    val i3 = cvCreateImage(cvSize(pageSize.width, 1), IPL_DEPTH_32F, 1)

    // binarize
    cvThreshold(i0, i1, 0.5, 1.0, CV_THRESH_BINARY)
    // calculate average value of columns
    cvReduce(i1, i2, 0, CV_REDUCE_AVG)
    // binarize
    cvThreshold(i2, i3, 0.999, 1.0, CV_THRESH_BINARY)

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

    val fpx = finder.find(0 until pageSize.width, _ < 1.0).get
    val lpx = finder.find(pageSize.width-1 to 0 by -1, _ < 1.0).get
    (fpx, lpx)
  }

  def main(args: Array[String]) {

    println("enumerating supported file types")
    println("R:", ImageIO.getReaderFormatNames().mkString(", "))
    println("W:", ImageIO.getWriterFormatNames().mkString(", "))
    println("")

    for ((file, i) <- new File("C:\\ee_bmp\\header").listFiles.zipWithIndex if file.isFile) {
      println("-" * 20)
      println(file.getName)

      val image = ImageIO.read(file)
      val i0 = bufferedImageToIplImage(image)
      val pageSize = cvGetSize(i0)
      val result = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
      val debug = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
      val i1 = cvCreateImage(pageSize, IPL_DEPTH_32F, 3)
      val i2 = cvCreateImage(pageSize, IPL_DEPTH_32F, 1)

      // up-sampling 8bit integer to 32bit floating-point
      cvConvertScale(i0, i1, 1.0 / 255, 0)
      // convert color to gray-scale
      cvCvtColor(i1, i2, CV_BGR2GRAY)

      if (debugOn) {
        cvCopy(i0, debug)
      }

      val (fpx, lpx) = scanPage(i2, debug)
      val shift = 100 - fpx
      val BOLAdjusted = fpx + shift
      println("fpx", fpx)
      println("lpx", lpx)
      println("shift", shift)
      println("BOL Adjusted", BOLAdjusted)

      if (debugOn) {
        cvPutText(debug, "fpx", cvPoint(fpx, 50), FONT, BLUE)
        cvLine(debug, cvPoint(fpx, 0), cvPoint(fpx, pageSize.height), BLUE, 2, CV_AA, 0)
        cvPutText(debug, "lpx", cvPoint(lpx, 50), FONT, BLUE)
        cvLine(debug, cvPoint(lpx, 0), cvPoint(lpx, pageSize.height), BLUE, 2, CV_AA, 0)
        cvPutText(debug, "BOLAdjusted", cvPoint(BOLAdjusted, 50), FONT, GREEN)
        cvLine(debug, cvPoint(BOLAdjusted, 0), cvPoint(BOLAdjusted, pageSize.height), GREEN, 2, CV_AA, 0)
        cvCopy(debug, result)
      } else {
        cvSet(result, WHITE)
        if (shift > 0) {
          val r0 = cvRect(0, 0, pageSize.width - shift, pageSize.height)
          val r1 = cvRect(shift, 0, pageSize.width - shift, pageSize.height)
          cvSetImageROI(i0, r0)
          cvSetImageROI(result, r1)
          cvCopy(i0, result)
          cvResetImageROI(i0)
          cvResetImageROI(result)
        } else if (shift < 0) {
          val r0 = cvRect(Math.abs(shift), 0, pageSize.width + shift, pageSize.height)
          val r1 = cvRect(0, 0, pageSize.width + shift, pageSize.height)
          cvSetImageROI(i0, r0)
          cvSetImageROI(result, r1)
          cvCopy(i0, result)
          cvResetImageROI(i0)
          cvResetImageROI(result)
        } else {
          cvCopy(i0, result)
        }
      }

      if (guiOn) {
        val title = "Preview"
        val preview = cvCreateImage(cvSize(pageSize.width / 4, pageSize.height / 4), IPL_DEPTH_8U, 3)
        cvResize(result, preview, INTER_LINEAR)
        cvShowImage(title, preview)
        cvResizeWindow(title, preview.arrayWidth(), preview.arrayHeight())
        cvNamedWindow(title)
        cvWaitKey()
        cvReleaseImage(preview)
      }

      if (outputOn) {
        cvSaveImage(s"output/${file.getName}", result)
      }

      //cvReleaseImage(i0)
      cvReleaseImage(i1)
      cvReleaseImage(i2)
      cvReleaseImage(result)
      cvReleaseImage(debug)
    }

//    for ((file, i) <- new File("C:\\ee_bmp\\content").listFiles.zipWithIndex if file.isFile) {
//      println("-" * 20)
//      println(file.getName)
//
//      val headerName = file.getName.replace("00.bmp", "00-header.bmp")
//      val header =
//        if (file.getName.endsWith("00.bmp")) {
//          val f = new File("C:\\ee_bmp\\header\\" + headerName)
//          Some(ImageIO.read(f))
//        }
//        else None
//
//      val pageTop = if (header.isDefined) header.get.getHeight else 0
//
//      val image = ImageIO.read(file)
//      val i0 = bufferedImageToIplImage(image)
//      val pageSize = cvGetSize(i0)
//      val result = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val debug = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val i1 = cvCreateImage(pageSize, IPL_DEPTH_32F, 3)
//      val i2 = cvCreateImage(pageSize, IPL_DEPTH_32F, 1)
//
//      // up-sampling 8bit integer to 32bit floating-point
//      cvConvertScale(i0, i1, 1.0 / 255, 0)
//      // convert color to gray-scale
//      cvCvtColor(i1, i2, CV_BGR2GRAY)
//
//      if (debugOn) {
//        cvCopy(i0, debug)
//      }
//
//      val (fpx, lpx) =
//        if (pageTop > 0 && pageSize.height > pageTop) {
//          println("crop header")
//          cvSetImageROI(i2, cvRect(0, pageTop, pageSize.width, pageSize.height - pageTop))
//          val i3 = cvCreateImage(cvSize(pageSize.width, pageSize.height - pageTop), IPL_DEPTH_32F, 1)
//          cvCopy(i2, i3)
//          try scanPage(i3, debug)
//          finally cvReleaseImage(i3)
//        } else {
//          scanPage(i2, debug)
//        }
//      val shift = 150 - fpx
//      val BOLAdjusted = fpx + shift
//      println("fpx", fpx)
//      println("lpx", lpx)
//      println("shift", shift)
//      println("BOL Adjusted", BOLAdjusted)
//
//      if (debugOn) {
//        cvPutText(debug, "top", cvPoint(50, pageTop+50), FONT, BLUE)
//        cvLine(debug, cvPoint(0, pageTop), cvPoint(pageSize.width, pageTop), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "fpx", cvPoint(fpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(fpx, 0), cvPoint(fpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "lpx", cvPoint(lpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(lpx, 0), cvPoint(lpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "BOLAdjusted", cvPoint(BOLAdjusted, 50), FONT, GREEN)
//        cvLine(debug, cvPoint(BOLAdjusted, 0), cvPoint(BOLAdjusted, pageSize.height), GREEN, 2, CV_AA, 0)
//        cvCopy(debug, result)
//      } else {
//        cvSet(result, WHITE)
//        if (shift > 0) {
//          val r0 = cvRect(0, 0, pageSize.width - shift, pageSize.height)
//          val r1 = cvRect(shift, 0, pageSize.width - shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        } else if (shift < 0) {
//          val r0 = cvRect(Math.abs(shift), 0, pageSize.width + shift, pageSize.height)
//          val r1 = cvRect(0, 0, pageSize.width + shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        } else {
//          cvCopy(i0, result)
//        }
//      }
//
//      if (guiOn) {
//        val title = "Preview"
//        val preview = cvCreateImage(cvSize(pageSize.width / 4, pageSize.height / 4), IPL_DEPTH_8U, 3)
//        cvResize(result, preview, INTER_LINEAR)
//        cvShowImage(title, preview)
//        cvResizeWindow(title, preview.arrayWidth(), preview.arrayHeight())
//        cvNamedWindow(title)
//        cvWaitKey()
//        cvReleaseImage(preview)
//      }
//
//      if (outputOn) {
//        cvSaveImage(s"output/${file.getName}", result)
//      }
//
//      //cvReleaseImage(i0)
//      cvReleaseImage(i1)
//      cvReleaseImage(i2)
//      cvReleaseImage(result)
//      cvReleaseImage(debug)
//    }

//    for ((file, i) <- new File("C:\\ee_bmp\\index\\2").listFiles.zipWithIndex if file.isFile) {
//      println("-" * 20)
//      println(file.getName)
//
//      val image = ImageIO.read(file)
//      val i0 = bufferedImageToIplImage(image)
//      val pageSize = cvGetSize(i0)
//      val result = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val debug = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val i1 = cvCreateImage(pageSize, IPL_DEPTH_32F, 3)
//      val i2 = cvCreateImage(pageSize, IPL_DEPTH_32F, 1)
//
//      // up-sampling 8bit integer to 32bit floating-point
//      cvConvertScale(i0, i1, 1.0 / 255, 0)
//      // convert color to gray-scale
//      cvCvtColor(i1, i2, CV_BGR2GRAY)
//
//      if (debugOn) {
//        cvCopy(i0, debug)
//      }
//
//      val (fpx, lpx) = scanPage(i2, debug)
//      val shift = 150 - fpx
//      val BOLAdjusted = fpx + shift
//      println("fpx", fpx)
//      println("lpx", lpx)
//      println("shift", shift)
//      println("BOL Adjusted", BOLAdjusted)
//
//      if (debugOn) {
//        cvPutText(debug, "fpx", cvPoint(fpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(fpx, 0), cvPoint(fpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "lpx", cvPoint(lpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(lpx, 0), cvPoint(lpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "BOLAdjusted", cvPoint(BOLAdjusted, 50), FONT, GREEN)
//        cvLine(debug, cvPoint(BOLAdjusted, 0), cvPoint(BOLAdjusted, pageSize.height), GREEN, 2, CV_AA, 0)
//        cvCopy(debug, result)
//      } else {
//        cvSet(result, WHITE)
//        if (shift == 0) {
//          cvCopy(i0, result)
//        } else if (shift > 0) {
//          val r0 = cvRect(0, 0, pageSize.width - shift, pageSize.height)
//          val r1 = cvRect(shift, 0, pageSize.width - shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        } else {
//          val r0 = cvRect(-shift, 0, pageSize.width + shift, pageSize.height)
//          val r1 = cvRect(0, 0, pageSize.width + shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        }
//      }
//
//      if (guiOn) {
//        val title = "Preview"
//        val preview = cvCreateImage(cvSize(pageSize.width / 4, pageSize.height / 4), IPL_DEPTH_8U, 3)
//        cvResize(result, preview, INTER_LINEAR)
//        cvShowImage(title, preview)
//        cvResizeWindow(title, preview.arrayWidth(), preview.arrayHeight())
//        cvNamedWindow(title)
//        cvWaitKey()
//        cvReleaseImage(preview)
//      }
//
//      if (outputOn) {
//        cvSaveImage(s"output/${file.getName}", result)
//      }
//
//      //cvReleaseImage(i0)
//      cvReleaseImage(i1)
//      cvReleaseImage(i2)
//      cvReleaseImage(result)
//      cvReleaseImage(debug)
//    }
//
//    for ((file, i) <- new File("C:\\ee_bmp\\index\\1").listFiles.zipWithIndex if file.isFile) {
//      println("-" * 20)
//      println(file.getName)
//
//      val image = ImageIO.read(file)
//      val i0 = bufferedImageToIplImage(image)
//      val pageSize = cvGetSize(i0)
//      val result = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val debug = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val i1 = cvCreateImage(pageSize, IPL_DEPTH_32F, 3)
//      val i2 = cvCreateImage(pageSize, IPL_DEPTH_32F, 1)
//
//      // up-sampling 8bit integer to 32bit floating-point
//      cvConvertScale(i0, i1, 1.0 / 255, 0)
//      // convert color to gray-scale
//      cvCvtColor(i1, i2, CV_BGR2GRAY)
//
//      if (debugOn) {
//        cvCopy(i0, debug)
//      }
//
//      val (fpx, lpx) = scanPage(i2, debug)
//      val shift = - (fpx - (pageSize.width - lpx)) / 2
//      val BOLAdjusted = fpx + shift
//      println("fpx", fpx)
//      println("lpx", lpx)
//      println("shift", shift)
//      println("BOL Adjusted", BOLAdjusted)
//
//      if (debugOn) {
//        cvPutText(debug, "fpx", cvPoint(fpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(fpx, 0), cvPoint(fpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "lpx", cvPoint(lpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(lpx, 0), cvPoint(lpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "BOLAdjusted", cvPoint(BOLAdjusted, 50), FONT, GREEN)
//        cvLine(debug, cvPoint(BOLAdjusted, 0), cvPoint(BOLAdjusted, pageSize.height), GREEN, 2, CV_AA, 0)
//        cvCopy(debug, result)
//      } else {
//        cvSet(result, WHITE)
//        if (shift == 0) {
//          cvCopy(i0, result)
//        } else if (shift > 0) {
//          val r0 = cvRect(0, 0, pageSize.width - shift, pageSize.height)
//          val r1 = cvRect(shift, 0, pageSize.width - shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        } else {
//          val r0 = cvRect(-shift, 0, pageSize.width + shift, pageSize.height)
//          val r1 = cvRect(0, 0, pageSize.width + shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        }
//      }
//
//      if (guiOn) {
//        val title = "Preview"
//        val preview = cvCreateImage(cvSize(pageSize.width / 4, pageSize.height / 4), IPL_DEPTH_8U, 3)
//        cvResize(result, preview, INTER_LINEAR)
//        cvShowImage(title, preview)
//        cvResizeWindow(title, preview.arrayWidth(), preview.arrayHeight())
//        cvNamedWindow(title)
//        cvWaitKey()
//        cvReleaseImage(preview)
//      }
//
//      if (outputOn) {
//        cvSaveImage(s"output/${file.getName}", result)
//      }
//
//      //cvReleaseImage(i0)
//      cvReleaseImage(i1)
//      cvReleaseImage(i2)
//      cvReleaseImage(result)
//      cvReleaseImage(debug)
//    }
//
//    for ((file, i) <- new File("C:\\ee_bmp\\index\\0").listFiles.zipWithIndex if file.isFile) {
//      println("-" * 20)
//      println(file.getName)
//
//      val image = ImageIO.read(file)
//      val i0 = bufferedImageToIplImage(image)
//      val pageSize = cvGetSize(i0)
//      val result = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val debug = cvCreateImage(pageSize, IPL_DEPTH_8U, 3)
//      val i1 = cvCreateImage(pageSize, IPL_DEPTH_32F, 3)
//      val i2 = cvCreateImage(pageSize, IPL_DEPTH_32F, 1)
//
//      // up-sampling 8bit integer to 32bit floating-point
//      cvConvertScale(i0, i1, 1.0 / 255, 0)
//      // convert color to gray-scale
//      cvCvtColor(i1, i2, CV_BGR2GRAY)
//
//      if (debugOn) {
//        cvCopy(i0, debug)
//      }
//
//      val (fpx, lpx) = scanPage(i2, debug)
//      val shift = - (fpx - (pageSize.width - lpx)) / 2
//      val BOLAdjusted = fpx + shift
//      println("fpx", fpx)
//      println("lpx", lpx)
//      println("shift", shift)
//      println("BOL Adjusted", BOLAdjusted)
//
//      if (debugOn) {
//        cvPutText(debug, "fpx", cvPoint(fpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(fpx, 0), cvPoint(fpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "lpx", cvPoint(lpx, 50), FONT, BLUE)
//        cvLine(debug, cvPoint(lpx, 0), cvPoint(lpx, pageSize.height), BLUE, 2, CV_AA, 0)
//        cvPutText(debug, "BOLAdjusted", cvPoint(BOLAdjusted, 50), FONT, GREEN)
//        cvLine(debug, cvPoint(BOLAdjusted, 0), cvPoint(BOLAdjusted, pageSize.height), GREEN, 2, CV_AA, 0)
//        cvCopy(debug, result)
//      } else {
//        cvSet(result, WHITE)
//        if (shift == 0) {
//          cvCopy(i0, result)
//        } else if (shift > 0) {
//          val r0 = cvRect(0, 0, pageSize.width - shift, pageSize.height)
//          val r1 = cvRect(shift, 0, pageSize.width - shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        } else {
//          val r0 = cvRect(shift, 0, pageSize.width - shift, pageSize.height)
//          val r1 = cvRect(0, 0, pageSize.width - shift, pageSize.height)
//          cvSetImageROI(i0, r0)
//          cvSetImageROI(result, r1)
//          cvCopy(i0, result)
//          cvResetImageROI(i0)
//          cvResetImageROI(result)
//        }
//      }
//
//      if (guiOn) {
//        val title = "Preview"
//        val preview = cvCreateImage(cvSize(pageSize.width / 4, pageSize.height / 4), IPL_DEPTH_8U, 3)
//        cvResize(result, preview, INTER_LINEAR)
//        cvShowImage(title, preview)
//        cvResizeWindow(title, preview.arrayWidth(), preview.arrayHeight())
//        cvNamedWindow(title)
//        cvWaitKey()
//        cvReleaseImage(preview)
//      }
//
//      if (outputOn) {
//        cvSaveImage(s"output/${file.getName}", result)
//      }
//
//      //cvReleaseImage(i0)
//      cvReleaseImage(i1)
//      cvReleaseImage(i2)
//      cvReleaseImage(result)
//      cvReleaseImage(debug)
//    }
  }
}