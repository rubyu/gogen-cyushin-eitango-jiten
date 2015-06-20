
import AssemblyKeys._

organization := "com.github.rubyu"

name := "ee"

version := "0.0.0"

scalaVersion := "2.10.4"

classpathTypes <<= classpathTypes(_ + "maven-plugin")

libraryDependencies ++= Seq(
    "org.bytedeco" % "javacv" % "0.11",
    "org.bytedeco" % "javacpp" % "0.11",
    "org.specs2" % "specs2_2.10" % "2.3.12" % "test"
  )

assemblySettings
