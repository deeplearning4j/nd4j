---
layout: default
title: "Full ND4J Installation"
description: "Install a Java development environment ready for N-Dimensional Array Algebra with ND4J"
---


This is a multistep install. We highly recommend you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback, so we can walk you through it. If you're feeling anti-social or brashly independent, you're still welcome to lurk and learn.

To get started with ND4J and DL4J, please read the following:

1. [Prerequisites](#prereq)
3. [Integrated Development Environment](#ide)
4. [New ND4J Project](#nd4j)
5. [Dev Tools](#devtools)
6. [GPUs](#gpu)
7. [Next Steps](#next-steps)

ND4J is an open-source project targetting professional Java developers familiar with production deployments, an IDE like IntelliJ and an automated build tool such as Apache Maven. Our tool will serve you best if you have those tools under your belt already.

## <a id="prereq"> Prerequisites </a>

System configuration requirements:

* [Java Development Kit 1.7 or later](#java)
* [Apache Maven 3.3.9 or later](#maven)

Above  0.7.2:

* [JavaCPP](#javacpp)
* [BLAS (MKL or OpenBLAS)](#blas)

Optional:

* [Cuda 7 for GPUs](http://docs.nvidia.com/cuda/index.html#axzz3dlfIdQjP)
* [Scala 2.10.x](#scala)
* [Windows](#windows)
* [Github](#github)

## JDK and Maven

For help installing the Java Development Kit or Maven, see the [DL4J quickstart](http://deeplearning4j.org/quickstart#Java).

## <a id="javacpp">JavaCPP</a>

[JavaCPP](https://github.com/bytedeco/javacpp) provides efficient access to native C++ inside Java.

## <a id="blas">BLAS</a>

BLAS is used as a backend for libnd4j computations. You can choose between [MKL](https://software.intel.com/en-us/mkl), or [OpenBLAS](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide). Note, if you use OpenBLAS check fortran requirements and make sure to configure for the number of cores on your machine.

## <a id="ide">Integrated Development Environment: IntelliJ</a>

An Integrated Development Environment ([IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)) will allow you to work with our API and build your nets with a few clicks. We suggest using **IntelliJ**, which works with your installed version of Java and communicates with [Maven](#maven) to handle the dependencies.

The free community edition of [IntelliJ](https://www.jetbrains.com/idea/download/) has installation instructions. While we prefer that, [Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html) and [Netbeans](http://wiki.netbeans.org/MavenBestPractices) are two other popular IDEs. Here is a guide to installing the [ND4J/DL4J package on Eclipse](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/).

## <a id="nd4j">Starting a New ND4J Project</a>

To create a new ND4J project within IntelliJ, either click on "Open Project" on IntelliJ's opening screen, or click on the  File/Open tab, and choose "nd4j." If you have cloned the source files from Github, the directory should be available from IntelliJ.

To create a new ND4J project within IntelliJ, just put the right dependencies in your project's POM.xml file. With those in place, Maven will be able to build ND4J for you. Pasting the right dependencies into your POM amounts to installing ND4J -- no other install is necessary.

Select `maven-archetype-quickstart`.

![Alt text](../img/new_maven_project.png)

The images below will walk you through the windows of the IntelliJ New Project Wizard using Maven. First, name your group and artifact as you please.

![Alt text](../img/maven2.png)

Click through the following screen with "Next", and on the screen after that, name your project ("ND4J-test", for example) and hit finish. Now go into your POM.xml file within the root of the new ND4J project in IntelliJ.

Update the POM file with the dependences you'll need. These will vary depending on whether you're running on CPUs or GPUs.

The default backend for CPUs is `nd4j-native-platform`, and for CUDA it is `nd4j-cuda-8.0-platform`. You can paste that into  the `<dependencies> ... </dependencies>` section of your POM like this:
```xml
<dependency>
 <groupId>org.nd4j</groupId>
 <artifactId>nd4j-native-platform</artifactId>
 <version>${nd4j.version}</version>
</dependency>
```
ND4J's version is a variable here. It will refer to another line higher in the POM, in the `<properties> ... </properties>` section, specifying the nd4j version and appearing similar to this:
```xml
<nd4j.version>1.0.0-alpha</nd4j.version>
```
*The dl4j version and DataVec version are also 1.0.0-alpha.*

Version `0.4.0` or higher now includes all backends by default and binaries for all platforms are automatically pulled. It is recommended to not alter this behavior *especially* if you are building on one platform but deploying to another (OS X vs. Linux). However, you can also explicitly pull binaries only for the platforms you are using. Information on how to do this can be found on the [dependencies](./dependencies) page.

Further, additional but optional binaries targeting processors with AVX2 or AVX512 instructions are also available since version 1.0.0-SNAPSHOT. You just need to add them to your dependencies and ND4J will automatically pick them up, as below for AVX2 (Linux, Mac, and Windows), and similarly for AVX512 (Linux and Mac only).
```xml
 <dependency>
   <groupId>org.nd4j</groupId>
   <artifactId>nd4j-native</artifactId>
   <version>${nd4j.version}</version>
   <classifier>linux-x86_64-avx2</classifier>
 </dependency>
 <dependency>
   <groupId>org.nd4j</groupId>
   <artifactId>nd4j-native</artifactId>
   <version>${nd4j.version}</version>
   <classifier>macosx-x86_64-avx2</classifier>
 </dependency>
 <dependency>
   <groupId>org.nd4j</groupId>
   <artifactId>nd4j-native</artifactId>
   <version>${nd4j.version}</version>
   <classifier>windows-x86_64-avx2</classifier>
 </dependency>
```

### Other Build Systems

If you are using a build tool such as Gradle or sbt, the artifacts work in the similar ways. For example, inside a `build.gradle` file:

```groovy
repositories {
    mavenCentral()
}
dependencies {
    compile 'org.nd4j:nd4j-native-platform:1.0.0-alpha'
}
```

Note: if your Gradle project fails with an error like the following:

```groovy
Warning:<i><b>root project 'search-classifier': Unable to resolve additional project configuration.</b>
Details: org.gradle.api.artifacts.ResolveException: Could not resolve all dependencies for configuration ':naive-classifier:compile'.
Caused by: org.gradle.internal.resolve.ArtifactNotFoundException: Could not find opencv-linux-x86_64.jar (org.bytedeco.javacpp-presets:opencv:3.2.0-1.3).
Searched in the following locations:
    file:/Users/user/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-linux-x86_64.jar</i>
```

You can add the required dependencies as below. This is because Gradle has limited support for classifiers.

```groovy
compile 'org.bytedeco.javacpp-presets:opencv:3.2.0-1.3'
compile 'org.bytedeco.javacpp-presets:opencv:3.2.0-1.3:linux-x86_64'
compile 'org.bytedeco.javacpp-presets:openblas:0.2.20-1.3'
compile 'org.bytedeco.javacpp-presets:openblas:0.2.20-1.3:linux-x86_64'
```

Similarly, for sbt, we need to include something like the following inside `build.sbt`:

```scala
classpathTypes += "maven-plugin"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-alpha"
```

### Stay Up-to-date

The number of the version will vary as we progress with new releases. Make sure you check [the latest version available on Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cnd4j). If you paste in the right dependency and ND4J version, Maven will automatically install the required libraries and you should be able to run ND4J.

### Switching Backends

The backend does not have to be `nd4j-native`; it can be switched to Jcublas for GPUs. That's explained on our [dependencies](./dependencies) page, alongside more advanced configuration changes. The same page also explains how to check on the [latest version](http://search.maven.org/#search%7Cga%7C1%7Cnd4j) of the libraries.

### Your Main Class

You can now create a new Java file within IntelliJ, and start using ND4J's API for distributed linear algebra.

Open App.java file that is created with every new Intellij project, and start writing code between the curly brackets you see after **public static void main( String[] args )**.

Many of the classes will appear in red, since you haven't imported the right packages, but IntelliJ will add those packages automatically to the top of your file. Lookup the appropriate hot keys based on your OS to help automatically load the packages.

(See our [intro](./introduction) for a couple beginning operations. ND4J in IntelliJ has **autocomplete**, so starting a new line with any letter will show you a list of all ND4J commands including that letter.)

## <a id="github">GitHub & Source</a>

Github is a web-based [Revision Control System](http://en.wikipedia.org/wiki/Revision_control), the [de facto host](http://opensource.com/life/12/11/code-hosting-comparison) for open-source projects.

If you are not planning to contribute to ND4J as a committer, or don't need the latest alpha version, we recommend downloading the most recent stable release of ND4J from [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j). The JAR files can be downloaded directly from [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cnd4j). Otherwise, please see [source](./source).

## <a id="scala">Scala</a>

While Scala doesn't need to be installed with work with ND4J, we do have a [Scala API](./scala) under a repository known as [ND4S](https://github.com/deeplearning4j/nd4s).

Scala is a multiparadigm language with a strong static type system that runs on the JVM. As such, Scala has functional programming features similar to Scheme and Haskell as well as OOP features like Java, and its structure keeps programs concise. You can use Java libraries with Scala. There are neural net examples you can run written in Scala, and it's required for the Spark implementation.

To test which version of Scala you have (and whether you have it at all), type the following into your command line:

		scala -version

To install Scala, visit the [Scala download page](http://www.scala-lang.org/download/2.1.html). ND4J is compatible with Scala 2.10.4, and Scala is not backwards compatible. [Homebrew](http://brew.sh/) will help Mac users install Scala. `brew install scala` will get you the latest version, which is `2.11.x`. To install Scala 2.10.x with Homebrew, please see [this page](https://github.com/ofishel/hb-scala-2.10.4).

You can also work with Scala via an IntelliJ plugin. (To add a plugin to IntelliJ, go to the tab `IntelliJ IDEA`/`Preferences`/`IDE Setting`/`Plugins`/ and search for Scala.)

## <a id="datavec">DataVec</a>

[DataVec](https://github.com/deeplearning4j/DataVec) is a general vectorization lib we built for machine-learning tools. It vectorizes raw data into usable vector formats like *svmLight*, *libsvm* and *ARFF*, which our neural nets can work with. ND4J does not require DataVec, but it is useful for loading data into Deeplearning4j neural nets.

### Installing DataVec

Take the same steps using Maven to install [DataVec](https://github.com/deeplearning4j/DataVec) that you followed for ND4J. Make sure you have the most recent version of [Maven](#maven). Please see the [examples](https://github.com/deeplearning4j/dl4j-examples) for the most recent versions.

### Installing Deeplearning4j

Deeplearning4j versions should be specified in the same way you did for ND4J, with the version hard-coded in the properties section of the POM, and the version variable cited in each dependency.

The DL4J dependencies you add to the POM will vary with the nature of your project.

In addition to the core dependency, given below, you may also want to install `deeplearning4j-cli` for the command-line interface, `deeplearning4j-scaleout` for running parallel on Hadoop or Spark, and others as needed. <!--A full list can be seen by searching for *deeplearning4j* on Maven Central.-->
```xml
<dependency>
 <groupId>org.deeplearning4j</groupId>
 <artifactId>deeplearning4j-core</artifactId>
 <version>${deeplearning4j.version}</version>
</dependency>
```
More information on installing Deeplearning4j is available on its [Getting Started page](http://deeplearning4j.org/gettingstarted.html).

## <a id="gpu"> GPUs </a>

We support CUDA versions 7.5 and higher.

Once you begin training neural networks on GPUs, you will want to monitor whether and how well the GPUs are working. There are several measures you can take:

* Make sure you have [nvcc, the Nvidia compiler](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/), in your classpath (`src/main/resources`). We compile the kernels on the fly.
* Install the [Nvidia System Management Interface (SMI)](https://developer.nvidia.com/nvidia-system-management-interface). Look for `Java` in the output.

## <a id="next">Next Steps</a>

Now you're ready to run the [examples](introduction.html). We recommend that you launch your IDE, load the ND4J project and open the examples subdirectory. Locate an example in the file tree on the lefthand side of the IntelliJ window, right click on it, and select the green arrow for "Run" on the drop-down menu.

If everything was installed correctly, you should see numbers appear as the program output at the bottom of the IntelliJ window. Please use these as a sandbox to start experimenting.  

Once you're comfortable with the examples, you might want to change the dependencies defined in the POM files. Learn how to change the [dependencies here](gpu_native_backends.html).

## Useful Links

* [ND4J Maven Repository](http://mvnrepository.com/artifact/org.nd4j)
* [DeepLearning4j.org](http://deeplearning4j.org/)
* [DeepLearning4j Maven Repository](http://mvnrepository.com/artifact/org.deeplearning4j)
