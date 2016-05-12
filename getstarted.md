---
layout: page
title: "Full ND4J Installation"
description: "Install a Java development environment ready for N-Dimensional Array Algebra with ND4J"
---
{% include JB/setup %}

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

Above  0.4-rc3.8:

* [JavaCPP](#javacpp)
* [BLAS (ATLAS, MKL, or OpenBLAS)](#blas) 

Optional:

* [Cuda 7 for GPUs](http://docs.nvidia.com/cuda/index.html#axzz3dlfIdQjP)
* [Scala 2.10.x](#scala)
* [Windows](#windows)
* [Github](#github) 

## JDK and Maven

For help installing the Java Development Kit or Maven, see the [DL4J quickstart](http://deeplearning4j.org/quickstart#Java).

## <a id="javacpp">JavaCPP</a>

[JavaCPP](https://github.com/bytedeco/javacpp) provides efficient access to native C++ inside Java. Git clone and install when using version above 0.4-rc3.8.

## <a id="blas">BLAS</a>

BLAS is used as a backend for libnd4j computations. You can choose between ATAL, MKL, or [OpenBLAS](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide). Note, if you use OpenBLAS check fortran requirements and make sure to configure for the number of cores on your machine.

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

The default backend for CPUs is x86, which relies heavily on Netlib-blas. You can paste that into  the `<dependencies> ... </dependencies>` section of your POM like this:

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-x86</artifactId>
	   <version>${nd4j.version}</version>
	 </dependency>

ND4J's version is a variable here. It will refer to another line higher in the POM, in the `<properties> ... </properties>` section, specifying the nd4j version and appearing similar to this:

		<nd4j.version>0.4-rc3.8</nd4j.version>

*The dl4j version is also 0.4-rc3.8, and Canova is 0.0.0.13.*

### Platform-specific binaries

Version `0.4-rc3.9` or higher now includes all backends by default. However, if you are using a build tool such as SBT or Gradle, you will need to explicitly pull binaries for the platform you are using. *Especially* if you are building on one platform but deploying to another (OS X vs. Linux). Information on how to do this can be found on the [dependencies](../dependencies.html) page.

For example, a `build.gradle` file will include an extra definition for the targeted platform:

```groovy
dependencies {
  compile 'org.nd4j:nd4j-native:0.4-rc3.9-SNAPSHOT'
  compile 'org.nd4j:nd4j-native:0.4-rc3.9-SNAPSHOT:macosx-x86_64'
}
```

Similarly, for sbt, we can include something like the following:

```scala
classpathTypes += "maven-plugin"

libraryDependencies += "org.nd4j" % "nd4j-native" % "0.4-rc3.9-SNAPSHOT" classifier "" classifier "linux-x86_64"
```

### Stay Up-to-date

The number of the version will vary as we progress with new releases. Make sure you check [the latest version available on Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cnd4j). If you paste in the right dependency and nd4j version, Maven will automatically install the required libraries and you should be able to run ND4J. 

### Switching Backends

The backend does not have to be x86; it can be switched to Jcublas for GPUs. That's explained on our [dependencies](../dependencies.html) page, alongside more advanced configuration changes. The same page also explains how to check on the [latest version](http://search.maven.org/#search%7Cga%7C1%7Cnd4j) of the libraries.

### Your Main Class

You can now create a new Java file within IntelliJ, and start using ND4J's API for distributed linear algebra. 

Open App.java file that is created with every new Intellij project, and start writing code between the curly brackets you see after **public static void main( String[] args )**. 

Many of the classes will appear in red, since you haven't imported the right packages, but IntelliJ will add those packages automatically to the top of your file. Lookup the appropriate hot keys based on your OS to help automatically load the packages. 

(See our [intro](http://nd4j.org/introduction.html) for a couple beginning operations. ND4J in IntelliJ has **autocomplete**, so starting a new line with any letter will show you a list of all ND4J commands including that letter.) 

## <a id="github">GitHub & Source</a>

Github is a web-based [Revision Control System](http://en.wikipedia.org/wiki/Revision_control), the [de facto host](http://opensource.com/life/12/11/code-hosting-comparison) for open-source projects. 

If you are not planning to contribute to ND4J as a committer, or don't need the latest alpha version, we recommend downloading the most recent stable release of ND4J from [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j), 0.4-rc1.x. The JAR files can be downloaded directly from [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cnd4j). Otherwise, please see [source](http://nd4j.org/source). 

## <a id="scala">Scala</a>

While Scala doesn't need to be installed with work with ND4J, we do have a [Scala API](http://nd4j.org/scala.html) under a repository known as [ND4S](https://github.com/deeplearning4j/nd4s). 

Scala is a multiparadigm language with a strong static type system that runs on the JVM. As such, Scala has functional programming features similar to Scheme and Haskell as well as OOP features like Java, and its structure keeps programs concise. You can use Java libraries with Scala. There are neural net examples you can run written in Scala, and it's required for the Spark implementation.

To test which version of Scala you have (and whether you have it at all), type the following into your command line:

		scala -version

To install Scala, visit the [Scala download page](http://www.scala-lang.org/download/2.10.4.html). ND4J is compatible with Scala 2.10.4, and Scala is not backwards compatible. [Homebrew](http://brew.sh/) will help Mac users install Scala. `brew install scala` will get you the latest version, which is `2.11.x`. To install Scala 2.10.x with Homebrew, please see [this page](https://github.com/ofishel/hb-scala-2.10.4). 

You can also work with Scala via an IntelliJ plugin. (To add a plugin to IntelliJ, go to the tab `IntelliJ IDEA`/`Preferences`/`IDE Setting`/`Plugins`/ and search for Scala.)

## <a id="canova">Canova</a>

[Canova](https://github.com/deeplearning4j/Canova) is a general vectorization lib we built for machine-learning tools. It vectorizes raw data into usable vector formats like *svmLight*, *libsvm* and *ARFF*, which our neural nets can work with. ND4J does not require Canova, but it is useful for loading data into Deeplearning4j neural nets. 

### Installing Canova

Take the same steps using Maven to install [Canova](https://github.com/deeplearning4j/Canova) that you followed for ND4J. Make sure you have the most recent version of [Maven](#maven). We are currently using version 0.0.0.5-SNAPSHOT.

### Installing Deeplearning4j

Deeplearning4j versions should be specified in the same way you did for ND4J, with the version hard-coded in the properties section of the POM, and the version variable cited in each dependency. 

The DL4J dependencies you add to the POM will vary with the nature of your project. 

In addition to the core dependency, given below, you may also want to install `deeplearning4j-cli` for the command-line interface, `deeplearning4j-scaleout` for running parallel on Hadoop or Spark, and others as needed. <!--A full list can be seen by searching for *deeplearning4j* on Maven Central.-->

		   <dependency>
		     <groupId>org.deeplearning4j</groupId>
		     <artifactId>deeplearning4j-core</artifactId>
		     <version>${deeplearning4j.version}</version>
		   </dependency>

More information on installing Deeplearning4j is available on its [Getting Started page](http://deeplearning4j.org/gettingstarted.html).

## <a id="devtools">Dev Tools for C on OSX, Windows & Linux</a>

To compile certain ND4J dependencies on Windows or a Linux OS, you will need to install some dev tools for C, including gcc. To check if you have *gcc*, enter `gcc -v` on your terminal or command prompt.

### OSX

Some versions of the [Apple developer tool Xcode](https://developer.apple.com/xcode/downloads/) will install *gcc* for you. If you don't already have gcc, enter `brew install gcc` into your command prompt.

### <a id="open"> OpenBlas </a>

To make sure the native libs on the x86 backend work, you need `/opt/OpenBLAS/lib` on the system path. After that, enter these commands in the prompt

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

We added this so that [Spark](http://deeplearning4j.org/spark) would work with OpenBlas.

If OpenBlas is not working correctly, follow these steps:

* Remove Openblas if you installed it.
* Run `sudo apt-get remove libopenblas-base`
* Download the development version of OpenBLAS
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* With **Linux**, double check if the symlinks for `libblas.so.3` and `liblapack.so.3` are present anywhere in your `LD_LIBRARY_PATH`. If they aren't, add the links to `/usr/lib`. A symlink is a "symbolic link." You can set it up like this (the -s makes the link symbolic):

		ln -s TARGET LINK_NAME
		// interpretation: ln -s "to-here" <- "from-here"

* The "from-here" is the symbolic link that does not exist yet, and which you are creating. Here's StackOverflow on [how to create a symlink](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux). And here's the [Linux man page](http://linux.die.net/man/1/ln).
* As a last step, restart your IDE. 
* For complete instructions on how to get native Blas running with Centos 6, [see this page](https://gist.github.com/jarutis/912e2a4693accee42a94).

For OpenBlas on **Ubuntu** (15.10), please see [these instructions](http://pastebin.com/F0Rv2uEk). Some additional gymnastics may be necessary to get OpenBlas running:

		cd OpenBLAS
		make FC=gfortran
		sudo make PREFIX=/usr/local/ install
		
		sudo ln -s libopenblas.so libblas.so.3 
		sudo ln -s libopenblas.so liblapack.so.3

For OpenBlas on **Windows**, download [this file](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1). Extract it to a location such as `C:/BLAS`. Finally, add that directory to your system's `PATH` environment variable.

### <a id="windows"> Windows </a>

Windows users may need to install [Visual Studio Community 2013 or later](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx), which is free. You will need to add its path to your PATH environment variable manually. The path will look something like this: `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin`

Type `cl` into your CMD. You may get a message informing you that certain `.dll` files are missing. Make sure that your VS/IDE folder is in the path (see above). If your CMD returns usage info for `cl`, then it's in the right place. 

If you use Visual Studio: 

* Set up `PATH` environment variable to point to `\bin\` (for `cl.exe` etc)
* Also try running `vcvars32.bat` (also in bin) to set up environment before doing `mvn clean install` on ND4J (it may save you from copying headers around)
* `vcvars32` may be temporary, so you might need to run it every time you want to do ND4J `mvn install`.
* After installing Visual Studio 2015 and setting the PATH variable, you need to run the `vcvars32.bat` to set up the environment variables (INCLUDE, LIB, LIBPATH) properly so that you don't have to copy header files. But if you run the bat file from Explorer, since the settings are temporary, they're not properly set. So run `vcvars32.bat` from the same CMD window as your `mvn install`, and all the environment variables will be set correctly.
* Here is how they should be set: 

		INCLUDE = C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include
		LIB = "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib"
		//so you can link to .lib files^^
* In Visual Studio, you also have to click on C++. It is no longer set by default. 
(*In addition, the include path for [Java CPP](https://github.com/bytedeco/javacpp) doesn't always work on Windows. One workaround is to take the the header files from the include directory of Visual Studio, and put them in the include directory of the Java Run-Time Environment (JRE), where Java is installed. This will affect files such as `standardio.h`.*)
* For a walkthrough of installing our examples with Git, IntelliJ and Maven, please see our [Quickstart page](http://deeplearning4j.org/quickstart.html#walk). 
* [This page](http://avulanov.blogspot.cz/2014/09/howto-to-run-netlib-javabreeze-in.html) describes how to obtain dll for the Windows 64 platform. 
* Download dll libraries and place them in the Java bin folder (e.g. `C:\prg\Java\jdk1.7.0_45\bin`).
* Library `netlib-native_system-win-x86_64.dll` depends on: 
`libgcc_s_seh-1.dll
libgfortran-3.dll
libquadmath-0.dll
libwinpthread-1.dll
libblas3.dll
liblapack3.dll`
* (`liblapack3.dll` and `libblas3.dll` are just renamed copies of `libopeblas.dll`)
* You can download compiled libs from [here](http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Automated%20Builds/), [here](http://www.openblas.net/), or [here](http://search.maven.org/#search%7Cga%7C1%7Ca%3A%22netlib-native_system-win-x86_64%22)

### Linux

With Linux, Ubuntu and Centos users will need to follow two separate sets of instructions:

### Ubuntu

For Ubuntu, first type:

		sudo apt-get update

Then you'll need to run a version of this command:

		sudo apt-get install linux-headers-$(uname -r) build-essential

`$(uname -r)` will vary according to your Linux version. To get your Linux version, open a new window of your terminal and enter this command:

		uname -r

You will see something like this -- `3.2.0-23-generic`. Whatever you see, copy and paste it into the first line of script in place of `$(uname -r)`. Then insert one space and type `build-essential`. Watch out for typos. You can press tab to complete any command. 

### Centos

Enter the following in your terminal (or ssh session) as a root user:

		yum groupinstall 'Development Tools'

After that, you should see a lot of activity and installs on the terminal. To verify that you have, for example, *gcc*, enter this line:

		gcc --version

For more complete instructions, [go here](http://www.cyberciti.biz/faq/centos-linux-install-gcc-c-c-compiler/). 

## <a id="gpu"> GPUs </a>

Instructions on adding a [Jcublas backend are here](../gpu_native_backends.html). We support CUDA versions 5.5, 6.0, 6.5 and 7.

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
