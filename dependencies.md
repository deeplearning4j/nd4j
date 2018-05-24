---
layout: default
title: "Dependencies"
description: "Dependencies - How to change backends"
---


## Platform Specific Binaries

Valid for version `0.9.0` and higher. (As of this writing, we're on `1.0.0-beta`.)

Certain build tools such as [Gradle](http://www.gradle.org) and [SBT](http://www.scala-sbt.org/) cannot resolve transitive dependencies for specific platforms. When using a build tool such as Gradle, you will need to either explicitly state the platform binary you need at runtime or create a command line parameter that will specify your required platform. Creating command line parameters will allow you to switch between multiple platforms, such as testing on OS X and submitting to an Apache Spark cluster using a Linux operating system.

### Explicitly Requiring Binaries (Gradle)

Add the following to your dependencies in `build.gradle`:

```groovy
dependencies {
  compile 'org.nd4j:nd4j-native:1.0.0-beta'
  compile 'org.nd4j:nd4j-native:1.0.0-beta:macosx-x86_64'
  compile 'org.bytedeco.javacpp-presets:openblas:0.2.20-1.4.1'
  compile 'org.bytedeco.javacpp-presets:openblas:0.2.20-1.4.1:macosx-x86_64'
}
```

### Explicitly Requiring Binaries (sbt)

Add the following to your dependencies in `build.sbt`:

```scala
classpathTypes += "maven-plugin"

libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-beta" classifier "" classifier "linux-x86_64"
libraryDependencies += "org.bytedeco.javacpp-presets" % "openblas" % "0.2.20-1.4.1" classifier "" classifier "linux-x86_64"
```

### Command Line Option (Gradle)

Add the following to your `build.gradle`:

```groovy
switch(libnd4jOS) {
  case 'windows':
    libnd4jOS = 'windows-x86_64'
    break
  case 'linux':
    libnd4jOS = 'linux-x86_64'
    break
  case 'linux-ppc64':
    libnd4jOS = 'linux-ppc64'
    break
  case 'linux-ppc64le':
    libnd4jOS = 'linux-ppc64le'
    break
  case 'macosx':
    libnd4jOS = 'macosx-x86_64'
    break
  default:
    throw new Exception('Unknown OS defined for -Plibnd4jOS parameter. ND4J will be unable to find platform-specific binaries and thus unable to run.')
}

dependencies {
  compile 'org.nd4j:nd4j-native:1.0.0-beta'
  compile 'org.nd4j:nd4j-native:1.0.0-beta:'+libnd4jOS
  compile 'org.bytedeco.javacpp-presets:openblas:0.2.20-1.4.1'
  compile 'org.bytedeco.javacpp-presets:openblas:0.2.20-1.4.1:'+libnd4jOS
}
```

Finally, when running a Gradle command, add the parameter via the `-P` flag:

```
gradle run `-Plibnd4jOS=macosx`
```

## Configuring the POM.xml file

Maven can automatically install the required dependencies once we select one of these backends:

* [nd4j-native]
* [nd4j-cuda-7.5](./gpu_native_backends) (for GPUs)

Go to your root directory -- e.g. nd4j or deeplearning4j -- and inspect the [pom.xml file](https://maven.apache.org/pom.html). You should see one backend defined in the `<dependencies> ... </dependencies>` section. You can switch among:

### native

After version `4.0-RC3.8`, you can now include nd4j-native for all platforms.

```xml
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```
### CUDA (GPUs)

See our [GPU page](./gpu_native_backends) for the versions you can choose.

```xml
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-7.5</artifactId>
    <version>${nd4j.version}</version>
</dependency>
```

### JOCL

JOCL is a WIP. Please see the [source code here](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-jocl-parent).

## Finding and specifying the latest version of the libraries (Advanced)

They can be found on: [search.maven.org](http://search.maven.org/#search%7Cga%7C1%7Cnd4j). Click on the "latest version" on this screen. From there, you want to copy the dependency information:

![Alt text](../img/nd4j_maven.png)

And paste it into the "dependencies" section of your pom.xml, which should end up looking like this in IntelliJ:

![Alt text](../img/nd4j_pom_after.png)
