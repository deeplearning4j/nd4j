ND4J: Scientific Computing on the JVM (Forked from [deeplearning4j/libnd4j
](https://github.com/deeplearning4j/libnd4j) to support Raspberry PI)
===========================================
#This is not the [orignal](https://github.com/deeplearning4j/nd4j) repository.

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.nd4j/nd4j/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.nd4j/nd4j)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.nd4j/nd4j/badge.svg)](http://nd4j.org/doc)

ND4J is an Apache2 Licensed open-sourced scientific computing library for the JVM. It is meant to be used in production environments
rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements.



This repo is forked for raspberry pi support. Here is the method to follow :
	1. {In build machine machine}compile libnd4j
				git clone https://github.com/dschowta/libnd4j.git
			- For cross compilation use this link:http://stackoverflow.com/questions/19162072/installing-raspberry-pi-cross-compiler
			- Use the 4.9 version of gcc (raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++)
		3.{In build machine}  build using "./buildnativeoperations.sh -o rp2"
	
		4. {In build machine} Install maven 3.3.9 (3.0.X does not work)
		5. {In build machine} followed instructions mentioned in https://deeplearning4j.org/buildinglocally :
				export LIBND4J_HOME=<pathTond4JNI>
				
				# build and install nd4j to maven locally (using the forked nd4j specifically changed for raspberry pi)
				git clone https://github.com/dschowta/nd4j.git
				cd nd4j
				mvn clean  install -DskipTests -Dplatform=linux-arm -Dmaven.javadoc.skip=true -P linux,arm -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests
		6. {In build machine } Copy following jars from the build machine :
			mkdir myjarFolder &&  \
			cp nd4j-backends/nd4j-backend-impls/nd4j-native/target/nd4j-native-0.6.1-pi.jar\
			nd4j-backends/nd4j-backend-impls/nd4j-native/target/nd4j-native-0.6.1-pi-linux-arm.jar \
			nd4j-backends/nd4j-backend-impls/nd4j-native-platform/target/nd4j-native-platform-0.6.1-pi.jar\
			nd4j-backends/nd4j-api-parent/nd4j-api/target/nd4j-api-0.6.1-pi.jar\
			nd4j-backends/nd4j-api-parent/nd4j-native-api/target/nd4j-native-api-0.6.1-pi.jar \
			myjarFolder/
		7. {In build machine }install these jars in maven using :
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-native -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-0.6.1-pi-linux-arm.jar -DgroupId=org.nd4j -DartifactId=nd4j-native -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true -Dclassifier=linux-arm
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-api-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-api -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-platform-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-native-platform -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			mvn install:install-file -Dfile=<path to myjarFolder>\nd4j-native-api-0.6.1-pi.jar -DgroupId=org.nd4j -DartifactId=nd4j-native-api -Dversion=0.6.1-pi -Dpackaging=jar -DgeneratePom=true
			
		8. {In build machine } build the source of dependant appllication with above (step 7) dependencies.
		9. {In raspbian }copy the generated jar of dependant application to raspberry
		10. {In raspbian }download the libraries inside the folders of build machine nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-arm/  to a permanent folder (if possible to a system folder)
		11.{In raspbian }export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<the path to libraryy>
		12. {In raspbian }java -jar myjar.jar