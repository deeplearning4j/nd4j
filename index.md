---
layout: page
title: 
tagline: 
---
{% include JB/setup %}

# ND4J: N-Dimentional-arrays (ND) for Java (4J)

### N-Dimensional Linear Algebra, made *intuitive* and *fast*.

ND4J is a **scientific computing library** for **linear algebra and matrix manipulation** (see [examples](../elementwise.html)) on the JVM. It is meant to be used in **production environments** rather than as a research tool, which means that the routines have been carefully crafted to run **fast** and with **minimum RAM** requirements.

ND4J includes:

* **Versatile** ND-array object.
* **Multiplatform** functionality including GPUs.
* **Linear algebra** and **signal processing** functions.

A usability gap has long enough separated Java, Scala and Clojure programmers from the most powerful tools in data analysis, like [NumPy](http://www.numpy.org/) or [Matlab](http://www.mathworks.com/). With ND4J, which is like NumPy, but for Java instead of Python, scientific computing tools are now **open source**, **distributed**, and **integrated** with **GPUs**.

Think [SLF4J](http://www.slf4j.org/). Now imagine Numpy with [Theano](http://deeplearning.net/software/theano/) built into it. That's what you get with ND4J. It gives engineers in **production environments** an easy way to port their algorithms and interface with other libraries native to the Java ecosystem. 

#[Get started!](../getstarted.html)


### ND4J Specifics

* Supports GPUs via CUDA and Native via Jblas.
* All of this is wrapped in a unifying interface.
* The API is a mix of Numpy and Jblas.

### Simple code example

Creating a 2 x 2 NDarray:

           INDArray arr1 = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
           System.out.println(arr1);
           
Which will result in:

           [[1.0 ,3.0]
           [2.0 ,4.0]
           ]

Adding scalar with in-place operations:

           arr1.addi(1);
           System.out.println(arr1);

So each element is incremented by one:

           [[2.0 ,4.0]
           [3.0 ,5.0]
           ]

Creating a second array (_arr2_) and adding it to the first one (_arr1_):

           INDArray arr2 = ND4j.create(new float[]{5,6,7,8},new int[]{2,2});
           arr1.addi(arr2);
           System.out.println(arr1);

Which results in:

           [[7.0 ,11.0]
           [9.0 ,13.0]
           ]

#[Get started!](../getstarted.html)
