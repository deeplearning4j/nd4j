---
layout: page
title: "ND4J: Java를 위한 N 차원 배열 (N-Dimensional Arrays)"
description: 
---
{% include JB/setup %}

ND4J와 ND4S는 JVM을 위한 과학 컴퓨팅 라이브러리 입니다. 이들은 생산 환경에서 사용되고자 만들어졌으며, 이는 최소 RAM 요건으로 routines이 빠르개 실행되도록 설계 되었슴을 의미합니다.

### 주요 특징:

* 다목적 N 차원 어레이 개체
* GPUs를 포함한 멀티플랫폼 기능
* 선형 대수학 및 신호 처리 기능

유저빌리티 (usability) 차이가 NumPy 또는 Matlab과 같은 데이터 분석에서 가장 강력한 도구들로부터 Java, [Scala](http://nd4j.org/scala.html) 및 Clojure 프로그래머들을 분류해 왔습니다. Breeze와 같은 라이브러리는 딥 러닝 및 기타 작업에 필요한 N 차원 어레이, 또는 tensors를 지원하지 않습니다. ND4J 및 ND4S 는 연산적으로 집약적인 시뮬레이션을 필요로 하는, 기후 모델링과 같은 작업들을 위한 국립 라이브러리들에 의해 사용됩니다.

ND4J는 Python 커뮤니티의 직관적으로 과학적인 컴퓨팅 도구들을 오픈 소스로 배포된 GPU 기반의 라이브러리에서의 JVM에게 제공합니다. 구조적으로는 [SLF4J](http://www.slf4j.org)와 유사합니다. ND4J는 생산 환경에 있는 엔지니어들에게 그들의 알고리즘을 포트하고 자바와 스칼라 에코시스템에 있는 다른 라이브러리들과 인터페이스 하는 쉬운 방법을 제공합니다.

[시작하시려면 여기를 클릭하시거나](http://nd4j.org/kr-getstarted.html) 계속 읽어주십시오.

### ND4J/ND4S 세부사항

* CUDA를 통해 GPUs를, Jblas와 Netlib Blas를 통해 Native를 지원합니다.
* 앤드로이드에서 구축합니다.
* 하둡(Hadoop) 및 스파크(Spark)와 통합합니다.
* [ND4S의 API](https://github.com/deeplearning4j/nd4s)는 Numpy의 semantics을 모방합니다.

### 코드 예제

2 × 2 NDarray를 생성하십시오:

		INDArray arr1 = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
		System.out.println(arr1);

### 산출(Output)

		[[1.0 ,3.0]
		[2.0 ,4.0]
		]

인플레이스 오퍼레이션(in-place operations)와 함께 scalar를 추가:

    arr1.addi(1);
    System.out.println(arr1);

각 요소는 하나씩 증가 합니다.

    [[2.0 ,4.0]
    [3.0 ,5.0]
    ]

두번째 어레이 (arr2)를 생성해 첫번째 어레이 (arr1)에 추가 하십시오:

    INDArray arr2 = ND4j.create(new float[]{5,6,7,8},new int[]{2,2});
    arr1.addi(arr2);
    System.out.println(arr1);

### 산출:

    [[7.0 ,11.0]
    [9.0 ,13.0]
    ]

모두 이해 하셨습니다. [이제 시작하도록 하겠습니다](http://nd4j.org/kr-getstarted.html).
