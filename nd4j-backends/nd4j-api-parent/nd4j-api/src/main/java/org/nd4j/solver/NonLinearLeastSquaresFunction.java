package org.nd4j.solver;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface NonLinearLeastSquaresFunction {

    int n();

    int numParams();

    INDArray x();



    INDArray score(INDArray input,INDArray...params);
}
