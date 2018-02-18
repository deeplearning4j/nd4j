package org.nd4j.solver;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NonLinearLeastSquares {
    private NonLinearLeastSquaresFunction function;
    private int n;
    private int numParams;
    private INDArray sigma;
    private boolean absoluteSigma;
    private boolean checkFinite;
    private INDArray bounds;
    private OptimizationMethod optimizationMethod;
    private JacobianMethod jacobianMethod;
    //results
    private INDArray output,covariance;

    public enum JacobianMethod {
        LM,TRF,DOGBOX,CUSTOM
    }

    public enum OptimizationMethod {
        LM,TRF,DOGBOX
    }


    @Builder
    public NonLinearLeastSquares(NonLinearLeastSquaresFunction function) {
        this.n = function.n();
        this.numParams = function.numParams();

    }

    public void invoke() {

    }


}
