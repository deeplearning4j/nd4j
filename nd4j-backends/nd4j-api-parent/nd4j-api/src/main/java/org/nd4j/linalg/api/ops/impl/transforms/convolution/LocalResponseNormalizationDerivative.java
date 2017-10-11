package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class LocalResponseNormalizationDerivative extends LocalResponseNormalization {
    @Builder(builderMethodName = "sameDiffDerivativeBuilder")
    public LocalResponseNormalizationDerivative(SameDiff sameDiff, DifferentialFunction[] inputs, boolean inPlace, double alpha, double beta, double bias, double depth) {
        super(sameDiff, inputs, inPlace, alpha, beta, bias, depth);
    }

    @Builder(builderMethodName = "execDerivativeBuilder")
    public LocalResponseNormalizationDerivative(INDArray[] inputs, INDArray[] outputs, double alpha, double beta, double bias, double depth) {
        super(inputs, outputs, alpha, beta, bias, depth);
    }

    public LocalResponseNormalizationDerivative() {}

    @Override
    public String opName() {
        return "lrn_bp";
    }


    @Override
    public ArrayField doGetValue() {
        return null;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

}
