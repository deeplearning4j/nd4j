package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;

import java.util.List;


/**
 * DeConv2DDerivative operation
 */
@Slf4j
public class DeConv2DDerivative extends DeConv2D {

    public DeConv2DDerivative() {}

    @Builder(builderMethodName = "derivativeBuilder")
    public DeConv2DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs, INDArray[] inputArrays, INDArray[] outputs, boolean inPlace, DeConv2DConfig config) {
        super(sameDiff, inputs, inputArrays, outputs, inPlace, config);
    }

    @Override
    public String opName() {
        return "deconv2d_bp";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");

    }

}
