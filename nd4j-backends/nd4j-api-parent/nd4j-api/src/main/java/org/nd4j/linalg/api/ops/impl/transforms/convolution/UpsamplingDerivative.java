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
public class UpsamplingDerivative extends Upsampling {

    public UpsamplingDerivative() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    public UpsamplingDerivative(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int scaleFactor) {
        super(sameDiff, inputs, inPlace, scaleFactor);
    }

    @Builder(builderMethodName = "execBuilder")
    public UpsamplingDerivative(INDArray[] inputs, INDArray[] outputs, int scaleFactor) {
        super(inputs,outputs, scaleFactor);
    }


    @Override
    public String opName() {
        return "upsampling_bp";
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
