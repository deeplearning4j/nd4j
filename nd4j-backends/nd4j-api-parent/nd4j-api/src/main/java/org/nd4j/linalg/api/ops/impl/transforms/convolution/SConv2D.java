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
public class SConv2D extends Conv2D {

    @Builder(builderMethodName = "sameDiffBuilder")
    public SConv2D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(sameDiff, inputs, inPlace, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    @Builder(builderMethodName = "execBuilder")
    public SConv2D(INDArray[] inputs, INDArray[] outputs, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(inputs,outputs, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    public SConv2D() {}

    @Override
    public String opName() {
        return "sconv2d";
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
