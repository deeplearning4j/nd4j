package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;


/**
 * DeConv2DDerivative operation
 */
@Slf4j
public class DeConv2DDerivative extends DeConv2D {

    public DeConv2DDerivative() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    public DeConv2DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, boolean isSameMode) {
        super(sameDiff, inputs, inPlace, kY, kX, sY, sX, pY, pX, dY, dX, isSameMode);
    }

    @Builder(builderMethodName = "execBuilder")
    public DeConv2DDerivative(INDArray[] inputs, INDArray[] outputs, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, boolean isSameMode) {
        super(inputs,outputs, kY, kX, sY, sX, pY, pX, dY, dX, isSameMode);
    }


    @Override
    public String opName() {
        return "deconv2d_bp";
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
