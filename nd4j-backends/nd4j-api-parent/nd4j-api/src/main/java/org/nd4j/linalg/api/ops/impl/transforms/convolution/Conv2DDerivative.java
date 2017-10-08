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
 * Pooling2DDerivative operation
 */
@Slf4j
public class Conv2DDerivative extends Conv2D {
    @Builder(builderMethodName = "sameDiffBuilder")
    public Conv2DDerivative(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v, inPlace, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    @Builder(builderMethodName = "execBuilder")
    public Conv2DDerivative(INDArray x, INDArray z, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(x, z, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode);
    }

    public Conv2DDerivative() {}

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "conv2d_bp";
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
