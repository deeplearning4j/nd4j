package org.nd4j.linalg.api.ops.impl.transforms.convolution;

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
public class Pooling2DDerivative extends BaseTransformOp {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    private int kh, kw, sy, sx, ph, pw, dh, dw;
    private Pooling2DType type;
    boolean isSameMode;
    double extra;
    @Getter protected DataBuffer im2colShape;

    public Pooling2DDerivative() {}

    /*
    public Pooling2DDerivative(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, boolean isSameMode, Pooling2DType opType) {
        this(x, kh, kw, sy, sx, ph, pw, isSameMode, opType, getNewOutputArray(x, kh, kw, sy, sx, ph, pw, false));
    }
*/



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
        return "pooling2d";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {kh, kw, sy, sx, ph, pw, dh, dw, isSameMode ? 1 : 0};
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
