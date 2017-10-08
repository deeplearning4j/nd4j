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
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;


/**
 * Pooling2D operation
 */
@Slf4j
public class Pooling2D extends BaseTransformOp {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }



    private int kh, kw, sy, sx, ph, pw, dh, dw;
    private Pooling2DType type;
    private boolean isSameMode;

    public Pooling2D() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    public Pooling2D(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, Pooling2DType type, boolean isSameMode) {
        super(sameDiff, i_v, inPlace);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.type = type;
        this.isSameMode = isSameMode;
    }

    @Builder(builderMethodName = "execBuilder")
    public Pooling2D(INDArray x, INDArray z,int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, Pooling2DType type, boolean isSameMode) {
        super(x, z);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.type = type;
        this.isSameMode = isSameMode;
    }

    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return getPoolingPrefix() + "pool2d";
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

    public String getPoolingPrefix() {
        switch(type) {
            case AVG:return "avg";
            case MAX: return "max";
            case PNORM: return "pnorm";
            default: throw new IllegalStateException("No pooling type found.");
        }
    }

}
