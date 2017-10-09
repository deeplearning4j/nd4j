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
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;


/**
 * Pooling2D operation
 */
@Slf4j
@Getter
public class Pooling2D extends DynamicCustomOp {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }



    private int kh, kw, sy, sx, ph, pw, dh, dw;
    private Pooling2DType type;
    private boolean isSameMode;

    public Pooling2D() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    public Pooling2D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, Pooling2DType type, boolean isSameMode) {
        super(null,sameDiff, inputs, inPlace);
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
        addArgs();
    }

    @Builder(builderMethodName = "execBuilder")
    public Pooling2D(INDArray[] inputs, INDArray[] outputs,int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, Pooling2DType type, boolean isSameMode) {
        super(null,inputs,outputs);
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
        addArgs();
    }


    private void addArgs() {
        getIArguments().add(kh);
        getIArguments().add(kw);
        getIArguments().add(sy);
        getIArguments().add(sx);
        getIArguments().add(ph);
        getIArguments().add(pw);
        getIArguments().add(dh);
        getIArguments().add(fromBoolean(isSameMode));


    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool2d";
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
