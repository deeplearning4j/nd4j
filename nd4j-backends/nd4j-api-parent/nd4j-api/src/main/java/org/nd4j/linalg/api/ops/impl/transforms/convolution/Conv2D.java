package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;


/**
 * Conv2D operation
 */
@Slf4j
public class Conv2D extends BaseTransformOp {


    private int kh, kw, sy, sx, ph, pw, dh, dw;
    private boolean isSameMode;

    @Builder(builderMethodName = "sameDiffBuilder")
    public Conv2D(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v, inPlace);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    @Builder(builderMethodName = "execBuilder")
    public Conv2D(INDArray x, INDArray z, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        super(x, z);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Conv2D() {}



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
        return "conv2d";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {kh, kw, sy, sx, ph, pw, dh, dw, fromBoolean(isSameMode)};
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
