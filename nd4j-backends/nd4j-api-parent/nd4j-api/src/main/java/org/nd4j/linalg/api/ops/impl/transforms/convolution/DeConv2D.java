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
public class DeConv2D extends BaseTransformOp {


    private int kY,kX,sY,sX,pY,pX,dY,dX;
    private boolean isSameMode;

    public DeConv2D() {}


    @Builder(builderMethodName = "sameDiffBuilder")
    public DeConv2D(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, boolean isSameMode) {
        super(sameDiff, i_v, inPlace);
        this.kY = kY;
        this.kX = kX;
        this.sY = sY;
        this.sX = sX;
        this.pY = pY;
        this.pX = pX;
        this.dY = dY;
        this.dX = dX;
        this.isSameMode = isSameMode;
    }

    @Builder(builderMethodName = "execBuilder")
    public DeConv2D(INDArray x, INDArray z, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, boolean isSameMode) {
        super(x, z);
        this.kY = kY;
        this.kX = kX;
        this.sY = sY;
        this.sX = sX;
        this.pY = pY;
        this.pX = pX;
        this.dY = dY;
        this.dX = dX;
        this.isSameMode = isSameMode;
    }

    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "deconv2d";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] { kY,kX,sY,sX,pY,pX,dY,dX,fromBoolean(isSameMode)};
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
