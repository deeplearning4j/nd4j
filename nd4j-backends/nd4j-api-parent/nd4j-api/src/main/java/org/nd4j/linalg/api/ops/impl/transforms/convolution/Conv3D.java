package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class Conv3D extends BaseTransformOp {
   
     private int dT;
     private int dW;
     private int dH;
     private int pT;
     private int pW;
     private int pH;
     private int dilationT;
     private int dilationW;
     private int dilationH;
     private int aT;
     private int aW;
     private int aH;
     private boolean biasUsed;

    public Conv3D() {}

    @Builder(builderMethodName = "sameDiffBuilder")
    public Conv3D(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, boolean biasUsed) {
        super(sameDiff, i_v, inPlace);
        this.dT = dT;
        this.dW = dW;
        this.dH = dH;
        this.pT = pT;
        this.pW = pW;
        this.pH = pH;
        this.dilationT = dilationT;
        this.dilationW = dilationW;
        this.dilationH = dilationH;
        this.aT = aT;
        this.aW = aW;
        this.aH = aH;
        this.biasUsed = biasUsed;
    }

    @Builder(builderMethodName = "execBuilder")
    public Conv3D(INDArray x, INDArray z, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, boolean biasUsed) {
        super(x, z);
        this.dT = dT;
        this.dW = dW;
        this.dH = dH;
        this.pT = pT;
        this.pW = pW;
        this.pH = pH;
        this.dilationT = dilationT;
        this.dilationW = dilationW;
        this.dilationH = dilationH;
        this.aT = aT;
        this.aW = aW;
        this.aH = aH;
        this.biasUsed = biasUsed;
    }

    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "conv3d";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {dT,dW,dH,pT,pW,pH,dilationT,dilationW,dilationH,aT,aW,aH,fromBoolean(biasUsed)};
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
