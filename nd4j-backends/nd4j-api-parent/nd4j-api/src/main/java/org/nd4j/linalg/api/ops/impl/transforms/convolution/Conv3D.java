package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * Conv3D operation
 */
@Slf4j
@Getter
public class Conv3D extends DynamicCustomOp {
   
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
    public Conv3D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, boolean biasUsed) {
        super(null,sameDiff, inputs, inPlace);
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
       addArgs();

    }

    @Builder(builderMethodName = "execBuilder")
    public Conv3D(INDArray[] inputs, INDArray[] outputs, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, boolean biasUsed) {
        super(null,inputs,outputs);
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
        addArgs();
    }

    private void addArgs() {
        getIArguments().add(dT);
        getIArguments().add(dW);
        getIArguments().add(dH);
        getIArguments().add(pT);
        getIArguments().add(pW);
        getIArguments().add(pH);
        getIArguments().add(dilationT);
        getIArguments().add(dilationW);
        getIArguments().add(dilationH);
        getIArguments().add(aT);
        getIArguments().add(aW);
        getIArguments().add(aH);
        getIArguments().add(fromBoolean(biasUsed));

    }

    @Override
    public String opName() {
        return "conv3d";
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
