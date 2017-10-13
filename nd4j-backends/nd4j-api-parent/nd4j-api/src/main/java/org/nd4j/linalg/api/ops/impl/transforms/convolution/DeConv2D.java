package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * DeConv2D operation
 */
@Slf4j
@Getter
public class DeConv2D extends DynamicCustomOp {


    private int kY,kX,sY,sX,pY,pX,dY,dX;
    private boolean isSameMode;

    public DeConv2D() {}


    @Builder(builderMethodName = "sameDiffBuilder")
    public DeConv2D(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, boolean isSameMode) {
        super(null,sameDiff, inputs, inPlace);
        this.kY = kY;
        this.kX = kX;
        this.sY = sY;
        this.sX = sX;
        this.pY = pY;
        this.pX = pX;
        this.dY = dY;
        this.dX = dX;
        this.isSameMode = isSameMode;
        addArgs();
    }

    @Builder(builderMethodName = "execBuilder")
    public DeConv2D(INDArray[] inputs, INDArray[] outputs, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, boolean isSameMode) {
        super(null,inputs,outputs);
        this.kY = kY;
        this.kX = kX;
        this.sY = sY;
        this.sX = sX;
        this.pY = pY;
        this.pX = pX;
        this.dY = dY;
        this.dX = dX;
        this.isSameMode = isSameMode;
        addArgs();
    }


    private void addArgs() {
        getIArguments().add(kY);
        getIArguments().add(kX);
        getIArguments().add(sY);
        getIArguments().add(sX);
        getIArguments().add(pY);
        getIArguments().add(pX);
        getIArguments().add(dY);
        getIArguments().add(dX);
        getIArguments().add(fromBoolean(isSameMode));

    }


    @Override
    public String opName() {
        return "deconv2d";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

}
