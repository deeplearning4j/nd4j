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
public class Upsampling extends BaseTransformOp {


    private int scaleFactor;

    @Builder(builderMethodName = "sameDiffBuilder")
    public Upsampling(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int scaleFactor) {
        super(sameDiff, i_v, inPlace);
        this.scaleFactor = scaleFactor;
    }

    @Builder(builderMethodName = "execBuilder")
    public Upsampling(INDArray x, INDArray z, int scaleFactor) {
        super(x, z);
        this.scaleFactor = scaleFactor;
    }

    public Upsampling() {}


    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "upsampling";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {scaleFactor};
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
