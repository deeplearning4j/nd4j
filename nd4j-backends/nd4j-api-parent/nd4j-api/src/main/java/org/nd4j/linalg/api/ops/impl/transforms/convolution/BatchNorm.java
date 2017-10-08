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
 * BatchNorm operation
 */
@Slf4j
public class BatchNorm extends BaseTransformOp {

    private boolean training;
    private boolean isLockGammaBeta;
    private boolean isMiniBatch;

    @Builder(builderMethodName = "sameDiffBuilder")
    public BatchNorm(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(sameDiff, i_v, inPlace);
        this.training = training;
        this.isLockGammaBeta = isLockGammaBeta;
        this.isMiniBatch = isMiniBatch;
    }

    @Builder(builderMethodName = "execBuilder")
    public BatchNorm(INDArray x, INDArray z, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(x, z);
        this.training = training;
        this.isLockGammaBeta = isLockGammaBeta;
        this.isMiniBatch = isMiniBatch;
    }

    public BatchNorm() {}



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
        return "batchnorm";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {fromBoolean(training),fromBoolean(isLockGammaBeta),fromBoolean(isMiniBatch)};
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
