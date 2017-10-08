package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * BatchNormDerivative operation
 */
@Slf4j
public class BatchNormDerivative extends BatchNorm {

    @Builder(builderMethodName = "sameDiffBuilder")
    public BatchNormDerivative(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(sameDiff, i_v, inPlace, training, isLockGammaBeta, isMiniBatch);
    }

    @Builder(builderMethodName = "execBuilder")
    public BatchNormDerivative(INDArray x, INDArray z, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(x, z, training, isLockGammaBeta, isMiniBatch);
    }

    public BatchNormDerivative() {}


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
        return "batchnorm_bp";
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
