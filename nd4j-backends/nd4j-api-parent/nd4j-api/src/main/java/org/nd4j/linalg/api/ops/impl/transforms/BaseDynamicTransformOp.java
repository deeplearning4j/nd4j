package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.List;

public abstract class BaseDynamicTransformOp extends DynamicCustomOp {

    public BaseDynamicTransformOp() {}

    public BaseDynamicTransformOp(SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public BaseDynamicTransformOp(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }


    @Override
    public List<int[]> calculateOutputShape() {
        val args = args();
        val firstLength = ArrayUtil.prod(args[0].getResultShape());
        val secondLength = ArrayUtil.prod(args[1].getResultShape());
        if(firstLength > secondLength)
            return Arrays.asList(args[0].getResultShape());
        else
            return Arrays.asList(args[1].getResultShape());
    }
}
