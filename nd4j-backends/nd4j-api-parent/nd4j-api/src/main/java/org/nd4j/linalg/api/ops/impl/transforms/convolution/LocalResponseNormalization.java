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
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
@Getter
public class LocalResponseNormalization extends DynamicCustomOp {



    private double alpha,beta,bias,depth;

    @Builder(builderMethodName = "sameDiffBuilder")
    public LocalResponseNormalization(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, double alpha, double beta, double bias, double depth) {
        super(null,sameDiff, inputs, inPlace);
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
        this.depth = depth;
        addArgs();
    }

    @Builder(builderMethodName = "execBuilder")
    public LocalResponseNormalization(INDArray[] inputs, INDArray[] outputs,double alpha, double beta, double bias, double depth) {
        super(null,inputs,outputs);
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
        this.depth = depth;
        addArgs();
    }

    public LocalResponseNormalization() {}


    private void addArgs() {
        getTArguments().add(alpha);
        getTArguments().add(beta);
        getTArguments().add(bias);
        getTArguments().add(depth);
    }

    @Override
    public String opName() {
        return "lrn";
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
