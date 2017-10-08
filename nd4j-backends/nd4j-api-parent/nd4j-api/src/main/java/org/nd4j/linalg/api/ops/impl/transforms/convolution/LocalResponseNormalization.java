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
public class LocalResponseNormalization extends BaseTransformOp {

    /**
     *    T alpha = block.getTArguments()->at(0);
     T beta = block.getTArguments()->at(1);
     T bias = block.getTArguments()->at(2);
     T depth = block.getTArguments()->at(3);

     */

    private double alpha,beta,bias,depth;

    @Builder(builderMethodName = "sameDiffBuilder")
    public LocalResponseNormalization(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, double alpha, double beta, double bias, double depth) {
        super(sameDiff, i_v, inPlace);
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
        this.depth = depth;
    }

    @Builder(builderMethodName = "execBuilder")
    public LocalResponseNormalization(INDArray x, INDArray z,double alpha, double beta, double bias, double depth) {
        super(x, z);
        this.alpha = alpha;
        this.beta = beta;
        this.bias = bias;
        this.depth = depth;
    }

    public LocalResponseNormalization() {}

    @Override
    public int opNum() {
        return 71;
    }


    @Override
    public String name() {
        return "lrn";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {alpha,beta,bias,depth};
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
