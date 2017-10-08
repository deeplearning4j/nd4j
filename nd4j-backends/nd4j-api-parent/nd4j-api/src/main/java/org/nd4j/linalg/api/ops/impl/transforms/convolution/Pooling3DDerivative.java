package org.nd4j.linalg.api.ops.impl.transforms.convolution;

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
public class Pooling3DDerivative extends Pooling3D {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    public Pooling3DDerivative() {}

    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "pooling3d_bp";
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
