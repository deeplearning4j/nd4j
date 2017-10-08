package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class Conv3DDerivative extends Conv3D {

    public Conv3DDerivative() {}


    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "conv3d_bp";
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
