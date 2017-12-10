/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.List;

/**
 * Scalar max operation.
 * Returns the max of an element
 * in the ndarray of the specified number.
 *
 * @author Adam Gibson
 */
public class ScalarSet extends BaseScalarOp {
    public ScalarSet() {}

    public ScalarSet(INDArray x, INDArray y, INDArray z, long n, Number num) {
        super(x, y, z, n, num);
    }

    public ScalarSet(INDArray x, Number num) {
        super(x, num);
    }

    public ScalarSet(INDArray x, INDArray y, INDArray z, long n, IComplexNumber num) {
        super(x, y, z, n, num);
    }

    public ScalarSet(SameDiff sameDiff, SDVariable i_v, Number scalar) {
        super(sameDiff, i_v, scalar);
    }

    public ScalarSet(SameDiff sameDiff, SDVariable i_v, Number scalar, boolean inPlace) {
        super(sameDiff, i_v, scalar, inPlace);
    }

    public ScalarSet(SameDiff sameDiff, SDVariable i_v, Number scalar, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, inPlace, extraArgs);
    }

    public ScalarSet(SameDiff sameDiff, SDVariable i_v, Number scalar, Object[] extraArgs) {
        super(sameDiff, i_v, scalar, extraArgs);
    }

    public ScalarSet(INDArray x, IComplexNumber num) {
        super(x, num);
    }

    @Override
    public int opNum() {
        return 13;
    }

    @Override
    public String opName() {
        return "set_scalar";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (num != null) {
            this.extraArgs = new Object[] {num};
        } else {
            this.extraArgs = new Object[] {complexNumber};
        }

    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        throw new UnsupportedOperationException();
    }

}
