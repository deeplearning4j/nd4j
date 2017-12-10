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

package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Bit mask over the ndarrays as to whether
 * the components are less than or equal or not
 *
 * @author Adam Gibson
 */
public class OldLessThanOrEqual extends BaseTransformOp {
    public OldLessThanOrEqual(SameDiff sameDiff, SDVariable i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public OldLessThanOrEqual(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public OldLessThanOrEqual(SameDiff sameDiff) {
        super(sameDiff);
    }

    public OldLessThanOrEqual(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public OldLessThanOrEqual(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public OldLessThanOrEqual(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public OldLessThanOrEqual(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public OldLessThanOrEqual() {}

    public OldLessThanOrEqual(INDArray x, INDArray z) {
        super(x, z);
    }

    public OldLessThanOrEqual(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public OldLessThanOrEqual(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public OldLessThanOrEqual(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 12;
    }

    @Override
    public String opName() {
        return "lte";
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
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.<DifferentialFunction>asList(f().val(getResult()));
    }
}
