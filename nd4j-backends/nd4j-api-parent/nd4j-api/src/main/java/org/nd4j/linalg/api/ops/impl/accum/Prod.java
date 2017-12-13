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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;

import java.util.Collections;
import java.util.List;

/**
 * Prod the components
 *
 * @author Adam Gibson
 */
public class Prod extends BaseAccumulation {
    public Prod(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Prod(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Prod() {}

    public Prod(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Prod(INDArray x, INDArray y, long n) {
        super(x, y, n);
    }

    public Prod(INDArray x) {
        super(x);
    }

    public Prod(INDArray x, INDArray y) {
        super(x, y);
    }


    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "prod";
    }


    @Override
    public double zeroDouble() {
        return 1.0;
    }

    @Override
    public float zeroFloat() {
        return 1.0f;
    }

    @Override
    public float zeroHalf() {
        return zeroFloat();
    }

    @Override
    public String onnxName() {
       return "ReduceProd";
    }

    @Override
    public String tensorflowName() {
       return "Prod";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        SDVariable ret = f().div(f().doRepeat(
                outputVariables()[0],
                i_v1.get(0)),f().mul(f().one(outputVariables()[0].getShape()),f()
                .getInputLength(i_v1.get(0))));

        return Collections.singletonList(ret);
    }

}
