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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * Element-wise "Replace NaN" implementation as Op
 *
 * @author raver119@gmail.com
 */
public class ReplaceNans extends BaseTransformOp {

    private double set;

    public ReplaceNans(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double set) {
        super(sameDiff, i_v, inPlace);
        this.set = set;
    }

    public ReplaceNans(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs, double set) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.set = set;
    }

    public ReplaceNans(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double set) {
        super(sameDiff, i_v, extraArgs);
        this.set = set;
    }

    public ReplaceNans() {

    }

    public ReplaceNans(INDArray x, double set) {
        super(x);
        this.set = set;
        init(x, null, x, x.length());
    }

    public ReplaceNans(INDArray x, INDArray z, double set) {
        super(x, z);
        this.set = set;
        init(x, null, z, x.length());
    }

    public ReplaceNans(INDArray x, INDArray z, double set, long n) {
        super(x, z, n);
        this.set = set;
        init(x, null, x, n);
    }

    @Override
    public int opNum() {
        return 46;
    }

    @Override
    public String opName() {
        return "replace_nans";
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
        this.extraArgs = new Object[] {set, (double) n};
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}

