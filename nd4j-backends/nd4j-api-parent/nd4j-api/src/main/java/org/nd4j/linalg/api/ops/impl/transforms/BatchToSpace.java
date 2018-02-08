/*-
 *
 *  * Copyright 2018 Skymind,Inc.
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


import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.List;

/**
 * N-dimensional batch to space operation. Transforms data from a tensor from batch dimension into M spatial dimensions
 * according to the "blocks" specified (a vector of length M). Afterwards the spatial dimensions are optionally cropped,
 * as specified in "crops", a tensor of dim (M, 2), denoting the crop range.
 *
 * Example:
 *      input:        [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
 *      input shape:  [4, 1, 1, 1]
 *      blocks:       [2, 2]
 *      crops:        [[0, 0], [0, 0]]
 *
 *      output:       [[[[1], [2]], [[3], [4]]]]
 *      output shape: [1, 2, 2, 1]
 *
 * @author Max Pumperla
 */
public class BatchToSpace extends BaseDynamicTransformOp {

    private INDArray blocks;
    private int spatialDimensions;
    private int[] inputShape;
    private INDArray crops;

    public BatchToSpace() {}

    public BatchToSpace(SameDiff sameDiff, SDVariable[] args, INDArray blocks, INDArray crops, boolean inPlace) {
        super(sameDiff, args, inPlace);

        this.blocks = blocks;
        this.spatialDimensions = blocks.shape()[0];
        this.crops = crops;
    }

    @Override
    public INDArray[] inputArguments() {
        /**
         * This op has 1 input variable coming from SameDiff, and 2 static input arrays
         */
        val array =  super.inputArguments();

        return new INDArray[]{array[0], blocks, crops};
    }

    @Override
    public INDArray getInputArgument(int index) {
        return inputArguments()[index];
    }

    @Override
    public int numInputArguments() {
        return 3;
    }

    @Override
    public String opName() {
        return "batch_to_space";
    }

    @Override
    public String onnxName() { return "batch_to_space"; }

    @Override
    public String tensorflowName() {
        return "batch_to_space_nd";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Inverse of batch to space is space to batch with same blocks and padding as crops
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        return Collections.singletonList(sameDiff.spaceToBatch(gradient, blocks, crops));

    }

}
