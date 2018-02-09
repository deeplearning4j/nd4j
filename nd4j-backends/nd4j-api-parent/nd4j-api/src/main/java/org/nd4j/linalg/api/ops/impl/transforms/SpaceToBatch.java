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
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * N-dimensional space to batch operation. Transforms data from a tensor from M spatial dimensions into batch dimension
 * according to the "blocks" specified (a vector of length M). Afterwards the spatial dimensions are optionally padded,
 * as specified in "padding", a tensor of dim (M, 2), denoting the padding range.
 * <p>
 * Example:
 * input:         [[[[1], [2]], [[3], [4]]]]
 * input shape:   [1, 2, 2, 1]
 * blocks:        [2, 2]
 * padding:       [[0, 0], [0, 0]]
 * <p>
 * output:        [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
 * output shape:  [4, 1, 1, 1]
 * *
 *
 * @author Max Pumperla
 */
public class SpaceToBatch extends DynamicCustomOp {

    protected INDArray blocks;
    private int spatialDimensions;
    private int[] inputShape;
    protected INDArray padding;

    public SpaceToBatch() {
    }

    public SpaceToBatch(SameDiff sameDiff, SDVariable[] args, INDArray blocks, INDArray padding, boolean inPlace) {
        super(null, sameDiff, args, inPlace);

        INDArray input = args[0].getArr();
        this.inputShape = input.shape();
        this.blocks = blocks;
        this.spatialDimensions = blocks.shape()[0];
        this.padding = padding;
    }

    @Override
    public String opName() {
        return "space_to_batch";
    }

    @Override
    public INDArray[] inputArguments() {
        /**
         * This op has 1 input variable coming from SameDiff, and 2 static input arrays
         */
        val array = super.inputArguments();

        return new INDArray[]{array[0], blocks, padding};
    }

    public List<int[]> calculateOutputShape() {
        int batchSize = inputShape[0];
        int[] outputShape = inputShape.clone();
        for (int i = 0; i < spatialDimensions; i++) {
            int block = (int) blocks.getDouble(i, 0);
            batchSize = batchSize * block;
            outputShape[i + 1] = outputShape[i + 1] / block + (int) padding.getDouble(i, 0) + (int) padding.getDouble(i, 1);

        }
        outputShape[0] = batchSize;
        return Collections.singletonList(outputShape);
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
    public String onnxName() {
        return "space_to_batch";
    }

    @Override
    public String tensorflowName() {
        return "space_to_batch_nd";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Inverse of space to batch is batch to space with same blocks and crops as padding
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        return Collections.singletonList(sameDiff.batchToSpace(gradient, blocks, padding));
    }

}
