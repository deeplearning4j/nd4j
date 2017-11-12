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

package org.nd4j.linalg.api.ops.impl.shape;

import com.google.common.primitives.Ints;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ShapeOp;
import org.nd4j.linalg.util.ComplexUtil;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Reshape function
 *
 * @author Adam Gibson
 */
@Slf4j
public class StridedSlice extends ShapeOp {

    private int[] shape;

    public StridedSlice(SameDiff sameDiff, DifferentialFunction i_v, int[] shape) {
        super(sameDiff, i_v, false);
        this.shape = shape;
    }

    public StridedSlice(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, Object[] extraArgs, int[] shape1) {
        super(sameDiff, i_v, shape, false, extraArgs);
        this.shape = shape1;
    }

    public StridedSlice() {}

    public StridedSlice(INDArray x, INDArray z) {
        super(x, z);
    }

    public StridedSlice(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public StridedSlice(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public StridedSlice(INDArray x) {
        super(x);
    }

    @Override
    public void exec(int... dimensions) {
        exec();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public void exec() {
        if(x != z) {
            z.assign(x.reshape(shape));
        }
        else {
            this.z = x.reshape(shape);
        }

    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "reshape";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public float op(float origin, float other) {
        return FastMath.abs(origin);
    }

    @Override
    public double op(double origin, double other) {
        return FastMath.abs(origin);
    }

    @Override
    public double op(double origin) {
        return FastMath.abs(origin);
    }

    @Override
    public float op(float origin) {
        return FastMath.abs(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return ComplexUtil.abs(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new StridedSlice(xAlongDimension, y.vectorAlongDimension(index, dimension),
                            z.vectorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new StridedSlice(xAlongDimension, z.vectorAlongDimension(index, dimension), xAlongDimension.length());

    }


    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new StridedSlice(xAlongDimension, y.tensorAlongDimension(index, dimension),
                            z.tensorAlongDimension(index, dimension), xAlongDimension.length());
        else
            return new StridedSlice(xAlongDimension, z.tensorAlongDimension(index, dimension), xAlongDimension.length());

    }
    @Override
    public TOp asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);
        /*
            strided slice typically takes 4 tensor arguments:
            0) input, it's shape determines number of elements in other arguments
            1) begin indices
            2) end indices
            3) strides
         */

        val strides = graph.getVariableSpace().getVariable(tNode.getInputs().remove(3));
        val end = graph.getVariableSpace().getVariable(tNode.getInputs().remove(2));
        val begin = graph.getVariableSpace().getVariable(tNode.getInputs().remove(1));

        val iArgs = new ArrayList<Integer>();

        for (int e = 0; e < begin.getArray().length(); e++)
            iArgs.add((int) begin.getArray().getInt(e));

        for (int e = 0; e < end.getArray().length(); e++)
            iArgs.add((int) end.getArray().getInt(e));

        for (int e = 0; e < strides.getArray().length(); e++)
            iArgs.add((int) strides.getArray().getInt(e));


        val bits = Ints.toArray(iArgs);
        tNode.getOpState().setExtraBits(bits);

        return tNode;
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = this;

        return Collections.singletonList(ret);
    }

}
