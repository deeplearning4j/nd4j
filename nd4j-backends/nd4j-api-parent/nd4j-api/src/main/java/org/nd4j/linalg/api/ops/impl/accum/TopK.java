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
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * Top k
 *
 * @author raver119@gmail.com
 */
public class TopK extends DynamicCustomOp {

    private int k;
    private boolean needsSort = true;



    public TopK(SameDiff sameDiff, SDVariable[] args, int k, boolean needsSort) {
        super(null, sameDiff, args);
        this.k = k;
        this.needsSort = needsSort;
        addArgs();
    }

    public TopK(INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, int k, boolean needsSort) {
        super(null, inputs, outputs, tArguments, iArguments);
        this.k = k;
        this.needsSort = needsSort;
        addArgs();
    }

    public TopK(INDArray[] inputs, INDArray[] outputs, int k, boolean needsSort) {
        super(null, inputs, outputs);
        this.k = k;
        this.needsSort = needsSort;
        addArgs();
    }

    public TopK(SameDiff sameDiff, SDVariable[] args, boolean inPlace, int k, boolean needsSort) {
        super(null, sameDiff, args, inPlace);
        this.k = k;
        this.needsSort = needsSort;
        addArgs();
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(this,attributesForNode,nodeDef,graph);
    }

    @Override
    public String opName() {
        return "top_k";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

    @Override
    public String onnxName() {
        return "TopK";
    }

    @Override
    public String tensorflowName() {
        return "TopK";
    }

    private void addArgs() {
        addIArgument(k,needsSort ? 1 : 0);
    }

}
