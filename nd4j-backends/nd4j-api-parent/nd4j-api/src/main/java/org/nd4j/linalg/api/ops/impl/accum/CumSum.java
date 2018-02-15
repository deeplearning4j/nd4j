package org.nd4j.linalg.api.ops.impl.accum;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Cumulative sum operation, optionally along dimension.
 *
 * @author Alex Black
 */
public class CumSum extends DynamicCustomOp {

    public CumSum(){}

    public CumSum(SameDiff sameDiff, SDVariable x, int... dimension){
        super(null, sameDiff, new SDVariable[]{x});
        this.sameDiff = sameDiff;
        this.dimensions = dimension;
        addIArgument(dimension);
    }


    @Override
    public INDArray getInputArgument(int index) {
        return inputArguments()[index];
    }

    @Override
    public String opName() {
        return "cumsum";
    }

    @Override
    public String tensorflowName() {
        return "Cumsum";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        // Output gradient is the reversed cumulative sum of the reversed input gradient
        SDVariable gradient = sameDiff.setupFunction(grad.get(0));

        SDVariable reverseGrad = sameDiff.reverse(gradient, 1- dimensions[0]);
        SDVariable ret = sameDiff.cumsum(reverseGrad, dimensions);
        SDVariable reversedRet = sameDiff.reverse(ret, 1- dimensions[0]);
        return Collections.singletonList(reversedRet);
    }

}
