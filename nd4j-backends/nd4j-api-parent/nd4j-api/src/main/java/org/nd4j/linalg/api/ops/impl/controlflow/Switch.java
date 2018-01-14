package org.nd4j.linalg.api.ops.impl.controlflow;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

public class Switch extends DynamicCustomOp {

    @Override
    public String opName() {
        return "switch";
    }



    @Override
    public String tensorflowName() {
        return "Switch";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CONDITIONAL;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }
}
