package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.Map;

/**
 * GatherND op
 */
@NoArgsConstructor
public class GatherNd extends DynamicCustomOp {

    private int[] broadcast;
    private int axis = 0;


    @Override
    public String onnxName() {
        return "GatherND";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"GatherNd"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode,nodeDef, graph);

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        OnnxGraphMapper.getInstance().initFunctionFromProperties(node.getOpType(), this, attributesForNode,node, graph);
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        super.resolvePropertiesFromSameDiffBeforeExecution();
        if(broadcast != null && numInputArguments() < 2) {
            if(numInputArguments() == 0) {
                addInputArgument(args()[0].getArr(),Nd4j.create( ArrayUtil.toFloats(broadcast)).reshape(broadcast.length));

            }
            else if(numInputArguments() == 1) {
                addInputArgument(Nd4j.create( ArrayUtil.toFloats(broadcast)));
            }

        }

        if(numIArguments() < 1) {
            addIArgument(axis);
        }

        if(numOutputArguments() < getDescriptor().getNumOutputs()) {
            val outputs = outputVariables();
            for(int i = 0; i < outputs.length; i++) {
                val output = outputs[i].getArr();
                addOutputArgument(output);
            }
        }



    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();

        Map<String,PropertyMapping> mapNd = new HashMap<>();
        val broadcastNd = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"broadcast"}).build();

        mapNd.put("broadcast",broadcastNd);

        ret.put("GatherNd",mapNd);

        return ret;
    }

    @Override
    public String opName() {
        return "gather_nd";
    }
}
