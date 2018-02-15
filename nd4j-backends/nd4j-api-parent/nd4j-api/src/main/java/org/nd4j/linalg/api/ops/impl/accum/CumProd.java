package org.nd4j.linalg.api.ops.impl.accum;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.BooleanAdapter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class CumProd extends DynamicCustomOp {
    protected boolean exclusive = false;
    protected boolean reverse = false;

    @Override
    public String opName() {
        return "cumprod";
    }



    @Override
    public String tensorflowName() {
        return "Cumprod";
    }

    @Override
    public Map<String, Map<String, AttributeAdapter>> attributeAdaptersForFunction() {
        Map<String, Map<String, AttributeAdapter>> ret = new HashMap<>();
        Map<String,AttributeAdapter> tfMappings = new LinkedHashMap<>();

        tfMappings.put("exclusive", new BooleanAdapter());
        tfMappings.put("reverse", new BooleanAdapter());


        ret.put(tensorflowName(), tfMappings);

        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val exclusiveMapper = PropertyMapping.builder()
                .tfAttrName("exclusive")
                .propertyNames(new String[]{"exclusive"})
                .build();

        val reverseMapper = PropertyMapping.builder()
                .tfAttrName("reverse")
                .propertyNames(new String[]{"reverse"})
                .build();


        map.put("exclusive", exclusiveMapper);
        map.put("reverse", reverseMapper);

        ret.put(tensorflowName(),map);

        return ret;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode,nodeDef, graph);
        addArgs();
    }

    protected void addArgs() {
        addIArgument(exclusive ? 1 : 0, reverse ? 1 : 0);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }
}
