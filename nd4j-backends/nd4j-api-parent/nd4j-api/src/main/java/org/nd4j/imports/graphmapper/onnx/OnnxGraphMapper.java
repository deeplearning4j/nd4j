package org.nd4j.imports.graphmapper.onnx;

import com.google.common.primitives.Ints;
import com.google.protobuf.Message;
import onnx.OnnxProto3;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OnnxGraphMapper extends BaseGraphMapper<OnnxProto3.GraphProto, OnnxProto3.NodeProto, OnnxProto3.AttributeProto, OnnxProto3.TensorProto> {
    @Override
    public Message.Builder getNewGraphBuilder() {
        return OnnxProto3.GraphProto.newBuilder();
    }

    @Override
    public OnnxProto3.GraphProto parseGraphFrom(InputStream inputStream) throws IOException {
        return OnnxProto3.ModelProto.parseFrom(inputStream).getGraph();
    }

    @Override
    public void mapNodeType(OnnxProto3.NodeProto tfNode, ImportState<OnnxProto3.GraphProto> importState) {
        for(int i = 0; i < tfNode.getInputCount(); i++) {
            String input = tfNode.getInput(i);
            OnnxProto3.ValueInfoProto valueInfo = importState.getGraph().getValueInfo(Integer.parseInt(input));
            if(importState.getSameDiff().getVariable(input) == null) {
                addVarFromValueInfo(valueInfo,importState,i);
            }
        }

        for(int i = 0; i < tfNode.getOutputCount(); i++) {
            String input = tfNode.getOutput(i);
            OnnxProto3.ValueInfoProto valueInfo = importState.getGraph().getValueInfo(Integer.parseInt(input));
            if(importState.getSameDiff().getVariable(input) == null) {
                addVarFromValueInfo(valueInfo,importState,i);
            }
        }
    }

    protected void addVarFromValueInfo( OnnxProto3.ValueInfoProto valueInfo,ImportState<OnnxProto3.GraphProto> importState,int i) {
        int[] shape = shapeFrom(valueInfo.getType().getTensorType().getShape().getDimList());
        OnnxProto3.TensorProto tensorProto = importState.getGraph().getInitializer(i);
        if(tensorProto != null) {
            importState.getSameDiff().var(String.valueOf(i),getNDArrayFromTensor(tensorProto));
        }
        else {
            importState.getSameDiff().var(String.valueOf(i),shape);
        }
    }


    @Override
    public DataBuffer.Type dataTypeForTensor(OnnxProto3.TensorProto tensorProto) {
        switch (tensorProto.getDataType()) {
            case DOUBLE: return DataBuffer.Type.DOUBLE;
            case FLOAT: return DataBuffer.Type.FLOAT;
            case FLOAT16: return DataBuffer.Type.HALF;
            case INT32:
            case INT64: return DataBuffer.Type.INT;
            default: return DataBuffer.Type.UNKNOWN;
        }
    }

    @Override
    public TOp asIntermediate(OnnxProto3.NodeProto nodeProto, TGraph intermediateGraph) {
        return null;
    }

    @Override
    public String getAttrValueFromNode(OnnxProto3.NodeProto nodeProto, String key) {
        return null;
    }

    @Override
    public int[] getShapeFromAttribute(OnnxProto3.AttributeProto attributeProto) {
        return Ints.toArray(attributeProto.getT().getDimsList());
    }

    @Override
    public boolean isPlaceHolder(OnnxProto3.NodeProto nodeProto) {
        return false;
    }

    @Override
    public INDArray getNDArrayFromTensor(OnnxProto3.TensorProto tensorProto) {
        return null;
    }

    @Override
    public int[] getShapeFromTensor(OnnxProto3.TensorProto tensorProto) {
        return Ints.toArray(tensorProto.getDimsList());
    }

    @Override
    public OnnxProto3.TensorProto getTensorFrom(OnnxProto3.AttributeProto attributeProto) {
        return attributeProto.getT();
    }

    @Override
    public String getInputFromNode(OnnxProto3.NodeProto node, int index) {
        return node.getInput(index);
    }

    @Override
    public int numInputsFor(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getInputCount();
    }

    @Override
    public String valueKey() {
        return null;
    }

    @Override
    public String shapeKey() {
        return null;
    }

    @Override
    public String dTypeKey() {
        return null;
    }

    @Override
    public int[] getShapeFromAttr(OnnxProto3.AttributeProto attr) {
        return Ints.toArray(attr.getT().getDimsList());
    }

    @Override
    public Map<String, OnnxProto3.AttributeProto> getAttrMap(OnnxProto3.NodeProto nodeProto) {
        Map<String,OnnxProto3.AttributeProto> proto = new HashMap<>();
        for(int i = 0; i < nodeProto.getAttributeCount(); i++) {
            OnnxProto3.AttributeProto attributeProto = nodeProto.getAttribute(i);
            proto.put(attributeProto.getName(),attributeProto);
        }
        return proto;
    }

    @Override
    public String getName(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getName();
    }

    @Override
    public boolean alreadySeen(OnnxProto3.NodeProto nodeProto) {
        return false;
    }

    @Override
    public boolean isVariableNode(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getOpType().contains("Var");
    }

    @Override
    public boolean shouldSkip(OnnxProto3.NodeProto opType) {
        return false;
    }

    @Override
    public boolean hasShape(OnnxProto3.NodeProto nodeProto) {
        return false;
    }

    @Override
    public int[] getShape(OnnxProto3.NodeProto nodeProto) {
        return null;
    }

    @Override
    public INDArray getArrayFrom(OnnxProto3.NodeProto nodeProto) {
        return null;
    }

    @Override
    public String getOpType(OnnxProto3.NodeProto nodeProto) {
        return nodeProto.getOpType();
    }

    @Override
    public List<OnnxProto3.NodeProto> getNodeList(OnnxProto3.GraphProto graphProto) {
        return graphProto.getNodeList();
    }

    private int[] shapeFrom(List<OnnxProto3.TypeProto.TensorShapeProto.Dimension> dims) {
        int[] ret = new int[dims.size()];
        for(int i = 0; i < dims.size(); i++) {
            ret[i] = (int) dims.get(i).getDimValue();
        }

        return ret;

    }
}
