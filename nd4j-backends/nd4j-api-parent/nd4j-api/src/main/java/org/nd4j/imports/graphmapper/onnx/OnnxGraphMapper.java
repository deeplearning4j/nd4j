package org.nd4j.imports.graphmapper.onnx;

import com.google.common.primitives.Ints;
import com.google.protobuf.Message;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.opstate.OpStateEdge;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DefaultOpConverter;
import org.nd4j.linalg.api.ops.Op;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A mapper for onnx graphs to {@link org.nd4j.autodiff.samediff.SameDiff} instances.
 *
 * @author Adam Gibson
 */
public class OnnxGraphMapper extends BaseGraphMapper<OnnxProto3.GraphProto, OnnxProto3.NodeProto, OnnxProto3.AttributeProto, OnnxProto3.TensorProto> {
    @Override
    public Op.Type opTypeForNode(OnnxProto3.NodeProto nodeProto) {
        return DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(nodeProto.getOpType()).opType();
    }

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
        int[] inputVertexIds = new int[tfNode.getInputCount()];
        int[] outputVertexIds = new int[tfNode.getOutputCount()];
        String[] vertexIdsForOpState = new String[tfNode.getInputCount() + tfNode.getOutputCount()];
        int vertexIdForOpStateIdx = 0;
        for(int i = 0; i < tfNode.getInputCount(); i++) {
            String input = tfNode.getInput(i);
            inputVertexIds[i] = Integer.parseInt(input);
            OnnxProto3.ValueInfoProto valueInfo = importState.getGraph().getValueInfo(Integer.parseInt(input));
            if(importState.getSameDiff().getVariable(input) == null) {
                addVarFromValueInfo(valueInfo,importState,i);
            }

            vertexIdsForOpState[vertexIdForOpStateIdx++] = input;
        }

        for(int i = 0; i < tfNode.getOutputCount(); i++) {
            String input = tfNode.getOutput(i);
            outputVertexIds[i] = Integer.parseInt(input);
            OnnxProto3.ValueInfoProto valueInfo = importState.getGraph().getValueInfo(Integer.parseInt(input));
            if(importState.getSameDiff().getVariable(input) == null) {
                addVarFromValueInfo(valueInfo,importState,i);
            }

            vertexIdsForOpState[vertexIdForOpStateIdx++] = input;

        }


        OpState opState = OpState.builder()
                .opType(opTypeForNode(tfNode))
                .vertexIds(vertexIdsForOpState)
                .opName(DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(tfNode.getOpType()).opName())
                .build();


        OpStateEdge opStateEdge = new OpStateEdge(inputVertexIds,outputVertexIds,opState,true);
        importState.getSameDiff().graph().addEdge(opStateEdge);

    }

    protected void addVarFromValueInfo(OnnxProto3.ValueInfoProto valueInfo,ImportState<OnnxProto3.GraphProto> importState,int i) {
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
        // first we try to use special converters
        DifferentialFunction converter = DifferentialFunctionClassHolder.getInstance().getInstance(nodeProto.getName().toLowerCase());
        if(converter == null)
            converter = DifferentialFunctionClassHolder.getInstance().getInstance(DefaultOpConverter.getInstance().opName());
        return converter.asIntermediateRepresentation(nodeProto, intermediateGraph);

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
