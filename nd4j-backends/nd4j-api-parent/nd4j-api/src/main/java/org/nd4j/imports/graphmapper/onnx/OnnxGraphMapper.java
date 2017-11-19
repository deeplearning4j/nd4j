package org.nd4j.imports.graphmapper.onnx;

import com.google.common.primitives.Ints;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DefaultOpConverter;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A mapper for onnx graphs to
 * {@link org.nd4j.autodiff.samediff.SameDiff} instances.
 *
 * @author Adam Gibson
 */
public class OnnxGraphMapper extends BaseGraphMapper<OnnxProto3.GraphProto, OnnxProto3.NodeProto, OnnxProto3.AttributeProto, OnnxProto3.TensorProto> {

    /**
     *
     * @param name the tensorflow or onnx name
     * @return
     */
    @Override
    public DifferentialFunction getMappedOp(String name) {
        return DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(name);
    }


    @Override
    public Map<String, Pair<int[], int[]>> inputsAndOutputsForGraph(OnnxProto3.GraphProto graph, Map<String, Integer> nodeNameToVertexId) {
        val ret = new HashMap<String, Pair<int[], int[]>>(graph.getNodeCount());
        for(val node : graph.getNodeList()) {
            val inputs = new int[node.getInputCount()];
            val outputs = new int[node.getOutputCount()];
            for(int i = 0; i < inputs.length; i++) {
                inputs[i] = nodeNameToVertexId.get(node.getInput(i));
            }

            for(int i = 0; i < outputs.length; i++) {
                outputs[i] = nodeNameToVertexId.get(node.getOutput(i));
            }

            ret.put(node.getName(),Pair.of(inputs,outputs));
        }

        return ret;
    }

    @Override
    public Map<String, OnnxProto3.TensorProto> variablesForGraph(OnnxProto3.GraphProto graphProto) {
        Map<String, OnnxProto3.TensorProto> ret = new HashMap<>();
        for(int i = 0; i < graphProto.getInitializerCount(); i++) {
            ret.put(graphProto.getInitializer(i).getName(),graphProto.getInitializer(i));
        }

        return ret;
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
    public void mapNodeType(OnnxProto3.NodeProto tfNode, ImportState<OnnxProto3.GraphProto, OnnxProto3.TensorProto> importState) {
        val differentialFunction = DifferentialFunctionClassHolder.getInstance().getOpWithOnnxName(tfNode.getName());
        val diff = importState.getSameDiff();

        try {
            val newInstance = differentialFunction.getClass().newInstance();
            newInstance.initFromOnnx(tfNode,diff,getAttrMap(tfNode),importState.getGraph());
            val indices = importState.getVertexIdMap().get(tfNode.getName());
            val opStateEdge = getOpStateEdge(indices.getFirst(),indices.getSecond(),tfNode);
            diff.graph().addEdge(opStateEdge);
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
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
    public TOp asIntermediate(OnnxProto3.NodeProto nodeProto, TGraph intermediateGraph, Map<String, OnnxProto3.AttributeProto> attributes) {
        // first we try to use special converters
        DifferentialFunction converter = DifferentialFunctionClassHolder.getInstance().getInstance(nodeProto.getName().toLowerCase());
        if(converter == null)
            converter = DifferentialFunctionClassHolder.getInstance().getInstance(DefaultOpConverter.getInstance().opName());
        return converter.asIntermediateRepresentation(nodeProto, intermediateGraph, attributes);

    }

    @Override
    public String getAttrValueFromNode(OnnxProto3.NodeProto nodeProto, String key) {
        for(OnnxProto3.AttributeProto attributeProto : nodeProto.getAttributeList()) {
            if(attributeProto.getName().equals(key)) {
                return attributeProto.getS().toString();
            }
        }

        throw new ND4JIllegalStateException("No key found for " + key);
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
        DataBuffer.Type type = dataTypeForTensor(tensorProto);
        ByteString bytes = tensorProto.getRawData();
        ByteBuffer byteBuffer = bytes.asReadOnlyByteBuffer();
        ByteBuffer directAlloc = ByteBuffer.allocateDirect(byteBuffer.capacity()).order(ByteOrder.nativeOrder());
        directAlloc.put(byteBuffer);
        directAlloc.rewind();
        int[] shape = getShapeFromTensor(tensorProto);
        DataBuffer buffer = Nd4j.createBuffer(directAlloc,type, ArrayUtil.prod(shape));
        INDArray arr = Nd4j.create(buffer);
        return arr;
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


}
