package org.nd4j.imports;

import com.google.common.primitives.Ints;
import com.google.protobuf.TextFormat;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.imports.intermediate.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.*;

import java.io.*;
import java.nio.IntBuffer;
import java.util.*;

/**
 * This class provides TensorFlow graphs & models import
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TensorFlowImport {

    /**
     *
     * @param graphFile
     * @return
     */
    public static SameDiff importGraph(File graphFile) {
        GraphDef def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = GraphDef.parseFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = builder.build();
            } catch (Exception e2) {
                //
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format");


        return importGraph(def);
    }

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    public static SameDiff importGraph(GraphDef tfGraph) {
        SDGraph graph = new SDGraph(true);

        SameDiff diff = SameDiff.builder()
                .graph(graph)
                .vertexToArray(new HashMap<>())
                .variableMap(new HashMap<>())
                .vertexIdxToInfo(new HashMap<>())
                .build();

        graph.setSameDiff(diff);

        Map<String, Integer> reverseVertexMap = new HashMap<>();

        int nodesCnt = 0;
        for (NodeDef tfNode :tfGraph.getNodeList()) {
            log.info("Node name: {}; Op: {};", tfNode.getName(), tfNode.getOp());


            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

            Map<String, AttrValue> attributes = tfNode.getAttrMap();





            if (isConst || isVar || isPlaceholder) {
                List<Integer> dimensions = new ArrayList<>();
                SDVariable variable = SDVariable.builder()
                        .sameDiff(diff)
                        .varName(tfNode.getName())
                        .build();

                NDArrayInformation varInformation = NDArrayInformation.builder()
                        .id(tfNode.getName())
                        .arrId(tfNode.getName())
                        .build();

                NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt,0, varInformation);

                reverseVertexMap.put(tfNode.getName(), nodesCnt);

                int[] arrayShape = null;

                if (attributes.containsKey("dtype")) {
                    AttrValue dtype = attributes.get("dtype");

                    dtype.getList();
                }

                if (attributes.containsKey("shape")) {
                    AttrValue shape = attributes.get("shape");
                    int dims = shape.getShape().getDimCount();
                    if (dims > 0) {

                        // even vector is 2d in nd4j
                        if (dims == 1)
                            dimensions.add(1);

                        for (int e = 0; e < dims; e++) {
                            // TODO: eventually we want long shapes :(
                            dimensions.add((int) shape.getShape().getDim(e).getSize());
                        }
                    }

                    arrayShape = Ints.toArray(dimensions);

                    variable.setShape(arrayShape);
                }

                if (attributes.containsKey("value")) {
                    // value of?
                    AttrValue value = attributes.get("value");

                    //DataType type = value.

                    TensorProto tensor = value.getTensor();
                    log.info("Dtype: {}", tensor.getDtype());

                    INDArray array = getNDArrayFromTensor(tensor);
                    variable.setShape(array.shape());
                    variable.setArr(array);
                }

                diff.addVariable(variable);
                graph.addVertex(vertex);
            } else {
                // operation node
                NDArrayInformation varInformation = NDArrayInformation.builder()
                        .id(tfNode.getName())
                        .build();

                NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt, 0,varInformation);
                graph.addVertex(vertex);

                OpState opState = getOpStateFromNodeDef(tfNode, tfNode.getInputCount());
                opState.setResult(varInformation);

                reverseVertexMap.put(tfNode.getName(), nodesCnt);


                for (int e = 0; e < tfNode.getInputCount(); e++) {
                    String input = tfNode.getInput(e);

                    Integer id = reverseVertexMap.get(input);

                    if (id == null)
                        throw new ND4JIllegalStateException("Unknown input: [" + input + "]");

                    graph.addEdge(new int[]{id}, new int[]{nodesCnt}, opState, true);
                }
            }
        }
        return diff;
    }

    /**
     * This method returns intermediate representation from TF protobuf file
     *
     * @param graphFile
     * @return
     */
    public static TGraph importIntermediate(File graphFile) {
        GraphDef def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = GraphDef.parseFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = builder.build();
            } catch (Exception e2) {
                //
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format");


        return importIntermediate(def);
    }

    /**
     * This method returns intermediate representation from TF GraphDef instance
     *
     * @return
     */
    public static TGraph importIntermediate(GraphDef tfGraph) {
        TGraph intermediateGraph = new TGraph();

        Map<String, Integer> reverseVertexMap = new HashMap<>();

        int varsCnt = 0;
        int nodesCnt = 0;
        for (NodeDef tfNode :tfGraph.getNodeList()) {

            log.info("Node name: {}; Op: {};", tfNode.getName(), tfNode.getOp());


            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

            Map<String, AttrValue> attributes = tfNode.getAttrMap();



            if (isConst || isVar || isPlaceholder) {
                varsCnt--;
                val variable = new TVariable();
                List<Integer> dimensions = new ArrayList<>();

                reverseVertexMap.put(tfNode.getName(), varsCnt);

                variable.setName(tfNode.getName());
                variable.setId(varsCnt);
                variable.setPlaceholder(isPlaceholder);

                int[] arrayShape = null;

                if (tfNode.getName().equalsIgnoreCase("mixed4b/concat_dim")) {
                    log.info("concat found!");
                }

                if (attributes.containsKey("dtype")) {
                    AttrValue dtype = attributes.get("dtype");

                    dtype.getList();
                }

                if (attributes.containsKey("shape")) {
                    AttrValue shape = attributes.get("shape");
                    int dims = shape.getShape().getDimCount();
                    if (dims > 0) {

                        // even vector is 2d in nd4j
                        if (dims == 1)
                            dimensions.add(1);

                        for (int e = 0; e < dims; e++) {
                            // TODO: eventually we want long shapes :(
                            dimensions.add((int) shape.getShape().getDim(e).getSize());
                        }
                    }

                    arrayShape = Ints.toArray(dimensions);

                    variable.setShape(arrayShape);
                }

                if (attributes.containsKey("value")) {
                    // value of?
                    AttrValue value = attributes.get("value");

                    //DataType type = value.

                    TensorProto tensor = value.getTensor();
                    log.info("Dtype: {}", tensor.getDtype());
                    if (tensor.getDtype() == DataType.DT_FLOAT || tensor.getDtype() == DataType.DT_DOUBLE) {

                        INDArray array = getNDArrayFromTensor(tensor);
                        variable.setShape(array.shape());
                        variable.setArray(array);
                    } else {
                        val shape = getShapeFromTensor(tensor);

                        assert shape != null;
                        assert shape.length > 0;

                        variable.setShape(shape);
                    }
                }

                //diff.addVariable(variable);
                //graph.addVertex(vertex);

                if (!variable.isPlaceholder())
                    log.info("Variable: id: {}; name: {}; shape: {}", variable.getId(), variable.getName(), Arrays.toString(variable.getShape()));
                else
                    log.info("Placeholder shape: {}", Arrays.toString(variable.getShape()));

                intermediateGraph.getVariableSpace().addVariable(variable.getId(), variable);
            } else {
                nodesCnt++;
                log.info("Adding op [{}]", tfNode.getOp());
                // operation node

                //NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt, 0,varInformation);
                //graph.addVertex(vertex);


//                opState.setResult(varInformation);

                reverseVertexMap.put(tfNode.getName(), nodesCnt);
                val tNode = TNode.builder()
                        .name(tfNode.getName())
                        .id(nodesCnt)
                        .opName(tfNode.getOp())
                        .build();


                for (int e = 0; e < tfNode.getInputCount(); e++) {
                    String input = tfNode.getInput(e);


                    // input taken from mult
                    if (input.startsWith("^")) {
                        log.info("Wow");
                    } else if (input.contains(":")) {
                        val split = input.split(":");

                        if (split.length == 1) {
                            Integer id = reverseVertexMap.get(split[0]);

                            tNode.addInput(id);
                        } else if (split.length == 2) {
                            Integer node = reverseVertexMap.get(split[0]);
                            Integer idx = Integer.valueOf(split[1]);

                            tNode.addInput(node, idx);
                        } else
                            throw new RuntimeException("Unknown input passed in: [" + input + "]");

                    } else {
                        Integer id = reverseVertexMap.get(input);
                        tNode.addInput(id);

                        if (id == null)
                            throw new ND4JIllegalStateException("Unknown input: [" + input + "]");
                    }
                }

                OpState opState = getOpStateFromNodeDef(tfNode, tfNode.getInputCount(), tNode, intermediateGraph.getVariableSpace());
                tNode.setOpState(opState);

                for (val index: tNode.getInputs()) {
                    if (index.getNode() < 0)
                        continue;

                    val node = intermediateGraph.getNode(index.getNode());
                    node.getOutputs().add(tNode.getId());
                }

                log.info("Node: {}", tNode);
                intermediateGraph.addNode(tNode);
            }

            System.out.println();
        }

        return intermediateGraph;
    }

    protected static int[]  getShapeFromTensor(TensorProto tfTensor) {
        int[] arrayShape = null;

        if (tfTensor.getIntValCount() == 1) {
            int val = tfTensor.getIntVal(0);

            arrayShape = new int[]{val};

        } else if (tfTensor.getInt64ValCount() > 0) {
            arrayShape = new int[tfTensor.getInt64ValCount()];
            for (int e = 0; e < tfTensor.getInt64ValCount(); e++)
                arrayShape[e] = (int) tfTensor.getInt64Val(e);

        } else {
            // FIXME: INT bytebuffers should be converted to floating point
            if (tfTensor.getDtype() == DataType.DT_INT32) {
                val buffer = tfTensor.getTensorContent().asReadOnlyByteBuffer().asIntBuffer();

                arrayShape = new int[buffer.capacity()];
                for (int e = 0; e < buffer.capacity(); e++)
                    arrayShape[e] = (int) buffer.get(e);
            } else if (tfTensor.getDtype() ==DataType.DT_INT64) {
                val buffer = tfTensor.getTensorContent().asReadOnlyByteBuffer().asLongBuffer();

                arrayShape = new int[buffer.capacity()];
                for (int e = 0; e < buffer.capacity(); e++)
                    arrayShape[e] = (int) buffer.get(e);
            }

            log.info("Array shape: {}", Arrays.toString(arrayShape));
        }

        return arrayShape;
    }

    protected static INDArray getNDArrayFromTensor(TensorProto tfTensor) {
        int[] arrayShape = null;
        List<Integer> dimensions = new ArrayList<>();

        // building shape first
        int dims = tfTensor.getTensorShape().getDimCount();
        if (dims > 0) {
            // even vector is 2d in nd4j
            if (dims == 1)
                dimensions.add(1);

            for (int e = 0; e < dims; e++) {
                // TODO: eventually we want long shapes :(
                int dim = (int) tfTensor.getTensorShape().getDim(e).getSize();

                dimensions.add(dim);
            }
        }
        arrayShape = Ints.toArray(dimensions);

        if (tfTensor.getDtype() == DataType.DT_INT32 || tfTensor.getDtype() == DataType.DT_INT16 || tfTensor.getDtype() == DataType.DT_INT8) {
            // valueOf
            if (tfTensor.getIntValCount() == 1) {
                int val = tfTensor.getIntVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, (double) val);
                return array;
            } else if (tfTensor.getInt64ValCount() > 0) {
                double[] jArray = new double[tfTensor.getIntValCount()];
                for (int e = 0; e < tfTensor.getIntValCount(); e++) {
                    jArray[e] = (double) tfTensor.getIntVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            } else {
                // FIXME: INT bytebuffers should be converted to floating point
                throw new UnsupportedOperationException("To be implemented yet");
//                return Nd4j.create(5,5);
            }
        } else if (tfTensor.getDtype() == DataType.DT_FLOAT) {
            if (tfTensor.getFloatValCount() == 1) {
                float val = tfTensor.getFloatVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, (double) val);
                return array;
            } else if (tfTensor.getFloatValCount() > 0) {
                float[] jArray = new float[tfTensor.getFloatValCount()];
                for (int e = 0; e < tfTensor.getFloatValCount(); e++) {
                    jArray[e] = tfTensor.getFloatVal(e);
                }

                // FIXME: we're missing float[] signature
                INDArray array = Nd4j.create(ArrayUtil.toDoubles(jArray), arrayShape, 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0){

                long length = ArrayUtil.prodLong(arrayShape);
                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.asFloatBuffer();
                val fa = new float[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    fa[e] = fb.get(e);

                val array = Nd4j.create(fa, arrayShape, 'c', 0);
                //log.info("SUM1: {}", array.sumNumber());
                //log.info("Data: {}", Arrays.toString(array.data().asFloat()));
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_DOUBLE) {
            if (tfTensor.getDoubleValCount() == 1) {
                double val = tfTensor.getDoubleVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, val);
                return array;
            } else if (tfTensor.getDoubleValCount() > 0) {
                double[] jArray = new double[tfTensor.getDoubleValCount()];
                for (int e = 0; e < tfTensor.getDoubleValCount(); e++) {
                    jArray[e] =  tfTensor.getDoubleVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0) {
                long length = ArrayUtil.prodLong(arrayShape);
                // binary representation
                //DataBuffer buffer = Nd4j.createBuffer(tfTensor.getTensorContent().asReadOnlyByteBuffer(), DataBuffer.Type.FLOAT, (int) length);
                //INDArray array = Nd4j.createArrayFromShapeBuffer(buffer, Nd4j.getShapeInfoProvider().createShapeInformation(arrayShape, 'c'));

                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.asFloatBuffer();
                val da = new double[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    da[e] = fb.get(e);

                val array = Nd4j.create(da, arrayShape, 0, 'c');
                //log.info("SUM1: {}", array.sumNumber());
                //log.info("Data: {}", Arrays.toString(array.data().asFloat()));

                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_INT64) {
            if (tfTensor.getInt64ValCount() == 1) {
                double val = (double) tfTensor.getInt64Val(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, val);
                return array;
            } else if (tfTensor.getInt64ValCount() > 0)  {
                double[] jArray = new double[tfTensor.getInt64ValCount()];
                for (int e = 0; e < tfTensor.getInt64ValCount(); e++) {
                    jArray[e] =  (double) tfTensor.getInt64Val(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0){
                // FIXME: INT bytebuffers should be converted to floating point
                throw new UnsupportedOperationException("To be implemented yet");
            }
        }  else {
            throw new UnsupportedOperationException("Unknown dataType found: [" + tfTensor.getDtype() + "]");
        }

        throw new RuntimeException("Wtf?");
    }

    protected static OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs) {
        return getOpStateFromNodeDef(tfNode, numInputs, null, null);
    }

    protected static OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs, TNode tNode, TVariableSpace variableSpace) {
        String lc = tfNode.getOp().toLowerCase();
        log.info("Looking for [{}] op...", lc);
        if (numInputs > 0 && numInputs <= 2) {
            int opNum = Nd4j.getOpFactory().getOpNumIfExists(lc);

            if (opNum >= 0) {
                val type = OpState.opTypeFromOp(Nd4j.getOpFactory().getOpByName(lc));

                if (type != OpState.OpType.SHAPE) {
                    OpState opState = OpState.builder()
                            .opType(OpState.opTypeFromOp(Nd4j.getOpFactory().getOpByName(lc)))
                            .opNum(opNum)
                            .opName(lc)
                            .build();

                    return opState;
                }
            }
        }

        OpState opState = OpState.builder()
                .opType(OpState.OpType.CUSTOM)
                .opNum(-1)
                .opName(tfNode.getOp())
                .build();

         if (lc.equalsIgnoreCase("conv2d")) {


             val aStrides = tfNode.getAttrOrThrow("strides");
             val tfStrides = aStrides.getList().getIList();
             val sY = tfStrides.get(1);
             val sX = tfStrides.get(2);

             val aPadding = tfNode.getAttrOrDefault("padding", null);

             val paddingMode = aPadding.getS().toStringUtf8();

             // we know that second input to conv2d is weights array
             val weightsIndex = tNode.getInputs().get(1);
             val variable = variableSpace.getVariable(weightsIndex);

             val kY = variable.getArray().size(0);
             val kX = variable.getArray().size(1);

            log.info("Conv2D: k: [{}, {}]; s: [{}, {}]; padding: {}", kY, kX, sY, sX,  paddingMode);

             opState.setExtraBits(new int[] {kY, kX, sY.intValue(), sX.intValue(), 0, 0, 1, 1, paddingMode.equalsIgnoreCase("SAME") ? 1 : 0});
         } else if (lc.equalsIgnoreCase("avgpool") || lc.equalsIgnoreCase("maxpool")) {
             val aStrides = tfNode.getAttrOrThrow("strides");
             val tfStrides = aStrides.getList().getIList();
             val sY = tfStrides.get(1);
             val sX = tfStrides.get(2);

             val aKernels = tfNode.getAttrOrThrow("ksize");
             val tfKernels = aKernels.getList().getIList();

             val kY = tfKernels.get(1);
             val kX = tfKernels.get(2);

             val aPadding = tfNode.getAttrOrDefault("padding", null);

             val paddingMode = aPadding.getS().toStringUtf8();

             log.info("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);

             opState.setExtraBits(new int[] {kY.intValue(), kX.intValue(), sY.intValue(), sX.intValue(), 1, 1});

         } else if (lc.equalsIgnoreCase("lrn")) {
             val aAlpha = tfNode.getAttrOrThrow("alpha");
             val aBeta = tfNode.getAttrOrThrow("beta");
             val aBias = tfNode.getAttrOrThrow("bias");
             val aDepth = tfNode.getAttrOrThrow("depth_radius");

             val alpha = aAlpha.getF();
             val beta = aBeta.getF();
             val bias = aBias.getF();
             val depth = aDepth.getF();


             opState.setExtraArgs(new Object[]{alpha, beta, bias, depth});
             log.info("LRN: alpha: {}; beta: {}; bias: {}; depth: {};", alpha, beta, bias, depth);
         } else if (lc.equalsIgnoreCase("reshape")) {
             // in reshape operation we replace second input, and replace it with extra args
             log.info("TNode inputs: {}", tNode.getInputs());
             val shapeIndex = tNode.getInputs().remove(1);
             val variable = variableSpace.getVariable(shapeIndex);

             assert variable != null;
             assert variable.getShape() != null;

             // new shape goes here
             opState.setExtraBits(variable.getShape());
         } else if (lc.equalsIgnoreCase("concat")) {
             log.info("TNode inputs: {}", tNode.getInputs());
             TIndex dimIndex;
             int idx = -1;
             int cnt = 0;
             int concatDimension = 0;
             for (val index:tNode.getInputs()) {
                 log.info("Trying to find node: [{}]", index);
                 val variable = variableSpace.getVariable(index);

                 // concat dimension is only possible
                 if (variable != null && variable.getId() < 0 && variable.getArray() == null) {
                     idx = cnt;
                     concatDimension = variable.getShape()[0];
                 }
                 cnt++;
             }

             if (idx < 0)
                 throw new ND4JIllegalStateException("Can't find dimension for concatenatiion");

             // deleting index of concat dimension
             tNode.getInputs().remove(idx);

             opState.setExtraBits(new int[]{concatDimension});
             log.info("Concat dimension: {}", concatDimension);
         }

         if (!Nd4j.getExecutioner().getCustomOperations().containsKey(lc))
             throw new ND4JIllegalStateException("Unknown operation requested: ["+ tfNode.getOp() +"]");

        return opState;
    }
}
