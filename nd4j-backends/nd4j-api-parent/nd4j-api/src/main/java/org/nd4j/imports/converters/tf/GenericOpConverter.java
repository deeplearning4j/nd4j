package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TIndex;
import org.nd4j.graph.intermediate.TNode;
import org.nd4j.graph.intermediate.TVariableSpace;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.NodeDef;

/**
 * This converter is used as default one, and used for ops that do not have own special converters
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class GenericOpConverter extends BaseTensorFlowNode {
    @Override
    public TNode asIntermediateRepresentation(NodeDef node, TGraph graph) {
        val tNode = buildBasicNode(node, graph);

        tNode.setOpState(getOpStateFromNodeDef(node, node.getInputCount(), tNode, graph.getVariableSpace()));

        return tNode;
    }

    @Override
    public String opName() {
        return "GenericOpConverter";
    }


    protected OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs) {
        return getOpStateFromNodeDef(tfNode, numInputs, null, null);
    }

    protected OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs, TNode tNode, TVariableSpace variableSpace) {
        String lc = tfNode.getOp().toLowerCase();
        if (lc.equalsIgnoreCase("while"))
            log.info("While found");

        log.debug("Looking for [{}] op...", lc);
        if (numInputs > 0 && numInputs <= 2) {
            int opNum = Nd4j.getOpFactory().getOpNumIfExists(lc);

            if (opNum >= 0) {
                /*
                OpState opState = OpState.builder()
                        .opType(BaseOp.getOpType(Nd4j.getOpFactory().getOpByName(lc)))
                        .opNum(opNum)
                        .opName(lc)
                        .build();
                        */
                val type = BaseOp.getOpType(Nd4j.getOpFactory().getOpByName(lc));

                if (type != Op.Type.SHAPE && type != Op.Type.CUSTOM) {
                    val op = Nd4j.getOpFactory().getOpByName(lc);
                    OpState opState = OpState.builder()
                            .opType(type)
                            .extraArgs(op.extraArgs())
                            .opNum(opNum)
                            .opName(lc)
                            .build();

                    return opState;
                }
            }
        }

        OpState opState = OpState.builder()
                .opType(Op.Type.CUSTOM)
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

            variable.setArray(variable.getArray().permute(3, 2, 0, 1).dup('c'));

            boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

            if (!isSameMode)
                log.debug("Mode: {}", paddingMode);

            log.debug("Conv2D: k: [{}, {}]; s: [{}, {}]; padding: {}", kY, kX, sY, sX,  paddingMode);

            opState.setExtraBits(new int[] {kY, kX, sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0});
        } else if (lc.equalsIgnoreCase("avgpool") || lc.equalsIgnoreCase("maxpool")) {
            val aStrides = tfNode.getAttrOrThrow("strides");
            val tfStrides = aStrides.getList().getIList();
            val sY = tfStrides.get(1);
            val sX = tfStrides.get(2);

            val aKernels = tfNode.getAttrOrThrow("ksize");
            val tfKernels = aKernels.getList().getIList();

            val kY = tfKernels.get(1);
            val kX = tfKernels.get(2);

            val aPadding = tfNode.getAttrOrThrow("padding");

            val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"","");

            boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

            if (!isSameMode)
                log.debug("Mode: {}", paddingMode);

            log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);

            opState.setExtraBits(new int[] {kY.intValue(), kX.intValue(), sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0 });

        } else if (lc.equalsIgnoreCase("lrn")) {
             /*
             val aAlpha = tfNode.getAttrOrThrow("alpha");
             val aBeta = tfNode.getAttrOrThrow("beta");
             val aBias = tfNode.getAttrOrThrow("bias");
             val aDepth = tfNode.getAttrOrThrow("depth_radius");

             val alpha = aAlpha.getF();
             val beta = aBeta.getF();
             val bias = aBias.getF();
             val depth = aDepth.getF();


             opState.setExtraArgs(new Object[]{alpha, beta, bias, depth});
             log.debug("LRN: alpha: {}; beta: {}; bias: {}; depth: {};", alpha, beta, bias, depth);
             */
        } else if (lc.equalsIgnoreCase("reshape")) {
             /*
             // in reshape operation we replace second input, and replace it with extra args
             log.debug("TNode inputs: {}", tNode.getInputs());
             val shapeIndex = tNode.getInputs().remove(1);
             val variable = variableSpace.getVariable(shapeIndex);

             assert variable != null;
             assert variable.getShape() != null;

             // we know that TF is always C order
             int[] args = ArrayUtils.add(variable.getShape(),  0, (int)'c');

             log.debug("Reshape node_{}, new shape: {}", tNode.getId(), Arrays.toString(args));

             // new shape goes here
             opState.setExtraBits(args);
             */
        } else if (lc.equalsIgnoreCase("concat")) {
            /*
            log.debug("TNode inputs: {}", tNode.getInputs());
            TIndex dimIndex;
            int idx = -1;
            int cnt = 0;
            int concatDimension = 0;
            for (val index:tNode.getInputs()) {
                log.debug("Trying to find node: [{}]", index);
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

            // if that's convolution graph, we should swap dimensions
            if (concatDimension == 3)
                concatDimension = 1;

            opState.setExtraBits(new int[]{concatDimension});
            log.debug("Concat dimension: {}", concatDimension);
            */
        }

        if (!Nd4j.getExecutioner().getCustomOperations().containsKey(lc))
            log.warn("Unknown op: [{}]", lc);
        //throw new ND4JIllegalStateException("Unknown operation requested: ["+ tfNode.getOp() +"]");

        return opState;
    }
}
